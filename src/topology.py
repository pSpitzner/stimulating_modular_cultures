# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-05 10:37:47
# @Last Modified: 2021-11-05 11:32:51
# ------------------------------------------------------------------------------ #
# Helper to load the topology from hdf5
# ------------------------------------------------------------------------------ #

# %%
import os
import sys
import h5py
import logging
import numpy as np
import hi5 as h5
# from hi5 import BetterDict

log = logging.getLogger(__name__)


def load_topology(input_path):
    """
        Loads variables needed for the topology, mainly connectivity matrix

        # Parameters
            input_path : to .hdf5 file

        # Returns
            num_n       : number of neurons
            a_ij_sparse : the connectivity matrix in sparse form
            mod_ids     : an array to find which module neurons belong to. index -> mod
    """

    log.info(f"Loading connectivity from {input_path} ")

    a_ij_sparse = h5.load(input_path, "/data/connectivity_matrix_sparse")
    a_ij_sparse = a_ij_sparse.astype(int, copy=False)  # brian doesnt like uints
    num_n = int(h5.load(input_path, "/meta/topology_num_neur"))

    # get the neurons sorted according to their modules
    mod_ids = h5.load(input_path, "/data/neuron_module_id")

    # if we would need to sort by modules
    # mods = np.unique(mod_ids)
    # mod_sorted = np.zeros(num_n, dtype=int)
    # temp = np.argsort(mod_ids)
    # for idx in range(0, num_n):
    #     mod_sorted[idx] = np.argwhere(temp == idx)

    # mod_sort = lambda x: mod_sorted[x]

    return num_n, a_ij_sparse, mod_ids


def load_bridging_neurons(input_path):
    try:
        return h5.load(input_path, "/data/neuron_bridge_ids", keepdim=True).astype(
            int, copy=False
        )  # brian doesnt like uints
    except Exception as e:
        log.debug(e)
        return []


def connect_synapses_from(S, a_ij):
    """
        Apply connectivity matrix to brian Synapses object `S`.
        Not used at the moment.
    """
    try:
        log.info("Applying connectivity (non-sparse) ... ")
        pre, post = np.where(a_ij == 1)
        for idx, i in enumerate(pre):
            j = post[idx]
            S.connect(i=i, j=j)
    except Exception as e:
        log.error(e)
        log.info(f"Creating Synapses randomly.")
        S.connect(condition="i != j", p=0.01)


def index_alignment(num_n, num_inhib, bridge_ids):
    """
        resort indices so that they are contiguous:

        * inhibitory, no bridge
        * inhibitory, and bridge neurons
        * excitatiory, and bridge
        * excitatiory, no bridge


    """
    inhib_ids_old = np.sort(np.random.choice(num_n, size=num_inhib, replace=False))
    bridge_ids_old = bridge_ids

    inhib_no_bridge = []
    inhib_bridge = []
    excit_bridge = []
    excit_no_bridge = []

    for n_id in range(0, num_n):
        if n_id in inhib_ids_old:
            if n_id in bridge_ids_old:
                inhib_bridge.append(n_id)
            else:
                inhib_no_bridge.append(n_id)
        else:
            if n_id in bridge_ids_old:
                excit_bridge.append(n_id)
            else:
                excit_no_bridge.append(n_id)
    brian_indices = np.concatenate(
        [
            np.sort(inhib_no_bridge),
            np.sort(inhib_bridge),
            np.sort(excit_bridge),
            np.sort(excit_no_bridge),
        ]
    ).astype("int64")
    assert len(brian_indices) == num_n

    # inverse mapping and resorted inputs
    topo_indices = np.zeros(num_n, dtype="int64")
    bridge_ids_new = []

    for n_id in brian_indices:
        topo_indices[n_id] = np.where(brian_indices == n_id)[0][0]
        if n_id in bridge_ids_old:
            bridge_ids_new.append(n_id)

    inhib_ids_new = brian_indices[0:num_inhib]
    excit_ids_new = brian_indices[num_inhib:]
    bridge_ids_new = np.array(bridge_ids_new, dtype="int64")

    return topo_indices, brian_indices, inhib_ids_new, excit_ids_new, bridge_ids_new


# ------------------------------------------------------------------------------ #
# Orlandi rewrite
# ------------------------------------------------------------------------------ #

try:
    from numba import jit, prange

    # raise ImportError
    log.info("Using numba")

except ImportError:
    log.info("Numba not available")
    # replace numba functions if numba not available:
    # we only use jit and prange
    # helper needed for decorators with kwargs
    def parametrized(dec):
        def layer(*args, **kwargs):
            def repl(f):
                return dec(f, *args, **kwargs)

            return repl

        return layer

    @parametrized
    def jit(func, **kwargs):
        return func

    def prange(*args):
        return range(*args)


# lets define parameters of the algorithm.
# Numba needs dictionaries that have the same data type
# fmt: off
par = dict (
    N   = 160.,        #  number neurons
    rho = -1.,        #  [1/um2] density
    L   = 600.0,        #  [um] linear dish size

    #  axons, variable length, segments of fixed length with variable angle
    std_l   = 800.0,  #  [um]  st. dev. of rayleigh dist. for axon len
                      #  expectation value: <l> = std_l*sqrt(pi/2)
    max_l   = 1500.0, #  [um]  max lentgh allowed for axons
    del_l   = 10.0,   #  [um]  length of each segment of axons
    std_phi = 0.1,    #  [rad] std of Gauss dist. for angles betw segs

    #  soma, hard sphere
    R_s     = 7.5,    #  [um] radius of soma

    #  dendritic tree, sphere with variable radius
    mu_d    = 150.,   #  [um] mean of Gauss dist. for radius
    std_d   = 20.0,   #  [um] std of Gauss dist. for radius

    #  connection probability alpha when axons intersect dendritic tree
    alpha = .5,

    # after hitting a wall, retry placing axon segment, default: 50
    axon_retry = 5000,

    # after hitting a wall, we multiply the next random angle with this value
    # so the path is a bit more flexible. default: 5
    angle_mod  = 5.0,
)

# fmt: on


def _grow_axon(par, start, end):
    pass


@jit(nopython=True)
def _soma_is_within_substrate(x, y, r=0):
    x_in = False
    y_in = False

    # default modular cultures are four modules of size 200 um with 200 um padding
    mods = np.array([[0.0, 200.0], [400.0, 600.0]])

    for m in range(0, len(mods)):
        if x >= mods[m][0] + r and x <= mods[m][1] - r:
            x_in = True
        if y >= mods[m][0] + r and y <= mods[m][1] - r:
            y_in = True

    return x_in and y_in

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _is_intersecting_circles(ref_pos, ref_rad, x, y, r=0):
    """
        Check whether x, y, is intersecting with any of the circles at ref_pos with
        radiu ref_rad
    """
    for i in range(ref_pos.shape[0]):
        x_, y_ = ref_pos[i, :]
        r_ = ref_rad[i]
        if (x-x_)*(x-x_) + (y-y_)*(y-y_) < (r+r_)*(r+r_):
            return True

    return False



@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def place_neurons(par_N, par_L, par_R_s, is_in_substrate):
    """
        make sure to set parameters correctly before calling this
    """

    rejections = 0

    # a 2d np array with x, y poitions, soma_radius and dendritic_tree diameter
    neuron_pos = np.ones(shape=(int(par_N), 2)) * - 93288.0
    neuron_rad = np.ones(shape=(int(par_N))) * par_R_s

    for i in range(int(par_N)):
        placed = False
        while (not placed):
            placed = True
            x = np.random.uniform(0.0, par_L)
            y = np.random.uniform(0.0, par_L)
            # print(i, x, y)
            if not is_in_substrate(x, y):
                placed = False
                rejections += 1
                continue
            if _is_intersecting_circles(neuron_pos[0:i], neuron_rad[0:i], x, y, par_R_s):
                print(i, "intersecting")
                placed = False
                rejections += 1
                continue
            neuron_pos[i, 0] = x
            neuron_pos[i, 1] = y

    return neuron_pos


# import plot_helper as ph
# neuron_pos = place_neurons(par["N"], par["L"], par["R_s"], _soma_is_within_substrate)
# fig, ax = plt.subplots()
# ph._circles(neuron_pos[:,0], neuron_pos[:,1], 7.5, ax=ax, fc="white", ec="black", alpha=1, lw=0.25, zorder=4)















