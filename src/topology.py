# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-05 10:37:47
# @Last Modified: 2021-11-17 10:58:59
# ------------------------------------------------------------------------------ #
# Helper to load the topology from hdf5
# ------------------------------------------------------------------------------ #

# %%
import os
import sys
import h5py
import logging
import functools
import numpy as np
import hi5 as h5

from benedict import benedict

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

    raise ImportError
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
# Numba does not like dicts, so we use global variables
# with a prefix. these guys should not change much!

par_N = 160  #  number neurons
par_L = 600.0  #  [um] maximum linear dish size

# axons, variable length, segments of fixed length with variable angle
par_std_l = 800.0  #  [um]  st. dev. of rayleigh dist. for axon len
# expectation value: <l> = std_l*sqrt(pi/2)
par_max_l = 1500.0  #  [um]  max lentgh allowed for axons
par_del_l = 10.0  #  [um]  length of each segment of axons
par_std_phi = 0.1  #  [rad] std of Gauss dist. for angles betw segs

# soma, hard sphere
par_R_s = 7.5  #  [um] radius of soma

# dendritic tree, sphere with variable radius
par_mu_d = 150.0  #  [um] mean of Gauss dist. for radius
par_std_d = 20.0  #  [um] std of Gauss dist. for radius

# whether to scale connection porbability with the amount of intersection
# between axon and dendritic tree
par_alpha_path_weighted = True

# connection probability alpha when axons intersect dendritic tree
if par_alpha_path_weighted:
    par_alpha = 0.0125
else:
    par_alpha = 0.33

# after hitting a wall, retry placing axon segment, default: 50
par_axon_retry = 5000

# after hitting a wall, we multiply the next random angle with this value
# so the path is a bit more flexible. default: 5
par_angle_mod = 5.0

# coordinates of rectangular modules: module_number, x | y, low | high
par_mods = np.ones(shape=(4, 2, 2)) * np.nan
par_mods[0] = np.array([[0, 200], [0, 200]])
par_mods[1] = np.array([[0, 200], [400, 600]])
par_mods[2] = np.array([[400, 600], [0, 200]])
par_mods[3] = np.array([[400, 600], [400, 600]])


aij_nested = None
def init_modular_topology(num_bridging_axons=5):
    global aij_nested

    # store everything in a nested dictionary, optionally with descriptions
    # so we can save to disk later as hdf5
    h5_data = benedict()
    h5_desc = benedict()

    neuron_pos = place_neurons()

    # it is convenient to have indices sorted by modules and position
    neuron_mid = np.ones(par_N, dtype="int") * -1
    for idx in range(0, par_N):
        x, y = neuron_pos[idx]
        neuron_mid[idx] = _mod_id_from_coordinate(x, y)

    sort_idx = np.lexsort((neuron_pos[:, 1], neuron_pos[:, 0], neuron_mid))
    neuron_pos = neuron_pos[sort_idx]
    neuron_mid = neuron_mid[sort_idx]

    # dendritic trees as circles
    neuron_den = draw_dendrite_radii()

    h5_data["data.neuron_pos_x"] = neuron_pos[:, 0]
    h5_data["data.neuron_pos_y"] = neuron_pos[:, 1]
    h5_data["data.neuron_module_id"] = neuron_mid[:]
    h5_data["data.neuron_radius_dendritic_tree"] = neuron_den

    # sparse connectivity matrix
    aij_nested = [np.array([], dtype="int")] * par_N

    # paths of all axons
    axon_paths = [np.array([])] * par_N

    # which neurons create bridges
    bridge_ids = []

    # generate and connect bridging axons
    # e.g. (0, 1, 2): "from 0 to 1 and 2"
    for b_ids in [(0, 1, 2), (1, 0, 3), (2, 0, 3), (3, 1, 2)]:
        src_mod = par_mods[b_ids[0]]
        tar_mods = [
            par_mods[b_ids[1]],
            par_mods[b_ids[2]],
        ]

        # get the ids and axon paths of all neurons that bridge between modules
        br_nids, br_paths = grow_bridging_axons(
            neuron_pos=neuron_pos,
            source_mod=src_mod,
            target_mods=tar_mods,
            num_bridging_axons=num_bridging_axons,
        )
        bridge_ids.extend(br_nids)

        # get connectivity for each neuron
        for idx, n_id in enumerate(br_nids):
            cids = get_connections_for_neuron(
                neuron_pos, neuron_den, n_id=n_id, axon_path=br_paths[idx]
            )
            axon_paths[n_id] = br_paths[idx]
            aij_nested[n_id] = np.sort(cids)

    h5_data["data.neuron_bridge_ids"] = np.array(bridge_ids, dtype="int")
    h5_desc["data.neuron_bridge_ids"] = "list of ids of neurons connecting two modules"

    # grow axons and get connections for the remaining neurons
    for n_id in range(0, par_N):
        if n_id in bridge_ids:
            continue

        path = grow_axon(start=neuron_pos[n_id, :])
        axon_paths[n_id] = path
        cids = get_connections_for_neuron(
            neuron_pos, neuron_den, n_id=n_id, axon_path=path
        )
        aij_nested[n_id] = np.sort(cids)

    k_in, k_out = _get_degrees(aij_nested)
    h5_data["data.neuron_k_in"] = k_in
    h5_data["data.neuron_k_out"] = k_out

    h5_data["data.connectivity_matrix_sparse"] = _nested_lists_to_sparse(aij_nested)
    h5_desc["data.connectivity_matrix_sparse"] = "first column is the id of the source neuron, second column is the id of the target neuron"

    # restructuring to a nan-padded 2d matrix allows us to save all the paths
    # into one dataset. (the padding does not cost storage on disc when compressing.)
    nan_padded_segments = _nested_lists_to_2d_nan_padded(axon_paths)
    h5_data["data.neuron_axon_segments_x"] = nan_padded_segments[:, :, 0]
    h5_data["data.neuron_axon_segments_y"] = nan_padded_segments[:, :, 1]

    h5_data["data.neuron_axon_length"] = np.array([len(x)*par_del_l for x in axon_paths])
    h5_data["data.neuron_axon_end_to_end_distance"] = _get_end_to_end_distances(axon_paths)


    return h5_data, h5_desc

def get_meta_data():
    h5_data = benedict()
    h5_data["meta.topology"] = "orlandi"
    h5_data["meta.topology_num_neur"] = par_N
    # h5_data["meta.topology_num_outgoing"] = todo
    h5_data["meta.topology_sys_size"] = par_L
    h5_data["meta.topology_alpha"] = par_alpha
    h5_data["meta.topology_alpha_is_weighted"] = int(par_alpha_path_weighted)
    # h5_data["meta.topology_k_inter"] = num_bridging_axons

    return h5_data

# x "data.neuron_pos_x"
# x "data.neuron_pos_y"
# "data.neuron_axon_length"
# "data.neuron_axon_end_to_end_distance"
# x "data.neuron_radius_dendritic_tree"
# x "data.neuron_bridge_ids"
# x "data.neuron_module_id"
# "data.neuron_k_in"
# "data.neuron_k_out"
# "/data/connectivity_matrix_sparse"

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def grow_axon(start, num_segments=None, start_phi=None):

    assert num_segments is None or num_segments > 0

    # we might not succeed placing all axons, retry full axon
    tries_to_grow = 0
    while tries_to_grow < par_axon_retry:
        tries_to_grow += 1

        last_x = start[0]
        last_y = start[1]

        if num_segments is None:
            length = np.random.rayleigh(par_std_l)
            num_segs = int(length / par_del_l)
        else:
            num_segs = num_segments

        if start_phi is None:
            last_phi = np.random.uniform(0, 1,) * 2 * np.pi
        else:
            last_phi = start_phi

        path = np.ones((num_segs, 2)) * np.nan

        # place segments
        sdx = 0
        for sdx in range(0, num_segs):
            placed = False
            tries_to_place = 0
            angle_mod = 1.0

            # retry placing segments
            while not placed and tries_to_place < par_axon_retry:
                placed = True
                tries_to_place += 1
                phi = last_phi + np.random.normal(0, par_std_phi) * angle_mod
                x = last_x + par_del_l * np.cos(phi)
                y = last_y + par_del_l * np.sin(phi)

                # check confinement, possibly allow larger bending angles at walls
                if not _is_within_substrate(x, y, 0):
                    placed = False
                    angle_mod = par_angle_mod
                    continue

            if placed:
                # update last coordinates
                last_x = x
                last_y = y
                last_phi = phi
                path[sdx, 0] = last_x
                path[sdx, 1] = last_y
            else:
                # failed to place segment within given number of attempts,
                # retry whole axon
                break

        # we succeeded growing the whole axon if all segments placed
        if sdx == num_segs - 1:
            break

    if tries_to_grow == par_axon_retry:
        raise Exception("Failed to grow axon with specified retries")

    return path

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def grow_axon_to_target(start, target, termination_criterion):

    last_x = start[0]
    last_y = start[1]

    tries_to_grow = 0
    while not termination_criterion(last_x, last_y) and tries_to_grow < par_axon_retry:
        tries_to_grow += 1

        length = np.random.rayleigh(par_std_l)
        num_segs = int(length / par_del_l)
        path = np.ones((num_segs, 2)) * np.nan

        last_x = start[0]
        last_y = start[1]

        # draw phi until it goes in the right direction
        for i in range(0, par_axon_retry):
            phi = np.random.uniform(-0.5, 0.5,) * 2 * np.pi
            if _is_angle_on_course(start, target, phi):
                last_phi = phi
                break

        # place segments
        sdx = 0
        for sdx in range(0, num_segs):
            # unless we have reached our target
            if termination_criterion(last_x, last_y):
                sdx -= 1
                break

            placed = False
            tries_to_place = 0

            while not placed and tries_to_place < par_axon_retry:
                placed = True
                tries_to_place += 1
                phi = last_phi + np.random.normal(0, par_std_phi)
                x = last_x + par_del_l * np.cos(phi)
                y = last_y + par_del_l * np.sin(phi)

                # check that we are not deflecting too far from a path towards the target
                if not _is_angle_on_course([last_x, last_y], target, phi):
                    placed = False
                    continue

            if placed:
                # update last coordinates
                last_x = x
                last_y = y
                last_phi = phi
                path[sdx, 0] = last_x
                path[sdx, 1] = last_y
            else:
                # failed to place segment within given number of attempts
                break

    num_segs_left = num_segs - sdx - 1
    return path, num_segs_left, last_phi


def grow_bridging_axons(neuron_pos, source_mod, target_mods, num_bridging_axons=5):

    # get the indices of neurons in the target module
    source_idx = [_is_within_rect(x, y, 0, source_mod) for x, y in neuron_pos]
    source_idx = np.where(source_idx)[0]
    source_pos = neuron_pos[source_idx]

    # get an idea where the modules are located. assuming some clustering in space,
    # then the center of mass is a good proxy... or we use the center of modules
    source_cm = _center_of_rect(source_mod)

    # pick the source closes to cm
    source_distances = np.sqrt(
        np.power(source_pos[:, 0] - source_cm[0], 2.0)
        + np.power(source_pos[:, 1] - source_cm[1], 2.0)
    )
    closest_idx = source_idx[np.argsort(source_distances)]

    rewired_idx = []
    rewired_paths = []

    # criteria when an axons has reached another module
    # we need a bit of boilerplate to get preset arguments into numba functions
    def make_criterion(mod):
        @jit(nopython=True, parallel=False, fastmath=True, cache=True)
        def f(x, y):
            # r=5um so we have a bit of a margin around the border, before accepting
            return _is_within_rect(x, y, r=5, rect=mod)

        return f

    termination_criteria = []
    for mod in target_mods:
        termination_criteria.append(make_criterion(mod))

    for k in range(0, num_bridging_axons):
        for mdx, mod in enumerate(target_mods):
            n_offset = len(rewired_idx)
            # ensure ~ 5um padding at module edges before terminating
            # f_term = functools.partial(_is_within_rect, r=5, rect=mod)
            criterion = termination_criteria[mdx]
            n_idx = closest_idx[n_offset]
            print(n_idx, neuron_pos[n_idx])
            if n_idx == 58:
                global debug_print
                debug_print = True

            # start by growing to our target
            path, num_segs_left, last_phi = grow_axon_to_target(
                start=neuron_pos[n_idx],
                target=_center_of_rect(mod),
                termination_criterion=criterion,
            )

            # debug
            # if not np.isfinite(path[-num_segs_left - 1][0]):
            #     print("\t", num_segs_left, path[-num_segs_left - 1])
            #     print(path)

            if num_segs_left > 0:
                # finish randomly
                last_pos = path[-num_segs_left - 1]
                rest_of_path = grow_axon(
                    start=last_pos, num_segments=num_segs_left, start_phi=last_phi
                )

                path[-num_segs_left:] = rest_of_path
            assert np.all(np.isfinite(path))

            rewired_idx.append(n_idx)
            rewired_paths.append(path)

    return rewired_idx, rewired_paths


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def get_connections_for_neuron(neuron_pos, neuron_dentrite_size, n_id, axon_path):
    """
        Get the connections due to a grown path

        # Parameters
        neuron_pos : 2d array
            positions of all neurons, x and y
        neuron_rad : array
            radii of the circular dendritic tree
        n_id : int
            the id of the source neuron
        axon_path : 2d array
            positions of all axons segments growing from n_id

        # Returns:
        ids_to_connect_with : array
            indices of neurons to form a connection with
    """

    num_intersections = np.zeros(len(neuron_pos), "int")

    src_mod = _mod_id_from_coordinate(*neuron_pos[n_id])

    # todo: this should become a class attribute
    tar_mods = np.ones(par_N, dtype="int") * -1
    for idx in range(0, par_N):
        x, y = neuron_pos[idx]
        tar_mods[idx] = _mod_id_from_coordinate(x, y)

    # print(tar_mods)

    for seg in axon_path:
        if not _is_within_substrate(seg[0], seg[1], r=0):
            # we dont count intersection that occur out of our substrate,
            # because we want dendritic trees to only be "on the substrate"
            continue

        intersecting_dendrites = _are_intersecting_circles(
           ref_pos=neuron_pos, ref_rad=neuron_dentrite_size, x=seg[0], y=seg[1]
        )

        # we also do not want to connect to a hypothetical super large tree
        # that comes from another module
        seg_mod = _mod_id_from_coordinate(*seg)
        valid_dendrites = tar_mods == seg_mod

        # only count intersections where valid and intersecting
        num_intersections += intersecting_dendrites & valid_dendrites

    # dont create self-connections:
    num_intersections[n_id] = 0
    idx = np.where(num_intersections > 0)[0]

    if not par_alpha_path_weighted:
        # flat probability, independent of the number of intersections
        idx = idx[np.random.uniform(0, 1, size=len(idx)) < par_alpha]
    else:
        # probability is applied once, for every intersecting segment
        jdx = []
        for i in idx:
            if np.any(np.random.uniform(0, 1, size=num_intersections[i]) < par_alpha):
                jdx.append(i)
        idx = np.array(jdx)

    return idx


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _center_of_rect(rect):
    x_center = rect[0][0] + (rect[0][1] - rect[0][0]) / 2
    y_center = rect[1][0] + (rect[1][1] - rect[1][0]) / 2
    return x_center, y_center


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _is_within_rect(x, y, r, rect):
    # rect is a 2d array x:low|high, y:low|high
    if x < rect[0][0] + r or x > rect[0][1] - r:
        return False
    if y < rect[1][0] + r or y > rect[1][1] - r:
        return False
    return True


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _is_within_substrate(x, y, r=0):
    for m in range(0, 4):
        if _is_within_rect(x, y, r, par_mods[m]):
            return True
    return False


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _is_angle_on_course(source_pos, target_pos, phi, threshold=0.1):
    src_x, src_y = source_pos
    tar_x, tar_y = target_pos

    atan = np.arctan2(tar_y - src_y, tar_x - src_x)
    # print(f"\t\t{tar_x - src_x:.2f} {tar_y - src_y:.2f} | {phi:.2f} {atan:.2f} | {phi-atan:.2f}")

    if np.fabs(phi - atan) > np.pi * threshold:
        return False

    return True


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _are_intersecting_circles(ref_pos, ref_rad, x, y, r=0):
    """
        Check whether x, y, is intersecting with any of the circles at ref_pos with
        radius ref_rad
    """
    x_ = ref_pos[:, 0]
    y_ = ref_pos[:, 1]
    r_ = ref_rad[:]
    return np.square(x - x_) + np.square(y - y_) < np.square(r + r_)


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _mod_id_from_coordinate(x, y):
    num_mods = len(par_mods)
    for i in range(0, num_mods):
        if _is_within_rect(x, y, 0, par_mods[i]):
            return i
    raise Exception("Coordinate is not within any known module")

def _nested_lists_to_2d_nan_padded(nested):
    """
        For segment paths,
        convert a list of lists/arrays (with varying length)
        into a 2d array of shape (num_lists, max_len)
        where missing elements are set to np.nan

        nested : list of 2d arrays with shape (num_segments, 2)
    """

    max_len = np.max([len(x) for x in nested])
    res = np.ones(shape=(len(nested), max_len, 2))*np.nan
    for idx, l in enumerate(nested):
        if len(l) > 0:
            res[idx, 0:len(l)] = l

    return res

def _nested_lists_to_sparse(nested):
    """
        Mostly for connectivity matrix,
        when stored as list of listes with variable lengths,
        convert to a sparse matrix like
        `i=a_ij_sparse[:, 0], j=a_ij_sparse[:, 1]`
    """
    sources = []
    targets = []
    for idx, l in enumerate(nested):
        sources.extend([idx]*len(l))
        targets.extend(l)

    res = np.array([sources, targets]).T
    return res

def _get_degrees(aij_nested):
    k_in = np.zeros(len(aij_nested), dtype="int")
    k_out = np.zeros(len(aij_nested), dtype="int")
    for n_id in range(0, len(aij_nested)):
        targets = aij_nested[n_id]
        k_out[n_id] = len(targets)
        k_in[targets] += 1

    return k_in, k_out

def _get_degrees_from_sparse(aij_sparse):
    k_in = np.zeros(len(aij_sparse), dtype="int")
    k_out = np.zeros(len(aij_sparse), dtype="int")
    for connection in aij_sparse:
        print(connection)
        k_in[connection[1]] += 1
        k_out[connection[0]] += 1
    return k_in, k_out

def _get_end_to_end_distances(axon_paths):

    res = np.zeros(len(axon_paths))
    for idx, path in enumerate(axon_paths):
        pos1 = path[0]
        pos2 = path[-1]
        res[idx] = np.sqrt(np.square(pos1[0] - pos2[0]) + np.square(pos1[1] - pos2[1]))

    return res


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def place_neurons():
    """
        make sure to set parameters correctly before calling this
    """

    rejections = 0

    # a 2d np array with x, y poitions, soma_radius and dendritic_tree diameter
    neuron_pos = np.ones(shape=(par_N, 2)) * -93288.0
    neuron_rad = np.ones(shape=(par_N)) * par_R_s

    # keep track of the number of neurons in every module
    neurons_per_module = np.zeros(len(par_mods), "int")

    for i in range(int(par_N)):
        placed = False
        while not placed:
            placed = True
            x = np.random.uniform(0.0, par_L)
            y = np.random.uniform(0.0, par_L)

            # dont place out of bounds
            if not _is_within_substrate(x, y, par_R_s):
                placed = False
                rejections += 1
                continue

            # dont overlap other neurons
            if np.any(
                _are_intersecting_circles(
                    neuron_pos[0:i], neuron_rad[0:i], x, y, par_R_s
                )
            ):
                placed = False
                rejections += 1
                continue

            # for modular topology, we want to ensure the same number of neurons
            # in every module
            mod_id = _mod_id_from_coordinate(x, y)
            if neurons_per_module[mod_id] != np.min(neurons_per_module):
                placed = False
                rejections += 1
                continue

            neuron_pos[i, 0] = x
            neuron_pos[i, 1] = y
            neurons_per_module[mod_id] += 1

    return neuron_pos


def draw_dendrite_radii():
    return np.random.normal(loc=par_mu_d, scale=par_std_d, size=par_N)


import plot_helper as ph


# numba has its own rng instance
@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _set_seed(seed):
    np.random.seed(seed)

def set_seed(seed):
    # we just want some sort of reproducibility, so set once out of numba
    np.random.seed(seed)
    # and once within numba
    _set_seed(seed)


set_seed(np.random.random_integers(0, 1000))
# set_seed(19)
h5_data, h5_desc = init_modular_topology()
h5_data.merge(get_meta_data())
import plot_helper as ph
h5_data = ph.ah.prepare_file(h5_data)
ph.overview_topology(h5_data)

import plot_helper as ph
import seaborn as sns

h5f = ph.ah.prepare_file("./dat/inhibition_sweep_rate_160/dyn/stim=off_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=90.0_rep=000.hdf5", hot=False)

fig, ax = plt.subplots()
sns.histplot(h5_data["data.connectivity_matrix_sparse"][:][:, 0], discrete=True, color="C0")
sns.histplot(h5_data["data.connectivity_matrix_sparse"][:][:, 1], discrete=True, color="C1")

fig, ax = plt.subplots()
sns.histplot(h5f["data.connectivity_matrix_sparse"][:][:, 0], discrete=True, color="C0")
sns.histplot(h5f["data.connectivity_matrix_sparse"][:][:, 1], discrete=True, color="C1")
ax.set_title("working")

ph.overview_topology(h5f)

# neuron_pos = place_neurons()
# dendr_rads = draw_dendrite_radii(par_N, par_mu_d, par_std_d)

# fig, ax = plt.subplots()
# ph._circles(
#     neuron_pos[:, 0],
#     neuron_pos[:, 1],
#     7.5,
#     ax=ax,
#     fc="white",
#     ec="black",
#     alpha=1,
#     lw=0.25,
#     zorder=4,
# )

# for n in range(0, par_N):
#     path = grow_axon(neuron_pos[n, :])
#     ax.plot(path[:, 0], path[:, 1], color="black", lw=0.35, zorder=0, alpha=0.5)

# nids, paths = grow_bridging_axons(neuron_pos, par_mods[0], [par_mods[1], par_mods[2]])

# nsel = 0
# cids = get_connections_for_neuron(
#     neuron_pos, dendr_rads, n_id=nids[nsel], axon_path=paths[nsel]
# )

# ph._circles(
#     neuron_pos[nids[nsel], 0],
#     neuron_pos[nids[nsel], 1],
#     6.5,
#     ax=ax,
#     fc="red",
#     ec=None,
#     zorder=5,
#     lw=0.25,
# )
# ax.plot(paths[nsel][:, 0], paths[nsel][:, 1], color="red", lw=1, zorder=1, alpha=0.8)

# ph._circles(
#     neuron_pos[cids, 0],
#     neuron_pos[cids, 1],
#     6.5,
#     ax=ax,
#     fc="red",
#     ec=None,
#     alpha=0.5,
#     zorder=5,
#     lw=0.25,
# )


# for i in range(0, 15):
# path, num_left = grow_axon_to_target(_center_of_rect(par_mods[0]), _center_of_rect(par_mods[1]), f)
# ax.plot(path[:,0], path[:,1])
