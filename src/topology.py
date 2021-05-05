# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-05 10:37:47
# @Last Modified: 2021-05-05 09:59:26
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
from hi5 import BetterDict

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
        return h5.load(input_path, "/data/neuron_bridge_ids").astype(
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
