# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-05 10:37:47
# @Last Modified: 2022-12-16 12:57:52
# ------------------------------------------------------------------------------ #
# Helper classes to simulate axonal growth described in
# Orlandi et al. 2013, DOI: 10.1038/nphys2686
# with additional constraints to confine axons and optionally,
# create modular topologies.
#
# Example
# ```python
# # check what is avaiable in terms of parameters
# MergedTopology().get_parameters()
#
# # default is 160 neurons
# tp = MergedTopology(par_N=100)
#
# tp = ModularTopology(par_k_inter=5)
# bridge_ids = tp.neuron_bridge_ids
# num_n = tp.par_N
# a_ij_sparse = tp.aij_sparse
# mod_ids = tp.neuron_module_ids
# h5_data, h5_desc = tp.get_everything_as_nested_dict(return_descriptions=True)
# ```
# ------------------------------------------------------------------------------ #

import os
import sys
import h5py
import logging
import functools
import numpy as np

from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.auto import tqdm
from benedict import benedict

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")

try:
    from numba import jit, prange

    raise ImportError
    log.info("Using numba")

except ImportError:
    log.info("Not using Numba")
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


class BaseTopology(object):
    def __init__(self, **kwargs):
        """
        all kwargs are assumed to be parameters and overwrite defaults.
        to see what parameters are avaiable, you could check a default instance:
        ```
        BaseTopology().get_parameters()
        ```

        """
        self.set_default_parameters()
        self.update_parameters(**kwargs)
        # per default, we have a quadratic substrate of size par_L
        self._substrate = np.array([[0, self.par_L], [0, self.par_L]])
        log.info("Base Topology")
        for k, v in self.get_parameters().items():
            log.info(f"{k}: {v}")
        self.init_topology()

    def set_default_parameters(self):
        # lets define parameters of the algorithm.
        # Numba does not like dicts, so we use global variables
        # with a prefix. these guys should not change much!

        self.par_N = 160  #  number neurons
        self.par_L = 600.0  #  [um] maximum linear dish size

        # axons, variable length, segments of fixed length with variable angle
        self.par_std_l = 800.0  #  [um]  st. dev. of rayleigh dist. for axon len
        # expectation value: <l> = std_l*sqrt(pi/2)
        self.par_max_l = 1500.0  #  [um]  max lentgh allowed for axons
        self.par_del_l = 10.0  #  [um]  length of each segment of axons
        self.par_std_phi = 0.1  #  [rad] std of Gauss dist. for angles betw segs

        # soma, hard sphere
        self.par_R_s = 7.5  #  [um] radius of soma

        # dendritic tree, sphere with variable radius
        self.par_mu_d = 150.0  #  [um] mean of Gauss dist. for radius
        self.par_std_d = 20.0  #  [um] std of Gauss dist. for radius

        # whether to scale connection porbability with the amount of intersection
        # between axon and dendritic tree
        self.par_alpha_path_weighted = True

        # connection probability alpha when axons intersect dendritic tree
        if self.par_alpha_path_weighted:
            self.par_alpha = 0.0125
        else:
            self.par_alpha = 0.33

        # after hitting a wall, retry placing axon segment, default: 50
        self.par_axon_retry = 5000

        # after hitting a wall, we multiply the next random angle with this value
        # so the path is a bit more flexible. default: 5
        self.par_angle_mod = 5.0

    def update_parameters(self, **kwargs):
        available_pars = self.get_parameters().keys()
        for key in kwargs.keys():
            assert key in available_pars, f"Unrecognized parameter `{key}`"
            setattr(self, key, kwargs[key])

    def init_topology(self):
        self.set_neuron_positions()

        # it is convenient to have indices sorted by position
        sort_idx = np.lexsort(
            (
                self.neuron_positions[:, 1],
                self.neuron_positions[:, 0],
            )
        )
        self.neuron_positions = self.neuron_positions[sort_idx]

        # dendritic trees as circles
        self.set_dendrite_radii()

        # every source neuron has a list of target neuron ids
        aij_nested = [np.array([], dtype="int")] * self.par_N
        # which are the axon segments that made the connections? (ids)
        segs_nested = [np.array([], dtype="int")] * self.par_N

        # paths of all axons
        axon_paths = [np.array([])] * self.par_N

        # grow axons and get connections for all neurons
        for n_id in range(0, self.par_N):
            path = self.grow_axon(start=self.neuron_positions[n_id, :])
            axon_paths[n_id] = path
            cids, seg_ids = self.connections_for_neuron(n_id=n_id, axon_path=path)

            argsorted = np.argsort(cids)
            aij_nested[n_id] = cids[argsorted]
            segs_nested[n_id] = seg_ids[argsorted]

        # save the details as attributes
        self.axon_paths = axon_paths
        # this has three columns
        aij_sparse = _nested_lists_to_sparse(aij_nested, vals=segs_nested)
        # vanilla sparse matrix has two columns
        self.aij_sparse = aij_sparse[:, 0:2]
        self.aij_nested = aij_nested
        # save the third column separately
        self.segs_sparse = aij_sparse[:, 2]
        self.k_in, self.k_out = _get_degrees(aij_nested)

    def get_parameters(self):
        res = self.__dict__
        res = {key: res[key] for key in res.keys() if key[0:4] == "par_"}
        return res

    def get_everything_as_nested_dict(self, return_descriptions=False):
        """get most relevant details of the topology as a benedict"""
        # fmt: off
        h5_data = benedict()
        h5_desc = benedict()
        h5_data["data.neuron_pos_x"] = self.neuron_positions[:, 0]
        h5_data["data.neuron_pos_y"] = self.neuron_positions[:, 1]
        h5_data["data.neuron_radius_dendritic_tree"] = self.dendritic_radii


        h5_data["data.connectivity_matrix_sparse"] = self.aij_sparse
        h5_desc["data.connectivity_matrix_sparse"] = "first column is the id of the source neuron, second column is the id of the target neuron"

        h5_data["data.connectivity_segments"] = self.segs_sparse
        h5_desc["data.connectivity_segments"] = "this is the third column to connectivity_matrix_sparse, which holds the id of the axon segment that made the connection"

        h5_data["data.neuron_k_in"] = self.k_in
        h5_data["data.neuron_k_out"] = self.k_out

        # restructuring to a nan-padded 2d matrix allows us to save all the paths
        # into one dataset. (the padding does not cost storage on disc when compressing.)
        nan_padded_segments = _nested_lists_to_2d_nan_padded(self.axon_paths)
        h5_data["data.neuron_axon_segments_x"] = nan_padded_segments[:, :, 0]
        h5_data["data.neuron_axon_segments_y"] = nan_padded_segments[:, :, 1]

        h5_data["data.neuron_axon_length"] = np.array(
            [len(x) * self.par_del_l for x in self.axon_paths]
        )
        h5_data["data.neuron_axon_end_to_end_distance"] = _get_end_to_end_distances(
            self.axon_paths
        )

        h5_data["meta.topology"] = "orlandi base topology"
        h5_data["meta.topology_num_neur"] = self.par_N
        h5_data["meta.topology_num_outgoing"] = np.mean(self.k_out)
        h5_data["meta.topology_sys_size"] = self.par_L
        h5_data["meta.topology_alpha"] = self.par_alpha
        h5_data["meta.topology_alpha_is_weighted"] = int(self.par_alpha_path_weighted)
        # fmt: on

        if return_descriptions:
            return h5_data, h5_desc
        return h5_data

    def set_neuron_positions(self):
        """
        make sure to set parameters correctly before calling this
        """

        rejections = 0

        # a 2d np array with x, y poitions, soma_radius and dendritic_tree diameter
        neuron_pos = np.ones(shape=(self.par_N, 2)) * -93288.0
        neuron_rad = np.ones(shape=(self.par_N)) * self.par_R_s

        for i in range(self.par_N):
            placed = False
            while not placed:
                placed = True
                x = np.random.uniform(0.0, self.par_L)
                y = np.random.uniform(0.0, self.par_L)

                # dont place out of bounds
                if not self.is_within_substrate(x, y, self.par_R_s):
                    placed = False
                    rejections += 1
                    continue

                # dont overlap other neurons
                if np.any(
                    _are_intersecting_circles(
                        neuron_pos[0:i], neuron_rad[0:i], x, y, self.par_R_s
                    )
                ):
                    placed = False
                    rejections += 1
                    continue

                neuron_pos[i, 0] = x
                neuron_pos[i, 1] = y

        self.neuron_positions = neuron_pos

    # lets have some overloads so we can use the generic helper functions
    # from class instances
    def grow_axon(self, start, num_segments=None, start_phi=None):
        return _grow_axon(
            self.par_std_l,
            self.par_del_l,
            self.par_axon_retry,
            self.par_std_phi,
            self.par_angle_mod,
            self.is_within_substrate,
            start=start,
            num_segments=num_segments,
            start_phi=start_phi,
        )

    def grow_axon_to_target(self, start, target, termination_criterion):
        return _grow_axon_to_target(
            self.par_std_l,
            self.par_del_l,
            self.par_axon_retry,
            self.par_std_phi,
            start,
            target,
            termination_criterion,
        )

    def set_dendrite_radii(self):
        self.dendritic_radii = np.random.normal(
            loc=self.par_mu_d, scale=self.par_std_d, size=self.par_N
        )

    def is_within_substrate(self, x, y, r=0):
        return _is_within_rect(x, y, r, self._substrate)

    def connections_for_neuron(self, n_id, axon_path):
        """
        Get the connections due to a grown path
        n_id : int
            the id of the source neuron
        axon_path : 2d array
            positions of all axons segments growing from n_id

        # Returns:
        ids_to_connect_with : array
            indices of neurons to form a connection with
        """

        # first, collect all dendritic trees which we intersect and how often
        num_intersections = np.zeros(self.par_N, "int")
        first_intersections = np.zeros(self.par_N, "int")

        for seg_id, seg in enumerate(axon_path):
            if not self.is_within_substrate(seg[0], seg[1], r=0):
                # we dont count intersection that occur out of our substrate,
                # because we want dendritic trees to only be "on the substrate"
                continue

            intersecting_dendrites = _are_intersecting_circles(
                ref_pos=self.neuron_positions,
                ref_rad=self.dendritic_radii,
                x=seg[0],
                y=seg[1],
            )

            # intersections by dendrite
            firsts = np.where(num_intersections == 0 & intersecting_dendrites)[0]
            first_intersections[firsts] = seg_id
            num_intersections += intersecting_dendrites

        num_intersections[n_id] = 0  # dont create self-connections
        t_ids = np.where(num_intersections > 0)[0]  # target neuron ids
        connection_segments = []  # which segment forms the connection

        if not self.par_alpha_path_weighted:
            # flat probability, independent of the number of intersections
            t_ids = t_ids[np.random.uniform(0, 1, size=len(t_ids)) < self.par_alpha]

            # find the segment that causes the connection. simply the first intersecting
            connection_segments = first_intersections[t_ids]

        else:
            # probability is applied once, for every intersecting segment
            filtered_t_ids = []
            for tid in t_ids:
                intersecting = (
                    np.random.uniform(0, 1, size=num_intersections[tid]) < self.par_alpha
                )

                if np.any(intersecting):
                    filtered_t_ids.append(tid)
                    # axons may grow out of trees and back in. the first index in
                    # of intersecing segs may not actually be the right one (or in tree)
                    ref = np.sum(intersecting)
                    for seg_id in range(first_intersections[tid], len(axon_path)):
                        seg = axon_path[seg_id]
                        if _is_intersecting_circle(
                            ref_pos=self.neuron_positions[tid],
                            ref_rad=self.dendritic_radii[tid],
                            x=seg[0],
                            y=seg[1],
                        ):
                            if ref == 0:
                                break
                            ref -= 1
                    connection_segments.append(seg_id)

            t_ids = np.array(filtered_t_ids, dtype="int")
            connection_segments = np.array(connection_segments, dtype="int")

        return t_ids, connection_segments


class OpenRoundTopology(BaseTopology):
    """
    This is where neurons are placed in a circle in the middle but axons can grow
    unconstrained.
    linear dish size par_L is the diameter of the circle
    (Implementationwise, we change the function that checks the substrate after neurons
    are placed)
    """

    def __init__(self, **kwargs):
        self.set_default_parameters()
        self.update_parameters(**kwargs)
        # per default, we have a quadratic substrate of size par_L
        self._substrate = np.array([[0, self.par_L], [0, self.par_L]])
        log.info("OpenRound Topology")
        for k, v in self.get_parameters().items():
            log.info(f"{k}: {v}")
        self.init_topology()

    def init_topology(self):

        log.info("Open Round Topology")
        for k, v in self.get_parameters().items():
            log.info(f"{k}: {v}")

        self.set_neuron_positions()

        # it is convenient to have indices sorted by position
        sort_idx = np.lexsort(
            (
                self.neuron_positions[:, 1],
                self.neuron_positions[:, 0],
            )
        )
        self.neuron_positions = self.neuron_positions[sort_idx]

        # dendritic trees as circles
        self.set_dendrite_radii()

        # every source neuron has a list of target neuron ids
        aij_nested = [np.array([], dtype="int")] * self.par_N
        # which are the axon segments that made the connections? (ids)
        segs_nested = [np.array([], dtype="int")] * self.par_N

        # paths of all axons
        axon_paths = [np.array([])] * self.par_N

        # do not confine axons: set substrate to be the whole plane
        self.is_within_substrate = lambda x, y, r=0: True

        # grow axons and get connections for all neurons
        for n_id in range(0, self.par_N):
            path = self.grow_axon(start=self.neuron_positions[n_id, :])
            axon_paths[n_id] = path
            cids, seg_ids = self.connections_for_neuron(n_id=n_id, axon_path=path)

            argsorted = np.argsort(cids)
            aij_nested[n_id] = cids[argsorted]
            segs_nested[n_id] = seg_ids[argsorted]

        # save the details as attributes
        self.axon_paths = axon_paths
        # this has three columns
        aij_sparse = _nested_lists_to_sparse(aij_nested, vals=segs_nested)
        # vanilla sparse matrix has two columns
        self.aij_sparse = aij_sparse[:, 0:2]
        self.aij_nested = aij_nested
        # save the third column separately
        self.segs_sparse = aij_sparse[:, 2]
        self.k_in, self.k_out = _get_degrees(aij_nested)

    def is_within_substrate(self, x, y, r=0):
        """
        This check will only be used to place the neurons.
        After that, we use a lambda that always returns true.
        """
        return _is_intersecting_circle(
            [self.par_center_x, self.par_center_y], self.par_L / 2, x, y, r
        )

    def set_default_parameters(self):
        super().set_default_parameters()
        self.par_center_x = self.par_L / 2
        self.par_center_y = self.par_L / 2

    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)
        self.par_center_x = self.par_L / 2
        self.par_center_y = self.par_L / 2
        if "par_center_x" in kwargs:
            self.par_center_x = kwargs["par_center_x"]
        if "par_center_y" in kwargs:
            self.par_center_y = kwargs["par_center_y"]

    def get_everything_as_nested_dict(self, return_descriptions=False):

        h5_data, h5_desc = super().get_everything_as_nested_dict(return_descriptions=True)

        h5_data["meta.topology"] = "orlandi open round topology"
        h5_data["meta.topology_center_x"] = self.par_center_x
        h5_data["meta.topology_center_y"] = self.par_center_y

        if return_descriptions:
            return h5_data, h5_desc
        return h5_data


class MergedTopology(BaseTopology):
    def __init__(self, assign_submodules=False, **kwargs):
        """
        The Base topology already largely does the job.
        Make sure to have right system size and set artificial "module ids"
        for the neurons that only depend on position, so we can better compare
        to modular cultures
        """
        self.set_default_parameters()
        self.update_parameters(**kwargs)
        # per default, we have a quadratic substrate of size par_L
        self._substrate = np.array([[0, self.par_L], [0, self.par_L]])
        log.info("Merged Topology")
        for k, v in self.get_parameters().items():
            log.info(f"{k}: {v}")
        self.init_topology()

        if assign_submodules:
            log.info("Assigning sub modules from neuron positions")
            # approximate module ids from position
            # we might need this to stimulate part of the merged culture
            # 0 lower left, 1 upper left, 2 lower right, 3 upper right
            self.neuron_module_ids = np.ones(self.par_N, dtype=int) * -1
            lim = 200
            for nid in range(0, self.par_N):
                x = self.neuron_positions[nid, 0]
                y = self.neuron_positions[nid, 1]
                if x < lim and y < lim:
                    self.neuron_module_ids[nid] = 0
                elif x < lim and y >= lim:
                    self.neuron_module_ids[nid] = 1
                elif x >= lim and y < lim:
                    self.neuron_module_ids[nid] = 2
                elif x >= lim and y >= lim:
                    self.neuron_module_ids[nid] = 3
        else:
            self.neuron_module_ids = np.zeros(self.par_N, dtype=int)

    def set_default_parameters(self):
        super().set_default_parameters()

        self.par_L = 400

        # to later use same code as for modular systems, use k_inter = -1 for merged
        self.par_k_inter = -1

    def get_everything_as_nested_dict(self, return_descriptions=False):

        h5_data, h5_desc = super().get_everything_as_nested_dict(return_descriptions=True)

        h5_data["data.neuron_module_id"] = self.neuron_module_ids[:]
        h5_data["meta.topology"] = "orlandi merged topology"
        h5_data["meta.topology_k_inter"] = self.par_k_inter

        if return_descriptions:
            return h5_data, h5_desc
        return h5_data


class ModularTopology(BaseTopology):
    def __init__(self, **kwargs):
        self.set_default_parameters()
        self.update_parameters(**kwargs)
        self.set_default_modules()
        log.info("Modular Topology")
        for k, v in self.get_parameters().items():
            log.info(f"{k}: {v}")
        self.init_topology()

    def set_default_parameters(self):
        super().set_default_parameters()

        # we have 2 200um modules with 200um space inbetween
        self.par_L = 600

        # number of axons going from each module to its neighbours
        self.par_k_inter = 5

    def init_topology(self):
        # only call this after initalizing parameters and modules!

        self.set_neuron_positions()
        self.set_neuron_module_ids()

        # it is convenient to have indices sorted by modules and position
        sort_idx = np.lexsort(
            (
                self.neuron_positions[:, 1],
                self.neuron_positions[:, 0],
                self.neuron_module_ids,
            )
        )
        self.neuron_positions = self.neuron_positions[sort_idx]
        self.neuron_module_ids = self.neuron_module_ids[sort_idx]

        # dendritic trees as circles
        self.set_dendrite_radii()

        # every source neuron has a list of target neuron ids
        aij_nested = [np.array([], dtype="int")] * self.par_N
        # which are the axon segments that made the connections? (ids)
        segs_nested = [np.array([], dtype="int")] * self.par_N

        # paths of all axons
        axon_paths = [np.array([])] * self.par_N

        # which neurons create bridges
        bridge_ids = []

        # generate and connect bridging axons
        # e.g. (0, 1, 2): "from 0 to 1 and 2"
        for b_ids in [(0, 1, 2), (1, 0, 3), (2, 0, 3), (3, 1, 2)]:
            src_mod = self.mods[b_ids[0]]
            tar_mods = [self.mods[b_ids[1]], self.mods[b_ids[2]]]

            # get the ids and axon paths of all neurons that bridge between modules
            br_nids, br_paths = self.grow_bridging_axons(
                source_mod=src_mod, target_mods=tar_mods
            )
            bridge_ids.extend(br_nids)

            # get connectivity for each neuron
            for idx, n_id in enumerate(br_nids):
                cids, seg_ids = self.connections_for_neuron(n_id, axon_path=br_paths[idx])
                axon_paths[n_id] = br_paths[idx]
                argsorted = np.argsort(cids)
                aij_nested[n_id] = cids[argsorted]
                segs_nested[n_id] = seg_ids[argsorted]

        self.neuron_bridge_ids = np.array(bridge_ids, dtype="int")

        # grow axons and get connections for the remaining neurons
        for n_id in range(0, self.par_N):
            if n_id in bridge_ids:
                continue

            path = self.grow_axon(start=self.neuron_positions[n_id, :])
            axon_paths[n_id] = path
            cids, seg_ids = self.connections_for_neuron(n_id=n_id, axon_path=path)
            argsorted = np.argsort(cids)
            aij_nested[n_id] = cids[argsorted]
            segs_nested[n_id] = seg_ids[argsorted]

        # save the details as attributes
        self.axon_paths = axon_paths
        # this has three columns
        aij_sparse = _nested_lists_to_sparse(aij_nested, vals=segs_nested)
        # vanilla sparse matrix has two columns
        self.aij_sparse = aij_sparse[:, 0:2]
        self.aij_nested = aij_nested
        # save the third column separately
        self.segs_sparse = aij_sparse[:, 2]
        self.k_in, self.k_out = _get_degrees(aij_nested)

    def get_everything_as_nested_dict(self, return_descriptions=False):

        h5_data, h5_desc = super().get_everything_as_nested_dict(return_descriptions=True)

        h5_data["data.neuron_module_id"] = self.neuron_module_ids[:]
        h5_data["data.neuron_bridge_ids"] = self.neuron_bridge_ids
        h5_desc[
            "data.neuron_bridge_ids"
        ] = "list of ids of neurons connecting two modules"

        h5_data["meta.topology"] = "orlandi modular topology"
        h5_data["meta.topology_k_inter"] = self.par_k_inter

        if return_descriptions:
            return h5_data, h5_desc
        return h5_data

    def set_default_modules(self):
        # coordinates of rectangular modules: module_number, x | y, low | high
        self.mods = np.ones(shape=(4, 2, 2)) * np.nan
        self.mods[0] = np.array([[0, 200], [0, 200]])
        self.mods[1] = np.array([[0, 200], [400, 600]])
        self.mods[2] = np.array([[400, 600], [0, 200]])
        self.mods[3] = np.array([[400, 600], [400, 600]])

    def grow_bridging_axons(self, source_mod, target_mods):

        # get the indices of neurons in the target module
        source_idx = [
            _is_within_rect(x, y, 0, source_mod) for x, y in self.neuron_positions
        ]
        source_idx = np.where(source_idx)[0]
        source_pos = self.neuron_positions[source_idx]

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

        for k in range(0, self.par_k_inter):
            for mdx, mod in enumerate(target_mods):
                n_offset = len(rewired_idx)
                # ensure ~ 5um padding at module edges before terminating
                # f_term = functools.partial(_is_within_rect, r=5, rect=mod)
                criterion = termination_criteria[mdx]
                n_idx = closest_idx[n_offset]
                # print(n_idx, self.neuron_positions[n_idx])
                if n_idx == 58:
                    global debug_print
                    debug_print = True

                # start by growing to our target
                path, num_segs_left, last_phi = self.grow_axon_to_target(
                    start=self.neuron_positions[n_idx],
                    target=_center_of_rect(mod),
                    termination_criterion=criterion,
                )

                if num_segs_left > 0:
                    # finish randomly
                    last_pos = path[-num_segs_left - 1]
                    rest_of_path = self.grow_axon(
                        start=last_pos,
                        num_segments=num_segs_left,
                        start_phi=last_phi,
                    )

                    path[-num_segs_left:] = rest_of_path
                if not np.all(np.isfinite(path)):
                    print("\t", num_segs_left, len(path), len(rest_of_path))
                    print(path)
                    print(rest_of_path)

                assert np.all(np.isfinite(path))

                rewired_idx.append(n_idx)
                rewired_paths.append(path)

        return rewired_idx, rewired_paths

    def set_neuron_positions(self):
        """
        make sure to set parameters correctly before calling this
        """

        rejections = 0

        # a 2d np array with x, y poitions, soma_radius and dendritic_tree diameter
        neuron_pos = np.ones(shape=(self.par_N, 2)) * -93288.0
        neuron_rad = np.ones(shape=(self.par_N)) * self.par_R_s

        # keep track of the number of neurons in every module
        neurons_per_module = np.zeros(len(self.mods), "int")

        for i in range(self.par_N):
            placed = False
            while not placed:
                placed = True
                x = np.random.uniform(0.0, self.par_L)
                y = np.random.uniform(0.0, self.par_L)

                # dont place out of bounds
                if not self.is_within_substrate(x, y, self.par_R_s):
                    placed = False
                    rejections += 1
                    continue

                # dont overlap other neurons
                if np.any(
                    _are_intersecting_circles(
                        neuron_pos[0:i], neuron_rad[0:i], x, y, self.par_R_s
                    )
                ):
                    placed = False
                    rejections += 1
                    continue

                # for modular topology, we want to ensure the same number of neurons
                # in every module
                mod_id = self.mod_id_from_coordinate(x, y)
                if neurons_per_module[mod_id] != np.min(neurons_per_module):
                    placed = False
                    rejections += 1
                    continue

                neuron_pos[i, 0] = x
                neuron_pos[i, 1] = y
                neurons_per_module[mod_id] += 1

        self.neuron_positions = neuron_pos

    def set_neuron_module_ids(self):
        """
        assign the module id of every neuron
        """
        tar_mods = np.ones(self.par_N, dtype="int") * -1
        for idx in range(0, self.par_N):
            x, y = self.neuron_positions[idx]
            tar_mods[idx] = self.mod_id_from_coordinate(x, y)
        self.neuron_module_ids = tar_mods

    def mod_id_from_coordinate(self, x, y):
        num_mods = len(self.mods)
        for i in range(0, num_mods):
            if _is_within_rect(x, y, 0, self.mods[i]):
                return i
        raise Exception("Coordinate is not within any known module")

    def is_within_substrate(self, x, y, r=0):
        for m in range(0, len(self.mods)):
            if _is_within_rect(x, y, r, self.mods[m]):
                return True
        return False

    def connections_for_neuron(self, n_id, axon_path):
        """
        Get the connections due to a grown path
        n_id : int
            the id of the source neuron
        axon_path : 2d array
            positions of all axons segments growing from n_id

        # Returns:
        ids_to_connect_with : array
            indices of neurons to form a connection with
        """

        # first, collect all dendritic trees which we intersect and how often
        num_intersections = np.zeros(self.par_N, "int")
        first_intersections = np.zeros(self.par_N, "int")

        for seg_id, seg in enumerate(axon_path):
            if not self.is_within_substrate(seg[0], seg[1], r=0):
                # we dont count intersection that occur out of our substrate,
                # because we want dendritic trees to only be "on the substrate"
                continue

            intersecting_dendrites = _are_intersecting_circles(
                ref_pos=self.neuron_positions,
                ref_rad=self.dendritic_radii,
                x=seg[0],
                y=seg[1],
            )

            # we also do not want to connect to a hypothetical super large tree
            # that comes from another module
            # this additioanl check is the only difference to the function in the
            # base class.
            # other than that, this a carbon copy of BaseTopology :/
            seg_mod = self.mod_id_from_coordinate(*seg)
            valid_dendrites = self.neuron_module_ids == seg_mod

            # only count intersections where valid and intersecting
            intersecting_dendrites = intersecting_dendrites & valid_dendrites

            # collect
            firsts = np.where(num_intersections == 0 & intersecting_dendrites)[0]
            first_intersections[firsts] = seg_id
            num_intersections += intersecting_dendrites

        num_intersections[n_id] = 0  # dont create self-connections
        t_ids = np.where(num_intersections > 0)[0]  # target neuron ids
        connection_segments = []  # which segment forms the connection

        if not self.par_alpha_path_weighted:
            # flat probability, independent of the number of intersections
            t_ids = t_ids[np.random.uniform(0, 1, size=len(t_ids)) < self.par_alpha]

            # find the segment that causes the connection. simply the first intersecting
            connection_segments = first_intersections[t_ids]

        else:
            # probability is applied once, for every intersecting segment
            filtered_t_ids = []
            for tid in t_ids:
                intersecting = (
                    np.random.uniform(0, 1, size=num_intersections[tid]) < self.par_alpha
                )

                if np.any(intersecting):
                    filtered_t_ids.append(tid)
                    # axons may grow out of trees and back in. the first index in
                    # of intersecing segs may not actually be the right one (or in tree)
                    ref = np.sum(intersecting)
                    for seg_id in range(first_intersections[tid], len(axon_path)):
                        seg = axon_path[seg_id]
                        if _is_intersecting_circle(
                            ref_pos=self.neuron_positions[tid],
                            ref_rad=self.dendritic_radii[tid],
                            x=seg[0],
                            y=seg[1],
                        ):
                            if ref == 0:
                                break
                            ref -= 1
                    connection_segments.append(seg_id)

            t_ids = np.array(filtered_t_ids, dtype="int")
            connection_segments = np.array(connection_segments, dtype="int")

        return t_ids, connection_segments


# ------------------------------------------------------------------------------ #
# constrained in-degrees
# a bit of iteration to find the right alpha
# for a desired in-degre distributions
# ------------------------------------------------------------------------------ #

# 2 levle dict, first: k_inter (topology), second: k_in (desired in-degree)
_alpha_for_kin = dict()
for k_inter in [0, 1, 3, 5, 10, 20, -1]:
    _alpha_for_kin[k_inter] = dict()

# k_in 30
_alpha_for_kin[0][30] = 0.0275
_alpha_for_kin[1][30] = 0.02641
_alpha_for_kin[3][30] = 0.02521
_alpha_for_kin[5][30] = 0.02437
_alpha_for_kin[10][30] = 0.0225
_alpha_for_kin[20][30] = 0.02021
_alpha_for_kin[-1][30] = 0.00824
# k_in 25
_alpha_for_kin[0][25] = 0.01728
_alpha_for_kin[1][25] = 0.01698
_alpha_for_kin[3][25] = 0.01667
_alpha_for_kin[5][25] = 0.01635
_alpha_for_kin[10][25] = 0.0156
_alpha_for_kin[20][25] = 0.01498
_alpha_for_kin[-1][25] = 0.00665
# k_in 20
_alpha_for_kin[0][20] = 0.01156
_alpha_for_kin[1][20] = 0.01129
_alpha_for_kin[3][20] = 0.01125
_alpha_for_kin[5][20] = 0.01091
_alpha_for_kin[10][20] = 0.01099
_alpha_for_kin[20][20] = 0.01057
_alpha_for_kin[-1][20] = 0.00519
# k_in 15
_alpha_for_kin[0][15] = 0.007448
_alpha_for_kin[1][15] = 0.007448
_alpha_for_kin[3][15] = 0.007344
_alpha_for_kin[5][15] = 0.007305
_alpha_for_kin[10][15] = 0.007266
_alpha_for_kin[20][15] = 0.007201
_alpha_for_kin[-1][15] = 0.00373


def find_alpha_for_target_in_degree(
    k_inter, k_in_target, num_reps=5, rerun=True, **kwargs
):
    """
    find the alpha value that leads to a desired in-degree distribution

    best set the log level to error before running this function
    `topo.log.setLevel("ERROR")`

    # Paramters
    k_inter : int
        number of bridging axons. -1 for merged topology, otherwise modular is used
    k_in_target : int
        target in-degree
    rerun : bool
        I have saved some values for k_in = 15, 20, 25, 30.
        By default I give you those and avoid the computation.
    num_reps : int
        how many independent estimates to create
    **kwargs : dict
        passed to `_iteration`. e.g. init, iterstep, threshold
    """

    if not rerun:
        try:
            log.debug("trying saved (hardcoded) alpha")
            return _alpha_for_kin[k_inter][k_in_target]
        except KeyError:
            log.debug(f"No precomputed value for {k_inter=}, {k_in_target=}")

    with logging_redirect_tqdm():
        res = np.ones(num_reps) * np.nan
        for rep in tqdm(range(num_reps), leave=False):
            log.info(f"finding alpha for {k_inter=}")
            res[rep] = _iteration(
                generator=lambda alpha: _generate_series_of_kin_from_topo(alpha, k_inter),
                target=k_in_target,
                **kwargs,
            )

    if not np.all(np.isfinite(res)):
        log.warning(f"some found alpha values do not make sense {res=}")

    return np.nanmean(res)


def _iteration(generator, target, init=0.0125, iterstep=0.0025, threshold=0.1):
    """
    yada yada standard stuff
    """

    alpha = init
    prev_was_greater = None
    while True:
        series = generator(alpha)
        # calculate the probability distribution from the series
        pdist = np.bincount(series, minlength=100) / len(series)
        # calculate the expectation value
        ev = _ev(np.arange(len(pdist)), pdist)

        log.debug(f"{alpha=:.4f}, {ev=:.3f}")

        if np.abs(ev - target) < threshold:
            break
        else:
            if prev_was_greater is None:
                prev_was_greater = ev > target
            else:
                if prev_was_greater != (ev > target):
                    iterstep /= 2
                    prev_was_greater = ev > target
                    log.debug(f"new {iterstep=:.3e}")
            # increase or decrease alpha if above or below
            if ev > target:
                alpha -= iterstep
            else:
                alpha += iterstep
    log.info(f"final {alpha=:.4f}, {ev=:.3f}")
    return alpha


def _generate_series_of_kin_from_topo(alpha, num_connections, num_reps=10):
    """
    generate a series of in-degrees (on value for each neuron)
    from a topology specified via num_connections.
    -1 for merged
    """
    series = []
    for _ in range(num_reps):
        if num_connections == -1:
            tp = MergedTopology(par_alpha=alpha)
        else:
            tp = ModularTopology(par_alpha=alpha, par_k_inter=num_connections)
        series.extend(tp.k_in)
    return np.array(series)


# ------------------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------------------ #


def _ev(x, pdist):
    """
    expectation value via x, p(x)
    """
    ev = 0
    for idx, p in enumerate(pdist):
        ev += x[idx] * pdist[idx]
    return ev


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _grow_axon(
    par_std_l,
    par_del_l,
    par_axon_retry,
    par_std_phi,
    par_angle_mod,
    is_within_substrate,
    start,
    num_segments=None,
    start_phi=None,
):

    assert num_segments is None or num_segments > 0

    # we might not succeed placing all axons, retry full axon
    tries_to_grow = 0
    while tries_to_grow < par_axon_retry:
        tries_to_grow += 1
        grown_successful = True

        last_x = start[0]
        last_y = start[1]

        if num_segments is None:
            length = np.random.rayleigh(par_std_l)
            num_segs = int(np.fmax(1, length / par_del_l))
        else:
            num_segs = num_segments

        if start_phi is None:
            last_phi = np.random.uniform(0, 1) * 2 * np.pi
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
                if not is_within_substrate(x, y, 0):
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
                grown_successful = False
                break

        # we succeeded growing the whole axon if all segments placed
        if grown_successful:
            break

    if tries_to_grow == par_axon_retry:
        raise Exception("Failed to grow axon with specified retries")

    return path


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _grow_axon_to_target(
    par_std_l,
    par_del_l,
    par_axon_retry,
    par_std_phi,
    start,
    target,
    termination_criterion,
):

    last_x = start[0]
    last_y = start[1]

    tries_to_grow = 0
    while not termination_criterion(last_x, last_y) and tries_to_grow < par_axon_retry:
        tries_to_grow += 1

        length = np.random.rayleigh(par_std_l)
        num_segs = int(np.fmax(1, length / par_del_l))
        path = np.ones((num_segs, 2)) * np.nan

        last_x = start[0]
        last_y = start[1]

        # draw phi until it goes in the right direction
        for i in range(0, par_axon_retry):
            phi = np.random.uniform(-0.5, 0.5) * 2 * np.pi
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
def _is_intersecting_circle(ref_pos, ref_rad, x, y, r=0):
    """
    Check whether x, y, is intersecting with any of the circles at ref_pos with
    radius ref_rad
    """
    x_ = ref_pos[0]
    y_ = ref_pos[1]
    r_ = ref_rad
    return np.square(x - x_) + np.square(y - y_) < np.square(r + r_)


def _nested_lists_to_2d_nan_padded(nested):
    """
    For segment paths,
    convert a list of lists/arrays (with varying length)
    into a 2d array of shape (num_lists, max_len)
    where missing elements are set to np.nan

    nested : list of 2d arrays with shape (num_segments, 2)
    """

    max_len = np.max([len(x) for x in nested])
    res = np.ones(shape=(len(nested), max_len, 2)) * np.nan
    for idx, l in enumerate(nested):
        if len(l) > 0:
            res[idx, 0 : len(l)] = l

    return res


def _nested_lists_to_sparse(nested, vals=None):
    """
    Mostly for connectivity matrix,
    when stored as list of listes with variable lengths,
    convert to a sparse matrix like
    `i=a_ij_sparse[:, 0], j=a_ij_sparse[:, 1]`
    # Paramters:
    nested : list of lists,
        every source neuron has a list of target neuron ids
    val : if not None, construct a third column in the sparse matrix with
        these values. shapes have to match `nested`
        (e.g. here the id of the segment that made the connection)
    # Returns:
    a_ij_sparse : array of shape (num_connections, 2) with [from_index, to_index]
        or shape (num_connections, 3) with [from_index, to_index, segment_id]
    """
    sources = []
    targets = []
    if vals is not None:
        extra = []

    for idx, l in enumerate(nested):
        sources.extend([idx] * len(l))
        targets.extend(l)
        if vals is not None:
            extra.extend(vals[idx])

    if vals is not None:
        res = np.array([sources, targets, extra]).T
    else:
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


def _get_in_degrees_by_internal_external(aij_nested, module_ids):
    k_internal = np.zeros(len(aij_nested), dtype="int")
    k_external = np.zeros(len(aij_nested), dtype="int")
    for n_id in range(0, len(aij_nested)):
        targets = aij_nested[n_id]
        same_mask = module_ids[targets] == module_ids[n_id]
        int_ids = targets[same_mask]
        ext_ids = targets[~same_mask]
        k_internal[int_ids] += 1
        k_external[ext_ids] += 1

    return k_internal, k_external


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
        try:
            pos1 = path[0]
            pos2 = path[-1]
            res[idx] = np.sqrt(
                np.square(pos1[0] - pos2[0]) + np.square(pos1[1] - pos2[1])
            )
        except Exception as e:
            log.exception(f"End to end distance failed for neuron {idx} with path {path}")
            log.exception(e)
            res[idx] = np.nan

    return res


# numba has its own rng instance
@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _set_seed(seed):
    np.random.seed(seed)


def set_seed(seed):
    # we just want some sort of reproducibility, so set once out of numba
    np.random.seed(seed)
    # and once within numba
    _set_seed(seed)


def index_alignment(num_n, num_inhib, bridge_ids):
    """
    resort indices so that they are contiguous, this is needed for brian:

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

    return (
        topo_indices,
        brian_indices,
        inhib_ids_new,
        excit_ids_new,
        bridge_ids_new,
    )


# ------------------------------------------------------------------------------ #
# legacy helpers to load topology that was generated in older c code
# ------------------------------------------------------------------------------ #


def _load_topology(input_path):
    """
    Loads variables needed for the topology, mainly connectivity matrix

    # Parameters
        input_path : to .hdf5 file

    # Returns
        num_n       : number of neurons
        a_ij_sparse : the connectivity matrix in sparse form
        mod_ids     : an array to find which module neurons belong to. index -> mod
    """

    from bitsandbobs import hi5 as h5

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


def _load_bridging_neurons(input_path):
    try:
        from bitsandbobs import hi5 as h5

        return h5.load(input_path, "/data/neuron_bridge_ids", keepdim=True).astype(
            int, copy=False
        )  # brian doesnt like uints
    except Exception as e:
        log.debug(e)
        return []


def _connect_synapses_from(S, a_ij):
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
