# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2022-12-30 11:41:45
# ------------------------------------------------------------------------------ #
# Scans the provided (wildcarded) filenames and merges individual realizsation
# into a single file, containing high-dimensional arrays.
# How the coordinates at which results from every input file are placed in the
# output are specified in the `d_obs` dictionary, below.
#
# Calls analysis routines from `ana/ana_helper.py` on each realization and places
# them at the right coordinate in parameter space.
#
# Essentially, we construct an xarray.Dataset. But when I wrote this, I did not
# know about xarrays so this is a poormans version. The resulting file can be
# read with the `ndim_helper` that also has a function to cast to xarray.
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import logging
import warnings
import functools
import itertools
import tempfile
import psutil
import re

# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
# import pandas as pd
from collections import OrderedDict
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from benedict import benedict

from contextlib import nullcontext, ExitStack
from dask_jobqueue import SGECluster
from dask.distributed import Client, SSHCluster, LocalCluster, as_completed

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("DEBUG")
warnings.filterwarnings("ignore")  # suppress numpy warnings

import ana_helper as ah
from bitsandbobs import hi5 as h5

# ------------------------------------------------------------------------------ #
# settings
# ------------------------------------------------------------------------------ #

# variables to span axes and how to get them from the hdf5 files
d_obs = OrderedDict()
d_obs["jA"] = "/meta/dynamics_jA"
d_obs["jG"] = "/meta/dynamics_jG"
# d_obs["jM"] = "/meta/dynamics_jM"
d_obs["rate"] = "/meta/dynamics_rate"
d_obs["tD"] = "/meta/dynamics_tD"
# d_obs["alpha"] = "/meta/topology_alpha"
d_obs["k_inter"] = "/meta/topology_k_inter"
d_obs["k_in"] = "/meta/topology_k_in"
d_obs["stim_rate"] = "/meta/dynamics_stimulation_rate"
d_obs["stim_mods"] = "/meta/dynamics_stimulation_mods"
# d_obs["k_frac"] = "/meta/dynamics_k_frac"

log.debug(f"d_obs: {d_obs}")

# these can be set via command line arguments, see `parse_args()`
threshold_factor = 2.5 / 100
smoothing_width = 20 / 1000
time_bin_size_for_rij = 500 / 1000
remove_null_sequences = False

# details of single bursts, such as involved modules. this can eat a lof of ram and disk
include_ragged_arrays = True


def ana_for_single_rep(candidate=None):
    """
    This function is called for each realization (potentially in parallel, using dask)
    and performs the complete barrage of analysis.

    # Parameters
    candidate : str or None
        file path to the hdf5 file, or None to get the expected structure.

    # Returns
    res : dict
        Every key becomes a dataset in the hdf5 file.
        The dict maps
        - key -> scalar or
        - key -> 1d array if the key starts with `vec_`
        - key -> shape if `candidate` was None. This is only needed because
            we have to init our data storage. Then again, it gives a nice overview
            of whats coming later.

    Note on naming convention:
        - `vec_` for 1d arrays, mostly relevant for histograms
        - `sys_` for observables that are calcualted for the whole system
            (e.g. Modularity is a system-wide porperty, `sys_modularity`)
        - `mod_` for observables that are calculated on the module level, but may
            span multiple modules (e.g. The number of bursts spanning 1, ..., 4 modules,
            `mod_num_b_1`)
        - `any_` when we want to be agnostic about how many or which module were involved,
            but still measured the obseravble on the module-level (e.g. how many
            module-level bursts took place, `any_num_b`)

    However, this is only a rough rule of thumb. As usual, things develop and these
    categories could always be applied.

    """
    if candidate is None:
        res = dict()
        # scalars, there are some comments on each near the implementation. ctrl+f, ftw.
        res["sys_modularity"] = 1
        res["any_num_b"] = 1
        res["mod_num_b_0"] = 1
        res["mod_num_b_1"] = 1
        res["mod_num_b_2"] = 1
        res["mod_num_b_3"] = 1
        res["mod_num_b_4"] = 1
        res["sys_rate_cv"] = 1
        res["sys_mean_rate"] = 1
        res["sys_rate_threshold"] = 1
        res["sys_blen"] = 1
        res["mod_blen_0"] = 1
        res["mod_blen_1"] = 1
        res["mod_blen_2"] = 1
        res["mod_blen_3"] = 1
        res["mod_blen_4"] = 1
        res["mod_num_spikes_in_bursts_1"] = 1
        res["sys_mean_any_ibis"] = 1
        res["sys_median_any_ibis"] = 1
        res["sys_mean_all_ibis"] = 1
        res["sys_median_all_ibis"] = 1
        res["any_ibis"] = 1
        res["sys_ibis_cv"] = 1
        res["any_ibis_cv"] = 1
        res["sys_functional_complexity"] = 1
        res["any_functional_complexity"] = 1
        res["sys_mean_depletion_correlation"] = 1
        res["sys_median_depletion_correlation"] = 1
        res["sys_mean_participating_fraction"] = 1
        res["sys_median_participating_fraction"] = 1
        res["sys_participating_fraction_complexity"] = 1
        res["any_num_spikes_in_bursts"] = 1
        res["sys_orderpar_fano_neuron"] = 1
        res["sys_orderpar_fano_population"] = 1
        res["sys_orderpar_baseline_neuron"] = 1
        res["sys_orderpar_baseline_population"] = 1
        res["sys_mean_core_delay"] = 1
        res["sys_orderpar_dist_low_end"] = 1
        res["sys_orderpar_dist_low_mid"] = 1
        res["sys_orderpar_dist_high_mid"] = 1
        res["sys_orderpar_dist_high_end"] = 1
        res["sys_orderpar_dist_median"] = 1
        res["sys_orderpar_dist_max"] = 1
        res["sys_mean_resources_at_burst_beg"] = 1
        res["sys_std_resources_at_burst_beg"] = 1
        # we started with neuorn level correlation, all above is this kind.
        res["sys_mean_correlation"] = 1
        res["sys_mean_correlation_across"] = 1
        res["sys_mean_correlation_within_stim"] = 1
        res["sys_mean_correlation_within_nonstim"] = 1
        res["sys_median_correlation"] = 1
        res["sys_median_correlation_across"] = 1
        res["sys_median_correlation_within_stim"] = 1
        res["sys_median_correlation_within_nonstim"] = 1
        # module level correlation, using the module-level firing rates
        res["mod_mean_correlation"] = 1
        res["mod_mean_correlation_across"] = 1
        res["mod_mean_correlation_within_stim"] = 1
        res["mod_mean_correlation_within_nonstim"] = 1
        res["mod_median_correlation"] = 1
        res["mod_median_correlation_across"] = 1
        res["mod_median_correlation_within_stim"] = 1
        res["mod_median_correlation_within_nonstim"] = 1

        # histograms, use "vec" prefix to indicate that higher dimensional data
        # hvals are the histogram values, hbins the bins ... obvio
        res["vec_sys_hbins_participating_fraction"] = 21
        res["vec_sys_hvals_participating_fraction"] = 20
        res["vec_sys_hbins_correlation_coefficients"] = 21
        res["vec_sys_hvals_correlation_coefficients"] = 20
        res["vec_sys_hbins_depletion_correlation_coefficients"] = 21
        res["vec_sys_hvals_depletion_correlation_coefficients"] = 20

        res["vec_sys_hbins_resource_dist"] = 101
        res["vec_sys_hvals_resource_dist"] = 100

        # integer histograms of out degrees, max k_out is num_neurons, 160
        res["vec_sys_hbins_kout_no_bridge"] = 161
        res["vec_sys_hvals_kout_no_bridge"] = 160
        res["vec_sys_hbins_kout_yes_bridge"] = 161
        res["vec_sys_hvals_kout_yes_bridge"] = 160

        # ragged arrays, use "rag" prefix to indicate that higher dimensional data,
        # of unknown length. for these guys, we do not specify the length,
        # but the data type is inferred.
        res["rag_burst_beg_times"] = 0.0
        res["rag_burst_end_times"] = 0.0
        res["rag_burst_ibis"] = 0.0
        res["rag_burst_core_delays"] = 0.0
        res["rag_resources_mod_0"] = 0.0
        res["rag_resources_mod_1"] = 0.0
        # we can now save sequences as single ints, see ana_helper seq_to_int()
        res["rag_burst_seqs"] = 0

        return res

    res = dict()

    # ------------------------------------------------------------------------------ #
    # load and prepare
    # ------------------------------------------------------------------------------ #
    # Note that the `ah.whatever_functions` __often (not always)__ modify h5f in place,
    # adding the results into the subdirectory "ana".
    # ------------------------------------------------------------------------------ #

    h5f = ah.prepare_file(
        candidate,
        # load everything to RAM directly
        hot=False,
        # we skip loading the matrix, as it slows things down and we often do not need it.
        skip=["connectivity_matrix"]
        # skip=["connectivity_matrix", "connectivity_matrix_sparse"]
    )

    # ------------------------------------------------------------------------------ #
    # modularity index Q
    # uses the Louvain algorithm in networkx nx.algorithms.community.modularity
    # and we know modules are the communities.
    # ------------------------------------------------------------------------------ #
    try:
        res["sys_modularity"] = ah.find_modularity(h5f)
    except Exception as e:
        log.exception(candidate)
        res["sys_modularity"] = np.nan

    # ------------------------------------------------------------------------------ #
    # Firing rates
    # calculated on system and module level (selecting corresponding neurons)
    # by putting a gaussian on every spike time (kernel convolution).
    #
    # We also detect module sequences (in which order did modules activate)
    # currently, we say a module "activates" when at least 20% of its neurons
    # fired at least one spike.
    # ------------------------------------------------------------------------------ #
    ah.find_rates(h5f, bs_large=smoothing_width)
    # threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
    threshold = 3.0 # for the two-mod at high noise, a fixed threshold was more robust.
    # this was not a problem when only stimulating 1/2, as we still saw high firing
    # rates in the non-targeted modules. thus the relative threshold was fine.
    # when targeting 2/2 and everything fluctuates, we start detecting bursts everywhere.
    res["sys_rate_threshold"] = threshold

    # a burst starts/stops when the rate exceeds/drops below rate_threshold
    # and two consecutive bursts are merged when less than `merge_threshold` seconds apart
    ah.find_system_bursts_from_global_rate(
        h5f, rate_threshold=threshold, merge_threshold=0.1
    )

    # detection is not flawless. it may happen that a (system) burst is detected
    # but no module alone fulfilled the 20% neurons "activation" requirement.
    if remove_null_sequences:
        ah.remove_bursts_with_sequence_length_null(h5f)

    # inter burst intervals
    ah.find_ibis(h5f)

    # ------------------------------------------------------------------------------ #
    # Bursts, Sequences, Burst durations
    # ------------------------------------------------------------------------------ #

    # general bursts, indep of number involved modules
    res["any_num_b"] = len(h5f["ana.bursts.system_level.beg_times"])
    # bursts where 0, 1, ... 4 modules were involved
    res["mod_num_b_0"] = len(
        [x for x in h5f["ana.bursts.system_level.module_sequences"] if len(x) == 0]
    )
    res["mod_num_b_1"] = len(
        [x for x in h5f["ana.bursts.system_level.module_sequences"] if len(x) == 1]
    )
    res["mod_num_b_2"] = len(
        [x for x in h5f["ana.bursts.system_level.module_sequences"] if len(x) == 2]
    )
    res["mod_num_b_3"] = len(
        [x for x in h5f["ana.bursts.system_level.module_sequences"] if len(x) == 3]
    )
    res["mod_num_b_4"] = len(
        [x for x in h5f["ana.bursts.system_level.module_sequences"] if len(x) == 4]
    )
    # coefficient of variation and mean of the system-wide firing rate
    res["sys_rate_cv"] = h5f["ana.rates.cv.system_level"]
    res["sys_mean_rate"] = np.nanmean(h5f["ana.rates.system_level"])

    # sequences of module activations and burst duration
    slen = np.array([len(x) for x in h5f["ana.bursts.system_level.module_sequences"]])
    blen = np.array(h5f["ana.bursts.system_level.end_times"]) - np.array(
        h5f["ana.bursts.system_level.beg_times"]
    )
    # mean burst duration, independent of the number of involved modules
    res["sys_blen"] = np.nanmean(blen)
    # burst duration for bursts involving 0, 1, ... 4 modules
    res["mod_blen_0"] = np.nanmean(blen[np.where(slen == 0)[0]])
    res["mod_blen_1"] = np.nanmean(blen[np.where(slen == 1)[0]])
    res["mod_blen_2"] = np.nanmean(blen[np.where(slen == 2)[0]])
    res["mod_blen_3"] = np.nanmean(blen[np.where(slen == 3)[0]])
    res["mod_blen_4"] = np.nanmean(blen[np.where(slen == 4)[0]])

    # inter burst intervals
    try:
        res["sys_mean_any_ibis"] = np.nanmean(h5f["ana.ibi.system_level.any_module"])
        res["sys_median_any_ibis"] = np.nanmedian(h5f["ana.ibi.system_level.any_module"])
        res["sys_mean_all_ibis"] = np.nanmean(h5f["ana.ibi.system_level.all_modules"])
        res["sys_median_all_ibis"] = np.nanmedian(h5f["ana.ibi.system_level.all_modules"])
        res["any_ibis_cv"] = np.nanmean(h5f["ana.ibi.system_level.cv_any_module"])
        res["sys_ibis_cv"] = np.nanmean(h5f["ana.ibi.system_level.cv_across_modules"])
        ibis_module = []
        for m_dc in h5f["ana.ibi.module_level"].keys():
            ibis_module.extend(h5f["ana.ibi.module_level"][m_dc])
        res["any_ibis"] = np.nanmean(ibis_module)
    except KeyError as e:
        log.debug(e)

    try:
        # core delay is a cool concept we came up with.
        # each modules firing rate roughly looks like a gaussian when the module bursts.
        # burst "cores" are the time points of the peak of this gaussian.
        # the "core delays" are the times between cores of different modules
        #    ... if > 1 mod involved
        ah.find_burst_core_delays(h5f)
        res["sys_mean_core_delay"] = np.nanmean(
            h5f["ana.bursts.system_level.core_delays_mean"]
        )
    except Exception as e:
        log.error(e)

    # ------------------------------------------------------------------------------ #
    # Correlation coefficients
    # ------------------------------------------------------------------------------ #
    try:
        # correlation between firing rates of different neurons
        # for the rij between pairs of neurons, we use time-binning.
        # this avoids the spurious correlations a sliding gaussian kernel would introduce.
        # Note: we also do this in the experimental analysis, but at a (larger) bin-size
        # that makes sense for the (slower) time resolution of Fluorescence data.
        rij_matrix = ah.find_rij(
            h5f, which="neurons", time_bin_size=time_bin_size_for_rij
        )
        # correlation between the resource variables of different neurons
        # (uses h5f["data.state_vars_D"] at the native rate of the simulation-recording)
        rij_depletion_matrix = ah.find_rij(h5f, which="depletion")
        np.fill_diagonal(rij_matrix, np.nan)
        np.fill_diagonal(rij_depletion_matrix, np.nan)
    except:
        log.error("Failed to find correlation coefficients")

    res["sys_mean_correlation"] = np.nanmean(rij_matrix)
    res["sys_median_correlation"] = np.nanmedian(rij_matrix)
    res["sys_mean_depletion_correlation"] = np.nanmean(rij_depletion_matrix)
    res["sys_median_depletion_correlation"] = np.nanmedian(rij_depletion_matrix)

    # correlations for different pairings
    # we are not showing those any more,
    # this is just in case I have to dig them back out during revisions
    # for mod in [0, 1, 2, 3]:
    #     try:
    #         res[f"vec_rij_within_{mod}"] = ah.find_rij_pairs(
    #             h5f, rij_matrix, pairing=f"within_group_{mod}"
    #         )
    #     except Exception as e:
    #         log.error(e)
    #         res[f"vec_rij_within_{mod}"] = np.ones(780) * np.nan

    # for pair in itertools.combinations("0123", 2):
    #     # here just 40^2, since in different modules
    #     try:
    #         res[f"vec_rij_across_{pair[0]}_{pair[1]}"] = ah.find_rij_pairs(
    #             h5f, rij_matrix, pairing=f"across_groups_{pair[0]}_{pair[1]}"
    #         )
    #     except Exception as e:
    #         log.exception(e)
    #         res[f"vec_rij_across_{pair[0]}_{pair[1]}"] = np.ones(1600) * np.nan

    # ------------------------------------------------------------------------------ #
    # Functional Complexity
    # this is closely related to the correlation coefficients, sharing bins etc.
    # ------------------------------------------------------------------------------ #

    bw = 1.0 / 20
    bins = np.arange(0, 1 + 0.1 * bw, bw)
    res["vec_sys_hbins_correlation_coefficients"] = bins.copy()
    res["vec_sys_hbins_participating_fraction"] = bins.copy()
    res["vec_sys_hbins_depletion_correlation_coefficients"] = bins.copy()

    try:
        C, _ = ah.find_functional_complexity(
            h5f, rij=rij_matrix, return_res=True, write_to_h5f=False, bins=bins
        )
        # this is not the place to do this, but ok
        # FC needs the rij as above, so we save time by computing only once.
        rij_hist, _ = np.histogram(rij_matrix.flatten(), bins=bins)
        rij_depletion_hist, _ = np.histogram(rij_depletion_matrix.flatten(), bins=bins)
    except Exception as e:
        log.exception(e)
        C = np.nan
        rij_hist = np.ones(20) * np.nan
        rij_depletion_hist = np.ones(20) * np.nan

    res["sys_functional_complexity"] = C
    res["vec_sys_hvals_correlation_coefficients"] = rij_hist.copy()
    res["vec_sys_hvals_depletion_correlation_coefficients"] = rij_depletion_hist.copy()

    # module level correlations.
    # Above we considered rij between pairs of neurons. here we take module-level
    # firing rates.
    try:
        rij_mod_level = ah.find_rij(h5f, which="modules")
        C, rij_mod_level = ah.find_functional_complexity(
            h5f, which="modules", return_res=True, write_to_h5f=False, bins=bins
        )
    except Exception as e:
        log.exception(e)
        C = np.nan
    res["any_functional_complexity"] = C

    try:
        np.fill_diagonal(rij_mod_level, np.nan)
        res["mod_mean_correlation"] = np.nanmean(rij_mod_level)
        res["mod_median_correlation"] = np.nanmedian(rij_mod_level)
    except:
        res["mod_mean_correlation"] = np.nan
        res["mod_median_correlation"] = np.nan

    # ------------------------------------------------------------------------------ #
    # Correlation coefficients for different pairings
    # ------------------------------------------------------------------------------ #

    cases = ["within_stim", "within_nonstim", "across"]
    for case in cases:
        try:
            if case == "within_stim":
                mod_rij_paired = ah.find_rij_pairs(
                    h5f, rij=rij_mod_level, pairing="across_groups_0_2", which="modules"
                )
                neuron_rij_paired = ah.find_rij_pairs(
                    h5f, rij=rij_matrix, pairing="across_groups_0_2", which="neurons"
                )

            elif case == "within_nonstim":
                mod_rij_paired = ah.find_rij_pairs(
                    h5f, rij=rij_mod_level, pairing="across_groups_1_3", which="modules"
                )
                neuron_rij_paired = ah.find_rij_pairs(
                    h5f, rij=rij_matrix, pairing="across_groups_1_3", which="neurons"
                )

            elif case == "across":
                mod_rij_paired = []
                mod_rij_paired.extend(
                    ah.find_rij_pairs(
                        h5f,
                        rij=rij_mod_level,
                        pairing="across_groups_0_1",
                        which="modules",
                    )
                )
                mod_rij_paired.extend(
                    ah.find_rij_pairs(
                        h5f,
                        rij=rij_mod_level,
                        pairing="across_groups_2_3",
                        which="modules",
                    )
                )

                neuron_rij_paired = []
                neuron_rij_paired.extend(
                    ah.find_rij_pairs(
                        h5f,
                        rij=rij_matrix,
                        pairing="across_groups_0_1",
                        which="neurons",
                    )
                )
                neuron_rij_paired.extend(
                    ah.find_rij_pairs(
                        h5f,
                        rij=rij_matrix,
                        pairing="across_groups_2_3",
                        which="neurons",
                    )
                )

            res[f"mod_median_correlation_{case}"] = np.nanmedian(mod_rij_paired)
            res[f"mod_mean_correlation_{case}"] = np.nanmean(mod_rij_paired)
            # I know, there are some variable naming inconstencies here.
            res[f"sys_median_correlation_{case}"] = np.nanmedian(neuron_rij_paired)
            res[f"sys_mean_correlation_{case}"] = np.nanmean(neuron_rij_paired)
        except:
            res[f"mod_median_correlation_{case}"] = np.nan
            res[f"mod_mean_correlation_{case}"] = np.nan
            res[f"sys_median_correlation_{case}"] = np.nan
            res[f"sys_mean_correlation_{case}"] = np.nan

    # ------------------------------------------------------------------------------ #
    # Event size
    # This is the fraction of neurons (in the whole system) involved in a
    # (bursting) event. I started off simply calling it "fraction",
    # and refactoring might break a lot of code.
    # ------------------------------------------------------------------------------ #

    try:
        ah.find_participating_fraction_in_bursts(h5f)
        fracs = h5f["ana.bursts.system_level.participating_fraction"]
        rij_hist, _ = np.histogram(fracs, bins=bins)
    except Exception as e:
        log.exception(e)
        fracs = np.array([np.nan])
        rij_hist = np.ones(20) * np.nan
    res["sys_mean_participating_fraction"] = np.nanmean(fracs)
    res["sys_median_participating_fraction"] = np.nanmedian(fracs)
    res["vec_sys_hvals_participating_fraction"] = rij_hist.copy()

    # this is the same as above but instead of using rij histograms (functional complxt.)
    # we use histograms of the fraction
    try:
        fractions = h5f["ana.bursts.system_level.participating_fraction"]
        C = ah._functional_complexity(np.array(fractions), num_bins=20)
    except Exception as e:
        log.exception(e)
        C = np.nan
    res["sys_participating_fraction_complexity"] = C

    # How many spikes were fired in a burst. this is normalized per contributing neuron
    try:
        C = np.nanmean(h5f["ana.bursts.system_level.num_spikes_in_bursts"])
    except Exception as e:
        log.exception(e)
        C = np.nan
    res["any_num_spikes_in_bursts"] = C

    try:
        spks = np.array(h5f["ana.bursts.system_level.num_spikes_in_bursts"])
        C = np.nanmean(spks[np.where(slen == 1)[0]])
    except:
        C = np.nan
    res["mod_num_spikes_in_bursts_1"] = C

    # Before setteling on charge-discharge cycles,
    # we tried to come up with order parameters to describe the dynamics of the resources.
    # Here a few candidates
    # - "fano_neuron": fano factor for every neuron, then average across neurons
    # - "fano_population": fano factor of the population resources (avg neurons first)
    # - "baseline_neuron": max resources found per neuron
    # - "baseline_population": max resources on population average
    # - "dist ... " : the distribution of resource values (histogram across time)
    #                 and different percentiles.
    ops = ah.find_resource_order_parameters(h5f)
    res["sys_orderpar_fano_neuron"] = ops["fano_neuron"]
    res["sys_orderpar_fano_population"] = ops["fano_population"]
    res["sys_orderpar_baseline_neuron"] = ops["baseline_neuron"]
    res["sys_orderpar_baseline_population"] = ops["baseline_population"]

    res["vec_sys_hbins_resource_dist"] = ops["dist_edges"]
    res["vec_sys_hvals_resource_dist"] = ops["dist_hist"]

    res["sys_orderpar_dist_low_end"] = ops["dist_low_end"]
    res["sys_orderpar_dist_low_mid"] = ops["dist_low_mid"]
    res["sys_orderpar_dist_high_mid"] = ops["dist_high_mid"]
    res["sys_orderpar_dist_high_end"] = ops["dist_high_end"]
    res["sys_orderpar_dist_median"] = ops["dist_median"]
    res["sys_orderpar_dist_max"] = ops["dist_max"]

    # ------------------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------------------ #

    # What is the amount of resources at the time of bursts starting?
    # first dim modules, second dim times, nans if not part of burst
    resources = ah.find_module_resources_at_burst_begin(
        h5f, write_to_h5f=False, return_res=True, nan_if_not_participating=False
    )

    if include_ragged_arrays:
        res["rag_resources_mod_0"] = resources[0, :]
        res["rag_resources_mod_1"] = resources[1, :]
    else:
        res["rag_resources_mod_0"] = np.array([])
        res["rag_resources_mod_1"] = np.array([])

    # this is about module-level resource cycles, so we want to treat all modules
    # as the ensemble -> flat list and then std and mean
    resources = resources.flatten()
    res["sys_mean_resources_at_burst_beg"] = np.nanmean(resources)
    res["sys_std_resources_at_burst_beg"] = np.nanstd(resources)

    # ------------------------------------------------------------------------------ #
    # Topology, out degrees for each neuron, sorted whether neuron is bridging or not
    # ------------------------------------------------------------------------------ #
    try:
        b_ids = h5f["data.neuron_bridge_ids"][:]
    except:
        # merged topo has no bridge ids
        b_ids = np.array([], dtype="int")
    nb_ids = np.isin(h5f["ana.neuron_ids"], b_ids, invert=True)

    bin_edges = np.arange(0, 161) - 0.5  # no self-cupling was allowed
    hist, _ = np.histogram(h5f["data.neuron_k_out"][nb_ids], bins=bin_edges)
    res["vec_sys_hbins_kout_no_bridge"] = bin_edges
    res["vec_sys_hvals_kout_no_bridge"] = hist

    hist, _ = np.histogram(h5f["data.neuron_k_out"][b_ids], bins=bin_edges)
    res["vec_sys_hbins_kout_yes_bridge"] = bin_edges
    res["vec_sys_hvals_kout_yes_bridge"] = hist

    # ------------------------------------------------------------------------------ #
    # ragged arrays, for bursts
    # ------------------------------------------------------------------------------ #

    if include_ragged_arrays:
        res["rag_burst_beg_times"] = h5f["ana.bursts.system_level.beg_times"]
        res["rag_burst_end_times"] = h5f["ana.bursts.system_level.end_times"]
        res["rag_burst_ibis"] = h5f["ana.ibi.system_level.any_module"]
        # if we use the ana.bursts.system_level.core_delays, we'd have an array for each burst
        res["rag_burst_core_delays"] = h5f["ana.bursts.system_level.core_delays_mean"]
        # we can now save sequences as single ints, see ana_helper seq_to_int()
        res["rag_burst_seqs"] = np.array(
            [ah.seq_to_int(s) for s in h5f["ana.bursts.system_level.module_sequences"]],
            dtype="int",
        )
    else:
        res["rag_burst_beg_times"] = np.array([])
        res["rag_burst_end_times"] = np.array([])
        res["rag_burst_ibis"] = np.array([])
        res["rag_burst_core_delays"] = np.array([])
        res["rag_burst_seqs"] = np.array([])

    h5.close_hot(h5f)
    h5f.clear()

    return res

    # ------------------------------------------------------------------------------ #
    # End of single-file analysis
    # ------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------------------ #
futures = None


def main(args, dask_client):
    """
    This relies on some global dask variables to be initialized.
    """

    # if a directory is provided as input, merge individual hdf5 files down
    if os.path.isdir(args.input_path):
        candidates = glob.glob(full_path(args.input_path + "/*.hdf5"))
        log.info(f"{args.input_path} is a directory, using contained hdf5 files")
    elif len(glob.glob(full_path(args.input_path))) <= 1:
        log.error(
            "Provide a directory with hdf5 files or wildcarded path as string:"
            " 'path/to/file_ptrn*.hdf5''"
        )
        sys.exit()
    else:
        candidates = glob.glob(full_path(args.input_path))
        log.info(f"{args.input_path} is a (list of) file")

    merge_path = full_path(args.output_path)
    log.info(f"Merging (overwriting) to {merge_path}")

    # which values occur across files
    d_axes = OrderedDict()
    for obs in d_obs.keys():
        d_axes[obs] = []

    # check what's in the files and create axes labels for n-dim tensor
    log.info(f"Checking {len(candidates)} files:")

    # set arguments for variables needed by every worker
    f = functools.partial(check_candidate, d_obs=d_obs)

    # dispatch, reading in parallel may be faster
    futures = dask_client.map(f, candidates)

    # gather
    l_valid = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        res, candidate = future.result()
        log.debug(f"{candidate} -> {res}")
        if res is None:
            log.warning(f"file seems invalid: {candidate}")
        else:
            l_valid.append(candidate)
            for odx, obs in enumerate(d_obs.keys()):
                val = res[odx]

                # somehow `is nan in [nan, nan]` is dodgy, works better with np.nan
                if isinstance(val, float) and np.isnan(val):
                    val = np.nan

                if val not in d_axes[obs]:
                    d_axes[obs].append(val)

    # sort axes and count unique axes entries
    axes_size = 1
    axes_shape = ()
    for obs in d_axes.keys():
        try:
            d_axes[obs] = np.array(sorted(d_axes[obs]))
        except:
            log.debug(f"Could not sort {obs} axis, keeping as is")
        axes_size *= len(d_axes[obs])
        axes_shape += (len(d_axes[obs]),)

    log.info(f"Found axes: {d_axes}")

    # we might have repetitions but estimating num_rep proved unreliable.
    log.info(f"Finding number of repetitions:")
    # num_rep = int(np.ceil(len(l_valid) / axes_size))
    sampled = np.zeros(shape=axes_shape, dtype=int)
    for candidate in tqdm(candidates):
        index = ()
        for obs in d_axes.keys():
            # get value
            temp = h5.load(candidate, d_obs[obs], silent=True)
            # same problem as above, nan == nan is dodgy
            if isinstance(temp, float) and np.isnan(temp):
                temp = np.where(np.isnan(d_axes[obs]))[0][0]
            else:
                temp = np.where(d_axes[obs] == temp)[0][0]
            index += (temp,)

        sampled[index] += 1

    num_rep = np.max(sampled)
    sampled = np.zeros(shape=axes_shape, dtype=int)

    log.info(f"Found axes:")
    for obs in d_axes.keys():
        log.info(f"{obs: >10}: {d_axes[obs]}")
    log.info(f"Repetitions: {num_rep}")

    # ------------------------------------------------------------------------------ #
    # write to a new hdf5 file, meta data first
    # ------------------------------------------------------------------------------ #

    try:
        os.makedirs(os.path.dirname(merge_path), exist_ok=True)
    except:
        pass

    with h5py.File(merge_path, "w", swmr=True) as f_tar:

        # contained axis in right order
        # workaround to store a list of strings (via object array) to hdf5
        dset = f_tar.create_dataset(
            "/meta/axis_overview",
            data=np.array(list(d_axes.keys()) + ["repetition"], dtype=object),
            dtype=h5py.special_dtype(vlen=str),
        )
        dset.attrs["description"] = (
            "ordered list of all coordinates (axis) spanning data."
            + "if observable name starts with `vec`, another dim follows the repetitions."
        )

        desc_axes = f"{len(axes_shape)+1}-dim array with axis: "
        for obs in d_axes.keys():
            dset = f_tar.create_dataset("/meta/axis_" + obs, data=d_axes[obs])
            desc_axes += obs + ", "

        dset = f_tar.create_dataset(
            "/meta/axis_repetition", data=np.arange(0, num_rep), dtype="int"
        )

        # meta data
        dset = f_tar.create_dataset(
            "/meta/ana_par/threshold_factor", data=threshold_factor
        )
        dset = f_tar.create_dataset("/meta/ana_par/smoothing_width", data=smoothing_width)
        dset = f_tar.create_dataset("/meta/num_samples", compression="gzip", data=sampled)
        dset.attrs["description"] = "measured number of repetitions"

        # ------------------------------------------------------------------------------ #
        # Main analysis loop
        # ------------------------------------------------------------------------------ #

        log.info(f"Analysing:")

        # key -> h5 dsets
        res_ndim = dict()

        # get a reference of the keys (obersvables) we will encounter
        res_ref = ana_for_single_rep(None)

        # setup data storage
        for key in res_ref.keys():
            if key[0:3] != "vec" and key[0:3] != "rag":
                # keep repetitions always as the last axes for scalars,
                vals = np.ones(shape=axes_shape + (num_rep,)) * np.nan
                res_ndim[key] = f_tar.create_dataset(
                    "/data/" + key, compression="gzip", data=vals
                )
            elif key[0:3] == "vec":
                # vectors (inconsistent length across observables), keep data-dim last
                # only floating-type data supported, thus we return only the length
                vals = np.ones(shape=axes_shape + (num_rep, res_ref[key])) * np.nan
                res_ndim[key] = f_tar.create_dataset(
                    "/data/" + key, compression="gzip", data=vals
                )
            elif key[0:3] == "rag":
                # ragged arrays (inconsistent length across repetitions), data-dim last
                try:
                    dtype = np.dtype(type(res_ref[key]))
                except:
                    dtype = "float64"
                log.debug(f"ragged array {key} has dtype {dtype}")
                shape = axes_shape + (num_rep,)
                res_ndim[key] = f_tar.create_dataset(
                    "/data/" + key, shape=shape, dtype=h5py.vlen_dtype(dtype)
                )

        # set arguments for variables needed by every worker
        f = functools.partial(analyse_candidate, d_axes=d_axes, d_obs=d_obs)

        # dispatch
        futures = dask_client.map(f, candidates)

        # for some analysis, we need repetitions to be indexed consistently
        # but we have to find them from filename since I usually do not save rep to meta
        # Note: repetitions have to go from 0=N, our indexing is rudimentary.
        reps_from_files = True

        # check first file, if we cannot infer there, skip ordering everywhere
        if find_rep(candidates[0]) is None:
            reps_from_files = False

        # gather
        for future in tqdm(as_completed(futures), total=len(futures)):
            index, rep, res = future.result()

            if reps_from_files:
                assert rep is not None, "Could not infer repetition from all file names"
            else:
                # get rep from already counted repetitions
                rep = sampled[index]

            # consider repetitions for each data point and stack values
            if rep <= num_rep:
                sampled[index] += 1
                index += (rep,)

                for key in res.keys():
                    try:
                        res_ndim[key][index] = res[key]
                    except Exception as e:
                        log.exception(f"{key} at {index} (rep {rep})")
                        log.exception(e)
            else:
                log.debug(f"unexpected repetition {rep} for index {index}")

        if not np.all(sampled == sampled.flat[0]):
            log.info(
                f"repetitions vary across data points, from {np.min(sampled)} to"
                f" {np.max(sampled)}"
            )

    return res_ndim, d_obs, d_axes


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def parse_arguments():

    # we use global variables for the threshold so we dont have to pass them
    # through a bunch of calls. not neat but yay, scientific coding.
    global threshold_factor
    global smoothing_width
    global time_bin_size_for_rij
    global remove_null_sequences

    parser = argparse.ArgumentParser(description="Merge Multidm")
    parser.add_argument(
        "-i",
        dest="input_path",
        required=True,
        help="input path with *.hdf5 files",
        metavar="FILE",
    )
    parser.add_argument(
        "-o", dest="output_path", help="output path", metavar="FILE", required=True
    )
    parser.add_argument(
        "-c",
        "--cores",
        dest="num_cores",
        help="number of dask cores",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-t",
        dest="threshold_factor",
        help="% of peak height to use for thresholding of burst detection",
        default=threshold_factor,
        type=float,
    )
    parser.add_argument(
        "-s",
        dest="smoothing_width",
        help="% of peak height to use for thresholding of burst detection",
        default=smoothing_width,
        type=float,
    )
    parser.add_argument(
        "-r",
        dest="time_bin_size_for_rij",
        help="bin size for spike counting, for correlation coefficients, in seconds",
        default=time_bin_size_for_rij,
        type=float,
    )

    args = parser.parse_args()
    threshold_factor = args.threshold_factor
    smoothing_width = args.smoothing_width
    time_bin_size_for_rij = args.time_bin_size_for_rij

    log.info(args)

    return args


# Find the repetition from the file name
def find_rep(candidate):
    search = re.search("((rep=*)|(_r=*))(\d+)", candidate, re.IGNORECASE)
    if search is not None and len(search.groups()) != 0:
        return search.groups()[-1]


# we have to pass global variables so that they are available in each worker.
# simple set them via frunctools.partial so only `candidate` varies between workers
def analyse_candidate(candidate, d_axes, d_obs):
    # for candidate in tqdm(l_valid, desc="Files"):
    index = ()
    for obs in d_axes.keys():
        # get value
        temp = h5.load(candidate, d_obs[obs], silent=True)
        if isinstance(temp, float) and np.isnan(temp):
            temp = np.where(np.isnan(d_axes[obs]))[0][0]
        else:
            temp = np.where(d_axes[obs] == temp)[0][0]
        index += (temp,)

        # psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    res = ana_for_single_rep(candidate)
    rep = find_rep(candidate)
    try:
        rep = int(rep)
    except:
        rep = None

    return index, rep, res


def check_candidate(candidate, d_obs):
    # returns a list matching the keys in d_obs
    # with a value for each key
    res = []
    for obs in d_obs.keys():
        try:
            temp = h5.load(candidate, d_obs[obs], silent=True)
            try:
                # sometimes we find nested arrays
                if len(temp == 1):
                    temp = temp[0]
            except:
                # was a number already
                pass

            res.append(temp)
        except:
            # something was fishy with this file, do not use it!
            return None, candidate

    return res, candidate


def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


if __name__ == "__main__":
    args = parse_arguments()

    with ExitStack() as stack:
        # init dask using a context manager to ensure proper clenaup
        # when using remote compute
        dask_cluster = stack.enter_context(
            # rudabeh
            # TODO: remove this before release
            SGECluster(
               cores=32,
                memory="192GB",
                processes=16,
                job_extra=["-pe mvapich2-sam 32"],
                log_directory="/scratch01.local/pspitzner/dask/logs",
                local_directory="/scratch01.local/pspitzner/dask/scratch",
                interface="ib0",
               walltime='02:30:00',
                extra=[
                    '--preload \'import sys; sys.path.append("./ana/"); sys.path.append("/home/pspitzner/code/pyhelpers/");\''
                ],
            )
            # local cluster
            # LocalCluster(local_directory=f"{tempfile.gettempdir()}/dask/")
        )
        dask_cluster.scale(cores=args.num_cores)

        dask_client = stack.enter_context(Client(dask_cluster))

        with logging_redirect_tqdm():
            res_ndim, d_obs, d_axes = main(args, dask_client)
