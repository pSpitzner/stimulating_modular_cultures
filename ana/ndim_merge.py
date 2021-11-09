# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-11-09 14:48:06
#
# Scans the provided directory for .hdf5 files and merges individual realizsation
# into an ndim array
#
# use `ndim_plot.py` to visualize the result
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
from tqdm import tqdm
from benedict import benedict

# from addict import Dict

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

import dask_helper as dh
import ana_helper as ah
import hi5 as h5


# ------------------------------------------------------------------------------ #
# settings
# ------------------------------------------------------------------------------ #

# variables to span axes and how to get them from the hdf5 files
d_obs = OrderedDict()
d_obs["jA"] = "/meta/dynamics_jA"
d_obs["jG"] = "/meta/dynamics_jG"
# d_obs["jM"] = "/meta/dynamics_jM"
# d_obs["jE"] = "/meta/dynamics_jE"
d_obs["rate"] = "/meta/dynamics_rate"
d_obs["tD"] = "/meta/dynamics_tD"
# d_obs["alpha"] = "/meta/topology_alpha"
# d_obs["k_inter"] = "/meta/topology_k_inter"
# d_obs["k_frac"] = "/meta/dynamics_k_frac"

threshold_factor = 0.025
smoothing_width = 20 / 1000
remove_null_sequences = False

# functions for analysis. candidate is the file path (e.g. to a hdf5 file)
# need to return a dict where the key becomes the hdf5 data set name
# and a scalar entry as the value
# todo: add description
def all_in_one(candidate=None):
    if candidate is None:
        res = dict()
        # scalars
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
        res["sys_mean_ibis"] = 1
        res["sys_median_ibis"] = 1
        res["any_ibis"] = 1
        res["sys_ibis_cv"] = 1
        res["any_ibis_cv"] = 1
        res["sys_functional_complexity"] = 1
        res["any_functional_complexity"] = 1
        res["sys_mean_correlation"] = 1
        res["sys_median_correlation"] = 1
        res["sys_mean_participating_fraction"] = 1
        res["sys_median_participating_fraction"] = 1
        res["sys_participating_fraction_complexity"] = 1
        res["any_num_spikes_in_bursts"] = 1

        # histograms, use "vec" prefix to indicate that higher dimensional data
        res["vec_sys_hbins_participating_fraction"] = 21
        res["vec_sys_hvals_participating_fraction"] = 20
        res["vec_sys_hbins_correlation_coefficients"] = 21
        res["vec_sys_hvals_correlation_coefficients"] = 20

        # correlation coefficients, within
        for mod in [0, 1, 2, 3]:
            # 40 neurons per module, count pairs only once, exclude 20 self-to-self
            # 40^2 / 2 - 20
            res[f"vec_rij_within_{mod}"] = 780

        # correlation coefficients, across
        for pair in itertools.combinations("0123", 2):
            # here just 40^2, since in different modules
            res[f"vec_rij_across_{pair[0]}_{pair[1]}"] = 1600

        return res

    # load and process
    h5f = ah.prepare_file(
        candidate, hot=False, skip=["connectivity_matrix", "connectivity_matrix_sparse"]
    )
    ah.find_rates(h5f, bs_large=smoothing_width)
    threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])

    # this also finds module sequences, currently when at least 20% of module neurons
    # fired at least one spike.
    ah.find_system_bursts_from_global_rate(h5f, rate_threshold=threshold, merge_threshold=0.1)
    # ah.find_bursts_from_rates(h5f, rate_threshold=5.0)
    if remove_null_sequences:
        ah.remove_bursts_with_sequence_length_null(h5f)
    ah.find_ibis(h5f)

    # number of bursts and duration
    res = dict()
    res["sys_rate_threshold"] = threshold
    res["any_num_b"] = len(h5f["ana.bursts.system_level.beg_times"])
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
    res["sys_rate_cv"] = h5f["ana.rates.cv.system_level"]
    res["sys_mean_rate"] = np.nanmean(h5f["ana.rates.system_level"])

    slen = np.array([len(x) for x in h5f["ana.bursts.system_level.module_sequences"]])
    blen = np.array(h5f["ana.bursts.system_level.end_times"]) - np.array(
        h5f["ana.bursts.system_level.beg_times"]
    )
    res["sys_blen"] = np.nanmean(blen)
    res["mod_blen_0"] = np.nanmean(blen[np.where(slen == 0)[0]])
    res["mod_blen_1"] = np.nanmean(blen[np.where(slen == 1)[0]])
    res["mod_blen_2"] = np.nanmean(blen[np.where(slen == 2)[0]])
    res["mod_blen_3"] = np.nanmean(blen[np.where(slen == 3)[0]])
    res["mod_blen_4"] = np.nanmean(blen[np.where(slen == 4)[0]])

    # ibis
    try:
        res["sys_mean_ibis"] = np.nanmean(h5f["ana.ibi.system_level.all_modules"])
        res["sys_median_ibis"] = np.nanmedian(h5f["ana.ibi.system_level.all_modules"])
        res["any_ibis_cv"] = np.nanmean(h5f["ana.ibi.system_level.cv_any_module"])
        res["sys_ibis_cv"] = np.nanmean(h5f["ana.ibi.system_level.cv_across_modules"])
        ibis_module = []
        for m_dc in h5f["ana.ibi.module_level"].keys():
            ibis_module.extend(h5f["ana.ibi.module_level"][m_dc])
        res["any_ibis"] = np.nanmean(ibis_module)
    except KeyError as e:
        log.debug(e)

    # correlation coefficients
    try:
        rij_matrix = ah.find_rij(h5f, which="neurons", time_bin_size=200 / 1000)
    except:
        log.error("Failed to find correlation coefficients")

    for mod in [0, 1, 2, 3]:
        try:
            res[f"vec_rij_within_{mod}"] = ah.find_rij_pairs(
                h5f, rij_matrix, pairing=f"within_group_{mod}"
            )
        except Exception as e:
            log.error(e)
            res[f"vec_rij_within_{mod}"] = np.ones(780) * np.nan

    # correlation coefficients, across
    for pair in itertools.combinations("0123", 2):
        # here just 40^2, since in different modules
        try:
            res[f"vec_rij_across_{pair[0]}_{pair[1]}"] = ah.find_rij_pairs(
                h5f, rij_matrix, pairing=f"across_groups_{pair[0]}_{pair[1]}"
            )
        except Exception as e:
            log.error(e)
            res[f"vec_rij_across_{pair[0]}_{pair[1]}"] = np.ones(1600) * np.nan

    # functional complexity
    bw = 1.0 / 20
    bins = np.arange(0, 1 + 0.1 * bw, bw)
    res["vec_sys_hbins_correlation_coefficients"] = bins.copy()
    res["vec_sys_hbins_participating_fraction"] = bins.copy()

    try:
        C, rij = ah.find_functional_complexity(
            h5f, rij=rij_matrix, return_res=True, write_to_h5f=False, bins=bins
        )
        # this is not the place to do this, but ok
        np.fill_diagonal(rij, np.nan)
        rij, _ = np.histogram(rij.flatten(), bins=bins)
    except Exception as e:
        log.debug(e)
        C = np.nan
        rij = np.ones(21) * np.nan

    res["sys_functional_complexity"] = C
    res["vec_sys_hvals_correlation_coefficients"] = rij.copy()
    res["sys_mean_correlation"] = np.nanmean(rij)
    res["sys_median_correlation"] = np.nanmedian(rij)

    # complexity on module level
    try:
        C, _ = ah.find_functional_complexity(
            h5f, which="modules", return_res=True, write_to_h5f=False, bins=bins
        )
    except Exception as e:
        log.debug(e)
        C = np.nan
    res["any_functional_complexity"] = C

    # participating fraction
    try:
        ah.find_participating_fraction_in_bursts(h5f)
        fracs = h5f["ana.bursts.system_level.participating_fraction"]
        rij, _ = np.histogram(fracs, bins=bins)
    except Exception as e:
        log.debug(e)
        fracs = np.array([np.nan])
        rij = np.ones(21) * np.nan
    res["sys_mean_participating_fraction"] = np.nanmean(fracs)
    res["sys_median_participating_fraction"] = np.nanmedian(fracs)
    res["vec_sys_hvals_participating_fraction"] = rij.copy()

    try:
        fractions = h5f["ana.bursts.system_level.participating_fraction"]
        C = ah._functional_complexity(np.array(fractions), num_bins=20)
    except Exception as e:
        log.debug(e)
        C = np.nan
    res["sys_participating_fraction_complexity"] = C

    try:
        C = np.nanmean(h5f["ana.bursts.system_level.num_spikes_in_bursts"])
    except Exception as e:
        log.debug(e)
        C = np.nan
    res["any_num_spikes_in_bursts"] = C

    try:
        spks = np.array(h5f["ana.bursts.system_level.num_spikes_in_bursts"])
        C = np.nanmean(spks[np.where(slen == 1)[0]])
    except:
        C = np.nan
    res["mod_num_spikes_in_bursts_1"] = C

    h5.close_hot(h5f)
    h5f.clear()

    return res


# ------------------------------------------------------------------------------ #
# arguments
# ------------------------------------------------------------------------------ #


def parse_arguments():

    # we use global variables for the threshold so we dont have to pass them
    # through a bunch of calls. not neat.
    global threshold_factor
    global smoothing_width

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
        default=256,
        type=int,
    )
    parser.add_argument(
        "-t",
        dest="threshold_factor",
        help="% of peak height to use for thresholding of burst detection",
        default=threshold_factor,
        type=float
    )
    parser.add_argument(
        "-s",
        dest="smoothing_width",
        help="% of peak height to use for thresholding of burst detection",
        default=smoothing_width,
        type=float
    )

    args = parser.parse_args()
    threshold_factor = args.threshold_factor
    smoothing_width = args.smoothing_width

    log.info(args)

    return args


# ------------------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------------------ #
futures = None


def main(args):

    # if a directory is provided as input, merge individual hdf5 files down
    if os.path.isdir(args.input_path):
        candidates = glob.glob(full_path(args.input_path + "/*.hdf5"))
        log.info(f"{args.input_path} is a directory, using contained hdf5 files")
    elif len(glob.glob(full_path(args.input_path))) <= 1:
        log.error(
            "Provide a directory with hdf5 files or wildcarded path as string: 'path/to/file_ptrn*.hdf5''"
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
    futures = dh.client.map(f, candidates)

    # gather
    l_valid = []
    for future in tqdm(dh.as_completed(futures), total=len(futures)):
        res, candidate = future.result()
        if res is None:
            log.warning(f"file seems invalid: {candidate}")
        else:
            l_valid.append(candidate)
            for odx, obs in enumerate(d_obs.keys()):
                val = res[odx]
                if val not in d_axes[obs]:
                    d_axes[obs].append(val)

    # for candidate in tqdm(candidates):
    #     # todo parallelize this
    #     try:
    #         for obs in d_obs:
    #             temp = h5.load(candidate, d_obs[obs], silent=True)
    #             try:
    #                 # sometimes we find nested arrays
    #                 if len(temp == 1):
    #                     temp = temp[0]
    #             except:
    #                 pass
    #             if temp not in d_axes[obs]:
    #                 d_axes[obs].append(temp)
    #         l_valid.append(candidate)
    #     except Exception as e:
    #         log.error(f"incompatible file: {candidate}")

    # sort axes and count unique axes entries
    axes_size = 1
    axes_shape = ()
    for obs in d_axes.keys():
        d_axes[obs] = np.array(sorted(d_axes[obs]))
        axes_size *= len(d_axes[obs])
        axes_shape += (len(d_axes[obs]),)

    # we might have repetitions but estimating num_rep proved unreliable.
    log.info(f"Finding number of repetitions:")
    # num_rep = int(np.ceil(len(l_valid) / axes_size))
    sampled = np.zeros(shape=axes_shape, dtype=int)
    for candidate in tqdm(candidates):
        index = ()
        for obs in d_axes.keys():
            # get value
            temp = h5.load(candidate, d_obs[obs], silent=True)
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
    # res_ndim
    # ------------------------------------------------------------------------------ #
    log.info(f"Analysing:")
    res_ndim = dict()
    # dict of key -> needed space
    res_dict = all_in_one(None)
    for key in res_dict.keys():
        # keep repetitions always as the last axes for scalars,
        # but for vectors (inconsistent length across observables), keep data-dim last
        if key[0:3] == "vec":
            res_ndim[key] = (
                np.ones(shape=axes_shape + (num_rep, res_dict[key])) * np.nan
            )
        else:
            res_ndim[key] = np.ones(shape=axes_shape + (num_rep,)) * np.nan

    # set arguments for variables needed by every worker
    f = functools.partial(analyse_candidate, d_axes=d_axes, d_obs=d_obs)

    # dispatch
    futures = dh.client.map(f, candidates)

    # for some analysis, we need repetitions to be indexed consistently
    # but we have to find them from filename since I usually do not save rep to meta
    reps_from_files = True

    # check first file, if we cannot infer there, skip ordering everywhere
    if find_rep(candidates[0]) is None:
        reps_from_files = False

    # gather
    for future in tqdm(dh.as_completed(futures), total=len(futures)):
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
                res_ndim[key][index] = res[key]
        else:
            log.error(f"unexpected repetitions (already read {num_rep}): {index}")

    if not np.all(sampled == sampled.flat[0]):
        log.info(
            f"repetitions vary across data points, from {np.min(sampled)} to {np.max(sampled)}"
        )

    # ------------------------------------------------------------------------------ #
    # write to a new hdf5 file
    # ------------------------------------------------------------------------------ #

    try:
        os.makedirs(os.path.dirname(merge_path), exist_ok=True)
    except:
        pass

    f_tar = h5py.File(merge_path, "w")

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

    # desc_axes += "repetition"
    dset = f_tar.create_dataset("/meta/axis_repetition",
        data=np.arange(0, num_rep), dtype="int")

    for key in res_ndim.keys():
        dset = f_tar.create_dataset(f"/data/{key}", data=res_ndim[key])
        if "hbins" in key:
            dset.attrs[
                "description"
            ] = "like scalars, but last dim are histogram bin edges"
        elif "weights" in key:
            dset.attrs[
                "description"
            ] = "like scalars, but last dim are histogram weights"
        else:
            dset.attrs["description"] = desc_axes

    dset = f_tar.create_dataset("/meta/num_samples", data=sampled)
    dset.attrs["description"] = "measured number of repetitions"

    # meta data
    dset = f_tar.create_dataset("/meta/ana_par/threshold_factor",
        data=threshold_factor)
    dset = f_tar.create_dataset("/meta/ana_par/smoothing_width",
        data=smoothing_width)

    f_tar.close()

    return res_ndim, d_obs, d_axes


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #

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
        # transform to index
        temp = np.where(d_axes[obs] == temp)[0][0]
        index += (temp,)

        # psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    res = all_in_one(candidate)
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
    dh.init_dask(args.num_cores)
    res_ndim, d_obs, d_axes = main(args)
