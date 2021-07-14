# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-07-14 15:46:07
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
import tempfile

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
d_obs["rate"] = "/meta/dynamics_rate"
# d_obs["tD"] = "/meta/dynamics_tD"
# d_obs["alpha"] = "/meta/topology_alpha"
# d_obs["k_inter"] = "/meta/topology_k_inter"
# d_obs["k_frac"] = "/meta/dynamics_k_frac"

# functions for analysis. candidate is the file path (e.g. to a hdf5 file)
# need to return a dict where the key becomes the hdf5 data set name
# and a scalar entry as the value
# todo: add description
def all_in_one(candidate=None):
    if candidate is None:
        return [
            "any_num_b",
            "mod_num_b_1",
            "mod_num_b_2",
            "mod_num_b_4",
            "sys_rate_cv",
            "sys_mean_rate",
            "sys_blen",
            "mod_blen_1",
            "mod_blen_2",
            "mod_blen_4",
            "sys_ibis",
            "any_ibis",
            "sys_ibis_cv",
            "any_ibis_cv",
            "sys_functional_complexity",
            "any_functional_complexity",
            "sys_participating_fraction",
            "sys_participating_fraction_complexity",
        ]

    # load and process
    h5f = ah.prepare_file(
        candidate, hot=False, skip=["connectivity_matrix", "connectivity_matrix_sparse"]
    )
    ah.find_bursts_from_rates(h5f, rate_threshold = 2.5)
    ah.find_ibis(h5f)

    res = dict()
    res["any_num_b"] = len(h5f["ana.bursts.system_level.beg_times"])
    res["mod_num_b_1"] = len(
        [x for x in h5f["ana.bursts.system_level.module_sequences"] if len(x) == 1]
    )
    res["mod_num_b_2"] = len(
        [x for x in h5f["ana.bursts.system_level.module_sequences"] if len(x) == 2]
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
    res["mod_blen_1"] = np.nanmean(blen[np.where(slen == 1)[0]])
    res["mod_blen_2"] = np.nanmean(blen[np.where(slen >= 2)[0]])
    res["mod_blen_4"] = np.nanmean(blen[np.where(slen == 4)[0]])

    # ibis
    try:
        ibis_module = []
        for m_dc in h5f["ana.ibi.module_level"].keys():
            ibis_module.extend(h5f["ana.ibi.module_level"][m_dc])
        res["any_ibis"] = np.nanmean(ibis_module)
        res["sys_ibis"] = np.nanmean(h5f["ana.ibi.system_level.all_modules"])
        res["any_ibis_cv"] = np.nanmean(h5f["ana.ibi.system_level.cv_any_module"])
        res["sys_ibis_cv"] = np.nanmean(
            h5f["ana.ibi.system_level.cv_across_modules"]
        )
    except KeyError as e:
        log.error(h5f.keypaths())
        raise e

    try:
        C, _ =  ah.find_functional_complexity(
            h5f, which="neurons", return_res=True, write_to_h5f=False
        )
    except Exception as e:
        log.debug(e)
        C = np.nan
    res["sys_functional_complexity"] = C

    try:
        C, _ =  ah.find_functional_complexity(
            h5f, which="modules", return_res=True, write_to_h5f=False
        )
    except Exception as e:
        log.debug(e)
        C = np.nan
    res["any_functional_complexity"] = C

    try:
        ah.find_participating_fraction_in_bursts(h5f)
        C = np.nanmean(h5f["ana.bursts.system_level.participating_fraction"])
    except Exception as e:
        log.debug(e)
        C = np.nan
    res["sys_participating_fraction"] = C

    try:
        fractions = h5f["ana.bursts.system_level.participating_fraction"]
        C = ah._functional_complexity(np.array(fractions) num_bins=20)
    except Exception as e:
        log.debug(e)
        C = np.nan
    res["sys_participating_fraction_complexity"] = C

    h5.close_hot(h5f)
    h5f.clear()

    return res


# a list of analysis functions to call on the candidate
# l_ana_functions = list()
# l_ana_functions.append(all_in_one)

# all the keys that will be returned from above
l_ana_keys = []
# for f in l_ana_functions:
l_ana_keys += all_in_one(candidate=None)

# ------------------------------------------------------------------------------ #
# arguments
# ------------------------------------------------------------------------------ #


def parse_arguments():
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
        "-c", "--cores", dest="num_cores", help="number of dask cores",
        default=256, type=int,
    )
    return parser.parse_args()


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
        log.info(
            "Provide a directory with hdf5 files or wildcarded path as string: 'path/to/file_ptrn*.hdf5''"
        )
        exit()
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

    log.info(f"Analysing:")
    # results for all scalars
    res_ndim = dict()
    for key in l_ana_keys:
        # keep repetitions always as the last axes
        res_ndim[key] = np.ones(shape=axes_shape + (num_rep,)) * np.nan

    # set arguments for variables needed by every worker
    f = functools.partial(analyse_candidate, d_axes=d_axes, d_obs=d_obs)

    # dispatch
    futures = dh.client.map(f, candidates)

    # gather
    for future in tqdm(dh.as_completed(futures), total=len(futures)):
        index, res = future.result()

        # consider repetitions for each data point and stack values
        rep = sampled[index]
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

    f_tar = h5py.File(merge_path, "w")

    # contained axis in right order
    # workaround to store a list of strings (via object array) to hdf5
    dset = f_tar.create_dataset(
        "/meta/axis_overview",
        data=np.array(list(d_axes.keys()), dtype=object),
        dtype=h5py.special_dtype(vlen=str),
    )

    desc_axes = f"{len(axes_shape)+1}-dim array with axis: "
    for obs in d_axes.keys():
        dset = f_tar.create_dataset("/data/axis_" + obs, data=d_axes[obs])
        desc_axes += obs + ", "
    desc_axes += "repetition"

    for key in res_ndim.keys():
        dset = f_tar.create_dataset(f"/data/{key}", data=res_ndim[key])
        dset.attrs["description"] = desc_axes

    dset = f_tar.create_dataset("/data/num_samples", data=sampled)
    dset.attrs["description"] = "number of repetitions in /data/ibi, same shape"

    f_tar.close()

    return res_ndim, d_obs, d_axes


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #

# we have to pass global variables so that they are available in each worker.
# simple set them as default arguments so only `candidate` varies between workers
def analyse_candidate(candidate, d_axes, d_obs):
    # for candidate in tqdm(l_valid, desc="Files"):
    index = ()
    for obs in d_axes.keys():
        # get value
        temp = h5.load(candidate, d_obs[obs], silent=True)
        # transform to index
        temp = np.where(d_axes[obs] == temp)[0][0]
        index += (temp,)

    res = all_in_one(candidate)

    return index, res


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
