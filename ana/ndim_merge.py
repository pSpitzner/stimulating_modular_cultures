# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-05-07 14:15:37
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

# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

import ana_helper as ah
import hi5 as h5

dbg_var = None

# ------------------------------------------------------------------------------ #
# settings
# ------------------------------------------------------------------------------ #

# variables to span axes and how to get them from the hdf5 files
d_obs = dict()
d_obs["gA"] = "/meta/dynamics_gA"
d_obs["gG"] = "/meta/dynamics_gG"
d_obs["gm"] = "/meta/dynamics_gm"
d_obs["rate"] = "/meta/dynamics_rate"
# d_obs["tD"] = "/meta/dynamics_tD"
# d_obs["alpha"] = "/meta/topology_alpha"
d_obs["k_inter"] = "/meta/topology_k_inter"

# functions for analysis. candidate is the hdf5 file
# need to return a dict where the key becomes the hdf5 data set name
# and a scalar entry as the value
# todo: add description
def all_in_one(candidate=None):
    if candidate is None:
        return ["num_bursts", "sys_rate_cv", "mean_rate"]

    # load and process
    h5f = ah.prepare_file(candidate)
    ah.find_bursts_from_rates(h5f)

    res = dict()
    res["num_bursts"] = len(h5f.ana.bursts.system_level.beg_times)
    res["sys_rate_cv"] = h5f.ana.rates.cv.system_level
    res["mean_rate"] = np.nanmean(h5f.ana.rates.system_level)

    h5.close_hot(h5f)
    h5f.clear()

    return res


# a list of analysis functions to call on the candidate
l_ana_functions = list()
l_ana_functions.append(all_in_one)

# all the keys that will be returned from above
l_ana_keys = []
for f in l_ana_functions:
    l_ana_keys += f(candidate=None)

# ------------------------------------------------------------------------------ #
# arguments
# ------------------------------------------------------------------------------ #

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
args = parser.parse_args()

# ------------------------------------------------------------------------------ #
# load, calcualte ibi, merge in a new hdf5 file
# ------------------------------------------------------------------------------ #


def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


# if a directory is provided as input, merge individual hdf5 files down
if os.path.isdir(args.input_path):
    candidates = glob.glob(full_path(args.input_path + "/*.hdf5"))
    print(f"{args.input_path} is a directory, using contained hdf5 files")
elif len(glob.glob(full_path(args.input_path))) <= 1:
    print(
        "Provide a directory with hdf5 files or wildcarded path as string: 'path/to/file_ptrn*.hdf5''"
    )
    exit()
else:
    candidates = glob.glob(full_path(args.input_path))
    print(f"{args.input_path} is a (list of) file")


merge_path = full_path(args.output_path)
print(f"Merging (overwriting) to {merge_path}")

# which values occur across files
d_axes = dict()
for obs in d_obs.keys():
    d_axes[obs] = []


# check what's in the files and create axes labels for n-dim tensor
print(f"Checking {len(candidates)} files:")
l_valid = []
for candidate in tqdm(candidates):
    try:
        for obs in d_obs:
            temp = h5.load(candidate, d_obs[obs], silent=True)
            try:
                # sometimes we find nested arrays
                if len(temp == 1):
                    temp = temp[0]
            except:
                pass
            if temp not in d_axes[obs]:
                d_axes[obs].append(temp)

        l_valid.append(candidate)
    except Exception as e:
        print(f"incompatible file: {candidate}")

# sort axes and count unique axes entries
axes_size = 1
axes_shape = ()
for obs in d_axes.keys():
    d_axes[obs] = np.array(sorted(d_axes[obs]))
    axes_size *= len(d_axes[obs])
    axes_shape += (len(d_axes[obs]),)

# we might have repetitions but estimating num_rep proved unreliable.
print(f"Finding number of repetitions:")
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


# results for all scalars
res_ndim = dict()
for key in l_ana_keys:
    # keep repetitions always as the last axes
    res_ndim[key] = np.ones(shape=axes_shape + (num_rep,)) * np.nan

print(f"Found axes:")
for obs in d_axes.keys():
    print(f"{obs: >10}: ", d_axes[obs])
print(f"Repetitions: ", num_rep)

print(f"Analysing:")
for candidate in tqdm(l_valid, desc="Files"):
    index = ()
    for obs in d_axes.keys():
        # get value
        temp = h5.load(candidate, d_obs[obs], silent=True)
        # transform to index
        temp = np.where(d_axes[obs] == temp)[0][0]
        # print(f"{obs} {temp}")
        index += (temp,)

    # consider repetitions for each data point and stack values
    rep = sampled[index]
    if rep <= num_rep:
        sampled[index] += 1
        index += (rep,)

        # calculate all scalars by calling the right function from d_ana_scalars
        for f in l_ana_functions:
            res = f(candidate)
            for key in res.keys():
                res_ndim[key][index] = res[key]

    else:
        print(f"Error: unexpected repetitions (already read {num_rep}): {candidate}")

if not np.all(sampled == sampled.flat[0]):
    print(
        f"repetitions vary across data points, from {np.min(sampled)} to {np.max(sampled)}"
    )

# write to a new hdf5 file
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
