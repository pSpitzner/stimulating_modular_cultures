# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2020-09-04 13:52:18
#
# Scans the provided directory for .hdf5 files and checks if they have the right
# data to plot a 2d ibi_mean_4d of ibi = f(gA, rate)
# uses ut.inter_burst_interval to get ibi from spiketimes
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

parser = argparse.ArgumentParser(description="Plot Ibi Sweep")
parser.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input path with *.hdf5 files",
    metavar="FILE",
)
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
args = parser.parse_args()


# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #


def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


# ------------------------------------------------------------------------------ #
# load, calcualte ibi, merge in a new hdf5 file
# ------------------------------------------------------------------------------ #


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

print(f"Checking {len(candidates)} files.")

merge_path = full_path(args.output_path)
print("Merging (overwriting) to {merge_path}")


# variables to span axes and how to get them from the hdf5 files
d_obs = dict()
d_obs["ga"] = "/meta/dynamics_gA"
d_obs["rate"] = "/meta/dynamics_rate"
d_obs["tD"] = "/meta/dynamics_tD"

# which values occur across files
d_axes = dict()
for obs in d_obs.keys():
    d_axes[obs] = []


# check what's in the files and create axes labels for n-dim tensor
l_valid = []
for candidate in candidates:
    try:
        for obs in d_obs:
            temp = ut.h5_load(candidate, d_obs[obs], silent=True)
            if temp not in d_axes[obs]:
                d_axes[obs].append(temp)

        l_valid.append(candidate)
    except Exception as e:
        print(f"incompatible file: {candidate}")

# sort axes and count unique axes entries
axes_size = 0
axes_shape = ()
for obs in d_axes.keys():
    d_axes[obs] = np.array(sorted(d_axes[obs]))
    axes_size += len(d_axes[obs])
    axes_shape += (len(d_axes[obs]),)

# we might have repetitions
num_rep = int(np.ceil(len(l_valid) / axes_size))
sampled = np.zeros(shape=axes_shape, dtype=int)

# keep repetitions always as the last axes
res_ndim = np.ones(shape=axes_shape + (num_rep,)) * np.nan

print(f"found axes:")
for obs in d_axes.keys():
    print(f"\t{obs}: ", d_axes[obs])
print(f"repetitions: ", num_rep)

for candidate in tqdm(l_valid):
    index = ()
    for obs in d_axes.keys():
        # get value
        temp = ut.h5_load(candidate, d_obs[obs], silent=True)
        # transform to index
        temp = np.where(d_axes[obs] == temp)[0][0]
        index += (temp,)

    sim_duration = ut.h5_load(candidate, "/meta/dynamics_simulation_duration")

    # consider repetitions for each data point and stack values
    rep = sampled[index]
    if rep <= num_rep:
        sampled[index] += 1
        index += (rep,)

        # load spiketimes and calculate ibi
        # spiketimes = ut.h5_load(candidate, "/data/spiketimes", silent=True)
        # bursttimes = ut.burst_times(spiketimes, bin_size=0.5, threshold=0.75)
        # ibis = ut.inter_burst_intervals(bursttimes=bursttimes)
        ibis = []
        res_ndim[index] = np.mean(ibis) if len(ibis) > 0 else np.inf

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

dset = f_tar.create_dataset("/data/ibi", data=res_ndim)
dset.attrs["description"] = desc_axes

dset = f_tar.create_dataset("/data/num_samples", data=sampled)
dset.attrs["description"] = "number of repetitions in /data/ibi, same shape"


f_tar.close()
