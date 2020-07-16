# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2020-07-16 16:55:58
#
# Scans the provided directory for .hdf5 files and checks if they have the right
# data to plot a 2d heatmap of ibi = f(gA, rate)
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

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #


def h5_load(filename, dsetname, raise_ex=True, silent=True):
    try:
        file = h5py.File(filename, "r")
        try:
            res = file[dsetname][:]
        except ValueError:
            res = file[dsetname][()]
        file.close()
        return res
    except Exception as e:
        if not silent:
            print(f"failed to load {dsetname} from {filename}: {e}")
        if raise_ex:
            raise e
        else:
            return np.nan


def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


# ------------------------------------------------------------------------------ #
# load and merge, if needed
# ------------------------------------------------------------------------------ #

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

# if a directory is provided as input, merge individual hdf5 files down, first
if os.path.isdir(args.input_path):
    merge_path = full_path(args.input_path + "_ibi.hdf5")
    print(f"{args.input_path} is a directory. Merging (overwriting) to {merge_path}")

    candidates = glob.glob(full_path(args.input_path + "/*.hdf5"))
    l_ga = []
    l_rate = []
    l_valid = []

    # check what's in the files and create axes labels for 2d matrix
    for candidate in candidates:
        try:
            ga = h5_load(candidate, "/meta/dynamics_gA")
            rate = h5_load(candidate, "/meta/dynamics_rate")
            ibi = h5_load(candidate, "/data/ibi")
            if ga not in l_ga:
                l_ga.append(ga)
            if rate not in l_rate:
                l_rate.append(rate)
            l_valid.append(candidate)
        except Exception as e:
            print(f"incompatible file: {candidate}")

    l_ga = np.array(sorted(l_ga))
    l_rate = np.array(sorted(l_rate))

    # we might have repetitions
    num_rep = int(np.ceil(len(l_valid) / len(l_ga) / len(l_rate)))
    sampled = np.zeros(shape=(len(l_ga), len(l_rate)), dtype=int)

    # 3d: x, y, repetition
    heatmap = np.ones(shape=(len(l_ga), len(l_rate), num_rep)) * np.nan

    print(f"ga: ", l_ga)
    print(f"rates: ", l_rate)
    print(f"repetitions: ", num_rep)

    for candidate in l_valid:
        ga = h5_load(candidate, "/meta/dynamics_gA")
        rate = h5_load(candidate, "/meta/dynamics_rate")
        ibi = h5_load(candidate, "/data/ibi")

        # print(f"{ga} {rate} {candidate}")
        # transform to indices
        ga = np.where(l_ga == ga)[0][0]
        rate = np.where(l_rate == rate)[0][0]

        # consider repetitions for each data point and stack values
        rep = sampled[ga, rate]
        if rep <= num_rep:
            sampled[ga, rate] += 1
            heatmap[ga, rate, rep] = ibi
        else:
            print(
                f"Error: unexpected repetitions (already read {num_rep}): {candidate}"
            )

    if not np.all(sampled == sampled[0, 0]):
        print(
            f"repetitions vary across data points, from {np.min(sampled)} to {np.max(sampled)}"
        )

    # write to a new hdf5 file
    f_tar = h5py.File(merge_path, "w")

    # this seems broken due to integer division
    dset = f_tar.create_dataset("/data/axis_rate", data=l_rate)
    dset = f_tar.create_dataset("/data/axis_ga", data=l_ga)
    dset = f_tar.create_dataset("/data/ibi", data=heatmap)
    dset.attrs[
        "description"
    ] = "inter burst interval, 3d array with axis_ga x axis_rate x repetition"
    dset = f_tar.create_dataset("/data/num_samples", data=sampled)
    dset.attrs["description"] = "number of repetitions in /data/ibi, same shape"

    f_tar.close()


elif os.path.isfile(args.input_path):
    print(f"{args.input_path} is a file, assuming merged data.")
    merge_path = full_path(args.input_path)

# ------------------------------------------------------------------------------ #
# load merged and plot
# ------------------------------------------------------------------------------ #

l_ga = h5_load(merge_path, "/data/axis_ga")
l_rate = h5_load(merge_path, "/data/axis_rate") * 1000  # to convert from 1/ms to Hz
ibi_3d = h5_load(merge_path, "/data/ibi")
sampled = h5_load(merge_path, "/data/num_samples")

ibi_median = pd.DataFrame(np.nanmedian(ibi_3d, axis=2), index=l_ga, columns=l_rate)
ibi_mean = pd.DataFrame(np.nanmean(ibi_3d, axis=2), index=l_ga, columns=l_rate)
ibi_std = pd.DataFrame(np.nanstd(ibi_3d, axis=2), index=l_ga, columns=l_rate)
ibi_not_nan = pd.DataFrame(
    np.sum(np.isfinite(ibi_3d), axis=2), index=l_ga, columns=l_rate
)

plt.ion()
fig, ax = plt.subplots()

sns.heatmap(
    ibi_mean,
    ax=ax,
    vmin=0,
    vmax=150,
    annot=True,
    fmt=".0f",
    linewidth=2.5,
    cmap="Blues",
    square=True,
    cbar_kws={"label": "IBI [seconds]"},
)
ax.set_xlabel("Noise, Rate [Hz]")
ax.set_ylabel("Ampa strength [mV]")
ax.invert_yaxis()
ax.set_title(args.input_path)

fig, ax = plt.subplots()
sns.heatmap(
    ibi_std,
    ax=ax,
    vmin=0,
    vmax=15,
    annot=True,
    fmt=".1f",
    linewidth=2.5,
    cmap="Reds",
    square=True,
    cbar_kws={"label": "IBI Std"},
)
ax.set_xlabel("Noise, Rate [Hz]")
ax.set_ylabel("Ampa strength [mV]")
ax.invert_yaxis()
ax.set_title(args.input_path)

fig, ax = plt.subplots()
sns.heatmap(
    ibi_not_nan,
    ax=ax,
    annot=True,
    fmt=".0f",
    linewidth=2.5,
    cmap="Greens",
    square=True,
    cbar_kws={"label": "Contributing samples"},
)
ax.set_xlabel("Noise, Rate [Hz]")
ax.set_ylabel("Ampa strength [mV]")
ax.invert_yaxis()
ax.set_title(args.input_path)
