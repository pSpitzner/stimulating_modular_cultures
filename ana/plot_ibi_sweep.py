# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2020-09-03 11:51:16
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

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #



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
parser.add_argument("-tD", dest="tD", help="recovery time", metavar="0.5", type=float)
args = parser.parse_args()

tD = args.tD

# if a directory is provided as input, merge individual hdf5 files down, first
if os.path.isdir(args.input_path):
    print("run merge_ibi_sweep.py first and provide the hdf5 file as input with -i")
    exit()

elif os.path.isfile(args.input_path):
    print(f"{args.input_path} is a file, assuming merged data.")
    merge_path = full_path(args.input_path)

# ------------------------------------------------------------------------------ #
# load merged and plot
# ------------------------------------------------------------------------------ #

l_ga = ut.h5_load(merge_path, "/data/axis_ga", silent=True)
l_tD = ut.h5_load(merge_path, "/data/axis_tD", silent=True)
l_rate = ut.h5_load(merge_path, "/data/axis_rate", silent=True) * 1000  # to convert from 1/ms to Hz
ibi_4d = ut.h5_load(merge_path, "/data/ibi", silent=True)
sampled = ut.h5_load(merge_path, "/data/num_samples", silent=True)

ibi_3d = ibi_4d[:, :, np.where(l_tD == tD)[0][0], :]
ibi_median = pd.DataFrame(np.nanmedian(ibi_3d, axis=2), index=l_ga, columns=l_rate)
ibi_mean = pd.DataFrame(np.nanmean(ibi_3d, axis=2), index=l_ga, columns=l_rate)
ibi_std = pd.DataFrame(np.nanstd(ibi_3d, axis=2), index=l_ga, columns=l_rate)
ibi_not_nan = pd.DataFrame(
    np.sum(np.isfinite(ibi_3d), axis=2), index=l_ga, columns=l_rate
)

plt.ion()
fig, ax = plt.subplots(figsize=(10,4))

sns.heatmap(
    ibi_mean,
    ax=ax,
    vmin=0,
    vmax=150,
    annot=True,
    fmt=".2g",
    linewidth=2.5,
    cmap="Blues",
    square=False,
    cbar_kws={"label": "IBI [seconds]"},
)
ax.set_xlabel("Noise, Rate [Hz]")
ax.set_ylabel("Ampa strength [mV]")
ax.invert_yaxis()
fig.suptitle(args.input_path)

fig.savefig(args.output_path, dpi=300)


fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(
    ibi_std,
    ax=ax,
    vmin=0,
    vmax=15,
    annot=True,
    fmt=".1f",
    linewidth=2.5,
    cmap="Reds",
    square=False,
    cbar_kws={"label": "IBI Std"},
)
ax.set_xlabel("Noise, Rate [Hz]")
ax.set_ylabel("Ampa strength [mV]")
ax.invert_yaxis()
fig.suptitle(args.input_path)

fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(
    ibi_not_nan,
    ax=ax,
    annot=True,
    fmt=".0f",
    linewidth=2.5,
    cmap="Greens",
    square=False,
    cbar_kws={"label": "Contributing samples"},
)
ax.set_xlabel("Noise, Rate [Hz]")
ax.set_ylabel("Ampa strength [mV]")
ax.invert_yaxis()
fig.suptitle(args.input_path)
