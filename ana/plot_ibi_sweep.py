# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2020-09-04 18:44:42
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
args = parser.parse_args()

# if a directory is provided as input, merge individual hdf5 files down, first
if os.path.isdir(args.input_path):
    print("run merge_ibi_sweep.py first and provide the hdf5 file as input with -i")
    exit()

elif os.path.isfile(args.input_path):
    print(f"{args.input_path} is a file, assuming merged data.")
    input_path = full_path(args.input_path)

# ------------------------------------------------------------------------------ #
# load merged and plot
# ------------------------------------------------------------------------------ #

l_axis_candidates = ut.h5_load(input_path, "/meta/axis_overview", silent=True)
data_nd = ut.h5_load(input_path, "/data/ibi", silent=True)

d_axes = dict()
for obs in l_axis_candidates:
    d_axes[obs] = ut.h5_load(input_path, "/data/axis_" + obs, silent=True)

# select which two axis to show
while True:
    txt = input(f"Choose 2 axis to plot {l_axis_candidates}: ")
    if len(txt) == 0:
        l_axis_selected = list(l_axis_candidates[:2])
    else:
        l_axis_selected = txt.split(' ')
    if len(l_axis_selected) == 2 and all(i in l_axis_candidates for i in l_axis_selected):
        print(f"Using {l_axis_selected}")
        break

if len(l_axis_candidates) > 2:
    print(f"Select cut-plane for remaining values,")
# for remaining axes, select one value
hidden_vals = dict()
for obs in [i for i in l_axis_candidates if i not in l_axis_selected]:
    while True:
        txt = input(f"{obs} from {d_axes[obs]}: ")
        if len(txt) == 0:
            val = d_axes[obs][0]
        else:
            val = float(txt)

        if val in d_axes[obs]:
            print(f"Using {val}")
            hidden_vals[obs] = val
            break

ax_idx = []
val_idx = []
for obs in hidden_vals.keys():
    # index of the axis in n-dim raw data
    ax_idx.append(np.where(l_axis_candidates == obs)[0][0])
    # index of the value along this axis
    val_idx.append(np.where(d_axes[obs] == hidden_vals[obs])[0][0])

# reduce the data, back to front, starting with last axis
data_3d = data_nd
for k in sorted(ax_idx, reverse=True):
    i = np.where(ax_idx == k)[0][0]
    data_3d = np.take(data_3d, val_idx[i], axis=ax_idx[i])

# swap axis if user gave selection in other order than in loaded data
if l_axis_selected != [i for i in l_axis_candidates if i in l_axis_selected]:
    data_3d = np.swapaxes(data_3d, 0, 1)

# plot using seaborn and pandas DataFrame (its just more convenient than manually)
y_obs = l_axis_selected[0]
x_obs = l_axis_selected[1]
data_mean = pd.DataFrame(
    # average across repetitions, which are last axis
    np.nanmean(data_3d, axis=2),
    index=d_axes[y_obs], columns=d_axes[x_obs])

plt.ion()
fig, ax = plt.subplots(figsize=(10,4))

sns.heatmap(
    data_mean,
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
ax.set_xlabel(x_obs)
ax.set_ylabel(y_obs)
ax.invert_yaxis()
fig.suptitle(args.input_path)

try:
    fig.savefig(args.output_path, dpi=300)
except:
    print("No output path, only showing figure.")
