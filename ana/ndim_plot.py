# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-05-08 10:49:20
#
# plot a merged down, multidimensional hdf5 file (from individual simulations)
# and select which dims to show where
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import numbers

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import hi5 as h5

from mpl_toolkits.mplot3d import Axes3D

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
parser.add_argument(
    "-c", "--center", dest="center_cmap_around", default=None, type=float,
)
parser.add_argument(
    "-a",
    "--annot",
    dest="enable_annotation",
    default=False,
    action="store_true",
)

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

# get all observables that qualify (i.e. are not axis)
l_obs_candidates = h5.ls(input_path, "/data/")
l_obs_candidates = [obs for obs in l_obs_candidates if obs.find("axis_") != 0]
assert len(l_obs_candidates) > 0

options = f""
for idx, opt in enumerate(l_obs_candidates):
    options += f"{idx:d} {opt}\n"

while True:
    txt = input(f"Choose observable to plot, e.g. '1'\n{options}> ")
    if len(txt) == 0:
        obs_to_plot = l_obs_candidates[0]
    else:
        if int(txt) >= len(l_obs_candidates):
            continue
        obs_to_plot = l_obs_candidates[int(txt)]
        print(f"Using {obs_to_plot}")
        break

data_nd = h5.load(input_path, f"/data/{obs_to_plot}", silent=True)

l_axis_candidates = h5.load(input_path, "/meta/axis_overview", silent=True)
l_axis_candidates = l_axis_candidates.astype("str")
d_axes = dict()
for obs in l_axis_candidates:
    d_axes[obs] = h5.load(input_path, "/data/axis_" + obs, silent=True)
    if not isinstance(d_axes[obs], np.ndarray):
        d_axes[obs] = np.array([d_axes[obs]])

# select which two axis to show
options = f""
for idx, opt in enumerate(l_axis_candidates):
    options += f"{idx:d} {opt}\n"
while True:
    txt = input(f"Choose 2 axis to plot, e.g. '1 3'\n{options}> ")
    if len(txt) == 0:
        l_axis_selected = list(l_axis_candidates[:2])
    else:
        txt = txt.split(" ")
        if (
            len(txt) < 2
            or int(txt[0]) > len(l_axis_candidates)
            or int(txt[1]) > len(l_axis_candidates)
        ):
            continue
        l_axis_selected = l_axis_candidates[[int(txt[0]), int(txt[1])]].tolist()

    if len(l_axis_selected) == 2 and all(
        i in l_axis_candidates for i in l_axis_selected
    ):
        print(f"Using {l_axis_selected}")
        break

if len(l_axis_candidates) > 2:
    print(f"Select cut-plane for remaining values,")
# for remaining axes, select one value
hidden_vals = dict()
for obs in [i for i in l_axis_candidates if i not in l_axis_selected]:
    while True:
        if len(d_axes[obs]) == 1:
            val = d_axes[obs][0]
        else:
            txt = input(f"{obs} from {d_axes[obs]}: ")
            if len(txt) == 0:
                val = d_axes[obs][0]
            else:
                val = float(txt)

        if val in d_axes[obs]:
            print(f"Using {obs} = {val}")
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
    index=d_axes[y_obs] if isinstance(d_axes[y_obs], np.ndarray) else [d_axes[y_obs]],
    columns=d_axes[x_obs] if isinstance(d_axes[x_obs], np.ndarray) else [d_axes[x_obs]],
)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))

if args.center_cmap_around is None:
    kwargs = {
        'vmin' : np.nanmin(data_mean[np.isfinite(data_mean)]),
        'vmax' : np.nanmax(data_mean[np.isfinite(data_mean)]),
        # 'vmin': 0,
        # 'vmax': 150,
        'cmap': "Blues",
    }
else:
    kwargs = {
        'vmin': 0,
        'vmax': args.center_cmap_around * 2,
        'center': args.center_cmap_around,
        'cmap': "twilight",
    }

sns.heatmap(
    data_mean,
    ax=ax,
    annot=args.enable_annotation,
    fmt=".2g",
    linewidth=2.5,
    square=False,
    cbar_kws={"label": obs_to_plot},
    **kwargs,
)
ax.set_xlabel(x_obs)
ax.set_ylabel(y_obs)
ax.invert_yaxis()

for text in args.input_path.split("/"):
    if "2x2" in text:
        fig.suptitle(text)

try:
    fig.savefig(args.output_path, dpi=300)
except:
    print("No output path, only showing figure.")
