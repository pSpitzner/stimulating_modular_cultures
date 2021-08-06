# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-08-06 18:18:49
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

import matplotlib

matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 150

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

parser = argparse.ArgumentParser(description="ndim_plot")
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

# get all observables that qualify (i.e. are not axis)
l_obs_candidates = h5.ls(input_path, "/data/")
l_obs_candidates = [
    obs
    for obs in l_obs_candidates
    if obs.find("axis_") != 0 and obs.find("hvals") != -1
]
assert len(l_obs_candidates) > 0


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


options = f""
for idx, opt in enumerate(l_obs_candidates):
    options += f"{idx:d} {opt}\n"

while True:
    txt = input(f"Choose observable to plot, e.g. '1'\n{options}> ")
    if len(txt) == 0:
        obs_to_plot = l_obs_candidates[0]
    elif not isint(txt):
        continue
    elif int(txt) >= len(l_obs_candidates):
        continue
    else:
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


print(f"Select point in phase space to plot")
# for remaining axes, select one value
hidden_vals = dict()
multiple_vals = None
for obs in [i for i in l_axis_candidates]:
    while True:
        if len(d_axes[obs]) == 1:
            val = d_axes[obs][0]
        else:
            txt = input(f"{obs} from {d_axes[obs]}: ")
            if len(txt) == 0:
                val = d_axes[obs][0]
            else:
                try:
                    # single digit
                    val = float(txt)
                except ValueError:
                    # multiple digits
                    txt = txt.split(" ")
                    if np.any([not isfloat(i) for i in txt]):
                        continue

                    val = [float(t) for t in txt]

        if np.all(np.in1d(val, d_axes[obs])):
            print(f"Using {obs} = {val}")
            hidden_vals[obs] = val
            if isinstance(val, list):
                if multiple_vals is not None:
                    print("Only select multiple values along one axis!")
                    continue
                multiple_vals = obs
            break

ax_idx = []
val_idx = []
for obs in hidden_vals.keys():
    # index of the axis in n-dim raw data
    ax_idx.append(np.where(l_axis_candidates == obs)[0][0])
    # index of the value along this axis
    val_idx.append(np.where(np.in1d(d_axes[obs], hidden_vals[obs]))[0])
    # val_idx.append(np.where(d_axes[obs] == hidden_vals[obs])[0][0])

# reduce the data, back to front, starting with last axis
data_3d = data_nd
bins_3d = h5.load(input_path, f"/data/{obs_to_plot.replace('hvals', 'hbins')}")
for k in sorted(ax_idx, reverse=True):
    i = np.where(ax_idx == k)[0][0]
    data_3d = np.take(data_3d, val_idx[i], axis=ax_idx[i])
    bins_3d = np.take(bins_3d, val_idx[i], axis=ax_idx[i])

# data_3d now has shape (repetition, hist_vals)
# or (1, 1, x, 1, rep, hist_vals) if multiple values were selected for one axis
assert np.sum(np.array(data_3d.shape) > 1) <= 2, "Only select multiple values along one axis!"

squeeze_ax = np.where(np.array(data_3d.shape)[0:-2] == 1)[0]
data_3d = np.squeeze(data_3d, axis=tuple(squeeze_ax))
bins_3d = np.squeeze(bins_3d, axis=tuple(squeeze_ax))

# need to squeeze bins a bit more
if multiple_vals is not None:
    bins_3d = bins_3d[0]

bins = bins_3d[0, :]
# assumte all bin edges match....
for b in range(0, bins_3d.shape[0]):
    assert np.all(bins == bins_3d[b, :])

centroids = (bins[1:] + bins[:-1]) / 2

# we could also sum up or so
weights = np.nanmean(data_3d, axis=-2)

plt.ion()
fig, ax = plt.subplots()

if len(data_3d.shape) == 2:
    # simple case
    sns.histplot(
        x=centroids,
        weights=weights,
        bins=len(centroids),
        binrange=(min(bins), max(bins)),
        ax=ax,
        element="step",
        color="black",
    )
elif len(data_3d.shape) == 3:
    # repeat for multiple values
    vdx = list(hidden_vals.keys()).index(multiple_vals)
    for idx in range(data_3d.shape[0]):
        val = d_axes[multiple_vals][val_idx[vdx]][idx]
        sns.histplot(
            x=centroids,
            weights=weights[idx,:],
            bins=len(centroids),
            binrange=(min(bins), max(bins)),
            ax=ax,
            element="step",
            stat= "probability",
            label=f"{multiple_vals}: {val}",
            color = f"C{idx}",
            alpha=0.25,
        )
        ax.legend()

else:
    raise ValueError

ax.set_xlabel(obs_to_plot.replace("hvals_", ""))
fig.tight_layout()
