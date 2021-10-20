# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-10-19 16:49:26
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
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
# matplotlib.rcParams["axes.spines.left"] = False
# matplotlib.rcParams["axes.spines.bottom"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams['figure.dpi'] = 150
from matplotlib.ticker import MultipleLocator, LogLocator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import hi5 as h5
import colors as cc

from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #

modular_cultures = True

def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


def o_labels(short):
    label = ""
    if "ratio_num_b" in short:
        label += "Fraction of bursts"
    elif "num_b" in short:
        label += "Number of bursts"
    elif "blen" in short:
        label += "Burst duration"
    elif "rate_cv" in short:
        label += "CV of the rate"
    elif "ibis_cv" in short:
        label += "CV of the IBI"
    elif "rate" in short:
        label += "Rate"
    elif "ibis" in short:
        label += "Inter-Burst-Interval"
    elif "functional_complexity" in short:
        label += "Functional complexity"
    elif "participating_fraction_complexity" in short:
        label += "Fraction complexity"
    elif "participating_fraction" in short:
        label += "Fraction of neurons in bursts"
    elif "num_spikes_in_bursts" in short:
        label += "Spikes per neuron per burst"
    else:
        return short

    if not modular_cultures:
        return label

    if "sys" in short:
        label += "\n(system-wide)"
    elif "any" in short:
        label += "\n(anywhere)"
    elif "mod" in short or "ratio" in short:
        label += "\n("
        label += short[-1]
        label += " modules)"

    return label

def a_labels(short):
    if "rate" in short:
        return "Noise rate (Hz)"
    elif "jG" in short:
        return "GABA strength jG (inhibition)"
    elif "jA" in short:
        return "AMPA strength jA (excitation)"
    elif "jE" in short:
        return "External current strength jE"
    elif "k_frac" in short:
        return "Fraction of Connections"


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
l_obs_candidates = [obs for obs in l_obs_candidates if obs.find("axis_") != 0 and obs.find("hbins") == -1 and obs.find("hvals") == -1]
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


l_axis_candidates = h5.load(input_path, "/meta/axis_overview", silent=True)
l_axis_candidates = l_axis_candidates.astype("str")
d_axes = dict()
for opt in l_axis_candidates:
    d_axes[opt] = h5.load(input_path, "/data/axis_" + opt, silent=True)
    if not isinstance(d_axes[opt], np.ndarray):
        d_axes[opt] = np.array([d_axes[opt]])

# select which two axis to show
options = f""
opt_ids = []
for idx, opt in enumerate(l_axis_candidates):
    if len(d_axes[opt]) > 1:
        opt_ids.append(idx)
        options += f"{idx:d} {opt}\n"
while True:
    if len(opt_ids) <= 1:
        break
    txt = input(f"Choose x-axis and which lines to plot, e.g. '1 3'\n{options}> ")
    if len(txt) == 0:
        l_axis_selected = list(opt_ids[:2])
    else:
        txt = txt.split(" ")
        if (
            len(txt) < 2
            or np.any([not isint(i) for i in txt])
            or np.any([int(i) > len(l_axis_candidates) for i in txt])
        ):
            continue
        l_axis_selected = l_axis_candidates[[int(txt[0]), int(txt[1])]].tolist()

    if len(l_axis_selected) == 2 and all(
        i in l_axis_candidates for i in l_axis_selected
    ):
        print(f"Using {l_axis_selected}")
        break

if len(opt_ids) > 2:
    print(f"Select cut-plane for remaining values,")
# for remaining axes, select one value
hidden_vals = dict()
for opt in [i for i in l_axis_candidates if i not in l_axis_selected]:
    while True:
        if len(d_axes[opt]) == 1:
            val = d_axes[opt][0]
        else:
            txt = input(f"{opt} from {d_axes[opt]}: ")
            if len(txt) == 0:
                val = d_axes[opt][0]
            else:
                val = float(txt)

        if val in d_axes[opt]:
            print(f"Using {opt} = {val}")
            hidden_vals[opt] = val
            break


x_obs = l_axis_selected[0]
y_obs = l_axis_selected[1]

# select which lines to show
options = f""
opt_ids = []
for idx, opt in enumerate(d_axes[y_obs]):
    opt_ids.append(idx)
    options += f"{idx:d} {opt}\n"
while True:
    if len(opt_ids) <= 1:
        break
    txt = input(f"Select lines for {y_obs} to plot (default: all), e.g. '1 3 5'\n{options}> ")
    if len(txt) == 0:
        l_opt_selected = list(opt_ids[:])
    else:
        txt = txt.split(" ")
        if (
            np.any([not isint(i) for i in txt])
            or np.any([int(i) > len(opt_ids) for i in txt])
        ):
            continue

        l_opt_selected = [int(t) for t in txt]

    print(f"Using {[d_axes[y_obs][i] for i in l_opt_selected]}")
    break



ax_idx = []
val_idx = []
for obs in hidden_vals.keys():
    # index of the axis in n-dim raw data
    ax_idx.append(np.where(l_axis_candidates == obs)[0][0])
    # index of the value along this axis
    val_idx.append(np.where(d_axes[obs] == hidden_vals[obs])[0][0])


def get_ratios():
    # returns a dict of arrays that still need to be reduced by one dim (the selected line)
    data_mean = dict()
    data_err = dict()

    for blen in [1, 2, 3, 4]:
        num_b_any = h5.load(input_path, f"/data/any_num_b", silent=True)
        num_b_this = h5.load(input_path, f"/data/mod_num_b_{blen}", silent=True)
        data_nd = num_b_this / num_b_any

        data_3d = data_nd
        for k in sorted(ax_idx, reverse=True):
            i = np.where(ax_idx == k)[0][0]
            data_3d = np.take(data_3d, val_idx[i], axis=ax_idx[i])

        # swap axis if user gave selection in other order than in loaded data
        if l_axis_selected != [i for i in l_axis_candidates if i in l_axis_selected]:
            data_3d = np.swapaxes(data_3d, 0, 1)

        num_reps = data_3d.shape[-1]
        data_mean[blen] = np.nanmean(data_3d, axis=2)
        data_err[blen] = np.nanstd(data_3d, axis=2) / np.sqrt(num_reps)

    return data_mean, data_err

data_mean, data_err = get_ratios()

for odx, opt in enumerate(l_opt_selected):

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(f"{y_obs} = {d_axes[y_obs][opt]:g}")

    x=d_axes[x_obs]
    selects = np.where((x >=70) & (x <= 120) )

    prev = np.zeros_like(x)
    for blen in [4, 3, 2, 1]:
        clr = cc.cmap_cycle('cold', edge=False, N=4)[int(blen-1)]

        nxt = np.nan_to_num(data_mean[blen][:, opt], nan=0.0)
        ax.fill_between(
            x[selects],
            prev[selects],
            prev[selects] + nxt[selects],
            linewidth=0,
            color = cc.alpha_to_solid_on_bg(clr, 0.2),
            )

        ax.errorbar(
            x=x[selects],
            y=prev[selects] + nxt[selects],
            yerr=data_err[blen][:, opt][selects],
            fmt="o",
            markersize=3,
            mfc = cc.alpha_to_solid_on_bg(clr, 0.2),
            elinewidth=0.5,
            capsize=2,
            label = f"blen {blen}",
            color = clr,
            clip_on=False
        )

        ycs = 6
        xcs = 1
        if blen == 1:
            xcs = -12
            ycs = -12
        if blen == 2:
            pass
        else:
            ax.text(
                x[selects][xcs],
                prev[selects][ycs] + (nxt[selects][ycs])/2,
                f"{blen} module" if blen == 1 else f"{blen} modules",
                color = clr,
                va="center",
            )

        prev += nxt

# ax.legend()
ax.set_xlabel(a_labels(x_obs))
ax.set_ylabel("Fraction of bursts\nspanning")
fig.tight_layout()

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1))
ax.set_xlim(70, 120)
ax.set_ylim(0, 1)
ax

ax.spines["left"].set_position(("outward", 5))
ax.spines["bottom"].set_position(("outward", 5))
