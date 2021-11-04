# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-27 18:10:11
# @Last Modified: 2021-11-04 15:36:43
# ------------------------------------------------------------------------------ #
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
matplotlib.rcParams["xtick.labelsize"]=8
matplotlib.rcParams["ytick.labelsize"]=8
matplotlib.rcParams["axes.titlesize"]= 8
matplotlib.rcParams["axes.labelsize"]= 8
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import xarray as xr
import hi5 as h5
import ndim_helper as nh
import colors as cc
import palettable as pt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, Normalize


parser = argparse.ArgumentParser(description="ndim_plot")
parser.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input path with *.hdf5 files",
    metavar="FILE",
)


args = parser.parse_args()

# ------------------------------------------------------------------------------ #
# load data and select what to plot
# ------------------------------------------------------------------------------ #


data = nh.load_ndim_h5f(args.input_path)
dim1 = "rate"

observables = [obs for obs in data.keys() if not "_hbins_" in obs]

# combined guys
# observables.append("ratio_num_b_1")
# observables.append("ratio_num_b_2")
# observables.append("ratio_num_b_3")
# observables.append("ratio_num_b_4")

data = nh.choose_all_dims(data, skip=dim1)

x = data["any_num_b"].coords[dim1]
selects = np.where((x >=70) & (x <= 120) )

fig, ax = plt.subplots()

prev = np.zeros_like(x)
for seq_len in [4, 3, 2, 1, 0]:

    ref = data["any_num_b"]
    dat = data[f"mod_num_b_{seq_len}"]

    num_reps = len(data["any_num_b"]["repetition"])

    ratio = dat / ref
    ratio_mean = ratio.mean(dim="repetition")
    ratio_errs = ratio.std(dim="repetition") / np.sqrt(num_reps)

    # buildup the graph area by area, using nxt and prev
    nxt = np.nan_to_num(ratio_mean, nan=0.0)

    clr = cc.cmap_cycle('cold', edge=False, N=5)[int(seq_len)]
    ax.fill_between(
        x[selects],
        prev[selects],
        prev[selects] + nxt[selects],
        linewidth=0,
        color = cc.alpha_to_solid_on_bg(clr, 0.2),
    )
    if seq_len != 0:
        ax.errorbar(
            x=x[selects],
            y=prev[selects] + nxt[selects],
            yerr=ratio_errs[selects],
            fmt="o",
            markersize=3,
            mfc = cc.alpha_to_solid_on_bg(clr, 0.2),
            elinewidth=0.5,
            capsize=2,
            label = f"{seq_len} module" if seq_len == 1 else f"{seq_len} modules",
            color = clr,
            clip_on=False
        )

    # we coult try to place text in the areas
    if seq_len == 4:
        ycs = 6
        xcs = 1
        ax.text(
            x[selects][xcs],
            prev[selects][ycs] + (nxt[selects][ycs])/2,
            f"{seq_len} module" if seq_len == 1 else f"{seq_len} modules",
            color = clr,
            va="center",
        )

    prev += nxt


# cc._legend_into_new_axes(ax)
ax.set_xlabel(nh.obs_labels(dim1))
ax.set_ylabel("Fraction of bursts spanning")
fig.tight_layout()

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1))
ax.set_xlim(70, 120)
ax.set_ylim(0, 1)

ax.spines["left"].set_position(("outward", 5))
ax.spines["bottom"].set_position(("outward", 5))
