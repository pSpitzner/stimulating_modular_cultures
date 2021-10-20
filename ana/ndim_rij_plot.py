# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-10-19 09:37:23
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
import colors as cc
import palettable as pt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, Normalize

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
    if obs.find("axis_") != 0 and obs.find("vec_") != -1
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


l_axis_candidates = h5.load(input_path, "/meta/axis_overview", silent=True)
l_axis_candidates = l_axis_candidates.astype("str")
d_axes = dict()
for obs in l_axis_candidates:
    d_axes[obs] = h5.load(input_path, "/data/axis_" + obs, silent=True)
    if not isinstance(d_axes[obs], np.ndarray):
        d_axes[obs] = np.array([d_axes[obs]])

hidden_vals = dict()
multiple_vals = None

# ------------------------------------------------------------------------------ #
# options
# ------------------------------------------------------------------------------ #

# # select observable to plot
# options = f""
# for idx, opt in enumerate(l_obs_candidates):
#     options += f"{idx:d} {opt}\n"

# while True:
#     txt = input(f"Choose observable to plot, e.g. '1'\n{options}> ")
#     if len(txt) == 0:
#         obs_to_plot = l_obs_candidates[0]
#     elif not isint(txt):
#         continue
#     elif int(txt) >= len(l_obs_candidates):
#         continue
#     else:
#         obs_to_plot = l_obs_candidates[int(txt)]
#     print(f"Using {obs_to_plot}")
#     break


# # select which two axis to show
# options = f""
# for idx, opt in enumerate(l_axis_candidates):
#     options += f"{idx:d} {opt}\n"


# print(f"Select point in phase space to plot")
# # for remaining axes, select one value
# for obs in [i for i in l_axis_candidates]:
#     while True:
#         if len(d_axes[obs]) == 1:
#             val = d_axes[obs][0]
#         else:
#             txt = input(f"{obs} from {d_axes[obs]}: ")
#             if len(txt) == 0:
#                 val = d_axes[obs][0]
#             else:
#                 try:
#                     # single digit
#                     val = float(txt)
#                 except ValueError:
#                     # multiple digits
#                     txt = txt.split(" ")
#                     if np.any([not isfloat(i) for i in txt]):
#                         continue

#                     val = [float(t) for t in txt]

#         if np.all(np.in1d(val, d_axes[obs])):
#             print(f"Using {obs} = {val}")
#             hidden_vals[obs] = val
#             if isinstance(val, list):
#                 if multiple_vals is not None:
#                     print("Only select multiple values along one axis!")
#                     continue
#                 multiple_vals = obs
#             break

# ------------------------------------------------------------------------------ #
# options retrieved from user
# ------------------------------------------------------------------------------ #

# multiple_vals = 'jE'
# hidden_vals = {'jA': 40.0, 'jG': 50.0, 'jE': [0.0, 30.0], 'rate': 80.0}
multiple_vals = 'rate'
hidden_vals = {'jA': 50.0, 'jG': 50.0, 'rate': [70, 100.0]}

def rij_from_ndim(obs_to_plot, multiple_vals, hidden_vals):

    ax_idx = []
    val_idx = []
    for obs in hidden_vals.keys():
        # index of the axis in n-dim raw data
        ax_idx.append(np.where(l_axis_candidates == obs)[0][0])
        # index of the value along this axis
        val_idx.append(np.where(np.in1d(d_axes[obs], hidden_vals[obs]))[0])
        # val_idx.append(np.where(d_axes[obs] == hidden_vals[obs])[0][0])

    data_nd = h5.load(input_path, f"/data/{obs_to_plot}", silent=True)
    # reduce the data, back to front, starting with last axis
    data_3d = data_nd

    for k in sorted(ax_idx, reverse=True):
        i = np.where(ax_idx == k)[0][0]
        data_3d = np.take(data_3d, val_idx[i], axis=ax_idx[i])

    # data_3d now has shape (repetition, hist_vals)
    # or (1, 1, x, 1, rep, hist_vals) if multiple values were selected for one axis
    assert np.sum(np.array(data_3d.shape) > 1) <= 3, "Only select multiple values along one axis!"

    squeeze_ax = np.where(np.array(data_3d.shape)[0:-2] == 1)[0]
    data_3d = np.squeeze(data_3d, axis=tuple(squeeze_ax))

    assert len(data_3d.shape) == 3

    rij = dict()
    desc = dict()
    vdx = list(hidden_vals.keys()).index(multiple_vals)

    assert data_3d.shape[0] == 2, "Only select two values to compare"

    for idx in range(data_3d.shape[0]):
        val = d_axes[multiple_vals][val_idx[vdx]][idx]
        rij[idx] = data_3d[idx].flatten()
        desc[idx] = f"{multiple_vals} = {val}"

    return rij, desc

rij_0 = []
rij_1 = []
desc_0 = None
desc_1 = None
# fig, ax = plt.subplots()

for obs_to_plot in [
    # 'vec_rij_across_0_1',
    # 'vec_rij_across_0_2',
    # 'vec_rij_across_0_3',
    # 'vec_rij_across_1_2',
    # 'vec_rij_across_1_3',
    # 'vec_rij_across_2_3',
    'vec_rij_within_0',
    'vec_rij_within_1',
    'vec_rij_within_2',
    'vec_rij_within_3',
]:
    rij, desc = rij_from_ndim(obs_to_plot, multiple_vals, hidden_vals)

    rij_0.extend(rij[0])
    rij_1.extend(rij[1])

    if desc_0 is None:
        desc_0 = desc[0]
        desc_1 = desc[1]
    else:
        assert desc_0 == desc[0] and desc_1 == desc[1]

    # ax.plot(
    #     rij[0],
    #     rij[1],
    #     ".",
    #     markersize=1.5,
    #     markeredgewidth=0,
    #     alpha=1,
    # )

    fig, ax = plt.subplots()

    clr = "black"
    if "within" in obs_to_plot:
        clr = f"C{obs_to_plot[-1]}"
    elif "across" in obs_to_plot:
        # this is useful when stimutating only 0_2
        # if obs_to_plot[-3:] == "0_2":
        #     # both stimulated
        #     clr = "#D46A0C"
        # elif obs_to_plot[-3:] == "1_3":
        #     # both unstimulated
        #     clr = "#1F77B4"
        # elif obs_to_plot[-3:] == "0_1" or obs_to_plot[-3:] == "2_3":
        #     # neighbours
        #     clr = "#009A00"

        if obs_to_plot[-3:] in ["0_2", "1_3", "0_1", "2_3"]:
            # both stimulated
            clr = "#009A00"
        else:
            clr = "#4A26BB"



    sns.histplot(
        x=rij[0],
        y=rij[1],
        ax=ax,
        # clip=[[0,1], [0,1]],
        # fill=True,
        # levels=10,
        cbar=True,
        thresh=None,
        bins=[np.arange(0.0,1.01,0.01)]*2,
        norm=LogNorm(),
        vmin=5e-4,
        vmax=1e-2,
        # vmax=1e-2,
        # vmin=0,
        stat='probability',
        # levels=np.logspace(-3,0,10, base=10),
        # cmap=pt.cmocean.sequential.Tempo_20.get_mpl_colormap(,
        # cmap=pt.cmocean.sequential.Ice_20_r.get_mpl_colormap(),
        # cmap=pt.cmocean.sequential.Amp_20.get_mpl_colormap(),
        cmap = cc.create_cmap(start="white", end=clr),
        # cmap= "mako",
        # cmap=pt.colorbrewer.sequential.Blues_9.get_mpl_colormap(),
        # color="gray",
        # alpha = 0.3,
        # linewidths=0.5
        kde=True,
    )
    # ax.set_facecolor('black')

    # sns.kdeplot(
    #     x=rij[0],
    #     y=rij[1],
    #     ax=ax,
    #     # color="gray",
    #     # alpha=0.1,
    #     # zorder=1,
    #     # levels=[0.2, 0.4, 0.6, 0.8, 1]
    #     fill=True,
    #     thresh=0,
    #     # levels=100,
    #     levels=np.arange(0,1,0.2),
    #     cmap="mako",
    #     cbar=True,
    #     clip=[[0,1], [0,1]],
    #     # common_norm=True,
    # )

    ax.plot([0, 1], [0, 1], zorder=2, ls=":", color="gray")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel(desc_0)
    ax.set_ylabel(desc_1)
    ax.set_title(obs_to_plot)
    fig.canvas.manager.set_window_title(obs_to_plot)


    fig.tight_layout()
