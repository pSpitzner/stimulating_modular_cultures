# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-27 18:10:11
# @Last Modified: 2021-11-10 18:36:36
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

data, selected_dims = nh.load_and_choose_two_dims(args.input_path)
dim1, dim2 = selected_dims

observables = [obs for obs in data.keys() if not "_hbins_" in obs]
observables = nh.choose(
    observables,
    prompt="Choose at least one observable, press enter for all:",
    min=1,
    via_int=True,
    always_return_list=True,
)

# combined guys
# observables.append("ratio_num_b_1")
# observables.append("ratio_num_b_2")
# observables.append("ratio_num_b_3")
# observables.append("ratio_num_b_4")

# if histograms are in observables, specify more selections
if np.any(["_hvals_" in obs for obs in observables]):

    # h_dim_iter is a str containing the dimension that we iterate over

    obs = observables[0]
    # ask along which direction to run over the hists, if both have more than one coord
    # use xarrays .sizes only has a dim if it has more than one coordinate.
    if np.all([dim in data[obs].sizes.keys() for dim in selected_dims]):
        h_dim_iter = nh.choose(
            selected_dims,
            prompt=f"For histograms, over which dim to iterate?",
            min=1,
            max=1,
            via_int=True,
            default=0,
            always_return_list=False,
        )
    else:
        h_dim_iter = dim2 if dim2 in data[obs].sizes.keys() else dim1
    h_dim_noiter = dim1 if h_dim_iter == dim2 else dim2

    # if more than one value in the not-to-iterate dim, we need a cutplane
    if h_dim_noiter in data[obs].sizes.keys():
        h_cs_noiter = nh.choose(
            data[obs].coords[h_dim_noiter].to_numpy(),
            prompt=f"For histograms, choose a cutplane for '{h_dim_noiter}':",
            min=1,
            max=1,
            default=0,
            always_return_list=False,
        )
    else:
        # we wont need data.sel() later
        h_cs_noiter = None

    # we may not want to show everything in the histograms
    h_cs_iter = nh.choose(
        data[obs].coords[h_dim_iter].to_numpy(),
        prompt=f"For histograms, which '{h_dim_iter}' to show?",
        min=1,
        default="all",
        always_return_list=True,
    )




for obs in observables:

    # average across repetitions
    num_reps = len(data[obs].coords["repetition"])
    dat_med = data[obs].mean(dim="repetition")
    dat_sem = data[obs].std(dim="repetition") / np.sqrt(num_reps)

    # pick a base color
    if "sys" in obs:
        base_color = "C0"
    elif "mod" in obs:
        base_color = "C1"
    elif "any" in obs:
        base_color = "C2"
    else:
        base_color = f"black"

    # treat scalars differently than vectors
    if obs[0:4] != "vec_":
        fig, ax = plt.subplots()
        # reshape(-1) to make sure this is a 1-dimensional array
        iter_coords = dat_med.coords[dim2].to_numpy()
        iter_coords = iter_coords.reshape(-1)
        for idx, cs2 in enumerate(iter_coords):

            # use one base color, light to dark for multiple lines
            alpha = np.fmin(1, 1 / (len(iter_coords) - idx))
            color = cc.alpha_to_solid_on_bg(base_color, alpha)

            x = dat_med.coords[dim1]
            if len(iter_coords) == 1:
                y = dat_med
                yerr = dat_sem
            else:
                y = dat_med.sel({dim2: cs2})
                yerr = dat_sem.sel({dim2: cs2})

            selects = np.where(np.isfinite(y))

            ax.errorbar(
                x=x[selects],
                y=y[selects],
                yerr=yerr[selects],
                fmt="-",
                markersize=1,
                color=color,
                elinewidth=0.5,
                capsize=3,
                zorder=0,
                label=f"{dim2} = {cs2:g}",
            )

        ax.set_xlabel(nh.dim_labels(dim1))
        ax.set_ylabel(nh.obs_labels(obs))
        ax.set_ylabel(obs)
        ax.legend()

    # histograms
    elif "_hvals_" in obs:

        fig, ax = plt.subplots()
        bin_obs = obs.replace("_hvals_", "_hbins_")

        # bins should be the same across all dimensions, pick zeroth element in each
        dims_to_drop = dict()
        for dim in data[bin_obs].dims:
            if dim == "vector_observable":
                continue
            dims_to_drop[dim] = 0
        bins = data[bin_obs][dims_to_drop].to_numpy()
        centroids = (bins[1:] + bins[:-1]) / 2

        # iterate over selected `h_dim_iter`
        # iter_coords = dat_med.coords[h_dim_iter].to_numpy()
        # iter_coords = iter_coords.reshape(-1)

        for idx, cs in enumerate(h_cs_iter):
            # use one base color, light to dark for multiple lines
            alpha = np.fmin(1, 1 / (len(h_cs_iter) - idx))
            color = cc.alpha_to_solid_on_bg(base_color, alpha)

            hvals = dat_med

            try:
                hvals = hvals.sel({h_dim_noiter: h_cs_noiter})
            except:
                # this happens when we only have one coordinate along h_dim_noiter
                pass

            if len(h_cs_iter) > 1:
                hvals = hvals.sel({h_dim_iter: cs})

            # this is a hack to use seaborn to plot a precomputed histogram
            sns.histplot(
                x=centroids,
                weights=hvals.to_numpy(),
                bins=len(centroids),
                binrange=(min(bins), max(bins)),
                ax=ax,
                element="step",
                stat="probability",
                label=f"{h_dim_iter}: {cs}",
                color=color,
                alpha=0.0,
            )

        ax.set_xlabel(nh.obs_labels(obs))
        ax.set_ylabel("Probability")
        ax.legend(loc='upper center')


    fig.canvas.manager.set_window_title(obs)
    # ax.legend()
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

    if "ratio" in obs:
        ax.set_ylim(0, 1)

    fig.tight_layout()
