# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

from os import read
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from benedict import benedict
import seaborn as sns
import gc

import mesos_plot_helper as mph

from ana_helper import find_system_bursts_from_module_bursts
import plot_helper as ph
import colors as cc
import hi5 as h5

# select things to draw for every panel
show_title = False
show_xlabel = True
show_ylabel = True
show_legend = False
show_legend_in_extra_panel = False
use_compact_size = True  # this recreates the small panel size of the manuscript

mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["axes.edgecolor"] = "black"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["xtick.labelsize"] = 6
mpl.rcParams["ytick.labelsize"] = 6
mpl.rcParams["lines.dash_capstyle"] = "round"
mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.titlesize"] = 6
mpl.rcParams["axes.labelsize"] = 6
mpl.rcParams["legend.fontsize"] = 6
mpl.rcParams["legend.facecolor"] = "#D4D4D4"
mpl.rcParams["legend.framealpha"] = 0.8
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)  # transparent figure bg
mpl.rcParams["axes.facecolor"] = (1.0, 0.0, 0.0, 0.0)  # developer mode, red axes


import xarray as xr
import re
import glob
import os
import functools
import colors as cc
from tqdm import tqdm


def write_xr_dset_to_hdf5(dset, output_path, **kwargs):
    """
    Wrapper to write xarray Dataset to disk as hdf5.
    to write a DataArray, you can cast beforehand e.g.
    `my_array.to_dataset(name="data")`

    turns on zlib compression for all data variables

    # Parameters
    kwargs : dict, passed to `xr.Dataset.to_netcdf()`
    """
    # enable compression
    encoding = {d: {"zlib": True, "complevel": 9} for d in dset.data_vars}

    dset.to_netcdf(
        output_path, format="NETCDF4", engine="h5netcdf", encoding=encoding, **kwargs
    )


def plot_obs_for_all_couplings(dset, obs):

    ax = None
    for cdx, coupling in enumerate(dset["coupling"].to_numpy()):
        ax = plot_xr_with_errors(
            dset[obs].sel(coupling=coupling),
            ax=ax,
            color=cc.alpha_to_solid_on_bg(
                "#333", cc.fade(cdx, dset["coupling"].size, invert=True)
            ),
            label=f"w = {coupling:.1f}",
        )
    ax.legend()

    if obs == "correlation_coefficient":
        ax.set_ylim(0, 1)
    # if obs == "event_size":
    # ax.set_ylim(0, 1)
    cc.set_size2(ax, w=3.0, h=2.2)


def plot_xr_with_errors(da, ax=None, apply_formatting=True, **kwargs):
    """
    Plot an observable provided via an xarray, with error bars over repetitions

    `da` needs to have specified all coordinate points except for two dimensions,
    the one that becomes the x axis and `repetitions`

    Use e.g. `da.loc[dict(dim_0=0.1, dim_2=0)]`

    # Parameters
    da : xarray.DataArray
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    assert (
        len(da.shape) == 2
    ), f"specify all coordinates except repetitions and one other {da.coords}"

    # this would fail if we only have one data point to plot
    x_name = [cs for cs in da.coords if (cs != "repetition" and da[cs].size > 1)][0]
    num_reps = da["repetition"].size

    if apply_formatting:
        ax.set_ylabel(da.name)
        ax.set_xlabel(x_name)

    plot_kwargs = kwargs.copy()
    plot_kwargs.setdefault("color", "#333")
    plot_kwargs.setdefault("fmt", "o")
    plot_kwargs.setdefault("markersize", 1.5)
    plot_kwargs.setdefault("elinewidth", 0.5)
    plot_kwargs.setdefault("capsize", 1.5)

    ax.errorbar(
        x=da[x_name],
        y=da.mean(dim="repetition", skipna=True),
        yerr=da.std(dim="repetition") / np.sqrt(num_reps),
        **plot_kwargs,
    )

    return ax


def plot_resource_cycle(input_file):
    h5f = mph.read_csv_and_format(input_file, "model")
    mph.module_contribution(h5f, "model", 1.0, area_min=0.7, roll_window=0.5)
    ax = ph.plot_resources_vs_activity(
        h5f, apply_formatting=False, max_traces_per_mod=20, clip_on=False
    )
    ax.set_xlabel("Synaptic resources")
    ax.set_ylabel("Module rate")
    ax.set_title(input_file)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.4, 4)
    cc.set_size3(ax, 3.5, 3)

    sns.despine(ax=ax, trim=True, offset=5)


def analyse_data(input_folder):
    # the structure we get from mesoscopic simulations is
    # ```
    # input_folder
    #   coup0.10-0
    #       noise0.hdf5
    #       noise1.hdf5
    #   coup0.10-1
    # ```
    # 0.10 is the coupling value, -0 and 1 are the repetitions
    # and the integer after noise is the noise value
    #

    # collect the folders and files
    # the `/**/` for all subdirectories is only supported by python >= 3.5
    candidates = glob.glob(
        os.path.abspath(os.path.expanduser(input_folder + "/**/*.hdf5"))
    )

    couplings = []
    noises = []
    reps = []

    # fetch the coordinates we may find in our folders
    for cdx, candidate in enumerate(candidates):
        regex = re.search("coup(\d+.\d+)-(\d+)/noise(\d+).hdf5", candidate, re.IGNORECASE)
        coupling = float(regex.group(1))
        rep = int(regex.group(2))
        noise = int(regex.group(3))

        if coupling not in couplings:
            couplings.append(coupling)
        if noise not in noises:
            noises.append(noise)
        if rep not in reps:
            reps.append(rep)

    print(f"found {len(couplings)} couplings, {len(noises)} noises, {len(reps)} reps")

    # create coords that work with xarrays
    coords = dict()
    coords["repetition"] = np.sort(reps)
    coords["coupling"] = np.sort(couplings)
    coords["noise"] = np.sort(noises)

    # lets have an xarray dataset that has an array for every scalar observable
    observables = ["correlation_coefficient", "event_size"]
    dset = xr.Dataset(coords=coords)
    for obs in observables:
        dset[obs] = xr.DataArray(np.nan, coords=dset.coords)

    # assigning default variables makes it easer to parallelise via dask, later
    f = functools.partial(
        mph.ps_process_data_from_file,
        processing_functions=[mph.ps_f_correlation_coefficients, mph.ps_f_event_size],
        labels=observables,
    )

    for candidate in tqdm(candidates, desc="analysing files"):
        regex = re.search("coup(\d+.\d+)-(\d+)/noise(\d+).hdf5", candidate, re.IGNORECASE)
        coupling = float(regex.group(1))
        rep = int(regex.group(2))
        noise = int(regex.group(3))
        cs = dict(noise=noise, coupling=coupling, repetition=rep)

        results = f(candidate)
        for obs in results.keys():
            dset[obs].loc[cs] = results[obs]

    return dset
