# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-11-08 17:51:24
# @Last Modified: 2022-08-10 17:21:42
# ------------------------------------------------------------------------------ #
#
# How to read this monstosity of a file?
#
# Start at the high-level functions for compound figures, they are named `fig_x()`
# and are placed in the beginning. From there, use your code editor to jump
# to lower-level functions that come further down. (in vscode option/alt+click)
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import re
import h5py
import argparse
import numbers
import logging
import warnings
import functools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import palettable
from tqdm.auto import tqdm
from scipy import stats
from benedict import benedict

# our tools
import bitsandbobs as bnb
from bitsandbobs import plt as cc

import plot_helper as ph
import ana_helper as ah
import ndim_helper as nh
import meso_helper as mh


logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")
warnings.filterwarnings("ignore")  # suppress numpy warnings

# ------------------------------------------------------------------------------ #
# Settings
# ------------------------------------------------------------------------------ #

reference_coordinates = dict()
reference_coordinates["jA"] = 45
reference_coordinates["jG"] = 50
reference_coordinates["tD"] = 20
reference_coordinates["k_inter"] = 5

# we had one trial for single bond where we saw more bursts than everywhere else
remove_outlier = True

# some default file paths, relative to this file
_p_base = os.path.dirname(os.path.realpath(__file__))
# output path for figure panels
p_fo =  os.path.abspath(_p_base + f"/../fig/paper/")
p_exp = os.path.abspath(_p_base + f"/../dat/experiments/")
p_sim = os.path.abspath(_p_base + f"/../dat/simulations/")


# ------------------------------------------------------------------------------ #
# Settings, Styling
# ------------------------------------------------------------------------------ #

# select default things to draw for every panel
show_title = True
show_xlabel = True
show_ylabel = True
# legends were mostly done in affinity designer so dont rely on those to work well
show_legend = True
show_legend_in_extra_panel = False

matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"] = 6
matplotlib.rcParams["ytick.labelsize"] = 6
matplotlib.rcParams["xtick.major.pad"] = 2  # padding between text and the tick
matplotlib.rcParams["ytick.major.pad"] = 2  # default 3.5
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["font.size"] = 6
matplotlib.rcParams["axes.titlesize"] = 6
matplotlib.rcParams["axes.labelsize"] = 6
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)  # transparent figure bg
matplotlib.rcParams["axes.facecolor"] = (1.0, 0.0, 0.0, 0.0)

# style of error bars 'butt' or 'round'
# "butt" gives precise errors, "round" looks much nicer but most people find it confusing.
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/joinstyle.html
_error_bar_cap_style = "butt"

colors = dict()

colors["pre"] = "#541854"
colors["Off"] = colors["pre"]

colors["stim"] = "#BD6B00"
colors["On"] = colors["stim"]
colors["90 Hz"] = colors["stim"]

colors["post"] = "#8C668C"

colors["KCl_0mM"] = "gray"
colors["KCl_2mM"] = "gray"
colors["spon_Bic_20uM"] = colors["pre"]
colors["stim_Bic_20uM"] = colors["stim"]

colors["rij_within_stim"] = "#BD6B00"
colors["rij_within_nonstim"] = "#135985"
colors["rij_across"] = "#B31518"
colors["rij_all"] = "#222"

colors["k=1"] = dict()
colors["k=1"]["75.0 Hz"] = colors["pre"]
colors["k=1"]["85.0 Hz"] = colors["stim"]
colors["k=5"] = dict()
colors["k=5"]["80.0 Hz"] = colors["pre"]
colors["k=5"]["90.0 Hz"] = colors["stim"]
colors["k=10"] = dict()
colors["k=10"]["85.0 Hz"] = colors["pre"]
colors["k=10"]["92.5 Hz"] = colors["stim"]
colors["partial"] = dict()
colors["partial"]["0.0 Hz"] = colors["pre"]
colors["partial"]["20.0 Hz"] = colors["stim"]

# ------------------------------------------------------------------------------ #
# Whole-figure wrapper functions
# ------------------------------------------------------------------------------ #

def fig_1(show_time_axis=False):
    """
    Wrapper for Figure 1 containing
    - an example raster plot for single-bond topology
        at the stimulation conditions pre | stim | post
    - an example raster plot of the single-bond controls exposed to KCl ("chemical")
    - Trial-level "stick" plots for Functional Complexity and Event Size,
        * comparing pre (left) vs stim (right)
        * for optogenetic stimulation (yellow) and chemical (gray)
    """

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(811)

    # ------------------------------------------------------------------------------ #
    # Create raster plots, need `raw` files
    # ------------------------------------------------------------------------------ #

    # optogenetic
    path = f"{p_exp}/raw/1b"
    experiment = "210719_B"
    conditions = ["1_pre", "2_stim", "3_post"]
    fig_widths = dict()
    fig_widths["1_pre"] = 2.4
    fig_widths["2_stim"] = 7.2
    fig_widths["3_post"] = 2.4
    time_ranges = dict()
    time_ranges["1_pre"] = [60, 180]
    time_ranges["2_stim"] = [58, 418]
    time_ranges["3_post"] = [100, 220]

    # ph.log.setLevel("DEBUG")

    for idx, condition in enumerate(conditions):
        log.info(condition)
        c_str = condition[2:]
        fig = exp_raster_plots(
            path,
            experiment,
            condition,
            time_range=time_ranges[condition],
            fig_width=fig_widths[condition] * 1.2 / 2.54,
            show_fluorescence=False,
        )
        # if we want axes to have the same size across figures,
        # we need to set the axes size again (as figures include padding for legends)
        _set_size(ax=fig.axes[0], w=fig_widths[condition], h=None)

        if not show_time_axis:
            fig.axes[-1].xaxis.set_visible(False)
            sns.despine(ax=fig.axes[-1], bottom=True, left=False)

        if show_title:
            fig.axes[1].set_title(f"{c_str}")
        if show_ylabel:
            fig.axes[-1].set_ylabel(f"Rate (Hz)")

        fig.savefig(f"{p_fo}/exp_combined_{c_str}.pdf", dpi=300, transparent=False)

    # chemical
    path = f"{p_exp}/raw/KCl_1b"
    experiment = "210720_B"
    conditions = ["1_KCl_0mM", "2_KCl_2mM"]
    fig_widths = dict()
    fig_widths["1_KCl_0mM"] = 2.4
    fig_widths["2_KCl_2mM"] = 3.6
    time_ranges = dict()
    time_ranges["1_KCl_0mM"] = [395, 395 + 120]
    time_ranges["2_KCl_2mM"] = [295, 295 + 180]

    for idx, condition in enumerate(conditions):
        log.info(condition)
        c_str = condition[2:]
        fig = exp_raster_plots(
            path,
            experiment,
            condition,
            time_range=time_ranges[condition],
            fig_width=fig_widths[condition] * 1.2 / 2.54,
            show_fluorescence=False,
        )
        _set_size(ax=fig.axes[0], w=fig_widths[condition], h=None)

        if not show_time_axis:
            fig.axes[-1].xaxis.set_visible(False)
            sns.despine(ax=fig.axes[-1], bottom=True, left=False)

        if show_title:
            fig.axes[1].set_title(f"{c_str}")
        if show_ylabel:
            fig.axes[-1].set_ylabel(f"Rate (Hz)")

        fig.savefig(f"{p_fo}/exp_combined_{c_str}.pdf", dpi=300, transparent=False)

    # ------------------------------------------------------------------------------ #
    # Stick plots for optogenetic vs chemical
    # ------------------------------------------------------------------------------ #

    for obs in ["Functional Complexity", "Mean Fraction"]:
        ax = exp_chemical_vs_opto(observable=obs, draw_error_bars=False)
        cc.set_size(ax, w=1.2, h=2, l=1.2, r=.7, b=.2, t=.5)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
        ax.get_figure().savefig(f"{p_fo}/exp_chem_vs_opto_{obs}.pdf", dpi=300)

        if show_title or show_ylabel:
            # we changed naming conventions at some point
            o_str = "Event size" if obs == "Mean Fraction" else obs
            ax.set_title(f"{o_str}")


def fig_2(skip_plots=False):
    """
    Wrapper for Figure 2 containing
    - pooled Violins that aggregate the results of all trials for
        * observables: Event size, Correclation Coefficient, IEI, Burst-Core delay
        * sorted by topology: single-bond | triple-bond | merged
        * for conditions: pre | stim | post,
    - Decomposition plots (scatter and bar) that show Correlation Coefficients
        * sorted by the location of the neuron pairs:
            - both targeted (yellow)
            - both not targeted (blue)
            - either one targeted (red)
        * for conditions: pre (dark) vs stim (light)
    - Trial level stick plots of FC for all topologies, 1-b, 3-b, merged
    """

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(812)

    if not skip_plots:
        exp_violins_for_layouts()
        exp_rij_for_layouts()
        exp_sticks_across_layouts(observable="Functional Complexity", hide_labels=False)

    log.debug("-------------------------")
    log.debug("Pairwise tests for trials")
    exp_pairwise_tests_for_trials(
        observables=[
            "Functional Complexity",
        ]
    )


def fig_3(pd_path=None, raw_paths=None, out_suffix=""):
    """
    Wrapper for Figure 3 on Simulations containing
    - pooled Violins that aggregate the results of all trials for
        * Event size and Correlation Coefficient
        * at 0Hz vs 20Hz additional stimulation in targeted modules (on top of 80Hz baseline)
    - Decomposition plots (scatter and bar) anaologous to Figure 2
    """

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(813)

    if pd_path is None:
        pd_path = f"{p_sim}/lif/processed/k=5.hdf5"

    if raw_paths is None:
        raw_paths = [
            f"{p_sim}/lif/raw/stim=02_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_stimrate=0.0_rep=000.hdf5",
            f"{p_sim}/lif/raw/stim=02_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_stimrate=20.0_rep=000.hdf5",
        ]

    osx = out_suffix

    # reproducing 2 module stimulation in simulations

    dfs = load_pd_hdf5(pd_path, ["bursts", "rij", "rij_paired"])
    df = dfs["rij_paired"]

    # ------------------------------------------------------------------------------ #
    # violins
    # ------------------------------------------------------------------------------ #

    def apply_formatting(ax, ylim=True, trim=True):
        if ylim:
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        sns.despine(ax=ax, bottom=True, left=False, trim=trim, offset=2)
        ax.tick_params(bottom=False)
        ax.set_xlabel(f"")
        ax.set_ylabel(f"")
        # ax.set_xticks([])
        cc.set_size(ax, 1.5, 2.0)

    log.info("")
    log.info("# simulation with two targeted modules")
    ax = custom_violins(
        dfs["bursts"],
        category="Condition",
        observable="Fraction",
        ylim=[0, 1],
        num_swarm_points=300,
        bw=0.2,
        palette=colors["partial"],
    )
    apply_formatting(ax)
    ax.set_xlabel("Event size")
    ax.get_figure().savefig(f"{p_fo}/sim_partial_violins_fraction{osx}.pdf", dpi=300)

    log.info("")
    ax = custom_violins(
        dfs["rij"],
        category="Condition",
        observable="Correlation Coefficient",
        ylim=[0, 1],
        num_swarm_points=600,
        bw=0.2,
        palette=colors["partial"],
    )
    apply_formatting(ax)
    ax.set_xlabel("Correlation")
    ax.get_figure().savefig(f"{p_fo}/sim_partial_violins_rij{osx}.pdf", dpi=300)

    # ------------------------------------------------------------------------------ #
    # tests for violins
    # ------------------------------------------------------------------------------ #

    # sim_tests_stimulating_two_modules(observables=["Fraction"])
    # sim_tests_stimulating_two_modules(observables=["Correlation Coefficient"])

    # ------------------------------------------------------------------------------ #
    # pairwise rij plots
    # ------------------------------------------------------------------------------ #

    log.info("barplot rij paired for simulations")

    ax = custom_rij_barplot(df, conditions=["0.0 Hz", "20.0 Hz"], recolor=True)

    ax.set_ylim(0, 1)
    cc.set_size(ax, 3, 1.5)
    ax.get_figure().savefig(f"{p_fo}/sim_rij_barplot{osx}.pdf", dpi=300)

    log.info("scattered 2d rij paired for simulations")
    ax = custom_rij_scatter(
        df, max_sample_size=2500, scatter=True, kde_levels=[0.9, 0.95, 0.975]
    )
    cc.set_size(ax, 3, 3)
    ax.get_figure().savefig(f"{p_fo}/sim_2drij{osx}.pdf", dpi=300)

    # ------------------------------------------------------------------------------ #
    # raster plots for 2 module stimulation
    # ------------------------------------------------------------------------------ #

    for pdx, path in enumerate(raw_paths):
        h5f = ph.ah.prepare_file(path)

        fig, ax = plt.subplots()
        ph.plot_raster(
            h5f,
            ax,
            clip_on=True,
            zorder=-2,
            markersize=0.75,
            alpha=0.5,
            color="#333",
        )

        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        sns.despine(ax=ax, left=True, right=True, bottom=True, top=True)
        ax.set_xlim(0, 180)

        cc.set_size(ax, 2.7, 0.9)

        fig.savefig(f"{p_fo}/sim_raster_stim_02_{pdx}{osx}.pdf", dpi=900)


def fig_4(skip_rasters=True, skip_cycles=True, style_for_sm=True):
    """
    Wrapper for Figure 4 (extended) on Simulations containing
    - As a function of increasing Synaptic Noise Rate:
        * For k=5, the "Fraction of events that span"
            across 4 (dark), 3, 2, or only 1 modules (light)
        * For k=1 (lightest), k=5, k=10 and merged (darkest)
            - Mean Event size (Fraction of neurons that contribute to the event)
            - Mean Correlation Coefficient (neuron pairs)
            - Median and Mean rij (module pairs)
            - Functional Complexity
            - Average Number of spikes each neuron fired during a detected event
            - Inter-event interval
            - Core delay (the delay between the time points of each involved
                modules maximum firing rate)
            - The average amount of synaptic resources at the time of the starting
                of the bursting event
    - Optionally, example raster plots
        * A sketch of the topology
        * Population-level rates in Hz (top)
        * Raster, color coded by module
        * Module-level synaptic resources available (bottom)
        * A zoomin of the raster of a single bursting event (right)
        Sorted by
            - the number of connections between modules (k)
            - and the "Synaptic Noise Rate" - a Poisson input provided to all neurons.
    - Optionally, charge-discharge cycles for the examples in the raster plots.

    """

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(814)


    # ------------------------------------------------------------------------------ #
    # Number of modules over which bursts / events extend
    # ------------------------------------------------------------------------------ #

    cs = reference_coordinates.copy()
    cs["k_inter"] = 5

    ax = sim_modules_participating_in_bursts(
        input_path=f"{p_sim}/lif/processed/ndim.hdf5",
        simulation_coordinates=cs,
        xlim_for_points=[65, 100],
        xlim_for_fill=[65, 100],
        drop_zero_len=True,
    )
    cc.set_size(ax, 3.5, 2.0)
    if show_legend:
        ax.legend(loc="lower left", ncol=1, frameon=False, handletextpad=0.01, fontsize=6)
    ax.get_figure().savefig(f"{p_fo}/sim_fractions.pdf", dpi=300)

    # ------------------------------------------------------------------------------ #
    # observables vs noise for differen k
    # this is also creates supplementary panels
    # ------------------------------------------------------------------------------ #

    unit_observables = [
        "sys_mean_participating_fraction",
        "sys_mean_correlation",
        "sys_functional_complexity",
        "mod_mean_correlation",
        "mod_median_correlation",
    ]
    observables = unit_observables + [
        "any_num_spikes_in_bursts",
        # "sys_median_any_ibis",
        "sys_mean_any_ibis",
        # testing order parameters
        # "sys_orderpar_fano_neuron",
        # "sys_orderpar_fano_population",
        # "sys_orderpar_baseline_neuron",
        # "sys_orderpar_baseline_population",
        "sys_mean_core_delay",
        "sys_mean_resources_at_burst_beg",
    ]

    ylabels = dict()
    ylabels["sys_mean_participating_fraction"] = "Event size"
    ylabels["sys_mean_correlation"] = "Correlation\ncoefficient"
    ylabels["sys_functional_complexity"] = "Functional\ncomplexity"
    ylabels["any_num_spikes_in_bursts"] = "Spikes\nper neuron in event"
    ylabels["sys_median_any_ibis"] = "Inter-event-interval\n(seconds)"
    ylabels["sys_mean_any_ibis"] = "Inter-event-interval\n(seconds)"
    ylabels["sys_orderpar_fano_neuron"] = "fano neuron"
    ylabels["sys_orderpar_fano_population"] = "fano population"
    ylabels["sys_orderpar_baseline_neuron"] = "baseline neuron"
    ylabels["sys_orderpar_baseline_population"] = "baseline population"
    ylabels["sys_mean_core_delay"] = "Core delay (seconds)"
    ylabels["sys_mean_resources_at_burst_beg"] = "Resources\nat event start"
    ylabels["mod_median_correlation"] = "mod rij median"
    ylabels["mod_mean_correlation"] = "mod rij mean"

    coords = reference_coordinates.copy()
    coords["k_inter"] = [1, 5, 10, -1]

    base_colors = [
        "#1D484F",
        "#371D4F",
        "#4F1D20",
        "#4F481D",
        "#333",
    ]

    for odx, obs in enumerate(observables):

        # setup colors
        # base_color = f"C{odx}"
        # base_color = palettable.cartocolors.qualitative.Antique_5.hex_colors[
        #     4 - odx
        # ]
        try:
            base_color = base_colors[odx]
            assert False  # we decided against colors
        except:
            base_color = "#333"
        clrs = dict()
        for kdx, k in enumerate([1, 5, 10, -1]):
            clrs[k] = cc.alpha_to_solid_on_bg(base_color, cc.fade(kdx, 4, invert=True))

        ax = sim_obs_vs_noise_for_all_k(
            observable=obs,
            path=f"{p_sim}/lif/processed/ndim.hdf5",
            simulation_coordinates=coords,
            colors=clrs,
            clip_on=False if obs in unit_observables else True,
        )
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
        ax.set_xlim(62.5, 112.5)
        if obs in unit_observables:
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
            ax.set_ylim(0, 1.0)

        if obs in ["sys_median_any_ibis", "sys_mean_any_ibis"]:
            ax.set_ylim(0, 70)
        if obs == "any_num_spikes_in_bursts":
            ax.set_ylim(0, 10)
        if "sys_orderpar_baseline" in obs:
            ax.set_ylim(0, 1.05)
        if "sys_orderpar_fano" in obs:
            ax.set_ylim(0, 0.1)
        if obs == "sys_mean_core_delay":
            ax.set_ylim(0, None)

        if show_ylabel:
            ax.set_ylabel(ylabels[obs])
        if obs in [
            "sys_mean_any_ibis",
            "sys_mean_core_delay",
            "any_num_spikes_in_bursts",
            "sys_mean_resources_at_burst_beg",
        ]:
            # these guys only go to the supplemental material
            cc.set_size(ax, 3.5, 2.5)
        else:
            cc.set_size(ax, 2.5, 1.8)

        if show_legend:
            leg = ax.legend(ncol=1, handletextpad=0.01, fontsize=6)
            cc.apply_default_legend_style(leg)

        ax.get_figure().savefig(f"{p_fo}/sim_ksweep_{obs}.pdf", dpi=300)

    # ------------------------------------------------------------------------------ #
    # raster plots
    # ------------------------------------------------------------------------------ #

    def path(k, rate, rep):
        # we additonally sampled a few simulations at higher time resolution. this gives
        # higher precision for the resource variable, but takes tons of disk space.
        # path = f"{p_sim}/lif/raw_highres/"
        path = f"{p_sim}/lif/raw/"
        path += f"stim=off_k={k:d}_jA=45.0_jG=50.0_jM=15.0_tD=20.0"
        path += f"_rate={rate:.1f}_rep={rep:03d}.hdf5"
        return path

    coords = []
    times = []  # start time for the large time window of ~ 180 seconds
    zooms = []  # start time for an interesting 250 ms time window showing a burst

    same_rates_for_all_topos = True
    if same_rates_for_all_topos:
        coords.append(dict(k=1, rate=80, rep=1))
        zooms.append(343.95)
        times.append(0)
        coords.append(dict(k=1, rate=90, rep=1))
        zooms.append(343.95)
        times.append(0)

        coords.append(dict(k=5, rate=80, rep=1))
        zooms.append(298.35)
        times.append(0)
        coords.append(dict(k=5, rate=90, rep=1))
        zooms.append(288.05)
        times.append(0)

        coords.append(dict(k=10, rate=80, rep=1))
        zooms.append(313.95)
        times.append(0)
        coords.append(dict(k=10, rate=90, rep=1))
        zooms.append(314.5)
        times.append(0)

        coords.append(dict(k=-1, rate=80, rep=1))
        zooms.append(333.40)
        times.append(0)
        coords.append(dict(k=-1, rate=90, rep=1))
        zooms.append(338.45)
        times.append(0)
    else:
        # these points give roughly matching IEI across realizations
        coords.append(dict(k=1, rate=75, rep=1))
        zooms.append(299.5)
        times.append(0)
        coords.append(dict(k=1, rate=85, rep=1))
        zooms.append(333.95)
        times.append(0)

        coords.append(dict(k=5, rate=80, rep=1))
        zooms.append(298.35)
        times.append(0)
        coords.append(dict(k=5, rate=90, rep=1))
        zooms.append(288.05)
        times.append(0)

        coords.append(dict(k=10, rate=85, rep=1))
        zooms.append(347.85)
        times.append(0)
        coords.append(dict(k=10, rate=92.5, rep=1))
        zooms.append(347.75)
        times.append(0)

        coords.append(dict(k=-1, rate=85.0, rep=1))
        zooms.append(0)
        times.append(0)
        coords.append(dict(k=-1, rate=100.0, rep=1))
        zooms.append(0)
        times.append(0)

    for idx in range(0, len(coords)):
        if skip_rasters:
            break

        cs = coords[idx]
        fig = sim_raster_plots(
            path=path(**cs),
            time_range=(times[idx], times[idx] + 360),
            zoom_time=zooms[idx],
            mark_zoomin_location=True,
        )
        # update to get exact axes width
        ax = fig.axes[0]
        if cs["k"] == 10:
            ax.set_ylim(0, 120)
        elif cs["k"] == -1:
            ax.set_ylim(0, 160)

        _set_size(ax=ax, w=3.5, h=None)
        k_str = f"merged" if cs["k"] == -1 else f"k={cs['k']}"
        ax.text(
            0.5,
            0.98,
            f"{k_str}    {cs['rate']}Hz",
            va="center",
            ha="center",
            transform=ax.transAxes,
        )
        fig.savefig(
            f"{p_fo}/sim_ts_combined_k={cs['k']}_nozoom_{cs['rate']}Hz.pdf",
            dpi=900,  # use higher res to get rasters smooth
            transparent=False,
        )

        # do we want a schematic of the topology?
        sim_layout_sketch(
            in_path=path(**cs),
            out_path=f"{p_fo}/sim_layout_sketch_{cs['k']}_{cs['rate']}Hz.png",
        )

    # ------------------------------------------------------------------------------ #
    # panel h, resource cycles
    # this has become quite horrible to read because we also do the sm version.
    # me sorry.
    # ------------------------------------------------------------------------------ #

    if not skip_cycles:
        # sim_resource_cycles(apply_formatting=True, k_list=[-1, 5])
        # for main manuscript, defaults above are fine, but for SI bigger overview:
        axes = sim_resource_cycles(apply_formatting=False, k_list=[-1, 1, 5, 10])
        for k in axes.keys():
            for rate in axes[k].keys():
                ax = axes[k][f"{rate}"]
                k_str = f"merged" if k == "-1" else f"k={k}"
                if show_title and style_for_sm:
                    ax.set_title(f"{k_str}    {rate}Hz")

                if not style_for_sm:
                    cc.set_size(ax, 1.6, 1.4)
                else:
                    cc.set_size(ax, 1.6, 1.4)
                ax.set_xlim(0.0, 1.0)
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
                # workaround to avoid cutting off figure content but limiting axes:
                # set, then despine, then set again
                ax.set_ylim(0, 150)
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(150))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))

                if not style_for_sm:
                    sns.despine(
                        ax=ax,
                        trim=True,
                        offset=2,
                        top=True,
                        right=True,
                        bottom=False,
                    )

                # if k != "1":
                #     cc.detick(ax.yaxis, keep_ticks=True)

                if style_for_sm:
                    if rate == "90":
                        cc.detick(ax.yaxis, keep_ticks=True, keep_labels=False)
                    ax.set_ylim(0, 200)
                    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
                    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))
                else:
                    if rate == "80":
                        cc.detick(ax.xaxis, keep_ticks=False, keep_labels=False)
                        sns.despine(ax=ax, bottom=True)
                    ax.set_ylim(0, 199)
                    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
                    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))

                ax.get_figure().savefig(
                    f"{p_fo}/sim_resource_cycle_{k_str}_{rate}Hz.pdf",
                    transparent=False,
                )




def fig_5_(
    dset=None,
    out_path=f"{p_fo}/meso_gates_on",
):
    """
    Wrapper for Figure 5 (extended) on Mesoscopic model containing
    - As a function of increasing External input h:
        * Module-level correlation coefficients (mean)"
            - for different coupling values (low coupling light, high coupling dark)
            - Note that this is only loosely comparable to the rij of the neuron model
            as those were pairwise correlations.
        * The average number of modules that contributed in an event (like in Fig 4)
            - one panel for each coupling value
            - dark blue: 4 modules, ... light: 1 module
    - A sketch of the Resrouces vs Probability to disconnect.

    """


    # ------------------------------------------------------------------------------ #
    # Observables changing as a function of input
    # ------------------------------------------------------------------------------ #

    if dset is None:
        try:
            dset = xr.load_dataset(f"{p_sim}/meso/processed/analysed.hdf5")
        except:
            dset = mh.process_data_from_folder(f"{p_sim}/meso/raw/")
            mh.write_xr_dset_to_hdf5(dset, output_path=f"{p_sim}/meso/processed/analysed.hdf5")

        # dset = dset.sel(coupling=[0.025, 0.04, 0.1])

    # since this is meso model, everything is module level. lets clean up a bit.
    ax = meso_obs_for_all_couplings(dset, "mean_correlation_coefficient")
    ax.set_xlabel("External input")
    ax.set_ylabel("Module-level\ncorrelation (mean)")

    if not show_xlabel:
        ax.set_xlabel("")
    if not show_ylabel:
        ax.set_ylabel("")

    ax.set_xlim(0, 0.3125)
    sns.despine(ax=ax, offset=2)
    cc.set_size(ax, w=3.0, h=1.41)
    ax.get_figure().savefig(f"{out_path}_mean_rij.pdf", dpi=300, transparent=True)

    for c in dset["coupling"].to_numpy():
        try:
            ax = meso_module_contribution(dset, coupling=c)
            if show_title:
                ax.set_title(f"coupling {c:.3f}")
                ax.get_figure().tight_layout()
            if show_xlabel:
                ax.set_xlabel("External input")
            else:
                ax.set_xlabel("")
            ax.set_xlim(0, 0.3125)
            cc.set_size(ax, w=3.0, h=1.41)
            ax.get_figure().savefig(
                f"{out_path}_module_contrib_{c:.3f}.pdf", dpi=300, transparent=True
            )
        except:
            log.error(f"failed for {c:3.f}")

    ax = meso_sketch_gate_deactivation()
    ax.get_figure().savefig(f"{out_path}_gate_sketch.pdf", dpi=300, transparent=True)

def fig_5_snapshots(
    rep_path=f"{p_sim}/meso/raw/",
    out_path=f"{p_fo}/meso_gates_on",
    skip_snapshots=False,
    skip_cycles=False,
    zoom_times=None,

):

    # ------------------------------------------------------------------------------ #
    # Snapshots and resource cycles use a single realization
    # ------------------------------------------------------------------------------ #

    # get file path, try to use the longer time series
    if rep_path[-1] == "/":
        rep_path = rep_path[:-1]
    if os.path.exists(rep_path + "_long_ts"):
        rep_path += "_long_ts"

    # for snapshots we have zoomed insets. where should they be anchored?
    # zoom_times[coupling_float][noise_float] = start_time of zoom
    if zoom_times is None:
        zoom_times = benedict(keypath_separator="/")
        zoom_times[f"0.1/0.025"] = 753
        zoom_times[f"0.1/0.05"] = 756
        zoom_times[f"0.025/0.025"] = 930

    r = 0  # repetition
    # for c in dset["coupling"].to_numpy():
    for c in [0.1, 0.025]:
        for n in [1, 2, 4, 6, 7]:

            input_file = f"{rep_path}/coup{c:0.2f}-{r:d}/noise{n}.hdf5"
            if not os.path.exists(input_file):
                log.error(f"File not found {input_file}")
                continue

            coupling, noise, rep = mh._coords_from_file(input_file)
            h5f = mh.prepare_file(input_file)
            mh.find_system_bursts_and_module_contributions2(h5f)
            gates = h5f["meta.gating_mechanism"]

            if ("gates_on" in out_path and not gates) or (
                "gates_off" in out_path and gates
            ):
                log.warning(f"out_path '{out_path}' seems to mismatch gates {gates}")

            if not skip_cycles:
                ode_coords = None
                max_rsrc = 1.0
                # defaults work well for low noise
                # if noise >= 0.125:
                #     ode_coords = np.concatenate(
                #         (np.linspace(0, 20, 1000), np.linspace(20.01, 60, 500))
                #     )
                #     max_rsrc = 1.3

                ax = meso_resource_cycle(h5f, ode_coords=ode_coords, max_rsrc=max_rsrc)
                if show_title:
                    ax.set_title(f"coupling={c:.3f}\ninput={noise:.3f}")
                # print(f"coupling={c}, noise={noise}")
                # cc.set_size(ax, 1.6, 1.4) # this is the size of microscopic
                ax.set_xlim(0, 1.2)
                ax.set_ylim(0, 15)
                cc.set_size(ax, 1.8, 1.1)
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(15.0))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5.0))
                sns.despine(ax=ax, trim=False, offset=2)

                ax.get_figure().savefig(
                    f"{out_path}_cycles_{c:.3f}_{noise:.3f}.pdf",
                    dpi=300,
                    transparent=True,
                )

            if not skip_snapshots:
                try:
                    z = zoom_times[f"{coupling}"][f"{noise}"]
                except:
                    z = 950
                fig = meso_activity_snapshot(h5f, zoom_start=z)
                if show_title:
                    fig.suptitle(f"coupling={c:.3f}, input={noise:.3f}", va="center")
                fig.savefig(
                    f"{out_path}_snapshot_{c:.3f}_{noise:.3f}.pdf",
                    dpi=300,
                    transparent=True,
                )


def fig_supplementary():
    """
    wrapper to produce the panels of most supplementary figures.
    """
    sm_exp_trialwise_observables(prefix=f"{p_fo}/exp_layouts_sticks")
    exp_pairwise_tests_for_trials(
        observables=[
            # "Mean Correlation",
            # "Mean IBI",
            "Mean Fraction", # this is the event size
            "Functional Complexity",
            # "Mean Core delays",
            # "Mean Rate",
            # "Median IBI",
            # "Median Core delays",
        ],
        layouts=["1b", "3b", "merged"],
    )

    sm_exp_trialwise_observables(
        prefix=f"{p_fo}/exp_layouts_sticks_only_chem",
        layouts=["KCl_1b"],
        conditions=dict(
            KCl_1b=["KCl_0mM", "KCl_2mM"],
            # this is a hack, conditions are passed through to the stick plotter
            draw_error_bars=False,
        ),
    )
    exp_pairwise_tests_for_trials(
        observables=[
            # "Mean Correlation",
            # "Mean IBI",
            "Mean Fraction", # this is the event size
            "Functional Complexity",
            # "Mean Core delays",
            # "Mean Rate",
            # "Median IBI",
            # "Median Core delays",
        ],
        layouts=["1b", "3b", "merged"],
    )


    sm_exp_bicuculline()
    fig_3(
        pd_path="./dat/sim_partial_out_20/k=5.hdf5",
        raw_paths=[
            "./dat/sim_partial_out_20/k=5.hdf5",
        ],
    )
    sim_degrees_sampled(-1)
    sim_degrees_sampled(1)
    sim_degrees_sampled(5)
    sim_degrees_sampled(10)


def tables(output_folder):

    funcs = dict(
        trials=table_for_trials,
        rij=table_for_rij,
        violins=table_for_violins,
    )

    for key in funcs.keys():
        func = funcs[key]
        df = func()
        # we want core delay in miliseconds
        try:
            df["Core delays (ms)"] = df["Core delays"].apply(lambda x: x * 1000)
            df = df.drop("Core delays", axis=1)
        except:
            pass
        try:
            df["Inter-event-interval (seconds)"] = df["Inter-event-interval"]
            df = df.drop("Inter-event-interval", axis=1)
        except:
            pass

        # reorder columns
        cols = df.columns.to_list()
        for col in [
            "Event size",
            "Correlation Coefficient",
            "Functional Complexity",
            "Inter-event-interval (seconds)",
            "Core delays (ms)",
        ]:
            try:
                df.insert(len(cols) - 1, col, df.pop(col))
            except:
                pass

        df.to_excel(f"{output_folder}/data_{key}.xlsx", engine="openpyxl")

        # add the number of trials of the trial data frame to the layout description
        if func == table_for_trials:
            df = df.reset_index()
            df["layout"] = df["layout"] + " ($N=" + df["trials"].map(str) + "$ trials)"
            df = df.drop("trials", axis=1)
            df = df.set_index(["layout", "condition", "kind"])

        df.to_latex(
            f"{output_folder}/data_{key}.tex",
            na_rep="",
            bold_rows=False,
            multirow=True,
            multicolumn=True,
            float_format="{:2.2f}".format,
        )


def sm_exp_trialwise_observables(
    prefix=None, layouts=None, conditions=None, draw_error_bars=True
):
    """
    We can calculate estimates for every trial and see how they change
    within each trial across the conditions (usually, pre  → stim → post).

    Each trial is denoted by a faint purple line, averages across trials
    are denoted as a white dot, with error bars (thick vertical lines) and
    maximal observed values (thin vertical lines).

    We used some custom settings for each observable (plot range etc),
    so they are hidden inside the function definition.

    This has some overlap with fig. 1 and 2
    """
    if conditions is None and layouts is None:
        conditions = dict()
        for layout in ["1b", "3b", "merged"]:
            conditions[layout] = ["pre", "stim", "post"]
        # conditions["KCl_1b"] = ["KCl_0mM", "KCl_2mM"]
        layouts = list(conditions.keys())

    kwargs = dict(
        save_path=None,
        hide_labels=False,
        layouts=layouts,
        conditions=conditions,
        draw_error_bars=draw_error_bars,
    )

    if prefix is None:
        prefix = f"{p_fo}/exp_layouts_sticks"

    axes = []
    ax = exp_sticks_across_layouts(observable="Functional Complexity", **kwargs)
    ax.get_figure().savefig(f"{prefix}_functional_complexity.pdf", dpi=300)
    axes.append(ax)

    ax = exp_sticks_across_layouts(observable="Mean Fraction", **kwargs)
    ax.set_ylabel("Mean Event size")
    ax.get_figure().savefig(f"{prefix}_mean_event_size.pdf", dpi=300)
    axes.append(ax)

    ax = exp_sticks_across_layouts(observable="Mean Correlation", **kwargs)
    ax.get_figure().savefig(f"{prefix}_mean_correlation.pdf", dpi=300)
    axes.append(ax)

    ax = exp_sticks_across_layouts(observable="Mean IBI", set_ylim=[0, None], **kwargs)
    ax.set_ylabel("Mean IEI (seconds)")
    ax.get_figure().savefig(f"{prefix}_mean_iei.pdf", dpi=300)
    axes.append(ax)

    ax = exp_sticks_across_layouts(observable="Mean Rate", set_ylim=[0, None], **kwargs)
    ax.set_ylabel("Mean Rate (Hz)")
    ax.get_figure().savefig(f"{prefix}_mean_rate.pdf", dpi=300)
    axes.append(ax)

    ax = exp_sticks_across_layouts(
        observable="Mean Core delays", set_ylim=[0, None], **kwargs
    )
    ax.set_ylabel("Mean Core delay\n(seconds)")
    # ax.set_ylim(0, None)
    # sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    # cc.set_size(ax, 2.2, 2)
    ax.get_figure().savefig(f"{prefix}_mean_core_delay.pdf", dpi=300)
    axes.append(ax)

    if show_xlabel:
        try:
            # this wont get saved but never mind. mainly for the jupyter notebook
            xlabel = " | ".join([layout for layout in layouts])
            for ax in axes:
                ax.set_xlabel(xlabel)
        except:
            pass



def sm_exp_bicuculline():
    """
    violins and sticks for blocked inhibition
    """

    exp_violins_for_layouts(layouts=["bic"], observables=["event_size", "rij"])

    kwargs = dict(
        draw_error_bars=False,
        hide_labels=False,
        layouts=["Bicuculline_1b"],
        conditions=["spon_Bic_20uM", "stim_Bic_20uM"],
    )
    save_path = f"{p_fo}/exp_layouts_sticks_bic"

    ax = exp_sticks_across_layouts(observable="Mean Fraction", save_path=None, **kwargs)
    ax.set_ylabel("Mean Event size")
    ax.get_figure().savefig(f"{save_path}_fraction.pdf", dpi=300)

    ax = exp_sticks_across_layouts(
        observable="Mean Correlation", save_path=f"{save_path}_correlation.pdf", **kwargs
    )

    ax = exp_sticks_across_layouts(
        observable="Functional Complexity",
        save_path=f"{save_path}_functional_complexity.pdf",
        **kwargs,
    )

    # ax = exp_sticks_across_layouts(
    #     observable="Mean Core delays",
    #     set_ylim=False,
    #     save_path=None,
    #     apply_formatting=False,
    #     **kwargs,
    # )
    # ax.set_ylim(0, 0.4)
    # sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    # ax.get_figure().savefig(f"{save_path}_coredelays.pdf", dpi=300)

    exp_pairwise_tests_for_trials(
        observables=[
            "Mean Correlation",
            # "Mean IBI",
            # "Median IBI",
            "Mean Fraction",
            "Functional Complexity",
            # "Mean Core delays",
            # "Median Core delays",
        ],
        layouts=["Bicuculline_1b"],
    )


def sm_exp_number_of_cells():
    """
    For simplicity, we just hard code this data
    """

    data = [
        ["merged", "210713_400_C", 91],
        ["merged", "210726_400_B", 113],
        ["merged", "210726_400_C", 89],
        ["merged", "210401_400_A", 174],
        ["merged", "210405_400_A", 173],
        ["merged", "210406_400_A", 151],
        ["merged", "210406_400_B", 128],
        ["3b", "210713_3b_A", 143],
        ["3b", "210713_3b_B", 108],
        ["3b", "210713_3b_C", 101],
        ["3b", "210316_3b_A", 184],
        ["3b", "210316_3b_C", 151],
        ["3b", "210401_A_3b_A", 180],
        ["3b", "210402_B", 154],
        ["1b", "210719_1b_B", 118],
        ["1b", "210719_1b_C", 105],
        ["1b", "210726_1b_B", 114],
        ["1b", "210315_1b_A", 146],
        ["1b", "210315_1b_C", 154],
        ["1b", "210405_1b_C", 162],
        ["1b", "210406_1b_B", 145],
        ["1b", "210406_1b_C", 177],
    ]

    df = pd.DataFrame(data, columns=["Layout", "Trial", "Num Cells"])

    # to reuse our `exp_sticks_across_layouts` function, we need a dict of dfs
    # instead of a single dataframe
    dfs = dict()
    layouts = df["Layout"].unique()
    for layout in layouts:
        dfs[layout] = df.query("Layout == @layout")

    print(layouts)

    ax = exp_sticks_across_layouts(
        observable="Num Cells",
        layouts=layouts,
        dfs=dfs,
        conditions=["Only one"],
        save_path=None,
        apply_formatting=False,
        set_ylim=False,
        x_offset=0.3,
    )

    ax.set_ylim(0, 200)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100))
    sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    cc.set_size(ax, 2.2, 1.8)
    ax.set_ylabel("Number of cells")

    ax.get_figure().savefig(f"{p_fo}/exp_layouts_sticks_num_cells.pdf", dpi=300)


def table_for_violins():
    """
    Generate a table (pandas data frame) that holds all analysis results with calculated
    errors. serves as a consistency check when comparing to the plots.

    `table.to_excel("/Users/paul/Desktop/test.xls", engine="openpyxl")`
    `table.to_latex("/Users/paul/Desktop/test.xls", na_rep="")`
    """

    np.random.seed(314)

    # collect conditions, since they depend on the layout and where they are stored
    ref = benedict()
    ref["single-bond.conditions"] = ["pre", "stim", "post"]
    ref["tripe-bond.conditions"] = ["pre", "stim", "post"]
    ref["merged.conditions"] = ["pre", "stim", "post"]
    ref["chem.conditions"] = ["KCl_0mM", "KCl_2mM"]
    ref["bic.conditions"] = ["spon_Bic_20uM", "stim_Bic_20uM"]
    # simulation of the partial stimulation in only two modules
    ref["simulation.conditions"] = ["0.0 Hz", "20.0 Hz"]

    # where are the data frames stored
    ref["single-bond.df_path"] = "./dat/exp_out/1b.hdf5"
    ref["tripe-bond.df_path"] = "./dat/exp_out/3b.hdf5"
    ref["merged.df_path"] = "./dat/exp_out/merged.hdf5"
    ref["chem.df_path"] = "./dat/exp_out/KCl_1b.hdf5"
    ref["bic.df_path"] = "./dat/exp_out/Bicuculline_1b.hdf5"
    ref["simulation.df_path"] = "./dat/sim_partial_out_20/k=5.hdf5"

    # depending on the observable, we may need a different data frame and different
    # querying. so we use retreiving functions of the form
    # f(layout, condition) -> observable value, upper error, lower error
    def f(df_path, df_key, condition, observable):
        try:
            df = load_pd_hdf5(input_path=df_path, keys=df_key)
            df = df.query(f"`Condition` == '{condition}'")
            mid, std, percentiles = ah.pd_bootstrap(
                df,
                obs=observable,
                num_boot=500,
                func=np.nanmedian,
                percentiles=[50, 2.5, 97.5],
            )
            # using 50% as the median, so we have median of medians
            return percentiles
        except:
            # should not occur, but I added new analysis and I dont want older data
            # versions to crash this
            return [np.nan, np.nan, np.nan]

    observables = benedict()
    observables["Correlation Coefficient"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        df_key="rij",
        condition=condition,
        observable="Correlation Coefficient",
    )
    observables["Event size"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        df_key="bursts",
        condition=condition,
        observable="Fraction",
    )
    observables["Inter-event-interval"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        df_key="bursts",
        condition=condition,
        observable="Inter-burst-interval",
    )
    observables["Core delays"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        df_key="bursts",
        condition=condition,
        observable="Core delay",
    )

    table = pd.DataFrame(
        columns=["layout", "condition", "percentile"]
        + [obs for obs in observables.keys()]
    )

    for layout in tqdm(ref.keys(), desc="layouts"):
        for condition in tqdm(ref[f"{layout}.conditions"], leave=False):
            new_rows = pd.DataFrame(
                dict(
                    layout=[layout] * 3,
                    condition=[condition] * 3,
                    percentile=["50", "2.5", "97.5"],
                )
            )
            for obs in observables.keys():
                new_rows[obs] = observables[obs](layout, condition)

            table = table.append(new_rows, ignore_index=True)

    table = table.set_index(["layout", "condition", "percentile"])
    return table


def table_for_trials():
    """
    Generate a table (pandas data frame) that holds all analysis results with calculated
    errors. serves as a consistency check when comparing to the plots.

    cf. `sticks_across_layouts()`
    """

    np.random.seed(815)

    # collect conditions, since they depend on the layout and where they are stored
    ref = benedict()
    ref["single-bond.conditions"] = ["pre", "stim", "post"]
    ref["tripe-bond.conditions"] = ["pre", "stim", "post"]
    ref["merged.conditions"] = ["pre", "stim", "post"]
    ref["chem.conditions"] = ["KCl_0mM", "KCl_2mM"]
    ref["bic.conditions"] = ["spon_Bic_20uM", "stim_Bic_20uM"]
    ref["simulation.conditions"] = ["0.0 Hz", "20.0 Hz"]

    # where are the data frames stored
    ref["single-bond.df_path"] = "./dat/exp_out/1b.hdf5"
    ref["tripe-bond.df_path"] = "./dat/exp_out/3b.hdf5"
    ref["merged.df_path"] = "./dat/exp_out/merged.hdf5"
    ref["chem.df_path"] = "./dat/exp_out/KCl_1b.hdf5"
    ref["bic.df_path"] = "./dat/exp_out/Bicuculline_1b.hdf5"
    ref["simulation.df_path"] = "./dat/sim_partial_out_20/k=5.hdf5"

    # same idea as before, but we use the trial data frame, where every row is a trial
    def f(df_path, condition, observable, df_key="trials"):
        try:
            df = load_pd_hdf5(input_path=df_path, keys=df_key)
            trials = df["Trial"].unique()
            df = df.query(f"`Condition` == '{condition}'")
            assert len(df) == len(trials)
            mid, std, percentiles = ah.pd_bootstrap(
                df,
                obs=observable,
                num_boot=500,
                # here we use the mean
                func=np.nanmean,
                # and will not further percentiles
                percentiles=[50, 2.5, 97.5],
            )
            df_max = np.nanmax(df[observable])
            df_min = np.nanmin(df[observable])
            error = std

            res = dict()
            res["mean"] = mid
            res["sem"] = error
            res["max"] = df_max
            res["min"] = df_min
            res["trials"] = len(trials)
        except:
            # should not occur, but I added new analysis and I dont want older data
            # versions to crash this
            res = dict()
            res["mean"] = np.nan
            res["sem"] = np.nan
            res["max"] = np.nan
            res["min"] = np.nan
            res["trials"] = np.nan
        return res

    observables = benedict()
    observables["Correlation Coefficient"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        condition=condition,
        observable="Mean Correlation",
    )
    observables["Event size"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        condition=condition,
        observable="Mean Fraction",
    )
    observables["Inter-event-interval"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        condition=condition,
        observable="Mean IBI",
    )
    observables["Core delays"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        condition=condition,
        observable="Mean Core delays",
    )
    observables["Functional Complexity"] = lambda layout, condition: f(
        df_path=ref[f"{layout}.df_path"],
        condition=condition,
        observable="Functional Complexity",
    )

    table = pd.DataFrame(
        columns=["layout", "condition", "kind"] + [obs for obs in observables.keys()]
    )

    for layout in tqdm(ref.keys(), desc="layouts"):
        for condition in tqdm(ref[f"{layout}.conditions"], leave=False):
            new_rows = pd.DataFrame(
                dict(
                    layout=[layout] * 4,
                    condition=[condition] * 4,
                    kind=["mean", "sem", "max", "min"],
                )
            )
            trials = 0
            for obs in observables.keys():
                res = observables[obs](layout, condition)
                trials = res.pop("trials")
                new_rows[obs] = res.values()
            # we want trials as multi index, so reorder a bit
            new_rows["trials"] = [trials] * 4

            table = table.append(new_rows, ignore_index=True)

    table = table.set_index(["layout", "trials", "condition", "kind"])
    return table


def table_for_rij():

    # collect conditions, since they depend on the layout and where they are stored
    ref = benedict()
    ref["single-bond.conditions"] = ["pre", "stim"]
    ref["tripe-bond.conditions"] = ["pre", "stim"]
    ref["merged.conditions"] = ["pre", "stim"]
    ref["simulation.conditions"] = ["0.0 Hz", "20.0 Hz"]

    # where are the data frames stored
    ref["single-bond.df_path"] = "./dat/exp_out/1b.hdf5"
    ref["tripe-bond.df_path"] = "./dat/exp_out/3b.hdf5"
    ref["merged.df_path"] = "./dat/exp_out/merged.hdf5"
    ref["simulation.df_path"] = "./dat/sim_partial_out_20/k=5.hdf5"

    pairings = ["within_stim", "within_nonstim", "across", "all"]
    observable = "Correlation Coefficient"

    def f(df_path, condition, df_key="rij_paired"):
        df = load_pd_hdf5(input_path=df_path, keys=df_key)
        df = df.query(f"`Condition` == '{condition}'")
        res = benedict(keypath_separator="/")
        for pairing in pairings:
            try:
                # we have calculated the pairings for merged system, but the are derived purely from neuron position, since no moduli exist. skip.
                if layout == "merged" and (pairing != "all"):
                    raise ValueError

                this_df = df.query(f"`Pairing` == '{pairing}'")
                np.random.seed(815)
                _, _, percentiles = ah.pd_bootstrap(
                    this_df,
                    obs=observable,
                    num_boot=500,
                    func=np.nanmedian,
                    percentiles=[50, 2.5, 97.5],
                )

                res[f"{pairing}/median"] = np.nanmedian(this_df[observable])
                res[f"{pairing}/2.5"] = percentiles[1]
                res[f"{pairing}/97.5"] = percentiles[2]

            except Exception as e:
                # some pairings do not exist in all data frames
                res[f"{pairing}/median"] = np.nan
                res[f"{pairing}/2.5"] = np.nan
                res[f"{pairing}/97.5"] = np.nan

        return res

    table = pd.DataFrame(columns=["layout", "condition", "kind"] + pairings)

    for layout in tqdm(ref.keys(), desc="layouts"):
        for condition in tqdm(ref[f"{layout}.conditions"], leave=False):
            res = f(df_path=ref[f"{layout}.df_path"], condition=condition)
            new_rows = pd.DataFrame(
                dict(
                    layout=[layout] * 3,
                    condition=[condition] * 3,
                    kind=["median", "2.5", "97.5"],
                )
            )
            for pairing in res.keys():
                new_rows[pairing] = res[pairing].values()

            table = table.append(new_rows, ignore_index=True)

    table = table.set_index(["layout", "condition", "kind"])
    return table


# Fig 1
def exp_raster_plots(
    path,
    experiment,
    condition,
    show_fluorescence=False,
    mark_bursts=False,
    time_range=None,
    fig_width=None,
    bs_large=200 / 1000,  # width of the gaussian kernel for rate
    threshold_factor=10 / 100,  # fraction of max peak height for burst
):

    # description usable for annotating
    c_str = condition[2:]

    h5f = ah.load_experimental_files(
        path_prefix=f"{path}/{experiment}/", condition=condition
    )

    ah.find_rates(h5f, bs_large=bs_large)
    threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
    ah.find_system_bursts_from_global_rate(
        h5f, rate_threshold=threshold, merge_threshold=0.1
    )

    if fig_width is None:
        fig_width = 3.5 / 2.54
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=[fig_width, 4 / 2.54])

    # only show 4 neurons per module, starting with module 0 on top
    if show_fluorescence:
        n_per_mod = 4
        npm = int(len(h5f["ana.neuron_ids"]) / len(h5f["ana.mods"]))
        neurons_to_show = []
        for m in range(0, len(h5f["ana.mods"])):
            neurons_to_show.extend(np.arange(0, n_per_mod) + m * npm)
    else:
        # show everything, we have space
        neurons_to_show = None

    # ------------------------------------------------------------------------------ #
    # Fluorescence
    # ------------------------------------------------------------------------------ #

    ax = axes[0]
    if not show_fluorescence:
        sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
    else:
        ax = ph.plot_fluorescence_trace(
            h5f,
            ax=ax,
            neurons=np.array(neurons_to_show),
            lw=0.25,
            base_color=colors[c_str],
        )

        sns.despine(ax=ax, left=True, bottom=True, trim=True, offset=2)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)

    # ------------------------------------------------------------------------------ #
    # Raster
    # ------------------------------------------------------------------------------ #

    # ph.log.setLevel("DEBUG")
    ax = axes[1]
    ph.plot_raster(
        h5f,
        ax,
        # base_color=colors[c_str],
        color=colors[c_str],
        sort_by_module=True,
        neuron_id_as_y=False,
        neurons=neurons_to_show,
        clip_on=True,
        markersize=2.5,
        alpha=1,
    )

    if mark_bursts:
        ph.plot_bursts_into_timeseries(
            h5f,
            ax,
            apply_formatting=False,
            style="fill_between",
            color=colors[c_str],
        )

    ax.set_ylim(-3, len(h5f["ana.neuron_ids"]) + 1.5)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    # _time_scale_bar(ax=ax, x1=340, x2=520, y=-2, ylabel=-3.5, label="180 s")

    # ------------------------------------------------------------------------------ #
    # Rates
    # ------------------------------------------------------------------------------ #

    ax = axes[2]
    ph.plot_system_rate(
        h5f,
        ax,
        mark_burst_threshold=False,
        color=colors[c_str],
        apply_formatting=False,
        clip_on=True,
        lw=1.0,
    )

    # ax.margins(x=0, y=0)
    if "KCl" in condition:
        ax.set_ylim(0, 5)
    else:
        ax.set_ylim(0, 5)

    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))

    # time bar
    # sns.despine(ax=ax, left=False, bottom=True, trim=False, offset=2)
    # ax.set_xticks([])
    # _time_scale_bar(ax=ax, x1=340, x2=520, y=-0.5, ylabel=-1.0, label="180 s")

    # normal time axis
    sns.despine(ax=ax, left=False, bottom=False, trim=False, offset=2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    # tighten space between panels
    plt.subplots_adjust(hspace=0.001)

    if time_range is None:
        time_range = [0, 540]
    ax.set_xlim(*time_range)

    return fig


def exp_chemical_vs_opto(observable="Functional Complexity", draw_error_bars=False):
    # here we decided to not show error bars because we had few realizations
    # and put more focus on individual trials
    chem = load_pd_hdf5(f"{p_exp}/processed/KCl_1b.hdf5")
    opto = load_pd_hdf5(f"{p_exp}/processed/1b.hdf5")

    # we want the precalculated summary statistics of each trial
    dfs = dict()
    dfs["exp"] = opto["trials"].query("`Condition` in ['pre', 'stim']")
    dfs["exp_chemical"] = chem["trials"]
    global df_merged
    df_merged = pd.concat([dfs["exp"], dfs["exp_chemical"]], ignore_index=True)

    fig, ax = plt.subplots()

    categories = ["exp", "exp_chemical"]
    conditions = ["Off", "On"]
    if draw_error_bars:
        small_dx = 0.3
        large_dx = 1.0
    else:
        small_dx = 0.5
        large_dx = 0.8
    x_pos = dict()
    x_pos_sticks = dict()
    for ldx, l in enumerate(categories):
        x_pos[l] = dict()
        x_pos_sticks[l] = dict()
        for cdx, c in enumerate(conditions):
            x_pos[l][c] = ldx * large_dx + cdx * small_dx
            x_pos_sticks[l][c] = ldx * large_dx + cdx * small_dx

    # opto
    for etype in categories:
        clr = colors["stim"] if etype == "exp" else colors["KCl_0mM"]
        trials = dfs[etype]["Trial"].unique()
        for trial in trials:
            df = dfs[etype].loc[dfs[etype]["Trial"] == trial]
            assert len(df) == 2
            x = []
            y = []
            for idx, row in df.iterrows():
                etype = row["Type"]
                stim = row["Stimulation"]
                x.append(x_pos[etype][stim])
                y.append(row[observable])

            if draw_error_bars:
                kwargs = dict(
                    lw=0.8,
                    color=cc.alpha_to_solid_on_bg(clr, 0.7),
                )
            else:
                kwargs = dict(
                    lw=1.0,
                    marker="o",
                    markersize=1.5,
                    color=cc.alpha_to_solid_on_bg(clr, 1.0),
                )

            ax.plot(
                x,
                y,
                label=trial,
                zorder=0,
                clip_on=False,
                **kwargs,
            )
        if not draw_error_bars:
            continue

        for stim in ["On", "Off"]:
            df = dfs[etype].query(f"Stimulation == '{stim}'")
            # sticklike error bar
            mid, std, percentiles = ah.pd_bootstrap(
                df,
                obs=observable,
                num_boot=500,
                func=np.nanmean,
                percentiles=[2.5, 50, 97.5],
            )
            df_max = np.nanmax(df[observable])
            df_min = np.nanmin(df[observable])
            error = std

            _draw_error_stick(
                ax,
                center=x_pos_sticks[etype][stim],
                # mid=percentiles[1],
                # errors=[percentiles[0], percentiles[2]],
                mid=mid,
                errors=[mid - error, mid + error],
                outliers=[df_min, df_max],
                orientation="v",
                color=clr,
                zorder=2,
            )

    # ax.legend()
    ax.set_xlim(-0.25, 1.5)
    ax.set_xticks([])

    return ax


def exp_sticks_across_layouts(
    observable="Functional Complexity",
    hide_labels=True,
    set_ylim=True,
    apply_formatting=True,
    layouts=None,
    conditions=None,
    save_path="automatic",
    draw_error_bars=True,
    dfs=None,
    x_offset=0,
):
    log.info(f"")
    log.info(f"# sticks for {observable}")

    if layouts is None:
        layouts = ["1b", "3b", "merged"]
    if conditions is None:
        conditions = dict()
        for etype in layouts:
            conditions[etype] = ["pre", "stim", "post"]

    # cast up if bool
    if not isinstance(draw_error_bars, dict):
        temp = draw_error_bars
        draw_error_bars = {layout: temp for layout in layouts}

    # we want the precalculated summary statistics of each trial
    if dfs is None:
        dfs = dict()
        for key in layouts:
            df = load_pd_hdf5(f"{p_exp}/processed/{key}.hdf5")
            local_conditions = conditions[key]
            dfs[key] = df["trials"].query("Condition == @local_conditions")

    fig, ax = plt.subplots()

    small_dx = 0.3
    large_dx = 1.5
    x_pos = dict()
    x_pos_sticks = dict()
    for ldx, l in enumerate(layouts):
        x_pos[l] = dict()
        x_pos_sticks[l] = dict()
        for cdx, c in enumerate(conditions[l]):
            x_pos[l][c] = x_offset + ldx * large_dx + cdx * small_dx
            x_pos_sticks[l][c] = x_offset + ldx * large_dx + cdx * small_dx

    # x_pos["1b"] = {"pre" : 0.0, "stim" : 1.0, "post" :  2.0 }
    # x_pos["3b"] = {"pre" : 0.25, "stim" : 1.25, "post"  :  2.25 }
    # x_pos["merged"] = {"pre" : 0.5, "stim" : 1.5, "post" :  2.5 }
    # x_pos_sticks["1b"] = {"pre" : 0.0, "stim" : 1.0, "post" :  2.0 }
    # x_pos_sticks["3b"] = {"pre" : 0.25, "stim" : 1.25, "post"  :  2.25 }
    # x_pos_sticks["merged"] = {"pre" : 0.5, "stim" : 1.5, "post" :  2.5 }

    # opto
    for edx, etype in enumerate(layouts):
        # clr = f"C{edx}"
        log.info(f"## {etype}")

        log.info(
            f"| Condition | Mean (across trials) | Standard error of the mean | min"
            f" observed | max observed |"
        )
        log.info(
            f"| --------- | -------------------- | -------------------------- |"
            f" ------------ | ------------ |"
        )

        clr = colors["pre"] if etype != "KCl_1b" else colors["KCl_0mM"]
        trials = dfs[etype]["Trial"].unique()
        for trial in trials:
            df = dfs[etype].loc[dfs[etype]["Trial"] == trial]
            # assert len(df) == len(layouts)
            x = []
            y = []
            try:
                for idx, row in df.iterrows():
                    cond = row["Condition"]
                    x.append(x_pos[etype][cond])
                    y.append(row[observable])

                if draw_error_bars[etype]:
                    kwargs = dict(
                        lw=0.6,
                        color=cc.alpha_to_solid_on_bg(clr, 0.4),
                    )
                else:
                    kwargs = dict(
                        lw=1.0,
                        marker="o",
                        markersize=1.5,
                        color=cc.alpha_to_solid_on_bg(clr, 1.0),
                    )

                ax.plot(
                    x,
                    y,
                    label=trial,
                    zorder=0,
                    clip_on=False,
                    **kwargs,
                )
            except KeyError as e:
                # this fails if we have no conditions in the df,
                # needed for number of cells
                # or for chemical where we do not have the "post" condition.
                log.debug(f"{e}")

        if not draw_error_bars[etype]:
            continue

        for cond in conditions[etype]:

            try:
                clr = colors[cond]
            except:
                clr = "#808080"

            # skip post condition for chemical and set clrs differently
            if etype == "KCl_1b":
                clr = colors["KCl_0mM"]
            try:
                df = dfs[etype].query(f"Condition == '{cond}'")
            except Exception as e:
                df = dfs[etype]

            # sticklike error bar
            mid, std, percentiles = ah.pd_bootstrap(
                df,
                obs=observable,
                num_boot=500,
                func=np.nanmean,
                percentiles=[2.5, 50, 97.5],
            )
            df_max = np.nanmax(df[observable])
            df_min = np.nanmin(df[observable])
            # error = std / np.sqrt(len(trials))
            error = std

            p_str = ""
            p_str += f"| {cond:>9} "
            p_str += f"| {mid:20.5f} "
            p_str += f"| {error:26.5f} "
            p_str += f"| {df_min:12.5f} "
            p_str += f"| {df_max:12.5f} |"

            log.info(p_str)

            _draw_error_stick(
                ax,
                center=x_pos_sticks[etype][cond],
                # mid=percentiles[1],
                # errors=[percentiles[0], percentiles[2]],
                mid=mid,
                errors=[mid - error, mid + error],
                outliers=[df_min, df_max],
                orientation="v",
                color=clr,
                zorder=2,
            )
        log.info(f"")

    # ax.legend()
    if isinstance(set_ylim, bool) and set_ylim is True:
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    elif isinstance(set_ylim, list):
        ax.set_ylim(*set_ylim)
    ax.set_xlim(-0.25, 3.5)
    if apply_formatting:
        sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    ax.set_xticks([])
    if not hide_labels:
        ax.set_ylabel(f"{observable}")

    if apply_formatting:
        cc.set_size(ax, 2.2, 2)

    if save_path == "automatic":
        save_path = f"{p_fo}/exp_layouts_sticks_{observable}.pdf"
    if save_path is not None:
        ax.get_figure().savefig(save_path, dpi=300)

    return ax


# Fig 2
def exp_violins_for_layouts(remove_outlier_for_ibis=True, layouts=None, observables=None):

    if layouts is None:
        layouts = ["single-bond", "triple-bond", "merged"]

    if observables is None:
        observables = ["event_size", "rij", "iei", "core_delay"]

    dfs = dict()
    if "single-bond" in layouts:
        dfs["single-bond"] = load_pd_hdf5(f"{p_exp}/processed/1b.hdf5")
    if "triple-bond" in layouts:
        dfs["triple-bond"] = load_pd_hdf5(f"{p_exp}/processed/3b.hdf5")
    if "merged" in layouts:
        dfs["merged"] = load_pd_hdf5(f"{p_exp}/processed/merged.hdf5")
    if "chem" in layouts:
        dfs["chem"] = load_pd_hdf5(f"{p_exp}/processed/KCl_1b.hdf5")
    if "bic" in layouts:
        dfs["bic"] = load_pd_hdf5(f"{p_exp}/processed/Bicuculline_1b.hdf5")

    def apply_formatting(ax, ylim=True, trim=True):
        if ylim:
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        sns.despine(ax=ax, bottom=True, left=False, trim=trim, offset=5)
        ax.tick_params(bottom=False)
        # reuse xlabel for title
        # ax.set_xlabel(f"{layout}")
        if not show_xlabel:
            ax.set_xlabel(f"")
        else:
            # dirty dirty hack
            ax.set_xlabel(f"pre → stim → post")
        if not show_ylabel:
            ax.set_ylabel(f"")

        ax.set_xticks([])
        cc.set_size(ax, 3, 2.0)

    # Event size, used to be called "Fraction" in the data frame
    for layout in dfs.keys():
        if "event_size" not in observables:
            continue
        log.info(f"")
        log.info(f"# {layout}")
        ax = custom_violins(
            dfs[layout]["bursts"],
            category="Condition",
            observable="Fraction",
            ylim=[0, 1],
            num_swarm_points=250,
            bw=0.2,
        )
        # we changed the naming convention while writing
        ax.set_ylabel("Event size")
        apply_formatting(ax)
        if show_title:
            ax.set_title(f"{layout}")
        ax.get_figure().savefig(f"{p_fo}/exp_violins_fraction_{layout}.pdf", dpi=300)

    # Correlation coefficients
    for layout in dfs.keys():
        if "rij" not in observables:
            continue
        log.info(f"")
        log.info(f"# {layout}")
        ax = custom_violins(
            dfs[layout]["rij"],
            category="Condition",
            observable="Correlation Coefficient",
            ylim=[0, 1],
            num_swarm_points=500,
            bw=0.2,
        )
        apply_formatting(ax)
        if show_title:
            ax.set_title(f"{layout}")
        ax.get_figure().savefig(f"{p_fo}/exp_violins_rij_{layout}.pdf", dpi=300)

    # IBI
    for layout in dfs.keys():
        if "iei" not in observables:
            continue
        log.info(f"")
        log.info(f"# {layout}")
        ax = custom_violins(
            dfs[layout]["bursts"].query("`Trial` != '210405_C'")
            if remove_outlier_for_ibis and layout == "single_bond"
            else dfs[layout]["bursts"],
            category="Condition",
            observable="Inter-burst-interval",
            ylim=[0, 70],
            num_swarm_points=250,
            bw=0.2,
        )
        # we changed the naming convention while writing
        ax.set_ylabel("Inter-event-interval\n(seconds)")
        if layout == "bic":
            ax.set_ylim(0, 200)
        else:
            ax.set_ylim(0, 70)
        apply_formatting(ax, ylim=False, trim=False)
        if show_title:
            ax.set_title(f"{layout}")
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
        ax.get_figure().savefig(f"{p_fo}/exp_violins_ibi_{layout}.pdf", dpi=300)

    # Core delay
    for layout in dfs.keys():
        if "core_delay" not in observables:
            continue
        # if layout == "merged":
        #     # for the merged system, we do not have modules, but the analysis still works
        #     # by using quadrants of the merged substrate.
        #     continue

        log.info(f"")
        log.info(f"# {layout}")
        ax = custom_violins(
            dfs[layout]["bursts"].query("`Trial` != '210405_C'")
            if remove_outlier_for_ibis and layout == "single_bond"
            else dfs[layout]["bursts"],
            category="Condition",
            observable="Core delay",
            ylim=[0, 0.4],
            num_swarm_points=250,
            bw=0.2,
        )
        ax.set_ylim(0, 0.4)
        ax.set_ylabel("Core delay\n(seconds)")
        apply_formatting(ax, ylim=False, trim=False)
        if show_title:
            ax.set_title(f"{layout}")
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.get_figure().savefig(
            f"{p_fo}/exp_violins_core_delay_{layout}.pdf", dpi=300
        )

    return ax


def exp_rij_for_layouts():
    dfs = dict()
    dfs["single-bond"] = load_pd_hdf5(f"{p_exp}/processed/1b.hdf5", ["rij_paired"])
    dfs["triple-bond"] = load_pd_hdf5(f"{p_exp}/processed/3b.hdf5", ["rij_paired"])
    dfs["merged"] = load_pd_hdf5(f"{p_exp}/processed/merged.hdf5", ["rij_paired"])

    # experimentally, we need to focus on pre vs stim, instead of on vs off,
    # to get _pairs_ of rij
    for key in dfs.keys():
        dfs[key]["rij_paired"] = dfs[key]["rij_paired"].query(
            "`Condition` in ['pre', 'stim']"
        )

    for layout in dfs.keys():
        log.debug(f"rijs for {layout}")
        pairings = None
        if layout == "merged":
            pairings = ["all"]

        # pairings = ["within_stim", "within_nonstim"]
        ax = custom_rij_scatter(dfs[layout]["rij_paired"], pairings=pairings)

        cc.set_size(ax, 3, 3)
        ax.get_figure().savefig(f"{p_fo}/exp_2drij_{layout}.pdf", dpi=300)

        ax = custom_rij_barplot(
            dfs[layout]["rij_paired"], pairings=pairings, recolor=True
        )
        ax.set_ylim(0, 1)
        # ax.set_xlabel(layout)
        if not show_ylabel:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Correlation")
        if not show_xlabel:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Pairing")

        cc.set_size(ax, 3, 1.5)
        ax.get_figure().savefig(f"{p_fo}/exp_rij_barplot_{layout}.pdf", dpi=300)

    return ax


# Fig 3
def sim_raster_plots(
    path,
    time_range=None,
    main_width=3.5,  # rough size of the main column, in cm, >1cm, zoom in is ~ 1cm wide
    zoom_time=None,
    zoom_duration=0.25,
    bs_large=20 / 1000,  # width of the gaussian kernel for rate
    threshold_factor=2.5 / 100,  # fraction of max peak height for burst
    mark_zoomin_location=False,
    **kwargs,
):

    total_width = main_width + 0.7 + 0.8  # 7mm for labels on the left, 8mm for zoom
    fig = plt.figure(figsize=[(total_width) / 2.54, 3.5 / 2.54])
    # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize)
    axes = []
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[main_width - 1.5, 0.8],
        wspace=0.05,
        hspace=0.1,
        left=0.7 / total_width,
        right=0.99,
        top=0.95,
        bottom=0.15,
    )
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[1, 0], sharex=axes[0]))
    axes.append(fig.add_subplot(gs[2, 0], sharex=axes[0]))
    axes.append(fig.add_subplot(gs[:, 1]))

    h5f = ah.prepare_file(path)

    # ------------------------------------------------------------------------------ #
    # rates
    # ------------------------------------------------------------------------------ #

    ax = axes[0]
    ah.find_rates(h5f, bs_large=bs_large)
    threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
    ah.find_system_bursts_from_global_rate(
        h5f, rate_threshold=threshold, merge_threshold=0.1
    )

    ph.plot_system_rate(
        h5f,
        ax,
        mark_burst_threshold=False,
        color="#333",
        apply_formatting=False,
        clip_on=True,
        lw=0.5,
    )
    ax.set_ylim(0, 80)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(20))
    sns.despine(ax=ax, left=False, bottom=True, trim=False, offset=2)
    ax.xaxis.set_visible(False)

    # ------------------------------------------------------------------------------ #
    # raster
    # ------------------------------------------------------------------------------ #
    kwargs = kwargs.copy()
    ax = axes[1]
    ax.set_rasterization_zorder(0)
    ph.plot_raster(h5f, ax, clip_on=True, zorder=-2, markersize=1.0, alpha=0.75, **kwargs)
    ax.set_ylim(-1, None)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    sns.despine(ax=ax, left=True, right=True, bottom=True, top=True)

    # ------------------------------------------------------------------------------ #
    # adaptation
    # ------------------------------------------------------------------------------ #

    ax = axes[2]
    ph.plot_state_variable(
        h5f, ax, variable="D", lw=0.5, apply_formatting=False, **kwargs
    )
    ax.margins(x=0, y=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
    if time_range is None:
        time_range = [0, 180]
    ax.set_xlim(*time_range)

    sns.despine(ax=ax, left=False, bottom=False, trim=True, offset=2)

    # ------------------------------------------------------------------------------ #
    # zoomin
    # ------------------------------------------------------------------------------ #

    ax = axes[-1]
    ax.set_rasterization_zorder(0)
    ph.plot_raster(h5f, ax, clip_on=True, zorder=-2, markersize=1.5, alpha=0.9, **kwargs)
    ylim = axes[1].get_ylim()
    ax.set_ylim(ylim[0] - 10, ylim[1] + 10)
    ax.set_xlim(zoom_time, zoom_time + zoom_duration)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    sns.despine(ax=ax, left=True, right=True, bottom=True, top=True)

    rate_as_background_in_zoomin = False
    if rate_as_background_in_zoomin:
        # replot the rate as a background to the raster, rescaling
        ylim_raster = axes[1].get_ylim()
        ylim_rate = axes[0].get_ylim()

        rate = h5f["ana.rates.system_level"].copy()
        # rescale to go from 0 to 1
        rate = (rate - ylim_rate[0]) / (ylim_rate[1] - ylim_rate[0])
        # rescale to match yrange of raster
        rate = rate * (ylim_raster[1] - ylim_raster[0]) + ylim_raster[0]
        h5f["ana.rates.system_level"] = rate
        ph.plot_system_rate(
            h5f,
            ax,
            mark_burst_threshold=False,
            color=cc.alpha_to_solid_on_bg("#333", 0.5),
            apply_formatting=False,
            clip_on=True,
            lw=0.5,
            zorder=-3,
        )

    if mark_zoomin_location:
        ph._plot_bursts_into_timeseries(
            ax=axes[2],
            beg_times=[zoom_time],
            end_times=[zoom_time + zoom_duration],
            style="markers",
            y_offset=0.05,
            markersize=1.5,
        )

    bnb.hi5.close_hot()

    return fig


def sim_vs_exp_violins(**kwargs):
    dfs = dict()
    dfs["exp"] = load_pd_hdf5(f"{p_exp}/processed/1b.hdf5")
    dfs["sim"] = load_pd_hdf5("./dat/sim_out/k=5.hdf5")

    for key in dfs["sim"].keys():
        dfs["sim"][key] = dfs["sim"][key].query(
            "`Condition` == '80 Hz' | `Condition` == '90 Hz'"
        )

    # burst size
    df = pd.concat(
        [
            dfs["exp"]["bursts"].query("Condition == 'pre' | Condition == 'stim'"),
            dfs["sim"]["bursts"],
        ]
    )
    ax = custom_violins(
        df,
        category="Condition",
        observable="Fraction",
        ylim=[0, 1],
        num_swarm_points=250,
        bw=0.2,
    )
    ax.set_ylim(-0.05, 1.05)
    sns.despine(ax=ax, bottom=True, left=False, trim=True)
    ax.tick_params(bottom=False)
    ax.set_xlabel(f"")
    cc.set_size(ax, 3, 3.5)
    ax.get_figure().savefig(f"{p_fo}/sim_vs_exp_violins_fraction.pdf", dpi=300)

    # correlation coefficients
    df = pd.concat(
        [
            dfs["exp"]["rij"].query("Condition == 'pre' | Condition == 'stim'"),
            dfs["sim"]["rij"],
        ]
    )
    ax = custom_violins(
        df,
        category="Condition",
        observable="Correlation Coefficient",
        ylim=[0, 1],
        num_swarm_points=250,
        bw=0.2,
    )
    ax.set_ylim(-0.05, 1.05)
    sns.despine(ax=ax, bottom=True, left=False, trim=True)
    ax.tick_params(bottom=False)
    ax.set_xlabel(f"")
    cc.set_size(ax, 3, 3.5)
    ax.get_figure().savefig(f"{p_fo}/sim_vs_exp_violins_rij.pdf", dpi=300)

    # ibis
    df = pd.concat(
        [
            dfs["exp"]["bursts"].query("Condition == 'pre' | Condition == 'stim'"),
            dfs["sim"]["bursts"],
        ]
    )
    ax = custom_violins(
        df,
        category="Condition",
        observable="Inter-burst-interval",
        ylim=[0, 90],
        num_swarm_points=250,
        bw=0.2,
    )
    ax.set_ylim(0, 90)
    sns.despine(ax=ax, bottom=True, left=False, trim=True)
    ax.tick_params(bottom=False)
    ax.set_xlabel(f"")
    cc.set_size(ax, 3, 3.5)
    ax.get_figure().savefig(f"{p_fo}/sim_vs_exp_violins_ibi.pdf", dpi=300)


def sim_vs_exp_ibi(
    input_path=None, ax=None, simulation_coordinates=reference_coordinates, **kwargs
):
    """
    Plot the inter-burst-interval of the k=5  simulation to compare with the experimental data.
    """
    kwargs = kwargs.copy()

    if input_path is None:
        input_path = f"{p_sim}/lif/processed/ndim.hdf5"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    data = nh.load_ndim_h5f(input_path)
    # obs = "sys_median_any_ibis"
    obs = "sys_mean_any_ibis"
    data[obs] = data[obs].sel(simulation_coordinates)

    num_reps = len(data[obs].coords["repetition"])
    log.info(f"{num_reps} repetitions")
    dat_med = data[obs].mean(dim="repetition")
    dat_sem = data[obs].std(dim="repetition") / np.sqrt(num_reps)

    x = dat_med.coords["rate"]
    y = dat_med
    yerr = dat_sem

    selects = np.where(np.isfinite(y))

    kwargs.setdefault("color", "#333")
    kwargs.setdefault("label", "simulation")

    ax.errorbar(
        x=x[selects],
        y=y[selects],
        yerr=yerr[selects],
        fmt="o",
        markersize=1.5,
        elinewidth=0.5,
        capsize=1.5,
        zorder=2,
        **kwargs,
    )

    kwargs.pop("label")
    kwargs["color"] = cc.alpha_to_solid_on_bg(kwargs["color"], 0.3)

    ax.plot(x[selects], y[selects], zorder=1, **kwargs)

    ax.plot(
        [0, 80, 80],
        [25, 25, 0],
        ls=":",
        color=cc.alpha_to_solid_on_bg(colors["pre"], 0.3),
        zorder=0,
    )
    ax.plot(
        [0, 90, 90],
        [6, 6, 0],
        ls=":",
        color=cc.alpha_to_solid_on_bg(colors["stim"], 0.3),
        zorder=0,
    )
    # ax.axhline(y=10, xmin=0, xmax=1, ls = ":", color="gray", zorder=0)
    # ax.axhline(y=40, xmin=0, xmax=1, ls = ":", color="gray", zorder=0)
    # ax.set_yscale("log")

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    ax.set_xlim(65, 100)
    ax.set_ylim(0, 70)

    if show_xlabel:
        ax.set_xlabel(r"Synaptic noise rate (Hz)")
    if show_ylabel:
        ax.set_ylabel("Inter-event-interval\n(seconds)")
    if show_title:
        ax.set_title(r"Simulation IEI")
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        cc._legend_into_new_axes(ax)
    # if use_compact_size:
    cc.set_size(ax, 3.5, 2.5)

    ax.get_figure().savefig(f"{p_fo}/ibi_sim_vs_exp.pdf", dpi=300)

    return ax


def sim_obs_vs_noise_for_all_k(
    path,
    observable,
    simulation_coordinates=reference_coordinates,
    ax=None,
    colors=None,
    **kwargs,
):
    """
    Plot selected observable as function of the noise (`rate`)
    Select the k to plot via `k_inter` in `simulation_coordinates`

    # Parameters:
    colors : dict with k_inter values as key that match the ones in simulation_coords
    """
    ndim = nh.load_ndim_h5f(path)
    ndim = ndim[observable]
    for coord in simulation_coordinates.keys():
        try:
            ndim = ndim.sel({coord: simulation_coordinates[coord]})
        except:
            log.debug(f"Could not select {coord}")

    log.debug(ndim.coords)
    x = ndim.coords["rate"].to_numpy()
    num_reps = len(ndim.coords["repetition"])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = dict()
        for kdx, k in enumerate(ndim.coords["k_inter"].to_numpy()):
            colors[k] = f"C{kdx}"

    # burst detection for modular systems starts failing for noise >~ 100 hz
    # when the systems enters an "up state"
    # we may want to avoid plotting some observables in this range
    burst_observables = [
        "sys_mean_participating_fraction",
        "any_num_spikes_in_bursts",
        "sys_median_any_ibis",
        "sys_mean_any_ibis",
        "sys_mean_core_delay",
    ]

    for kdx, k in enumerate(ndim.coords["k_inter"].to_numpy()):
        y = ndim.sel(k_inter=k).mean(dim="repetition")
        yerr = ndim.sel(k_inter=k).std(dim="repetition") / np.sqrt(num_reps)

        noise_range_to_show = (
            [-np.inf, 100.0]
            if (observable in burst_observables) and k != -1
            else [-np.inf, np.inf]
        )

        # 92.5 Hz was only simulated for k=10, so drop it
        selects = np.where(
            (x != 92.5) & (x >= noise_range_to_show[0]) & (x <= noise_range_to_show[1])
        )

        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault("color", colors[k])
        plot_kwargs.setdefault("label", f"k={k}" if k != -1 else "mrgd.")
        plot_kwargs.setdefault("fmt", "o")
        plot_kwargs.setdefault("markersize", 1.5)
        plot_kwargs.setdefault("elinewidth", 0.5)
        plot_kwargs.setdefault("capsize", 1.5)
        plot_kwargs.setdefault("zorder", kdx)

        if k == -1 and observable == "sys_median_any_ibis":
            plot_kwargs["zorder"] = -1

        ax.errorbar(
            x=x[selects],
            y=y[selects],
            yerr=yerr[selects],
            **plot_kwargs,
        )

        plot_kwargs.setdefault("lw", 0.5)
        plot_kwargs.pop("label")
        plot_kwargs.pop("fmt")
        plot_kwargs.pop("markersize")
        plot_kwargs.pop("elinewidth")
        plot_kwargs.pop("capsize")
        plot_kwargs["zorder"] -= 1
        plot_kwargs["color"] = cc.alpha_to_solid_on_bg(plot_kwargs["color"], 1)

        ax.plot(x[selects], y[selects], **plot_kwargs)

        if show_ylabel:
            ax.set_ylabel(observable)
        if show_xlabel:
            ax.set_xlabel("Synaptic noise rate (Hz)")

    return ax


def sim_resource_dist_vs_noise_for_all_k(
    path,
    simulation_coordinates=reference_coordinates,
    ax=None,
    colors=None,
    **kwargs,
):
    """
    similar to above, but now we want some fancier plotting than just a line.
    we have a low and high perecentiles, and peak position for the distribution
    fo resources.

    # Parameters
    par1 : type, description
    """
    ndims = nh.load_ndim_h5f(path)
    for obs in ndims.keys():
        for coord in simulation_coordinates.keys():
            try:
                ndim[obs] = ndim[obs].sel({coord: simulation_coordinates[coord]})
            except:
                log.debug(f"Could not select {coord}")

    ndim = ndims["sys_orderpar_dist_max"]

    log.debug(ndim.coords)
    x = ndim.coords["rate"].to_numpy()
    num_reps = len(ndim.coords["repetition"])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = dict()
        for kdx, k in enumerate(ndim.coords["k_inter"].to_numpy()):
            colors[k] = f"C{kdx}"

    for kdx, k in enumerate(ndim.coords["k_inter"].to_numpy()):
        if k in [4, 6]:
            continue
        fig, ax2 = plt.subplots()
        y = np.squeeze(ndim.sel(k_inter=k).mean(dim="repetition"))
        yh = np.squeeze(
            ndims["sys_orderpar_dist_high_end"].sel(k_inter=k).mean(dim="repetition")
        )
        yl = np.squeeze(
            ndims["sys_orderpar_dist_low_end"].sel(k_inter=k).mean(dim="repetition")
        )

        noise_range_to_show = [-np.inf, 100.0]
        # 92.5 Hz was only simulated for k=10, so drop it
        selects = np.where(
            (x != 92.5) & (x >= noise_range_to_show[0]) & (x <= noise_range_to_show[1])
        )

        ax2.plot(x[selects], y[selects], lw=0.5, color=colors[k])
        ax2.fill_between(
            x[selects], yl[selects], yh[selects], color=colors[k], alpha=0.3, lw=0
        )

        ax.plot(x[selects], y[selects], lw=0.5, color=colors[k], label=k)
        ax.plot(x[selects], yl[selects], ls=":", lw=0.5, color=colors[k])
        ax.plot(x[selects], yh[selects], ls=":", lw=0.5, color=colors[k])
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        ax2.set_title(k)
        ax2.set_xlabel("Synaptic noise rate (Hz)")
        ax2.set_ylabel("Distribution")
        ax.set_xlabel("Synaptic noise rate (Hz)")
        ax.set_ylabel("Distribution")
        ax.legend()
        cc.set_size(ax2, 4, 3)

    return ax


def sim_resource_density_vs_noise_for_all_k(
    path,
    simulation_coordinates=reference_coordinates,
    ax=None,
    colors=None,
    **kwargs,
):
    ndims = nh.load_ndim_h5f(path)
    for obs in ndims.keys():
        for coord in simulation_coordinates.keys():
            try:
                ndim[obs] = ndim[obs].sel({coord: simulation_coordinates[coord]})
            except:
                log.debug(f"Could not select {coord}")

    ndim = np.squeeze(ndims["vec_sys_hvals_resource_dist"])
    ndim_edges = np.squeeze(ndims["vec_sys_hbins_resource_dist"])[0, 0, 0, :]

    ndim = ndim.sel(k_inter=10)
    # ndim = ndim.sel(k_inter=5)
    # ndim_edges.sel(k_inter=5)

    log.debug(ndim.coords)
    x = ndim.coords["rate"].to_numpy()
    num_reps = len(ndim.coords["repetition"])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ndim /= ndim.sum(dim="vector_observable")

    noise_range_to_show = [-np.inf, 120.0]
    selects = np.where(
        (x != 92.5) & (x >= noise_range_to_show[0]) & (x <= noise_range_to_show[1])
    )
    ndim = ndim[selects].mean(dim="repetition")
    # ndim /= ndim.max(dim="vector_observable")

    # ndim = ndim.sel(rate=80)
    # for rdx, rep in enumerate(ndim.coords["repetition"]):
    #     ax.plot(ndim_edges[0:-1], ndim[rdx, :])

    centroids = ndim_edges[0:-1] + (ndim_edges[1] - ndim_edges[0]) / 2
    print(centroids)

    ndim = ndim.assign_coords(vector_observable=centroids.to_numpy())

    # ax.plot(ndim_edges[0:-1], ndim.mean(dim="repetition"), color="black")
    # ndim.groupby_bins("vector_observable", np.arange(0, 110, 5)).sum().plot.contourf(
    #     x="rate", cmap="Blues", vmin=0, vmax=0.4
    # )

    ndim.plot.imshow(
        x="rate", cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=0.001, vmax=0.2)
    )

    return ndim


def controls_sim_vs_exp_ibi():
    fig, ax = plt.subplots()

    p1 = "./dat/the_last_one/ndim_jM=15_tD=8_t2.5_k20.hdf5"
    p2 = "./dat/the_last_one/ndim_jM=15_tD=8_t2.5_k20_remove_null.hdf5"

    sim_vs_exp_ibi(p1, ax=ax, color="#333", label="all")
    sim_vs_exp_ibi(p2, ax=ax, color="red", label="0 removed")

    return ax


def sim_participating_fraction(
    input_path,
    ax=None,
    x_cs="rate",
    simulation_coordinates=reference_coordinates,
    **kwargs,
):
    """
    # Example
    ```
    cs = pp.reference_coordinates.copy()
    cs['rate'] = 80
    pp.sim_participating_fraction(
        "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/the_last_one/stim02_rij500.hdf5",
        x_cs="stim_rate",
        simulation_coordinates=cs
    )
    ```
    """

    kwargs = kwargs.copy()

    data = nh.load_ndim_h5f(input_path)
    obs = "sys_mean_participating_fraction"
    data[obs] = data[obs].sel(simulation_coordinates)

    log.debug(data[obs])

    num_reps = len(data[obs].coords["repetition"])
    dat_med = data[obs].mean(dim="repetition")
    dat_sem = data[obs].std(dim="repetition") / np.sqrt(num_reps)

    x = dat_med.coords[x_cs]
    y = dat_med
    yerr = dat_sem

    if x_cs == "rate":
        selects = np.where(x != 92.5)[0]
    else:
        selects = np.where(np.isfinite(y))[0]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kwargs.setdefault("color", "#333")

    ax.errorbar(
        x=x[selects],
        y=y[selects],
        yerr=yerr[selects],
        fmt="o",
        markersize=1.5,
        elinewidth=0.5,
        capsize=1.5,
        zorder=2,
        label=f"simulation",
        **kwargs,
    )

    kwargs["color"] = cc.alpha_to_solid_on_bg(kwargs["color"], 0.3)

    ax.plot(x[selects], y[selects], zorder=1, label=f"simulation", **kwargs)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

    ax.set_ylim(0, 1)

    if show_xlabel:
        if x_cs == "rate":
            ax.set_xlim(65, 110)
            ax.set_xlabel(r"Noise rate (Hz)")
        elif x_cs == "stim_rate":
            ax.set_xlim(0, 25)
            ax.set_xlabel(r"Stimulation rate (Hz)")
    if show_ylabel:
        ax.set_ylabel("Mean burst size")
    # if show_title:
    #     ax.set_title(r"")
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        cc._legend_into_new_axes(ax)
    cc.set_size(ax, 3.5, 2)

    ax.get_figure().savefig(f"{p_fo}/participating_fraction.pdf", dpi=300)

    return ax


def sim_modules_participating_in_bursts(
    input_path,
    simulation_coordinates,
    xlim_for_fill=[65, 110],
    xlim_for_points=[65, 110],
    dim1="rate",
    drop_zero_len=True,
    apply_formatting=True,
):

    if isinstance(input_path, str):
        data = nh.load_ndim_h5f(input_path)
    elif isinstance(input_path, xr.Dataset):
        data = input_path
    else:
        raise ValueError("Provide a file path or loaded xr dataArray")

    # memo from past paul: should have just used xr datasets...
    if not isinstance(data, xr.Dataset):
        for obs in data.keys():
            data[obs] = data[obs].sel(simulation_coordinates)
    else:
        data = data.sel(simulation_coordinates)

    x = data["any_num_b"].coords[dim1]

    fig, ax = plt.subplots()

    prev = np.zeros_like(x, dtype=float)
    for seq_len in [4, 3, 2, 1, 0]:

        ref = data["any_num_b"].copy()
        if drop_zero_len:
            ref -= data["mod_num_b_0"]
            if seq_len == 0:
                continue
        dat = data[f"mod_num_b_{seq_len}"]

        num_reps = len(data["any_num_b"]["repetition"])

        ratio = dat / ref
        ratio_mean = ratio.mean(dim="repetition")
        ratio_errs = ratio.std(dim="repetition") / np.sqrt(num_reps)

        ratio_mean = ratio_mean.to_numpy().reshape(-1)
        ratio_errs = ratio_errs.to_numpy().reshape(-1)

        # buildup the graph area by area, using nxt and prev
        nxt = np.nan_to_num(ratio_mean, nan=0.0)

        clr = cc.cmap_cycle("cold", edge=False, N=5)[int(seq_len)]

        # for 92.5 Hz we only sampled k = 10, hence drop the point
        # this should really go elsewhere, but, you know. stuff grows over time.
        try:
            selects = np.where(
                (x >= xlim_for_fill[0]) & (x <= xlim_for_fill[1]) & (x != 92.5)
            )
        except:
            selects = ...
        # selects = np.ones_like(x, dtype=bool)
        ax.fill_between(
            x[selects],
            prev[selects],
            prev[selects] + nxt[selects],
            linewidth=0,
            color=cc.alpha_to_solid_on_bg(clr, 0.2),
            clip_on=True,
        )

        # for the error bars, we might want different xlims.
        try:
            selects = np.where(
                (x >= xlim_for_points[0]) & (x <= xlim_for_points[1]) & (x != 92.5)
            )
        except:
            selects = ...
        if seq_len != 0 and seq_len != 1:
            ax.errorbar(
                x=x[selects],
                y=prev[selects] + nxt[selects],
                yerr=ratio_errs[selects],
                fmt="o",
                markersize=1.5,
                # mfc=cc.alpha_to_solid_on_bg(clr, 0.2),
                elinewidth=0.5,
                capsize=1.5,
                label=f"{seq_len} module" if seq_len == 1 else f"{seq_len} modules",
                color=clr,
                clip_on=True,
            )

        # we coult try to place text in the areas
        # if seq_len == 4:
        #     ycs = 6
        #     xcs = 1
        #     ax.text(
        #         x[selects][xcs],
        #         prev[selects][ycs] + (nxt[selects][ycs]) / 2,
        #         f"{seq_len} module" if seq_len == 1 else f"{seq_len} modules",
        #         color=clr,
        #         va="center",
        #     )

        prev += nxt

    if apply_formatting:
        fig.tight_layout()

        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.set_xlim(xlim_for_fill)
        ax.set_ylim(0, 1)

        # ax.spines["left"].set_position(("outward", 5))
        # ax.spines["bottom"].set_position(("outward", 5))

        ax.set_xlabel(r"Synaptic noise rate (Hz)")
        ax.set_ylabel("Fraction of events\nspanning")

    return ax


def sim_violins_for_all_k(ax_width=4):
    def apply_formatting(ax, ylim=True, trim=True):
        if ylim:
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        sns.despine(ax=ax, bottom=True, left=False, trim=trim, offset=5)
        ax.tick_params(bottom=False)
        ax.set_xlabel(f"")
        ax.set_ylabel(f"")
        # ax.set_xticks([])
        cc.set_size(ax, ax_width, 3.0)

    def render_plots():

        # burst size
        log.debug("burst size")
        ax = custom_violins(
            df["bursts"],
            category="Condition",
            observable="Fraction",
            palette=colors[k],
            ylim=[0, 1],
            num_swarm_points=500,
            bw=0.2,
            scale="area",
        )
        apply_formatting(ax)
        ax.set_ylabel("Burst size")
        ax.set_xlabel(f"{k}")
        ax.get_figure().savefig(f"{p_fo}/sim_violins_fraction_{k}.pdf", dpi=300)

        # correlation coefficients
        log.debug("rij")
        ax = custom_violins(
            df["rij"],
            category="Condition",
            observable="Correlation Coefficient",
            palette=colors[k],
            ylim=[0, 1],
            num_swarm_points=1000,
            bw=0.2,
            scale="area",
        )
        apply_formatting(ax)
        ax.set_ylabel("Correlation Coefficient")
        ax.set_xlabel(f"{k}")
        ax.get_figure().savefig(f"{p_fo}/sim_violins_rij_{k}.pdf", dpi=300)

        # depletion
        try:
            log.debug("depletion")
            ax = custom_violins(
                df["drij"],
                category="Condition",
                observable="Depletion rij",
                palette=colors[k],
                ylim=[-0.5, 1],
                num_swarm_points=1000,
                bw=0.2,
                scale="area",
            )
            apply_formatting(ax, ylim=False)
            ax.set_ylabel("Depletion rij")
            ax.set_xlabel(f"{k}")
            ax.get_figure().savefig(
                f"{p_fo}/sim_violins_depletion_rij_{k}.pdf", dpi=300
            )
        except:
            log.debug("depletion skipped")

        # ibi
        log.debug("ibi")
        ax = custom_violins(
            df["bursts"],
            category="Condition",
            observable="Inter-burst-interval",
            palette=colors[k],
            ylim=[0, 70],
            num_swarm_points=250,
            bw=0.2,
            scale="area",
        )
        apply_formatting(ax, ylim=False, trim=False)
        ax.set_ylim([0, 70])
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
        ax.set_ylabel("Inter-burst-interval")
        ax.set_xlabel(f"{k}")
        ax.get_figure().savefig(f"{p_fo}/sim_violins_ibi_{k}.pdf", dpi=300)

    for k in ["k=1", "k=5", "k=10"]:
        log.info("")
        log.info(k)
        log.info("Loading data")
        df = load_pd_hdf5(f"./dat/sim_out/{k}.hdf5", keys=["bursts", "rij", "drij"])
        render_plots()


def sim_prob_dist_rates_and_resources(k=5):
    if k == 5:
        h5f1 = ah.prepare_file(
            "./dat/the_last_one/dyn/stim=off_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_rep=001.hdf5"
        )
        h5f2 = ah.prepare_file(
            "./dat/the_last_one/dyn/stim=off_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=90.0_rep=001.hdf5"
        )
    elif k == -1:
        h5f1 = ah.prepare_file(
            "./dat/the_last_one/dyn/stim=off_k=-1_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=85.0_rep=001.hdf5"
        )
        h5f2 = ah.prepare_file(
            "./dat/the_last_one/dyn/stim=off_k=-1_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=100.0_rep=001.hdf5"
        )
    else:
        raise NotImplementedError

    bs_large = 20 / 1000  # width of the gaussian kernel for rate
    threshold_factor = 2.5 / 100  # fraction of max peak height for burst

    # ph.overview_dynamic(h5f1)
    # ph.overview_dynamic(h5f2)

    ah.find_rates(h5f1, bs_large=bs_large)
    ah.find_rates(h5f2, bs_large=bs_large)

    # bins = np.logspace(start=-1, stop=2, base=10)
    bins = np.arange(0, 101) - 0.5
    # ax = ph._plot_distribution_from_series(
    #     h5f1["ana.rates.system_level"], bins=bins, alpha=0.3, color="C0", label="80Hz"
    # )
    # ax = ph._plot_distribution_from_series(
    #     h5f2["ana.rates.system_level"],
    #     bins=bins,
    #     alpha=0.3,
    #     color="C1",
    #     label="90Hz",
    #     ax=ax,
    # )
    # ax.set_xlim(0, 80)
    # ax.set_yscale("log")
    # ax.legend(loc="upper right")
    # ax.set_xlabel("Populaton Rate (Hz)")
    # ax.set_ylabel("Probability")
    # cc.set_size(ax, 3.0, 3.0)
    # cc._fix_log_ticks(ax.yaxis, every=1, hide_label_condition=lambda idx: idx % 2 == 1)

    # adapt1 = np.nanmean(h5f1["data.state_vars_D"], axis=0)
    # adapt2 = np.nanmean(h5f2["data.state_vars_D"], axis=0)

    # random neurons
    # selects = np.sort(np.random.choice(h5f1["ana.neuron_ids"], size=20, replace=False))
    # adapt1 = h5f1["data.state_vars_D"][selects, :].flatten()
    # adapt2 = h5f2["data.state_vars_D"][selects, :].flatten()
    # ax = ph._plot_distribution_from_series(
    #     adapt1, alpha=0.3, color="C0", binwidth=0.02, label="80Hz"
    # )
    # ax = ph._plot_distribution_from_series(
    #     adapt2, alpha=0.3, color="C1", binwidth=0.02, label="90Hz", ax=ax
    # )

    # we want to use the adaptation as we have plotted it the rasters -> per module
    ax = None
    for hdx, h5f in enumerate([h5f1, h5f2]):
        mod_adapts = []
        for mod_id in np.unique(h5f["data.neuron_module_id"]):
            n_ids = np.where(h5f["data.neuron_module_id"][:] == mod_id)[0]
            mod_adapts.append(np.mean(h5f["data.state_vars_D"][n_ids, :], axis=0))
        mod_adapts = np.hstack(mod_adapts)

        if hdx == 0:
            ax = ph._plot_distribution_from_series(
                mod_adapts, alpha=0.3, color="C0", binwidth=0.01, label="low noise", ax=ax
            )
            print(ah.find_resource_order_parameters(h5f, which="dist_percentiles"))
        else:
            ax = ph._plot_distribution_from_series(
                mod_adapts,
                alpha=0.3,
                color="C1",
                binwidth=0.01,
                label="high noise",
                ax=ax,
            )
            print(ah.find_resource_order_parameters(h5f, which="dist_percentiles"))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Synaptic Resources")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left")
    cc.set_size(ax, 3.0, 3.0)

    return mod_adapts


def sim_layout_sketch(in_path, out_path, ax_width=1.5):

    h5f = ph.ah.prepare_file(in_path)
    ax = ph.plot_axon_layout(
        h5f,
        axon_kwargs=dict(color="gray", alpha=1, lw=0.1),
        soma_kwargs=dict(ec="black", lw=0.1),
    )
    k = re.search("_k=(-*\d+)", in_path, re.IGNORECASE).group(1)
    if k == "-1":
        cc.set_size(ax, ax_width * 2 / 3, ax_width * 2 / 3)
    else:
        cc.set_size(ax, ax_width, ax_width)
    ax.set_xlabel("")
    ax.set_xticks([])
    sns.despine(ax=ax, bottom=True, left=True)
    # ax.tick_params(axis="both", which="both", bottom=False)
    ax.get_figure().savefig(f"{out_path}", dpi=600, transparent=True)


def sim_resource_cycles(apply_formatting=True, k_list=None):
    """
    wrapper that creates the resource plots, fig. 4 h.

    this is horribly slow: we import and analyse the files (rate and burst detection)

    returns a nested benedict containing the matplotlib axes elements
    """
    axes = benedict()

    if k_list is None:
        k_list = [-1, 1, 5, 10]

    for k in k_list:
        axes[str(k)] = benedict()
        for rate in ["80", "90"]:
            log.info(f"Resource cycle for k={k} at {rate} Hz")
            # we additonally sampled a few simulations at higher time resolution. this gives
            # higher precision for the resource variable, but takes tons of disk space.
            path = f"{p_sim}/lif/raw/highres_stim=off_k={k}_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate={rate}.0_rep=001.hdf5"
            try:
                h5f = ah.prepare_file(path)
            except:
                log.debug(f"highres data file for resource cycle not found: {path}")
                path = path.replace("highres_", "")
                h5f = ah.prepare_file(path)

            # we keep hardcoding these values. TODO: set as global variables or sth
            bs_large = 20 / 1000  # width of the gaussian kernel for rate
            threshold_factor = 2.5 / 100  # fraction of max peak height for burst
            ah.find_rates(h5f, bs_large=bs_large)
            threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
            ah.find_system_bursts_from_global_rate(
                h5f, rate_threshold=threshold, merge_threshold=0.1
            )

            ax = ph.plot_resources_vs_activity(
                h5f,
                max_traces_per_mod=80 if k == -1 else 20,
                apply_formatting=False,
                lw=0.3,
                alpha=0.6,
                clip_on=False,
            )

            if apply_formatting:
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0, 150)

                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))

                cc.set_size(ax, 1.6, 1.0)
                sns.despine(
                    ax=ax,
                    trim=False,
                    offset=2,
                    right=True,
                    top=True,
                    bottom=True if rate == "80" else False,
                )
                if rate == "80":
                    cc.detick(ax.xaxis)

                if k != 1:
                    cc.detick(ax.yaxis, keep_ticks=True)

            axes[str(k)][rate] = ax

    return axes


def sim_out_degrees(
    path,
    simulation_coordinates=reference_coordinates,
    ax=None,
    colors=None,
    **kwargs,
):
    """Loads the ndim merged file and plot the out-degree distribution, depending on the type of neuron (bridging or not)"""
    ndims = nh.load_ndim_h5f(path)
    # rates do not matter, we just need reps
    if "rate" not in simulation_coordinates:
        simulation_coordinates["rate"] = 75.0

    for obs in ndims.keys():
        for coord in simulation_coordinates.keys():
            try:
                ndims[obs] = ndims[obs].sel({coord: simulation_coordinates[coord]})
            except:
                log.debug(f"Could not select {coord}")

    bin_edges = ndims["vec_sys_hbins_kout_no_bridge"][0].to_numpy()
    x = bin_edges[0:-1] + 0.5

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    def _plot(key, **kwargs):
        # reconstruct series of observations from histogram
        hist = ndims[key].sum(dim="repetition").to_numpy()
        series = []
        for idx in range(0, len(hist)):
            y = hist[idx]
            series.extend([x[idx]] * int(y))

        # print(series)

        sns.histplot(
            series,
            **kwargs,
        )

    bins = np.arange(-0.5, 160.5, 4)

    kwargs = dict(
        kde=False,
        stat="probability",
        element="step",
        alpha=0.25,
        bins=bins,
    )
    _plot(
        key="vec_sys_hvals_kout_no_bridge",
        label="no bridge",
        color="C0",
        **kwargs,
    )
    _plot(
        key="vec_sys_hvals_kout_yes_bridge",
        color="C1",
        label="yes bridge",
        **kwargs,
    )
    ax.legend()

    return ndims


def sim_degrees_sampled(k_inter=5, num_reps=50):
    """
    Sample realizations of the topology and plot the in-degree distribution
    """

    sys.path.append(os.path.dirname(__file__) + "/../src")
    from topology import (
        ModularTopology,
        MergedTopology,
        _get_in_degrees_by_internal_external,
    )

    res = dict(
        k_out=[],
        k_in_total=[],
        k_in_internal=[],
        k_in_external=[],
    )

    for rep in tqdm(range(0, num_reps), leave=False):
        if k_inter == -1:
            topo = MergedTopology()
        else:
            topo = ModularTopology(par_k_inter=k_inter)
        res["k_out"].extend(topo.k_out)
        res["k_in_total"].extend(topo.k_in)

        try:
            k_int, k_ext = _get_in_degrees_by_internal_external(
                topo.aij_nested, topo.neuron_module_ids
            )
            res["k_in_internal"].extend(k_int)
            res["k_in_external"].extend(k_ext)
        except:
            # merged topo
            pass

    fig, ax = plt.subplots()

    def _plot(series, **kwargs):

        sns.histplot(
            series,
            **kwargs,
        )

    bins = np.arange(-0.5, 160.5, 1)

    kwargs = dict(
        kde=False,
        stat="probability",
        element="step",
        alpha=0.25,
        bins=bins,
    )
    _plot(
        series=res["k_in_total"],
        color="C0",
        label="total",
        **kwargs,
    )
    _plot(
        series=res["k_in_external"],
        label="external",
        color="C3",
        linestyle=":",
        **kwargs,
    )
    _plot(
        series=res["k_in_internal"],
        color="C1",
        label="internal",
        linestyle="--",
        **kwargs,
    )

    ax.set_ylim(0, 0.25)
    ax.set_title("merged" if k_inter == -1 else f"k={k_inter}")
    ax.set_xlabel("Number of incoming connections")
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    if show_legend:
        leg = ax.legend(fontsize=6)
        cc.apply_default_legend_style(leg)

    if not show_xlabel:
        ax.set_xlabel("")
    if not show_ylabel:
        ax.set_ylabel("")
    if not show_title:
        ax.set_title("")

    if k_inter == -1:
        ax.set_xlim(0, 80)
        cc.set_size(ax, w=3.0*8/5, h=2.5)
    else:
        ax.set_xlim(0, 50)
        cc.set_size(ax, w=3.0, h=2.5)

    return res


# ------------------------------------------------------------------------------ #
# Mesoscopic model
# ------------------------------------------------------------------------------ #



def meso_explore_multiple(
    rep_path=f"{p_sim}/meso/raw_no_gates", out_path="{p_fo}/meso_gates_off"
):
    """
    Run meso_launcher before and save to a custom folder.
    Simply calls figure 5 with changed input / output paths

    Example
    ```
    # to produce no gates sm figure:
    pp.meso_explore_multiple(
        rep_path=f"{p_sim}/meso/raw_no_gates", out_path="{p_fo}/meso_gates_off"
    )
    ```

    """

    dset_path = rep_path.replace("meso_in", "meso_out/analysed")
    dset_path += ".hdf5"
    try:
        dset = xr.load_dataset(dset_path)
    except:
        dset = mh.process_data_from_folder(rep_path)
        mh.write_xr_dset_to_hdf5(dset, output_path=dset_path)

    zoom_times = benedict(keypath_separator="/")
    zoom_times[f"0.1/0.025"] = 938
    zoom_times[f"0.025/0.025"] = 865

    fig_5(
        dset=dset,
        rep_path=rep_path,
        out_path=out_path,
        skip_snapshots=False,
        skip_cycles=False,
        skip_observables=False,
        zoom_times=zoom_times,
    )


def meso_obs_for_all_couplings(dset, obs, base_clr="#333", **kwargs):
    """
    Wrapper tjat reproduces the plots of the microscopic plots ~ fig 4:
    Correlations and event size with error bars across realizations.
    Uses `meso_xr_with_errors`

    # Parameters
    dset : xarray dataset
        from `mh.process_data_from_folder` (or loaded from disk)
    obs : str,
        one of the observables in `dset.data_vars`

    # Example
    ```
    import paper_plots as pp
    import xarray as xr

    dset = xr.load_dataset(f"{p_sim}/meso/processed/analysed.hdf5"
    ax = pp.meso_obs_for_all_couplings(dset, "event_size")
    ```
    """
    ax = None
    for cdx, coupling in enumerate(dset["coupling"].to_numpy()):
        ax = meso_xr_with_errors(
            dset[obs].sel(coupling=coupling),
            ax=ax,
            color=cc.alpha_to_solid_on_bg(
                base_clr, cc.fade(cdx, dset["coupling"].size, invert=True)
            ),
            label=f"w = {coupling}",
            zorder=cdx,
            **kwargs,
        )

    if show_legend:
        ax.legend()

    if "correlation_coefficient" in obs:
        ax.set_ylim(0, 1)

        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))

    if not show_xlabel:
        ax.set_xlabel("")

    if not show_ylabel:
        ax.set_ylabel("")

    return ax


def meso_xr_with_errors(da, ax=None, apply_formatting=True, **kwargs):
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

    plot_kwargs.setdefault("lw", 0.5)
    plot_kwargs.pop("label")
    plot_kwargs.pop("fmt")
    plot_kwargs.pop("markersize")
    plot_kwargs.pop("elinewidth")
    plot_kwargs.pop("capsize")
    plot_kwargs["zorder"] -= 1

    ax.plot(da[x_name], da.mean(dim="repetition", skipna=True), **plot_kwargs)

    return ax


def meso_resource_cycle(
    input_file, show_nullclines=False, plot_kwargs=dict(), ax=None, **kwargs
):
    """
    Wrapper to plot a resource cycle for a single file created from the mesoscopic model

    kwargs are passed to mh.plot_ax_nullcline
    plot_kwargs are passed to ph.plot_resources_vs_activity
    """
    if isinstance(input_file, str):
        h5f = mh.prepare_file(input_file)
        # mh.find_system_bursts_and_module_contributions(h5f)
        # mh.module_contribution(h5f)
        mh.find_system_bursts_and_module_contributions2(h5f)
    else:
        h5f = input_file

    if ax is None:
        _, ax = plt.subplots()
    ax.set_rasterization_zorder(0)

    plot_kwargs = plot_kwargs.copy()
    plot_kwargs.setdefault("clip_on", False)
    plot_kwargs.setdefault("alpha", 0.1)
    plot_kwargs.setdefault("lw", 0.25)
    plot_kwargs.setdefault("zorder", -1)
    plot_kwargs.setdefault("max_traces_per_mod", 50)

    ax = ph.plot_resources_vs_activity(
        h5f,
        ax=ax,
        apply_formatting=False,
        **plot_kwargs,
    )
    ax.set_xlabel("Synaptic resources")
    ax.set_ylabel("Module rate")
    if show_title and isinstance(input_file, str):
        ax.set_title(input_file)
    # ax.set_xlim(0, 5)
    # ax.set_ylim(-0.4, 4)
    # cc.set_size(ax, 3.5, 3)

    # sns.despine(ax=ax, trim=True, offset=5)
    if show_nullclines:
        coupling, noise, rep = mh._coords_from_file(input_file)
        kwargs = kwargs.copy()
        # we have to overwrite this, cos
        kwargs["ext_str"] = noise
        try:
            kwargs.pop("simulation_time")
        except:
            pass
        mh.plot_nullcline(ax=ax, **kwargs.copy())

    if not show_xlabel:
        ax.set_xlabel("")

    if not show_ylabel:
        ax.set_ylabel("")

    return ax


def meso_sketch_gate_deactivation(**kwargs):
    if f"{_p_base}/../src" not in sys.path:
        sys.path.append(f"{_p_base}/../src")
    # if you get an import error here, we may need to append another location for
    # the src folder, this is a bit of a hack.
    from mesoscopic_model import probability_to_disconnect

    src_resources = np.arange(0.0, 1.3, 0.01)
    fig, ax = plt.subplots()
    ax.plot(src_resources, probability_to_disconnect(src_resources, **kwargs))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.002))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
    sns.despine(ax=ax, right=True, top=True, trim=True)

    if show_xlabel:
        ax.set_xlabel("Resources")

    if show_ylabel:
        ax.set_ylabel("Probability to disconnect")

    return ax


def meso_sketch_activation_function():
    if f"{_p_base}/../src" not in sys.path:
        sys.path.append(f"{_p_base}/../src")
    from mesoscopic_model import transfer_function, default_pars

    kwargs = dict()
    for key in ["gain_inpt", "k_inpt", "thrs_inpt"]:
        kwargs[key] = default_pars[key]

    total_input = np.arange(0.0, 3, 0.01)
    res = np.array([transfer_function(x, **kwargs) for x in total_input])

    fig, ax = plt.subplots()
    ax.plot(total_input, res)

    ax.axhline(
        kwargs["gain_inpt"],
        0,
        1,
        color="gray",
        linestyle=(0, (0.01, 2)),
        dash_capstyle="round",
    )

    ax.axvline(
        kwargs["thrs_inpt"],
        0,
        1,
        color="gray",
        linestyle=(0, (0.01, 2)),
        dash_capstyle="round",
    )

    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(20))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    sns.despine(ax=ax, right=True, top=True, trim=True)
    cc.set_size(ax, 2.7, 1.5)

    return ax


def meso_module_contribution(dset=None, coupling=0.3):
    """
    fig 4 c but for mesoscopic model, how many modules contributed to bursting
    events. so this is similar to `sim_modules_participating_in_bursts`
    """

    if dset is None:
        dset = xr.load_dataset(f"{p_sim}/meso/processed/analysed.hdf5")

    ax = sim_modules_participating_in_bursts(
        dset,
        simulation_coordinates=dict(coupling=coupling),
        xlim_for_fill=None,
        xlim_for_points=[0, 0.2],
        dim1="noise",
        drop_zero_len=False,
    )
    ax.set_xlabel("noise")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))

    if not show_xlabel:
        ax.set_xlabel("")

    if not show_ylabel:
        ax.set_ylabel("")

    return ax


def meso_activity_snapshot(
    h5f=None,
    main_width=3.5,
    zoom_duration=50,
    zoom_start=100,
    mark_zoomin_location=True,
    indicate_bursts=False,
):
    """
    Create one of our activity snapshots for the mesoscopic model,
    showing module rate, resources, gate state (and a zoom in?)

    # Parameters
    main_width : rough size of main column in cm
    """
    # get meta data
    # coupling, noise, rep = mh._coords_from_file(input_file)

    if isinstance(h5f, str):
        h5f = mh.prepare_file(h5f)
        mh.find_system_bursts_and_module_contributions2(h5f)

    total_width = main_width + 0.7 + 0.8  # 7mm for labels on the left, 8mm for zoom
    fig = plt.figure(figsize=[(total_width) / 2.54, 4.05 / 2.54])
    axes = []
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[main_width - 1.5, 0.8],
        height_ratios=[
            1,
            0.5,
            1,
        ],
        wspace=0.05,
        hspace=0.15,
        left=0.7 / total_width,
        right=0.99,
        top=0.95,
        bottom=0.15,
    )
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[1, 0], sharex=axes[0]))
    axes.append(fig.add_subplot(gs[2, 0], sharex=axes[0]))
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[1, 1], sharex=axes[3]))
    axes.append(fig.add_subplot(gs[2, 1], sharex=axes[3]))

    # rates
    ph.plot_module_rates(h5f, axes[0], alpha=1, lw=0.75)
    ph.plot_module_rates(h5f, axes[3], alpha=1, lw=0.75)
    ph.plot_system_rate(h5f, axes[0], mark_burst_threshold=False, lw=0.5)
    ph.plot_system_rate(h5f, axes[3], mark_burst_threshold=False, lw=0.5)

    if indicate_bursts:
        ph.plot_bursts_into_timeseries(h5f, axes[0], style="fill_between")
        ph.plot_bursts_into_timeseries(h5f, axes[3], style="fill_between")

    # gates
    ax = ph.plot_gate_history(h5f, axes[1])
    ax = ph.plot_gate_history(h5f, axes[4])

    # resources
    ax = ph.plot_state_variable(h5f, axes[2], variable="D", lw=0.5)
    ax = ph.plot_state_variable(h5f, axes[5], variable="D", lw=0.5)

    # formatting
    axes[3].set_xlim(zoom_start, zoom_start + zoom_duration)
    axes[0].set_xlim(0, 1000)

    # we did not share_y, do it manually
    # rates
    for a_id in [0, 3]:
        axes[a_id].set_ylim(-1, 15)
        axes[a_id].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(15))
        axes[a_id].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
        # workaround to show negative rate
        sns.despine(ax=axes[a_id], trim=True)

    # gates
    for a_id in [1, 4]:
        axes[a_id].set_ylim(0, 1)

    # resources
    for a_id in [2, 5]:
        axes[a_id].set_ylim(0, 1.2)
        axes[a_id].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        axes[a_id].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))

    for a_id in range(6):
        axes[a_id].set_xlabel("")
        axes[a_id].set_ylabel("")
        try:
            axes[a_id].get_legend().remove()
        except:
            pass
        sns.despine(ax=axes[a_id], bottom=True, offset=3, trim=False)

        # keep ticks for bottom left, we might need them
        if a_id != 2:
            axes[a_id].xaxis.set_visible(False)

    for a_id in [3, 4, 5]:
        # cc.detick(axis=axes[a_id].yaxis)
        axes[a_id].yaxis.set_visible(False)
        sns.despine(ax=axes[a_id], left=True, bottom=True)

    cc.detick(axis=axes[1].yaxis)
    sns.despine(ax=axes[1], left=True, bottom=True)
    sns.despine(ax=axes[0], left=False, bottom=True, trim=False)

    # reenable one time axis label
    ax = axes[2]
    sns.despine(ax=ax, left=False, bottom=False, offset=3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(250))

    if mark_zoomin_location:
        for ax in [axes[2], axes[5]]:
            ph._plot_bursts_into_timeseries(
                ax=ax,
                beg_times=[zoom_start + zoom_duration / 2],
                end_times=[zoom_start + zoom_duration / 2],
                style="markers",
                y_offset=-0.05,
                markersize=1.5,
                clip_on=False,
            )

    bnb.hi5.close_hot()

    return fig


def meso_explore_single(
    activity_snapshot=True,
    resource_cycle=True,
    flow_field=True,
    cycle_kwargs=dict(),
    ax=None,
    **kwargs,
):
    """
    Example
    ```
    pp.meso_explore_single(
        simulation_time=5000,
        ext_str=0.01,
        w0=0.01,
        gating_mechanism=False,
    )
    ```
    """
    if f"{_p_base}/../src" not in sys.path:
        sys.path.append(f"{_p_base}/../src")
    import mesoscopic_model as mm
    import tempfile

    path = tempfile.gettempdir() + "/meso_test.hdf5"
    pars = mm.default_pars.copy()
    for key, val in kwargs.items():
        pars[key] = val

    mm.simulate_and_save(
        output_filename=path,
        meta_data=dict(
            coupling=pars["w0"],
            noise=pars["ext_str"],
            rep=0,
            gating_mechanism=pars["gating_mechanism"],
        ),
        **pars,
    )

    ret = []

    if activity_snapshot:
        fig = meso_activity_snapshot(path)
        if (t := kwargs.get("simulation_time")) is not None:
            fig.axes[0].set_xlim(0, t)
        ret.append(fig)

    if resource_cycle:
        cycle_kwargs = cycle_kwargs.copy()
        cycle_kwargs.setdefault("alpha", 0.4)
        cycle_kwargs.setdefault("zorder", 1)
        cycle_kwargs.setdefault("clip_on", True)
        cycle_kwargs.setdefault("color", "C3")
        ax = meso_resource_cycle(
            path,
            ax=ax,
            show_nullclines=False,
            plot_kwargs=cycle_kwargs,
            **pars,
        )
        try:
            if show_title:
                # ax.set_title(f"input: {pars['ext_str']} | {pars['thrs_inpt']}")
                ax.set_title(f"input: {pars['ext_str']}, noise: {pars['sigma']}")
        except:
            pass

        ret.append(ax)

    # ax.set_xlim(0, 1.5)
    # ax.set_ylim(-1, 17)

    return ret


def sm_meso_noise_and_input_flowfields():

    cmap, norm = meso_stationary_points()

    total_width = 10.5
    fig = plt.figure(figsize=[(total_width) / 2.54, (total_width * 0.67) / 2.54])
    axes = []
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=[1.0] * 3,
        height_ratios=[1.0] * 3,
        wspace=0.1,
        hspace=0.1,
        left=0.15,
        right=0.99,
        top=0.95,
        bottom=0.15,
    )

    axes = [[] for _ in range(3)]
    #
    # gridspec starts counting in top left corner
    # high noise top, low noise bottom
    for sdx, sigma in enumerate([0.025, 0.1, 0.2]):
        row = sdx
        # low input left, high input right
        for hdx, h in enumerate([0.0, 0.1, 0.2]):
            col = hdx

            ax = fig.add_subplot(gs[row, col])
            axes[row].append(ax)

            pars = {
                "simulation_time": 5000,
                "ext_str": h,
                "w0": 0.0,
                "sigma": sigma,
                "gating_mechanism": False,
                "rseed": sdx * 103 + hdx,
            }

            meso_explore_single(
                activity_snapshot=False,
                ax=ax,
                cycle_kwargs=dict(
                    color=cmap(norm(h)),
                    alpha=0.6,
                ),
                **pars,
            )

            pars.pop("simulation_time")
            mh.plot_flow_field(
                ax=ax, plot_kwargs=dict(alpha=1, clip_on=True, color="#bbb"), **pars
            )

            txt = ""
            txt += r"$h={h}$".format(h=h)
            txt += "\n"
            txt += r"$\sigma={sigma}$".format(sigma=sigma)
            ax.text(
                0.95,
                0.95,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
                color="#666",
                fontsize=6,
            )

            ax.set_xlim(-0.1, 1.5)
            ax.set_ylim(-1, 15)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(15))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
            sns.despine(ax=ax, offset=0, trim=True)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
            ax.get_figure().tight_layout()
            if row != 2 or col != 0:
                clr = "#aaa"
                ax.tick_params(
                    axis="both",
                    which="both",
                    colors=clr,
                    labelleft=False,
                    labelbottom=False,
                )
                ax.spines["bottom"].set_color(clr)
                ax.spines["left"].set_color(clr)
                ax.xaxis.label.set_color(clr)

    ax.get_figure().savefig(
        f"{p_fo}/meso_noise_and_input_flowfields_combined.pdf",
        dpi=300,
    )


def meso_stationary_points(input_range=None):

    if input_range is None:
        input_range = np.arange(0.0, 0.35, 0.0005)

    rates, rsrcs = mh.get_stationary_solutions(input_range=input_range)

    special_input = [0.0, 0.1, 0.2]

    special_idx = [np.where(input_range == s)[0][0] for s in special_input]

    # z values for color map, ty this to input

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom",
        [
            (0, "#C31B2B"),
            # (0.25, "#FF9F68"),
            (0.5, "#DABE49"),
            (0.85, "#195571"),
            (1, "#011A39"),
        ],
        N=512,
    )
    cmap = cmap.reversed()
    norm = plt.Normalize(0, 0.2)

    plot_kwargs = dict(
        c=input_range,
        cmap=cmap,
        norm=norm,
        zorder=0,
        linewidths=0,
        s=1,
        clip_on=False,
    )

    str_to_data = dict(
        Resources=rsrcs,
        Rate=rates,
        Input=input_range,
    )
    str_to_lim = dict(
        Resources=[0, 1.15],
        Rate=[0, 5],
        Input=[0, 0.35],
    )
    str_to_minor = dict(
        Resources=matplotlib.ticker.MultipleLocator(0.5),
        Rate=matplotlib.ticker.MultipleLocator(1),
        Input=matplotlib.ticker.MultipleLocator(0.1),
    )
    str_to_major = dict(
        Resources=matplotlib.ticker.MultipleLocator(1),
        Rate=matplotlib.ticker.MultipleLocator(5),
        Input=matplotlib.ticker.MultipleLocator(0.2),
    )

    for combination in [
        ("Resources", "Rate"),
        ("Input", "Rate"),
        ("Input", "Resources"),
    ]:
        x_str = combination[0]
        y_str = combination[1]
        x = str_to_data[x_str]
        y = str_to_data[y_str]

        kwargs = plot_kwargs.copy()
        fig, ax = plt.subplots()
        ax.set_rasterization_zorder(0)
        ax.scatter(x=x, y=y, **kwargs)
        ax.set_xlabel(x_str, labelpad=1.5)
        ax.set_ylabel(y_str, labelpad=1.5)

        # mark special points as vectors
        kwargs["zorder"] = 2
        kwargs["s"] = 8
        if x_str == "Resources" and y_str == "Rate":
            # points coincide at Resources=1, make both visible
            kwargs["s"] = [8, 3, 8]
        kwargs["c"] = special_input
        ax.scatter(x=x[special_idx], y=y[special_idx], **kwargs)
        ax.set_xlim(str_to_lim[x_str])
        ax.set_ylim(str_to_lim[y_str])
        ax.xaxis.set_major_locator(str_to_major[x_str])
        ax.xaxis.set_minor_locator(str_to_minor[x_str])
        ax.yaxis.set_major_locator(str_to_major[y_str])
        ax.yaxis.set_minor_locator(str_to_minor[y_str])

        sns.despine(ax=ax, offset=3)

        cc.set_size(ax, 2.2, 1.5)

        ax.get_figure().savefig(
            f"fig/paper/meso_stationary_points_{x_str}_{y_str}.pdf", dpi=300
        )

    return cmap, norm


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def load_pd_hdf5(input_path, keys=None):
    """
    return a dict of data frames from processed conditions

    if global variable `remove_outlier` is set to true, the 210405_C single bond
    trial (with unusally short ibis) is filtered out.

    # possible `keys`:

    bursts : collection of all burst events across all trials and conditions
    isis : all inter-spike-intervals that occured. maybe we should make this optional,
        up to 70k spikes per trial in simulations
    rij : every row is a correlation coefficient, corresponding to one neuron pair
    rij_paired : rijs, grouped by certain conditions, e.g. between stimulated modules
    drij : simulations only,
        the correlation coefficients between all neuron pairs, calculated for the
        depletion variable ("D", modeling synaptic resources)
    trials : summary statistics, each row is a trial (or repetition in simulations)
    """

    single_key = False
    if keys is None:
        keys = ["bursts", "isis", "rij", "rij_paired", "drij", "trials"]
    elif isinstance(keys, str):
        single_key = True
        keys = [keys]
    res = dict()
    for key in keys:
        try:
            res[key] = pd.read_hdf(input_path, f"/data/df_{key}")
            if remove_outlier and "1b.hdf5" in input_path:
                res[key] = res[key].query("`Trial` != '210405_C'")
        except Exception as e:
            # log.exception(e)
            log.debug(f"/data/df_{key} not in {input_path}, skipping")

    if single_key:
        return res[keys[0]]
    else:
        return res


def custom_violins(
    df,
    category,
    observable,
    ax=None,
    num_swarm_points=400,
    same_points_per_swarm=True,
    replace=False,
    palette=None,
    bs_estimator=np.nanmedian,
    **violin_kwargs,
):

    log.debug(bs_estimator)
    # log.info(f'|{"":-^75}|')
    log.info(f"## Pooled violins for {observable}")
    # log.info(f'|{"":-^65}|')
    log.info(f" Condition | 2.5% percentile | 50% percentile | 97.5% percentile |")
    log.info(f" --------- | --------------- | -------------- | ---------------- |")
    violin_kwargs = violin_kwargs.copy()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.set_rasterization_zorder(-5)

    categories = df[category].unique()

    # we want half violins, so hack the data and abuse seaborns "hue" and "split"
    df["fake_hue"] = 0
    for cat in categories:
        dummy = pd.Series([np.nan] * len(df.columns), index=df.columns)
        dummy["fake_hue"] = 1
        dummy[category] = cat
        df = df.append(dummy, ignore_index=True)

    # lets use that seaborn looks up the `hue` variable as the key in palette dict,
    # maybe matching our global colors
    if palette is None:
        palette = colors.copy()

    # if we are plotting something, for which we have not defined color codes globally,
    # use defaults
    for idx, cat in enumerate(categories):
        if cat not in palette.keys():
            palette[cat] = f"C{idx}"

    light_palette = dict()
    for key in palette.keys():
        try:
            light_palette[key] = cc.alpha_to_solid_on_bg(palette[key], 0.5)
        except:
            # if we have keys that are not colors
            pass

    violin_defaults = dict(
        scale_hue=False,
        cut=0,
        scale="width",
        inner=None,
        bw=0.1,
        # palette=light_palette,
        hue="fake_hue",
        split=True,
    )

    for key in violin_defaults.keys():
        violin_kwargs.setdefault(key, violin_defaults[key])

    sns.violinplot(x=category, y=observable, data=df, **violin_kwargs)

    ylim = ax.get_ylim()

    # prequerry the data frames via category
    sub_dfs = dict()
    max_points = 0
    for idx, cat in enumerate(categories):
        df_for_cat = df.query(f"`{category}` == '{cat}'")
        sub_dfs[cat] = df_for_cat

        # for the swarm plot, fetch max height so we could tweak number of points and size
        hist, bins = np.histogram(df_for_cat[observable], _unit_bins(ylim[0], ylim[1]))
        max_points = np.max([max_points, np.max(hist)])

    for idx, cat in enumerate(categories):
        ax.collections[idx].set_color(light_palette[cat])
        ax.collections[idx].set_edgecolor(palette[cat])
        ax.collections[idx].set_linewidth(1.0)

        # custom error estimates
        df_for_cat = sub_dfs[cat]
        log.debug("bootstrapping")
        try:
            raise KeyError
            # we ended up not using nested bootstrapping
            mid, error, percentiles = ah.pd_nested_bootstrap(
                df_for_cat,
                grouping_col="Trial",
                obs=observable,
                num_boot=500,
                func=bs_estimator,
                resample_group_col=True,
                percentiles=[2.5, 50, 97.5],
            )
        except:
            # log.warning("Nested bootstrap failed")
            # this may also happen when category variable is not defined.
            mid, std, percentiles = ah.pd_bootstrap(
                df_for_cat,
                obs=observable,
                num_boot=500,
                func=bs_estimator,
                percentiles=[2.5, 50, 97.5],
            )

        log.debug(f"{cat}: estimator {mid:.3g}, std {std:.3g}")

        p_str = f" {cat:>9} "
        p_str += f"| {percentiles[0]:15.4f} "  # 2.5%
        p_str += f"| {percentiles[1]:14.4f} "  # 50%
        p_str += f"| {percentiles[2]:16.4f} |"  # 97.5%

        log.info(p_str)

        _draw_error_stick(
            ax,
            center=idx,
            mid=percentiles[1],
            errors=[percentiles[0], percentiles[2]],
            orientation="v",
            color=palette[cat],
            zorder=2,
        )

        # cover the part where the violins are with a rectangle to hide swarm plots
        # zorder of violins hard to change, probably it is 1
        # note from future paul: not needed when removing the points from collection
        # dy = ylim[1] - ylim[0]
        # ax.add_patch(
        #     matplotlib.patches.Rectangle(
        #         (idx, ylim[0] - 0.1 * dy),
        #         -0.5,
        #         dy * 1.2,
        #         linewidth=0,
        #         facecolor=ax.get_facecolor(),
        #         zorder=0,
        #     )
        # )

    # swarms
    swarm_defaults = dict(
        size=1.4,
        palette=light_palette,
        order=categories,
        zorder=-1,
        edgecolor=(1.0, 1.0, 1.0, 1),
        linewidth=0.1,
    )

    # so, for the swarms, we may want around the same number of points per swarm
    # then, apply subsampling for each category!
    if same_points_per_swarm:
        merged_df = []
        for idx, cat in enumerate(categories):
            sub_df = df.query(f"`{category}` == '{cat}'")
            if not replace:
                num_samples = np.min([num_swarm_points, len(sub_df)])
            else:
                num_samples = num_swarm_points
            merged_df.append(
                sub_df.sample(
                    n=num_samples,
                    replace=replace,
                    ignore_index=True,
                )
            )
        merged_df = pd.concat(merged_df, ignore_index=True)

    else:
        if not replace:
            num_samples = np.min([num_swarm_points, len(df)])
        else:
            num_samples = num_swarm_points
        merged_df = df.sample(n=num_samples, replace=replace, ignore_index=True)

    sns.swarmplot(
        x=category,
        y=observable,
        data=merged_df,
        **swarm_defaults,
    )

    # move the swarms slightly to the right and throw away left half
    for idx, cat in enumerate(categories):
        c = ax.collections[-len(categories) + idx]
        offsets = c.get_offsets()
        odx = np.where(offsets[:, 0] >= idx)
        offsets = offsets[odx]
        offsets[:, 0] += 0.05
        c.set_offsets(offsets)
        log.debug(f"{cat}: {len(offsets)} points survived")

    ax.get_legend().set_visible(False)

    # log.info(f'|{"":-^65}|')

    return ax


def custom_pointplot(df, category, observable, hue="Trial", ax=None, **point_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    point_kwargs = point_kwargs.copy()
    # point_kwargs.setdefault("palette", "YlGnBu_d")
    # point_kwargs.setdefault("dodge", True)
    point_kwargs.setdefault("scale", 0.5)
    point_kwargs.setdefault("ci", None)
    point_kwargs.setdefault("errwidth", 1)

    sns.pointplot(
        x=category,
        y=observable,
        hue=hue,
        data=df,
        **point_kwargs,
    )

    ax.get_legend().set_visible(False)
    plt.setp(ax.collections, sizes=[0.5])

    return ax


def custom_rij_barplot(df, ax=None, conditions=None, pairings=None, recolor=True):
    """
    query the layout of df first!
    provide a rij_paired dataframe
    """
    if pairings is None:
        pairings = ["within_stim", "within_nonstim", "across"]
        # pairings = df["Pairing"].unique()

    if conditions is None:
        conditions = ["pre", "stim"]
    df = df.query("Pairing == @pairings")
    df = df.query("Condition == @conditions")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    palette = colors.copy()
    for idx, c in enumerate(df["Condition"].unique()):
        try:
            palette[c]
        except:
            palette[c] = f"C{idx}"

    with matplotlib.rc_context(rc={"lines.solid_capstyle": "butt"}):
        sns.barplot(
            ax=ax,
            data=df,
            x="Pairing",
            hue="Condition",
            y="Correlation Coefficient",
            order=pairings,
            dodge=True,
            capsize=0.15,
            linewidth=1,
            errwidth=1.0,
            palette=palette,
            errcolor=".0",
            edgecolor=".0",
            # these settings reflect what we do manually in `table_rij()`
            # the bar height is given from the
            # estimator applied on the original data, not the bootstrap estimates
            estimator=np.nanmedian,
            ci=95,  # 2.5 and 97.5%
            n_boot=500,
            seed=815,
        )

    # so either we plot manually to tweak the styl or we alter after creation
    # barplot creates patches and lines
    if recolor:
        mult = len(pairings)
        for pdx, pairing in enumerate(pairings):
            for cdx, condition in enumerate(conditions):
                base_color = colors[f"rij_{pairing}"]
                c = ax.patches[pdx + cdx * mult]
                c.set_edgecolor(base_color)
                if cdx % len(conditions) == 0:
                    c.set_facecolor(cc.alpha_to_solid_on_bg(base_color, 0.7))
                else:
                    c.set_facecolor(cc.alpha_to_solid_on_bg(base_color, 0.3))

                # three lines for error bars with caps
                for ldx in range(0, 3):
                    c = ax.lines[3 * (pdx + cdx * mult) + ldx]
                    c.set_color(base_color)

        ax.get_legend().set_visible(False)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

    return ax


def custom_rij_scatter(
    df,
    ax=None,
    pairings=None,
    scatter=True,
    kde_levels=None,
    max_sample_size=np.inf,
    **kwargs,
):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if pairings is None:
        # pairings = ["within_stim", "within_nonstim", "across"]
        pairings = ["within_stim", "within_nonstim", "across"]
        # pairings = df["Pairing"].unique()

    kwargs = kwargs.copy()

    for pdx, pairing in enumerate(pairings):

        df_paired = df.query(f"Pairing == '{pairing}'")
        # print(pairing)

        # for each paring, make two long lists of rij: before stim and during stim.
        # add data from all experiments
        # rij may be np.nan if one of the neurons did not have any spikes.
        rijs = _rij_pairs_from_trials(df_paired)

        scatter_kwargs = kwargs.copy()
        try:
            scatter_kwargs.setdefault("color", colors[f"rij_{pairing}"])
        except:
            scatter_kwargs.setdefault("color", f"C{pdx}")

        scatter_kwargs["color"] = cc.alpha_to_solid_on_bg(scatter_kwargs["color"], 0.2)

        # scatter_kwargs.setdefault("alpha", .02)
        scatter_kwargs.setdefault("label", pairing)
        scatter_kwargs.setdefault("zorder", 1)
        scatter_kwargs.setdefault("edgecolor", None)
        scatter_kwargs.setdefault("linewidths", 0.0)

        marker = "o"
        if pairing == "across":
            scatter_kwargs.setdefault("marker", "D")
            scatter_kwargs.setdefault("s", 0.5)
        elif pairing == "within_stim":
            scatter_kwargs.setdefault("marker", "s")
            scatter_kwargs.setdefault("s", 0.5)
        else:
            scatter_kwargs.setdefault("marker", "o")
            scatter_kwargs.setdefault("s", 0.5)

        log.debug(f"{len(rijs['before'])} rij pairs")
        if len(rijs["before"]) > max_sample_size:
            idx = np.random.choice(
                np.arange(len(rijs["before"])),
                replace=False,
                size=max_sample_size,
            )
            rijs["before"] = np.array(rijs["before"])[idx]
            rijs["after"] = np.array(rijs["after"])[idx]

        if scatter:
            ax.scatter(
                rijs["before"],
                rijs["after"],
                **scatter_kwargs,
                # clip_on=False
            )

        if kde_levels is None:
            kde_levels = [0.5, 0.9, 0.95, 0.975]
        for low_l in kde_levels:
            sns.kdeplot(
                x=rijs["before"],
                y=rijs["after"],
                levels=[low_l, 1],
                fill=True,
                alpha=0.25,
                common_norm=False,
                color=colors[f"rij_{pairing}"],
                zorder=2,
                label=pairing,
            )

    ax.plot([0, 1], [0, 1], zorder=0, ls="-", color="gray", clip_on=False, lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.set_aspect(1)
    ax.set_xlabel("Correlation pre")
    ax.set_ylabel("Correlation stim")
    sns.despine(top=False, bottom=False, left=False, right=False)

    return ax


def _rij_pairs_from_trials(df_paired):
    """
    For each paring, make two long lists of rij: before stim and during stim.
    add data from all experiments
    rij may be np.nan if one of the neurons did not have any spikes.

    returned as a dict
    """
    rijs = dict()
    rijs["before"] = []
    rijs["after"] = []
    for trial in df_paired["Trial"].unique():
        df_trial = df_paired.query(f"`Trial` == '{trial}'")
        df_before = df_trial.query(f"`Stimulation` == 'Off'")
        df_after = df_trial.query(f"`Stimulation` == 'On'")
        # we make sure that on == stim and off == pre by querying df before.
        # otherwise, we would get shape missmatch, anyway.
        # df_before = df_trial.query(f"`Condition` == 'pre'")
        # df_after = df_trial.query(f"`Condition` == 'stim'")

        assert np.all(df_before["Pair ID"].to_numpy() == df_after["Pair ID"].to_numpy())

        rijs["before"].extend(df_before["Correlation Coefficient"].to_list())
        rijs["after"].extend(df_after["Correlation Coefficient"].to_list())

    return rijs


def _time_scale_bar(ax, x1, x2, y=-2, ylabel=-3, label=None, **kwargs):

    if label is None:
        label = f"{np.fabs(x2 - x1)}"

    kwargs = kwargs.copy()
    kwargs.setdefault("lw", 2)
    kwargs.setdefault("color", "black")
    kwargs.setdefault("clip_on", False)
    kwargs.setdefault("zorder", 5)

    ax.plot([x1, x2], [y, y], solid_capstyle="butt", **kwargs)
    ax.text(
        x1 + (x2 - x1) / 2,
        ylabel,
        label,
        transform=ax.transData,
        va="top",
        ha="center",
    )


# defaul bins when using histograms
def _unit_bins(low=0, high=1, num_bins=20):
    bw = (high - low) / num_bins
    return np.arange(low, high + 0.1 * bw, bw)


def _draw_error_stick(
    ax,
    center,
    mid,
    errors,
    outliers=None,
    orientation="v",
    linewidth=1.5,
    **kwargs,
):
    """
    Use this to draw errors likes seaborns nice error bars, but using our own
    error estimates.

    respects the global `_error_bar_cap_style` variable, set this to "butt" to be precise

    # Parameters
    ax : axis element
    center : number,
        where to align in off-data direction.
        seaborn uses integers 0, 1, 2, ... when plotting categorial violins.
    mid : float,
        middle data point to draw (white dot)
    errors : array like, length 2
        thick bar corresponding to errors
    outliers : array like, length 2
        thin (longer) bar corresponding to outliers
    orientation : "v" or "h"
    **kwargs are passed through to `ax.plot`
    """

    kwargs = kwargs.copy()
    kwargs.setdefault("color", "black")
    kwargs.setdefault("zorder", 3)
    kwargs.setdefault("clip_on", False)

    try:
        # "butt" gives precise errors, "round" looks much nicer but most people find
        # it confusing.
        kwargs.setdefault("solid_capstyle", _error_bar_cap_style)
    except:
        kwargs.setdefault("solid_capstyle", "round")

    if outliers is not None:
        assert len(outliers) == 2
        if orientation == "h":
            ax.plot(outliers, [center, center], linewidth=linewidth, **kwargs)
        else:
            ax.plot([center, center], outliers, linewidth=linewidth, **kwargs)

    assert len(errors) == 2
    if orientation == "h":
        ax.plot(errors, [center, center], linewidth=linewidth * 3, **kwargs)
    else:
        ax.plot([center, center], errors, linewidth=linewidth * 3, **kwargs)

    kwargs["zorder"] += 1
    kwargs["edgecolor"] = kwargs["color"]
    kwargs["color"] = "white"
    if kwargs["solid_capstyle"] == "round":
        kwargs["s"] = np.square(linewidth * 2)
    else:
        kwargs["s"] = np.square(linewidth * 1.5)
        # kwargs["lw"] = 0.75*linewidth
    kwargs.pop("solid_capstyle")

    if orientation == "h":
        ax.scatter(mid, center, **kwargs)
    else:
        ax.scatter(center, mid, **kwargs)


def _colorline(
    x,
    y,
    z=None,
    ax=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    **kwargs,
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, **kwargs
    )

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def _set_size(ax, w, h=None):
    """
    set the size of an axis, where the size describes the actual area of the plot,
    _excluding_ the axes, ticks, and labels.

    I later wrote a collection of tweaks that has some functionality in that
    sense but it does not (yet) support leaving `h` unspecified.

    w, h: width, height in cm

    # Example
    ```
        cm = 2.54
        fig, ax = plt.subplots()
        ax.plot(stuff)
        fig.tight_layout()
        _set_size(ax, 3.5*cm, 4.5*cm)
    ```
    """
    # https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w / 2.54) / (r - l)
    if h is None:
        ax.figure.set_figwidth(figw)
    else:
        figh = float(h / 2.54) / (t - b)
        ax.figure.set_size_inches(figw, figh)

# ------------------------------------------------------------------------------ #
# statistical tests
# ------------------------------------------------------------------------------ #


def exp_pairwise_tests_for_trials(observables, layouts=None):
    # observables = ["Mean Fraction", "Mean Correlation", "Functional Complexity"]
    kwargs = dict(observables=observables, col="Condition", return_num_samples=True)

    if layouts is None:
        layouts = ["1b", "3b", "merged", "KCl_1b", "Bicuculline_1b"]

    table = pd.DataFrame(
        columns=["layout", "kind", "N"] + observables,
    )

    # create an appendable row for each layout
    def row(layout, kind, p_dict, n=0):
        # p_dict is a dict of observable_name_as_str->scalar
        df_dict = dict(layout=[layout], kind=[kind], N=[n])
        for obs in p_dict.keys():
            df_dict[obs] = [p_dict[obs]]
        return pd.DataFrame(df_dict)

    for layout in layouts:
        log.info(f"\n{layout}")
        dfs = load_pd_hdf5(f"{p_exp}/processed/{layout}.hdf5")
        df = dfs["trials"]

        if layout in ["1b", "3b", "merged"]:
            p, n = _paired_sample_t_test(
                df, col_vals=["pre", "stim"], alternatives="two-sided", **kwargs
            )
            table = table.append(row(layout, "pre-stim", p, n), ignore_index=True)

            p, n = _paired_sample_t_test(
                df, col_vals=["stim", "post"], alternatives="two-sided", **kwargs
            )
            table = table.append(row(layout, "stim-post", p, n), ignore_index=True)

            p, n = _paired_sample_t_test(
                df, col_vals=["pre", "post"], alternatives="two-sided", **kwargs
            )
            table = table.append(row(layout, "pre-post", p, n), ignore_index=True)

        elif layout == "KCl_1b":
            p, n = _paired_sample_t_test(
                df, col_vals=["KCl_0mM", "KCl_2mM"], alternatives="two-sided", **kwargs
            )
            table = table.append(row(layout, "pre-stim", p, n), ignore_index=True)

        elif layout == "Bicuculline_1b":
            p, n = _paired_sample_t_test(
                df,
                col_vals=["spon_Bic_20uM", "stim_Bic_20uM"],
                alternatives="two-sided",
                **kwargs,
            )
            table = table.append(row(layout, "pre-stim", p, n), ignore_index=True)

    return table


def _paired_sample_t_test(
    df, col, col_vals, observables=None, alternatives=None, return_num_samples=False
):
    """
    # Parameters
    df : dataframe, each row an observable estimated over a whole trial (e.g. mean ibi)
    col : str, column label in which to check the `col_vals`
    observables : list
    alternatives : dict mapping observables to the alternative hypothesis:
        values can be "less", "greater", "two-sided"

    # Assumptions
    - dependent variable is continuous
        here: e.g. burst size or corr. coefficients.
    - observarions are independent
        here: observartions correspond to trials, measured independently.
    - dependent variable should be normally distributed.
        here: when using an observed variable in a trial, e.g. burst size,
        the we look at the mean of means (or mean of medians),
        hence central limit theorem applies.
        Does not hold for functional complexity, though.
    - no significant outliers

    """

    assert len(col_vals) == 2

    # make sure we only have rows where the column values are relevant
    # df = df.query(f"`{col}` == @col_vals")

    # we want to do a pairwise test, where a pair is before vs after in col_vals
    before = df.query(f"`{col}` == @col_vals[0]")
    after = df.query(f"`{col}` == @col_vals[1]")
    assert len(before) == len(after)
    num_samples = len(before)

    # log.debug(f"df.describe():\n{before.describe()}\n{after.describe()}")

    # using shapiro test we could _reject_ the H0 that the obs are normally
    # distributed
    # print(stats.shapiro(before[observable]))

    # focus on numeric values and do the test for selected observables
    if observables is None:
        before = before.select_dtypes(include="number")
        after = after.select_dtypes(include="number")
        observables = list(before.columns)

    if alternatives is None:
        alternatives = {obs: "two-sided" for obs in observables}
    elif isinstance(alternatives, str):
        alternatives = {obs: alternatives for obs in observables}

    p_values = dict()

    # H0 that two related and repeated samples have identical expectation value
    for obs in observables:
        alternative = alternatives[obs]
        if alternative == "one-sided":
            alternative = "two-sided"
        ttest = stats.ttest_rel(before[obs], after[obs], alternative=alternative)
        p = ttest.pvalue

        # this is a lazy workaround so we do not need to specify in which direction
        # our alternative hypothesis goes - since this will be different for
        # any of the passed observables!
        if alternatives[obs] == "one-sided":
            p /= 2.0

        p_values[obs] = p

    log.info(
        f"paired_sample_t_test for {col_vals}, {len(before)} samples."
        f" p_values:\n{_p_str(p_values, alternatives)}"
    )

    if return_num_samples:
        return p_values, num_samples
    return p_values


def exp_tests_for_joint_distributions(observables=["Fraction"], dfkind=None):

    kwargs = dict(
        observables=observables,
        col="Condition",
    )

    # depending on the observables, we may need a different dataframe.
    # make sure to provide observables that are in the same dataframe
    # try to find the right frame
    if dfkind is None:
        if "Fraction" in observables:
            dfkind = "bursts"
        elif "Correlation Coefficient":
            dfkind = "rij"
        else:
            raise ValueError(f"could not determine df, provide `dfkind`")

    layouts = ["1b", "3b", "merged"]
    for layout in layouts:
        log.info(f"\n{layout}")
        dfs = load_pd_hdf5(f"./dat/exp_out/{layout}.hdf5")

        # depending on dfkind, we need a different test
        if dfkind == "bursts":
            # groups are indpenendent
            _mann_whitney_u_test(dfs[dfkind], col_vals=["pre", "stim"], **kwargs)
            _mann_whitney_u_test(dfs[dfkind], col_vals=["stim", "post"], **kwargs)
            # _mann_whitney_u_test(
            #     dfs[dfkind], col_vals=["pre", "post"], **kwargs
            # )
            # _kolmogorov_smirnov_test(
            #     dfs[dfkind], col_vals=["pre", "post"], **kwargs
            # )
        elif dfkind == "rij":
            # groups are dependent
            _wilcoxon_signed_rank_test(dfs[dfkind], col_vals=["pre", "stim"], **kwargs)
            _wilcoxon_signed_rank_test(dfs[dfkind], col_vals=["stim", "post"], **kwargs)
            # _wilcoxon_signed_rank_test(
            #     dfs[dfkind], col_vals=["pre", "post"], **kwargs
            # )
            # _kolmogorov_smirnov_test(
            #     dfs[dfkind], col_vals=["pre", "post"], **kwargs
            # )
        else:
            raise NotImplementedError()


def _mann_whitney_u_test(df, col, col_vals, observables=None):
    """
    Use this guy to test whether our annealed distributions are identical.

    # Parameters
    df : dataframe where each row is one observation, distributions are pooled across
        trials.

    # Assumptions
    - oboservations of both groups are independant.
        Here, in our annealed description, an observation in one group (a burst in pre)
        has no direct observation in the other group. bursts in pre and stim are
        independent. Violated for correlation coefficients, where each pair of
        neurons exists in each condition.
    - responses can be compared (one larger than the other)
    - under H0 distribution of both are equal
    - H1 that distributions are not equal
    """

    # we compare pre with stim, and stim with post
    assert len(col_vals) == 2

    # make sure we only have rows where the column values are relevant
    # df = df.query(f"`{col}` == @col_vals")

    # we want to do a pairwise test, where a pair is before vs after in col_vals
    before = df.query(f"`{col}` == @col_vals[0]")
    after = df.query(f"`{col}` == @col_vals[1]")

    # before and after are independant and do not need to have the same sample size
    # assert len(before) == len(after)

    # log.debug(f"df.describe():\n{before.describe()}\n{after.describe()}")

    # focus on numeric values and do the test for selected observables
    if observables is None:
        before = before.select_dtypes(include="number")
        after = after.select_dtypes(include="number")
        observables = list(before.columns)

    p_values = dict()

    # H0 that two related and repeated samples have identical expectation value
    for obs in observables:
        # filter out nans. eg. the ibi of the last/first burst
        bf = before[obs].to_numpy()
        bf = bf[np.where(np.isfinite(bf))]

        af = after[obs].to_numpy()
        af = af[np.where(np.isfinite(af))]

        utest = stats.mannwhitneyu(bf, af)
        p_values[obs] = utest.pvalue

    log.info(
        f"mann_whitney_u_test for {col_vals}, {len(bf)} and"
        f" {len(af)} samples, respectively. p_values:\n{_p_str(p_values)}"
    )

    return p_values


def _wilcoxon_signed_rank_test(df, col, col_vals, observables=None):
    """
    Use this guy to test whether our annealed distributions are identical.

    # Parameters
    df : dataframe where each row is one observation, distributions are pooled across
        trials.

    # Assumptions
    - oboservations of both groups are dependant.
        same subject is present in both groups (here, each pair of neurons was present
        before and after stim)
    - independence of paired observations.
        here, pairs of neurons are indepenent from one another.
    - continuous dependent variable. here: correlation
    """

    # we compare pre with stim, and stim with post
    assert len(col_vals) == 2

    # make sure we only have rows where the column values are relevant
    # df = df.query(f"`{col}` == @col_vals")

    # we want to do a pairwise test, where a pair is before vs after in col_vals
    before = df.query(f"`{col}` == @col_vals[0]")
    after = df.query(f"`{col}` == @col_vals[1]")

    # paired observations
    assert len(before) == len(after)

    # log.debug(f"df.describe():\n{before.describe()}\n{after.describe()}")

    # focus on numeric values and do the test for selected observables
    if observables is None:
        before = before.select_dtypes(include="number")
        after = after.select_dtypes(include="number")
        observables = list(before.columns)

    p_values = dict()

    # H0 that two related and repeated samples have identical expectation value
    for obs in observables:
        bf = before[obs].to_numpy()
        af = after[obs].to_numpy()

        # filter out nans. correlation coefficients may become nan if no spikes
        # were found for a neuron
        idx = np.where(np.isfinite(bf) & np.isfinite(af))[0]
        bf = bf[idx]
        af = af[idx]

        wtest = stats.wilcoxon(bf, af)
        p_values[obs] = wtest.pvalue

    log.info(
        f"wilcoxon_signed_rank for {col_vals}, {len(bf)} and"
        f" {len(af)} samples, respectively. p_values:\n{_p_str(p_values)}"
    )

    return p_values


def _kolmogorov_smirnov_test(df, col, col_vals, observables=None):
    """ """

    # we compare pre with stim, and stim with post
    assert len(col_vals) == 2

    # make sure we only have rows where the column values are relevant
    # df = df.query(f"`{col}` == @col_vals")

    # we want to do a pairwise test, where a pair is before vs after in col_vals
    before = df.query(f"`{col}` == @col_vals[0]")
    after = df.query(f"`{col}` == @col_vals[1]")

    # paired observations
    # assert len(before) == len(after)

    # log.debug(f"df.describe():\n{before.describe()}\n{after.describe()}")

    # focus on numeric values and do the test for selected observables
    if observables is None:
        before = before.select_dtypes(include="number")
        after = after.select_dtypes(include="number")
        observables = list(before.columns)

    p_values = dict()

    # H0 that two related and repeated samples have identical expectation value
    for obs in observables:
        bf = before[obs].to_numpy()
        af = after[obs].to_numpy()

        # filter out nans. correlation coefficients may become nan if no spikes
        # were found for a neuron
        try:
            idx = np.where(np.isfinite(bf) & np.isfinite(af))[0]
            bf = bf[idx]
            af = af[idx]
        except:
            pass

        kstest = stats.ks_2samp(data1=bf, data2=af)
        p_values[obs] = kstest.pvalue

    log.info(
        f"kolmogorov_smirnov for {col_vals}, {len(bf)} and"
        f" {len(af)} samples, respectively. p_values:\n{_p_str(p_values)}"
    )

    return p_values


def sim_tests_stimulating_two_modules(observables=["Fraction"], dfkind=None):

    kwargs = dict(
        observables=observables,
        col="Condition",
    )

    # depending on the observables, we may need a different dataframe.
    # make sure to provide observables that are in the same dataframe
    # try to find the right frame
    if dfkind is None:
        if "Fraction" in observables:
            dfkind = "bursts"
        elif "Correlation Coefficient":
            dfkind = "rij"
        else:
            raise ValueError(f"could not determine df, provide `dfkind`")

    layouts = ["k=5"]
    for layout in layouts:
        log.info(f"\n{layout}")
        dfs = load_pd_hdf5(f"./dat/sim_partial_out_20/{layout}.hdf5")

        # depending on dfkind, we need a different test
        if dfkind == "bursts":
            # groups are indpenendent
            _mann_whitney_u_test(dfs[dfkind], col_vals=["0.0 Hz", "20.0 Hz"], **kwargs)
        elif dfkind == "rij":
            # groups are dependent
            _wilcoxon_signed_rank_test(
                dfs[dfkind], col_vals=["0.0 Hz", "20.0 Hz"], **kwargs
            )
        else:
            raise NotImplementedError()


def _p_str(p_values, alternatives=None):
    """
    # Parameters
    p_values : dict, with observable as key and float p value
    alternaives : None or dict, mapping observables (keys of `p_values`) to the
        alternative hypothesis
    """

    p_str = "\n"
    for obs in p_values.keys():
        p_str += f"{obs:>34}: "
        p = p_values[obs]
        if p <= 0.01:
            p_str += "** "
        elif p <= 0.05:
            p_str += "*  "
        else:
            p_str += "ns "
        if alternatives is not None:
            p_str += f"({p:.05f}, {alternatives[obs]})\n"
        else:
            p_str += f"({p:.05f})\n"

    return p_str
