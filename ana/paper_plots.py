# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-11-08 17:51:24
# @Last Modified: 2022-04-06 16:11:29
# ------------------------------------------------------------------------------ #
# collect the functions to create figure panels here
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import re
from ana.plot_helper import overview_burst_duration_and_isi
import h5py
import argparse
import numbers
import logging
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import palettable

from benedict import benedict
from tqdm import tqdm
from scipy import stats

import plot_helper as ph
import ana_helper as ah
import ndim_helper as nh
import meso_helper as mh
import colors as cc
import hi5 as h5


log = logging.getLogger(__name__)
log.setLevel("INFO")
warnings.filterwarnings("ignore")  # suppress numpy warnings

# select things to draw for every panel
show_title = False
show_xlabel = True
show_ylabel = True
show_legend = False
show_legend_in_extra_panel = False
use_compact_size = True  # this recreates the small panel size of the manuscript

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
matplotlib.rcParams["savefig.facecolor"] = (
    0.0,
    0.0,
    0.0,
    0.0,
)  # transparent figure bg
matplotlib.rcParams["axes.facecolor"] = (
    1.0,
    0.0,
    0.0,
    0.0,
)  # developer mode, red axes

# style of error bars 'butt' or 'round'
# "butt" gives precise errors, "round" looks much nicer but most people find it confusing.
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/joinstyle.html
_error_bar_cap_style = "butt"

colors = dict()
# colors["pre"] = "#135985"
# colors["pre"] = "#138567"
colors["pre"] = "#541854"
colors["Off"] = colors["pre"]
# colors["75 Hz"] = colors["pre"]
# colors["80 Hz"] = colors["pre"]

colors["stim"] = "#e09f3e"
colors["On"] = colors["stim"]
colors["90 Hz"] = colors["stim"]

# colors["post"] = "#80A4BB"
# colors["post"] = "#99CCB5"
colors["post"] = "#8C668C"

colors["KCl_0mM"] = "gray"
colors["KCl_2mM"] = "gray"
colors["spon_Bic_20uM"] = colors["pre"]
colors["stim_Bic_20uM"] = colors["stim"]

colors["rij_within_stim"] = "#e09f3e"
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

reference_coordinates = dict()
reference_coordinates["jA"] = 45
reference_coordinates["jG"] = 50
reference_coordinates["tD"] = 20
reference_coordinates["k_inter"] = 5

# we had one trial for single bond where we saw more bursts than everywhere else
remove_outlier = True


def fig_1(show_time_axis=False):

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(811)

    # ------------------------------------------------------------------------------ #
    # Create raster plots
    # ------------------------------------------------------------------------------ #

    # optogenetic
    path = "./dat/exp_in/1b"
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
        cc.set_size(ax=fig.axes[0], w=fig_widths[condition], h=None)

        if not show_time_axis:
            fig.axes[-1].xaxis.set_visible(False)
            sns.despine(ax=fig.axes[-1], bottom=True, left=False)

        fig.savefig(f"./fig/paper/exp_combined_{c_str}.pdf", dpi=300, transparent=False)

    # chemical
    path = "./dat/exp_in/KCl_1b"
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
        cc.set_size(ax=fig.axes[0], w=fig_widths[condition], h=None)

        if not show_time_axis:
            fig.axes[-1].xaxis.set_visible(False)
            sns.despine(ax=fig.axes[-1], bottom=True, left=False)

        fig.savefig(f"./fig/paper/exp_combined_{c_str}.pdf", dpi=300, transparent=False)

    # ------------------------------------------------------------------------------ #
    # Stick plots for optogenetic vs chemical
    # ------------------------------------------------------------------------------ #

    for obs in ["Functional Complexity", "Mean Fraction"]:
        ax = exp_chemical_vs_opto2(observable=obs)
        cc.set_size3(ax, 1.2, 2)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
        ax.get_figure().savefig(f"./fig/paper/exp_chem_vs_opto_{obs}.pdf", dpi=300)


def fig_2(skip_plots=False):

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(812)

    if not skip_plots:
        exp_violins_for_layouts()
        exp_rij_for_layouts()
        exp_sticks_across_layouts(observable="Functional Complexity", hide_labels=False)

    log.debug("\n\nPairwise tests for trials\n")
    exp_pairwise_tests_for_trials(
        observables=[
            "Functional Complexity",
        ]
    )


def fig_3():

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(813)

    # reproducing 2 module stimulation in simulations
    # choosing noise rate in lower modules to maximize fc, ibi are somewhat off at 15hz

    dfs = load_pd_hdf5(
        "./dat/sim_partial_out_20/k=5.hdf5", ["bursts", "rij", "rij_paired"]
    )
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
        cc.set_size2(ax, 1.5, 2.0)

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
    ax.set_xlabel("Burst size")
    ax.get_figure().savefig(f"./fig/paper/sim_partial_violins_fraction.pdf", dpi=300)

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
    ax.get_figure().savefig(f"./fig/paper/sim_partial_violins_rij.pdf", dpi=300)

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
    cc.set_size3(ax, 3, 1.5)
    ax.get_figure().savefig(f"./fig/paper/sim_rij_barplot.pdf", dpi=300)

    log.info("scattered 2d rij paired for simulations")
    ax = custom_rij_scatter(
        df, max_sample_size=2500, scatter=True, kde_levels=[0.9, 0.95, 0.975]
    )
    cc.set_size3(ax, 3, 3)
    ax.get_figure().savefig(f"./fig/paper/sim_2drij.pdf", dpi=300)

    # ------------------------------------------------------------------------------ #
    # raster plots for 2 module stimulation
    # ------------------------------------------------------------------------------ #

    h5f = ph.ah.prepare_file(
        "./dat/the_last_one/dyn/stim=02_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_stimrate=20.0_rep=000.hdf5"
    )

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

    fig.savefig(f"./fig/paper/sim_raster_stim_02_20hz.pdf", dpi=900)

    # and again, without stimulation
    h5f = ph.ah.prepare_file(
        "./dat/the_last_one/dyn/stim=02_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_stimrate=0.0_rep=000.hdf5"
    )

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

    fig.savefig(f"./fig/paper/sim_raster_stim_02_0hz.pdf", dpi=900)


def fig_4(skip_rasters=True, skip_cycles=True):

    # set the global seed once for each figure to produce consistent results, when
    # calling repeatedly.
    # many panels rely on bootstrapping and drawing random samples
    np.random.seed(814)

    # ------------------------------------------------------------------------------ #
    # raster plots
    # ------------------------------------------------------------------------------ #

    def path(k, rate, rep):
        # path = f"./dat/the_last_one/dyn/highres_stim=off"
        path = f"./dat/the_last_one/dyn/stim=off"
        path += f"_k={k:d}_jA=45.0_jG=50.0_jM=15.0_tD=20.0"
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

        cc.set_size(ax=ax, w=3.5, h=None)
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
            f"./fig/paper/sim_ts_combined_k={cs['k']}_nozoom_{cs['rate']}Hz.pdf",
            dpi=900,  # use higher res to get rasters smooth
            transparent=False,
        )

        # do we want a schematic of the topology?
        sim_layout_sketch(
            in_path=path(**cs),
            out_path=f"./fig/paper/sim_layout_sketch_{cs['k']}_{cs['rate']}Hz.png",
        )

    # ------------------------------------------------------------------------------ #
    # panel h, resource cycles
    # ------------------------------------------------------------------------------ #

    if not skip_cycles:
        # sim_resource_cycles(apply_formatting=True, k_list=[-1, 5])
        # for main manuscript, defaults above are fine, but for SI bigger overview:
        global axes
        axes = sim_resource_cycles(apply_formatting=False, k_list=[-1, 1, 5, 10])
        for k in axes.keys():
            for rate in axes[k].keys():
                ax = axes[k][f"{rate}"]
                k_str = f"merged" if k == "-1" else f"k={k}"
                # ax.set_title(f"{k_str}    {rate}Hz")
                cc.set_size2(ax, 1.6, 1.4)
                ax.set_xlim(0.0, 1.0)
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
                # workaround to avoid cutting off figure content but limiting axes:
                # set, then despine, then set again
                ax.set_ylim(0, 150)
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(150))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))

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

                if rate == "80":
                    cc.detick(ax.xaxis, keep_ticks=False, keep_labels=False)
                    sns.despine(ax=ax, bottom=True)

                ax.set_ylim(0, 199)
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))

                ax.get_figure().savefig(
                    f"./fig/paper/sim_resource_cycle_{k_str}_{rate}Hz.pdf",
                    transparent=False,
                )

    # ------------------------------------------------------------------------------ #
    # Number of modules over which bursts / events extend
    # ------------------------------------------------------------------------------ #

    cs = reference_coordinates.copy()
    cs["k_inter"] = 5

    ax = sim_modules_participating_in_bursts(
        input_path="./dat/the_last_one/k_sweep_with_merged.hdf5",
        simulation_coordinates=cs,
        xlim=[65, 100],
        drop_zero_len=True,
    )
    cc.set_size2(ax, 3.5, 2.0)
    ax.get_figure().savefig("./fig/paper/sim_fractions.pdf", dpi=300)

    # ------------------------------------------------------------------------------ #
    # observables vs noise for differen k
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
            path="./dat/the_last_one/k_sweep_with_merged.hdf5",
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
        ]:
            # these guys only go to the supplemental material
            cc.set_size3(ax, 3.5, 2.5)
        else:
            cc.set_size3(ax, 2.5, 1.8)
        ax.get_figure().savefig(f"./fig/paper/sim_ksweep_{obs}.pdf", dpi=300)


def fig_supplementary():
    """
    wrapper to produce the panels of most supplementary figures.
    """
    sm_exp_trialwise_observables()
    sm_exp_bicuculline()


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
                df.insert(len(cols)-1, col, df.pop(col))
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


def sm_exp_trialwise_observables():
    """
    We can calculate estimates for every trial and see how they change within each
    trial.
    Also does stat. significance tests.

    This has some overlap with fig. 1 and 2
    """
    kwargs = dict(save_path=None, hide_labels=False)
    prefix = "./fig/paper/exp_layouts_sticks"

    ax = exp_sticks_across_layouts(observable="Functional Complexity", **kwargs)
    ax.get_figure().savefig(f"{prefix}_functional_complexity.pdf", dpi=300)

    ax = exp_sticks_across_layouts(observable="Mean Fraction", **kwargs)
    ax.set_ylabel("Mean Event size")
    ax.get_figure().savefig(f"{prefix}_mean_event_size.pdf", dpi=300)

    ax = exp_sticks_across_layouts(observable="Mean Correlation", **kwargs)
    ax.get_figure().savefig(f"{prefix}_mean_correlation.pdf", dpi=300)

    ax = exp_sticks_across_layouts(observable="Mean IBI", set_ylim=False, **kwargs)
    ax.set_ylabel("Mean IEI (seconds)")
    ax.get_figure().savefig(f"{prefix}_mean_iei.pdf", dpi=300)

    ax = exp_sticks_across_layouts(
        observable="Mean Core delays", set_ylim=False, apply_formatting=False, **kwargs
    )
    ax.set_ylabel("Mean Core delay\n(seconds)")
    ax.set_ylim(0, None)
    sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    ax.get_figure().savefig(f"{prefix}_mean_core_delay.pdf", dpi=300)

    exp_pairwise_tests_for_trials(
        observables=[
            "Mean Correlation",
            "Mean IBI",
            "Mean Fraction",
            "Functional Complexity",
            "Mean Core delays",
            # "Median IBI",
            # "Median Core delays",
        ],
        layouts=["1b", "3b", "merged"],
    )


def sm_exp_bicuculline():
    """
    violins and sticks for blocked inhibition
    """

    exp_violins_for_layouts(layouts=["bic"], observables=["event_size", "rij"])

    kwargs = dict(
        hide_labels=False,
        layouts=["Bicuculline_1b"],
        conditions=["spon_Bic_20uM", "stim_Bic_20uM"],
    )
    save_path = "./fig/paper/exp_layouts_sticks_bic"

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
    cc.set_size3(ax, 2.2, 1.8)
    ax.set_ylabel("Number of cells")

    ax.get_figure().savefig(f"./fig/paper/exp_layouts_sticks_num_cells.pdf", dpi=300)


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
        markersize=1.5,
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
        lw=0.5,
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
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60))
    # ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    # tighten space between panels
    plt.subplots_adjust(hspace=0.001)

    if time_range is None:
        time_range = [0, 540]
    ax.set_xlim(*time_range)

    return fig


# this became impossible to tweak further when using seaborn
def exp_chemical_vs_opto_old(observable="Mean Fraction", df="trials"):
    chem = load_pd_hdf5("./dat/exp_out/KCl_1b.hdf5")
    opto = load_pd_hdf5("./dat/exp_out/1b.hdf5")

    fig, ax = plt.subplots()

    custom_pointplot(
        opto[df].query("`Condition` in ['pre', 'stim']"),
        category="Stimulation",
        observable=observable,
        ax=ax,
        palette="Wistia",
        scale=0.2,
    )
    custom_pointplot(
        chem[df],
        category="Stimulation",
        observable=observable,
        ax=ax,
        # palette="gist_yarg",
        color="#999",
        # palette=None,
        linestyles="-",
        scale=0.2,
    )
    # draw chemical in front
    from_last = len(chem[df]["Trial"].unique())
    num_lines = len(ax.collections)
    for idx in range(num_lines - from_last, num_lines):
        ax.collections[idx].set_zorder(num_lines - from_last + idx + 1)
        ax.lines[idx].set_zorder(num_lines - from_last + idx + 1)

    # disable clipping
    for idx in range(0, num_lines):
        ax.collections[idx].set_clip_on(False)
        ax.lines[idx].set_clip_on(False)

    ax.set_ylim(0, 1.0)
    sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=-10)
    ax.tick_params(bottom=False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

    cc.set_size(ax, 2.5, 2)

    ax.get_figure().savefig("./fig/paper/exp_chem_vs_opto.pdf", dpi=300)

    return ax


def exp_chemical_vs_opto2(observable="Functional Complexity"):
    chem = load_pd_hdf5("./dat/exp_out/KCl_1b.hdf5")
    opto = load_pd_hdf5("./dat/exp_out/1b.hdf5")

    # we want the precalculated summary statistics of each trial
    dfs = dict()
    dfs["exp"] = opto["trials"].query("`Condition` in ['pre', 'stim']")
    dfs["exp_chemical"] = chem["trials"]
    global df_merged
    df_merged = pd.concat([dfs["exp"], dfs["exp_chemical"]], ignore_index=True)

    fig, ax = plt.subplots()

    categories = ["exp", "exp_chemical"]
    conditions = ["Off", "On"]
    small_dx = 0.3
    large_dx = 1.0
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

            ax.plot(
                x,
                y,
                # marker="o",
                # markersize=1,
                lw=0.5,
                color=cc.alpha_to_solid_on_bg(clr, 0.5),
                # color=clr,
                # alpha = 0.3,
                label=trial,
                zorder=0,
                clip_on=False,
            )

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
    dfs=None,
    x_offset=0,
):
    log.info(f"")
    log.info(f"# sticks for {observable}")

    if layouts is None:
        layouts = ["1b", "3b", "merged"]
    if conditions is None:
        conditions = ["pre", "stim", "post"]

    # we want the precalculated summary statistics of each trial
    if dfs is None:
        dfs = dict()
        for key in layouts:
            df = load_pd_hdf5(f"./dat/exp_out/{key}.hdf5")
            dfs[key] = df["trials"].query("Condition == @conditions")

    fig, ax = plt.subplots()

    small_dx = 0.3
    large_dx = 1.5
    x_pos = dict()
    x_pos_sticks = dict()
    for ldx, l in enumerate(layouts):
        x_pos[l] = dict()
        x_pos_sticks[l] = dict()
        for cdx, c in enumerate(conditions):
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

        clr = colors["pre"]
        trials = dfs[etype]["Trial"].unique()
        for trial in trials:
            df = dfs[etype].loc[dfs[etype]["Trial"] == trial]
            # assert len(df) == len(layouts)
            x = []
            y = []
            try:
                for idx, row in df.iterrows():
                    stim = row["Condition"]
                    x.append(x_pos[etype][stim])
                    y.append(row[observable])

                ax.plot(
                    x,
                    y,
                    # marker="o",
                    # markersize=1,
                    lw=0.5,
                    color=cc.alpha_to_solid_on_bg(clr, 0.2),
                    # color=clr,
                    # alpha = 0.3,
                    label=trial,
                    zorder=0,
                    clip_on=False,
                )
            except KeyError as e:
                # this fails if we have no conditions in the df,
                # needed for number of cells
                log.debug(f"{e}")

        for stim in conditions:
            try:
                df = dfs[etype].query(f"Condition == '{stim}'")
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
            p_str += f"| {stim:>9} "
            p_str += f"| {mid:20.5f} "
            p_str += f"| {error:26.5f} "
            p_str += f"| {df_min:12.5f} "
            p_str += f"| {df_max:12.5f} |"

            log.info(p_str)

            try:
                clr = colors[stim]
            except:
                clr = "#808080"

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
        log.info(f"")

    # ax.legend()
    if set_ylim:
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.set_xlim(-0.25, 3.5)
    if apply_formatting:
        sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    ax.set_xticks([])
    if not hide_labels:
        ax.set_ylabel(f"{observable}")

    if apply_formatting:
        cc.set_size3(ax, 2.2, 2)

    if save_path is "automatic":
        save_path = f"./fig/paper/exp_layouts_sticks_{observable}.pdf"
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
        dfs["single-bond"] = load_pd_hdf5("./dat/exp_out/1b.hdf5")
    if "triple-bond" in layouts:
        dfs["triple-bond"] = load_pd_hdf5("./dat/exp_out/3b.hdf5")
    if "merged" in layouts:
        dfs["merged"] = load_pd_hdf5("./dat/exp_out/merged.hdf5")
    if "chem" in layouts:
        dfs["chem"] = load_pd_hdf5("./dat/exp_out/KCl_1b.hdf5")
    if "bic" in layouts:
        dfs["bic"] = load_pd_hdf5("./dat/exp_out/Bicuculline_1b.hdf5")

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
        if not show_ylabel:
            ax.set_ylabel(f"")

        ax.set_xticks([])
        cc.set_size2(ax, 3, 2.0)

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
        ax.get_figure().savefig(f"./fig/paper/exp_violins_fraction_{layout}.pdf", dpi=300)

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
        ax.get_figure().savefig(f"./fig/paper/exp_violins_rij_{layout}.pdf", dpi=300)

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
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
        ax.get_figure().savefig(f"./fig/paper/exp_violins_ibi_{layout}.pdf", dpi=300)

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
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.get_figure().savefig(
            f"./fig/paper/exp_violins_core_delay_{layout}.pdf", dpi=300
        )

    return ax


def exp_rij_for_layouts():
    dfs = dict()
    dfs["single-bond"] = load_pd_hdf5("./dat/exp_out/1b.hdf5", ["rij_paired"])
    dfs["triple-bond"] = load_pd_hdf5("./dat/exp_out/3b.hdf5", ["rij_paired"])
    dfs["merged"] = load_pd_hdf5("./dat/exp_out/merged.hdf5", ["rij_paired"])

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

        cc.set_size3(ax, 3, 3)
        ax.get_figure().savefig(f"./fig/paper/exp_2drij_{layout}.pdf", dpi=300)

        ax = custom_rij_barplot(
            dfs[layout]["rij_paired"], pairings=pairings, recolor=True
        )
        ax.set_ylim(0, 1)
        # ax.set_xlabel(layout)
        ax.set_xlabel("")
        ax.set_ylabel("")
        cc.set_size3(ax, 3, 1.5)
        ax.get_figure().savefig(f"./fig/paper/exp_rij_barplot_{layout}.pdf", dpi=300)

    return ax


# Fig 3
def sim_raster_plots_old(
    bs_large=20 / 1000,  # width of the gaussian kernel for rate
    threshold_factor=2.5 / 100,  # fraction of max peak height for burst
    zoomin=False,
):

    rates = [80, 90]

    # exclude_nids_from_raster = np.delete(np.arange(0, 160), np.arange(0, 160, 8))
    exclude_nids_from_raster = []

    for rdx, rate in enumerate(rates):

        c_str = f"{rate} Hz"

        h5f = ah.prepare_file(
            f"./dat/the_last_one/dyn/stim=off_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate={rate:.1f}_rep=001.hdf5"
        )

        # ------------------------------------------------------------------------------ #
        # raster
        # ------------------------------------------------------------------------------ #
        figsize = [3.5 / 2.54, 3 / 2.54]
        if zoomin:
            figsize = [1 / 2.54, 3 / 2.54]
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize)

        ax = axes[1]
        ax.set_rasterization_zorder(0)
        ph.plot_raster(
            h5f,
            ax,
            exclude_nids=exclude_nids_from_raster,
            clip_on=True,
            zorder=-1,
            markersize=0.75,
            alpha=0.5,
        )

        ax.set_ylim(-1, None)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="both", which="both", bottom=False)
        ax.set_yticks([])
        # sns.despine(ax=ax, left=False, right=False, bottom=False, top=False)
        sns.despine(ax=ax, left=True, right=True, bottom=True, top=True)

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
        ax.margins(x=0, y=0)
        ax.set_xticks([])

        ax.set_ylim(0, 80)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(20))
        if rate == 90:
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(([0, 40])))
            ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(([20])))
        sns.despine(ax=ax, left=False, bottom=False, trim=True, offset=0)
        ax.tick_params(axis="x", which="both", bottom=False)

        # ------------------------------------------------------------------------------ #
        # adaptation
        # ------------------------------------------------------------------------------ #

        # fig, ax = plt.subplots()
        ax = axes[2]
        ph.plot_state_variable(h5f, ax, variable="D", lw=0.5, apply_formatting=False)

        ax.margins(x=0, y=0)
        ax.set_xlim(0, 360)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))

        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
        sns.despine(ax=ax, left=False, bottom=False, trim=True, offset=0)
        plt.subplots_adjust(hspace=0.1)

        if zoomin == False:
            ax.get_figure().savefig(
                f"./fig/paper/sim_ts_combined_nozoom_{c_str}.pdf",
                dpi=300,
                transparent=False,
            )
        else:
            if rate == 80:
                x_ref = 298.35
            elif rate == 90:
                x_ref = 288.05
            for ax in [axes[0], axes[2]]:
                sns.despine(ax=ax, left=True, bottom=False, trim=False, offset=0)
                ax.tick_params(axis="both", which="both", left=False, bottom=False)
            ax.set_xlim(x_ref, x_ref + 0.25)
            ax.get_figure().savefig(
                f"./fig/paper/sim_ts_combined_zoom_{c_str}.pdf",
                dpi=300,
                transparent=False,
            )

        h5.close_hot()


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
    ph.plot_raster(h5f, ax, clip_on=True, zorder=-2, markersize=0.75, alpha=0.5, **kwargs)
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
    ph.plot_raster(h5f, ax, clip_on=True, zorder=-2, markersize=1.0, alpha=0.75, **kwargs)
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

    h5.close_hot()

    return fig


def sim_vs_exp_violins(**kwargs):
    dfs = dict()
    dfs["exp"] = load_pd_hdf5("./dat/exp_out/1b.hdf5")
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
    cc.set_size2(ax, 3, 3.5)
    ax.get_figure().savefig(f"./fig/paper/sim_vs_exp_violins_fraction.pdf", dpi=300)

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
    cc.set_size2(ax, 3, 3.5)
    ax.get_figure().savefig(f"./fig/paper/sim_vs_exp_violins_rij.pdf", dpi=300)

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
    cc.set_size2(ax, 3, 3.5)
    ax.get_figure().savefig(f"./fig/paper/sim_vs_exp_violins_ibi.pdf", dpi=300)


def sim_vs_exp_ibi(
    input_path=None, ax=None, simulation_coordinates=reference_coordinates, **kwargs
):
    """
    Plot the inter-burst-interval of the k=5  simulation to compare with the experimental data.
    """
    kwargs = kwargs.copy()

    if input_path is None:
        input_path = "./dat/the_last_one/k_sweep_with_merged.hdf5"

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
    cc.set_size2(ax, 3.5, 2.5)

    ax.get_figure().savefig(f"./fig/paper/ibi_sim_vs_exp.pdf", dpi=300)

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
        plot_kwargs.setdefault("label", f"k={k}")
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
        cc.set_size2(ax2, 4, 3)

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
    if use_compact_size:
        cc.set_size2(ax, 3.5, 2)

    ax.get_figure().savefig(f"./fig/paper/participating_fraction.pdf", dpi=300)

    return ax


def sim_modules_participating_in_bursts(
    input_path,
    simulation_coordinates,
    xlim=[65, 110],
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
    # for 92.5 Hz we only sampled k = 10, hence drop the point
    try:
        selects = np.where((x >= xlim[0]) & (x <= xlim[1]) & (x != 92.5))
    except:
        selects = ...
    # selects = np.ones_like(x, dtype=bool)

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
        ax.fill_between(
            x[selects],
            prev[selects],
            prev[selects] + nxt[selects],
            linewidth=0,
            color=cc.alpha_to_solid_on_bg(clr, 0.2),
            clip_on=True,
        )
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
        ax.set_xlim(xlim)
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
        cc.set_size2(ax, ax_width, 3.0)

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
        ax.get_figure().savefig(f"./fig/paper/sim_violins_fraction_{k}.pdf", dpi=300)

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
        ax.get_figure().savefig(f"./fig/paper/sim_violins_rij_{k}.pdf", dpi=300)

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
                f"./fig/paper/sim_violins_depletion_rij_{k}.pdf", dpi=300
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
        ax.get_figure().savefig(f"./fig/paper/sim_violins_ibi_{k}.pdf", dpi=300)

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
    # cc.set_size2(ax, 3.0, 3.0)
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
    cc.set_size2(ax, 3.0, 3.0)

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
        cc.set_size2(ax, ax_width * 2 / 3, ax_width * 2 / 3)
    else:
        cc.set_size2(ax, ax_width, ax_width)
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
            path = f"./dat/the_last_one/dyn/highres_stim=off_k={k}_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate={rate}.0_rep=001.hdf5"
            try:
                h5f = ah.prepare_file(path)
            except:
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
                lw=0.5,
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

                cc.set_size2(ax, 1.6, 1.0)
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


# ------------------------------------------------------------------------------ #
# Mesoscopic model
# ------------------------------------------------------------------------------ #


def fig_5(dset=None):

    if dset is None:
        try:
            dset = xr.load_dataset("./dat/meso_out/analysed.hdf5")
        except:
            dset = mh.process_data_from_folder("./dat/meso_in/")

    ax = meso_obs_for_all_couplings(dset, "mean_correlation_coefficient")
    ax = meso_module_contribution(dset, coupling=0.1)

    r = 1 # repetition
    for n in [1, 15]:
        for c in dset["coupling"].to_numpy():
            if c == 0.1:
                continue
            input_file = f"./dat/meso_in/coup{c:0.2f}-{r:d}/noise{n}.hdf5"
            coupling, noise, rep = mh._coords_from_file(input_file)
            ax = meso_resource_cycle(input_file)
            ax.set_title(f"coupling={c:.2f}, noise={noise:.3f}")
            # print(f"coupling={c}, noise={noise}")
            # cc.set_size2(ax, 1.6, 1.4) # this is the size of microscopic
            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, 8)
            cc.set_size3(ax, 4, 3)
            sns.despine(ax=ax, trim=True, offset=5)


def meso_obs_for_all_couplings(dset, obs):
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

    dset = xr.load_dataset("./dat/meso_out/analysed.hdf5"
    ax = pp.meso_obs_for_all_couplings(dset, "event_size")
    ```
    """
    ax = None
    for cdx, coupling in enumerate(dset["coupling"].to_numpy()):
        ax = meso_xr_with_errors(
            dset[obs].sel(coupling=coupling),
            ax=ax,
            color=cc.alpha_to_solid_on_bg(
                "#333", cc.fade(cdx, dset["coupling"].size, invert=True)
            ),
            label=f"w = {coupling}",
            zorder=cdx,
        )
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

    cc.set_size2(ax, w=3.0, h=2.2)
    # cc.set_size2(ax, 1.6, 1.4)


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


def meso_resource_cycle(input_file):
    """
    Wrapper to plot a resource cycle for a single file created from the mesoscopic model
    """
    if isinstance(input_file, str):
        h5f = mh.prepare_file(input_file)
        # mh.find_system_bursts_and_module_contributions(h5f)
        # mh.module_contribution(h5f)
        mh.find_system_bursts_and_module_contributions2(h5f)
    else:
        h5f = input_file

    ax = ph.plot_resources_vs_activity(
        h5f, apply_formatting=False, max_traces_per_mod=20, clip_on=False
    )
    ax.set_xlabel("Synaptic resources")
    ax.set_ylabel("Module rate")
    if isinstance(input_file, str):
        ax.set_title(input_file)
    # ax.set_xlim(0, 5)
    # ax.set_ylim(-0.4, 4)
    # cc.set_size3(ax, 3.5, 3)

    # sns.despine(ax=ax, trim=True, offset=5)

    return ax


def meso_sketch_gate_deactivation():
    sys.path.append("./src")
    from mesoscopic_model import gate_deactivation_function

    # currently using probabilities for y. better as rates?
    src_resources = np.arange(0.25, 0.75, 0.01)
    fig, ax = plt.subplots()
    ax.plot(src_resources, gate_deactivation_function(src_resources))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.002))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
    sns.despine(ax=ax, right=True, top=True, trim=True)


def meso_module_contribution(dset=None, coupling=0.3):
    """
    fig 4 c but for mesoscopic model, how many modules contributed to bursting
    events. so this is similar to `sim_modules_participating_in_bursts`
    """

    if dset is None:
        dset = xr.load_dataset("./dat/meso_out/analysed.hdf5")

    ax = sim_modules_participating_in_bursts(
        dset,
        simulation_coordinates=dict(coupling=coupling),
        xlim=None,
        dim1="noise",
        drop_zero_len=False,
    )
    ax.set_xlabel("noise")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))

    cc.set_size2(ax, w=3.0, h=2.2)

    if not show_xlabel:
        ax.set_xlabel("")

    if not show_ylabel:
        ax.set_ylabel("")

    return ax

def meso_activity_snapshot(input_file):

    # get meta data
    # coupling, noise, rep = mh._coords_from_file(input_file)

    h5f = mh.prepare_file(input_file)
    mh.find_system_bursts_and_module_contributions2(h5f)
    ph.overview_dynamic(h5f)





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
    **violin_kwargs,
):

    # log.info(f'|{"":-^75}|')
    log.info(f"## Pooled violins for {observable}")
    # log.info(f'|{"":-^65}|')
    log.info(f"| Condition | 2.5% percentile | 50% percentile | 97.5% percentile |")
    log.info(f"| --------- | --------------- | -------------- | ---------------- |")
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
                func=np.nanmedian,
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
                func=np.nanmean,
                percentiles=[2.5, 50, 97.5],
            )

        log.debug(f"{cat}: median {mid:.3g}, std {std:.3g}")

        p_str = f"| {cat:>9} "
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


# ------------------------------------------------------------------------------ #
# statistical tests
# ------------------------------------------------------------------------------ #


def exp_pairwise_tests_for_trials(observables, layouts=None):
    # observables = ["Mean Fraction", "Mean Correlation", "Functional Complexity"]
    kwargs = dict(
        observables=observables,
        col="Condition",
    )

    if layouts is None:
        layouts = ["1b", "3b", "merged", "KCl_1b", "Bicuculline_1b"]
    for layout in layouts:
        print(f"\n{layout}")
        dfs = load_pd_hdf5(f"./dat/exp_out/{layout}.hdf5")
        df = dfs["trials"]

        if layout in ["1b", "3b", "merged"]:
            _paired_sample_t_test(
                df, col_vals=["pre", "stim"], alternatives="one-sided", **kwargs
            )
            _paired_sample_t_test(
                df, col_vals=["stim", "post"], alternatives="one-sided", **kwargs
            )
            _paired_sample_t_test(
                df, col_vals=["pre", "post"], alternatives="two-sided", **kwargs
            )

        elif layout == "KCl_1b":
            _paired_sample_t_test(
                df, col_vals=["KCl_0mM", "KCl_2mM"], alternatives="one-sided", **kwargs
            )
        elif layout == "Bicuculline_1b":
            _paired_sample_t_test(
                df,
                col_vals=["spon_Bic_20uM", "stim_Bic_20uM"],
                alternatives="one-sided",
                **kwargs,
            )


def _paired_sample_t_test(df, col, col_vals, observables=None, alternatives=None):
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

    print(
        f"paired_sample_t_test for {col_vals}, {len(before)} samples."
        f" p_values:\n{_p_str(p_values, alternatives)}"
    )

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
        print(f"\n{layout}")
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

    print(
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

    print(
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

    print(
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
        print(f"\n{layout}")
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
