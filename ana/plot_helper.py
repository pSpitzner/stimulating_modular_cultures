# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-09 11:16:44
# @Last Modified: 2021-06-24 11:07:38
# ------------------------------------------------------------------------------ #
# All the plotting is in here.
#
# What's a good level of abstraction?
# * Basic routines that plot one thing or the other, directly from file.
# * target an mpl ax element with normal functions and
# * provide higher level ones that combine those to `overview` panels
# ------------------------------------------------------------------------------ #


# fmt: off
import os
import sys
import glob
import h5py
import argparse
import logging

import matplotlib
# matplotlib.rcParams['font.sans-serif'] = "Arial"
# matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
# matplotlib.rcParams["axes.spines.right"] = False
# matplotlib.rcParams["axes.spines.top"] = False
# matplotlib.rcParams["axes.spines.left"] = False
# matplotlib.rcParams["axes.spines.bottom"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
    "#233954", "#ea5e48", "#1e7d72", "#f49546", "#e8bf58", # dark
    "#5886be", "#f3a093", "#53d8c9", "#f2da9c", "#f9c192", # light
    ]) # qualitative, somewhat color-blind friendly, in mpl words 'tab5'


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from tqdm import tqdm
from brian2.units.allunits import *

log = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))

# import logisi as logisi
# requires my python helpers https://github.com/pSpitzner/pyhelpers
import hi5 as h5
# from hi5 import BetterDict
from benedict import benedict
from addict import Dict
import colors as cc
import ana_helper as ah

# fmt: on


# ------------------------------------------------------------------------------ #
# overview panels
# ------------------------------------------------------------------------------ #


def overview_topology(h5f, filenames=None, skip_graph=False):
    """
        Plots an overview figure of the topology of `h5f`.

        If `filenames` are provided, they are passed to the underlying plot
        functions and, where possible, ensemble statistics across filenames are
        used.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[12, 8])

    if not "ana" in h5f.keypaths():
        ah.prepare_file(h5f)

    plot_distribution_dendritic_tree_size(h5f, axes[0, 0], filenames=filenames)
    plot_distribution_axon_length(h5f, axes[1, 0], filenames=filenames)
    plot_distribution_degree_k(h5f, axes[1, 1], filenames=filenames)
    plot_parameter_info(h5f, axes[0, 1])

    if not skip_graph:
        plot_connectivity_layout(h5f, axes[0, 2])
        plot_axon_layout(h5f, axes[1, 2])
        axes[0, 2].set_title("Connectivity")
        axes[1, 2].set_title("Axons")
        # share zoom between connectivity and axons
        axes[0, 2].get_shared_x_axes().join(axes[0, 2], axes[1, 2])
        axes[0, 2].get_shared_y_axes().join(axes[0, 2], axes[1, 2])
        axes[1, 2].set_aspect(1)
        axes[1, 2].autoscale()


    fig.tight_layout()

    return fig


def overview_dynamic(h5f, filenames=None):

    # fig, axes = plt.subplots(4, 1, sharex=True, figsize=(4, 7))
    fig = plt.figure(figsize=(4, 7))
    axes = []
    gs = fig.add_gridspec(4, 2)  # [row, column]
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[1, :]))
    axes.append(fig.add_subplot(gs[2, :], sharex=axes[1]))
    axes.append(fig.add_subplot(gs[3, :], sharex=axes[1]))
    axes.append(fig.add_subplot(gs[0, 0]))

    if not "ana" in h5f.keypaths():
        ah.prepare_file(h5f)

    plot_parameter_info(h5f, axes[0])
    plot_raster(h5f, axes[1])
    plot_module_rates(h5f, axes[2])
    plot_system_rate(h5f, axes[2])
    plot_bursts_into_timeseries(h5f, axes[3])
    ax = plot_initiation_site(h5f, axes[4])
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())

    axes[1].set_xlabel("")
    axes[2].set_xlabel("")

    fig.tight_layout()

    return fig


def overview_burst_duration_and_isi(h5f, filenames=None, which="all"):
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(4, 6), gridspec_kw=dict(height_ratios=[1, 3, 3]),
    )
    plot_distribution_burst_duration(h5f, ax=axes[1], filenames=filenames)
    plot_distribution_isi(h5f, ax=axes[2], filenames=filenames, which=which)
    comments = [
        ["hist bin size", "1ms"],
        ["rate bin size", f'{h5f[f"ana.rates.dt"]*1000}ms'],
    ]
    plot_parameter_info(h5f, ax=axes[0], add=comments)

    axes[1].set_xlim(0, 0.4)
    axes[2].set_xlim(-3, 3)

    for i in range(4):
        fig.tight_layout()

    return fig


# ------------------------------------------------------------------------------ #
# main plot level functions
# ------------------------------------------------------------------------------ #


def plot_raster(h5f, ax=None, apply_formatting=True, sort_by_module=True):
    """
        Plot a raster plot

        # Parameters:
        h5f : BetterDict
            of a loaded hdf5 file with the conventional entries.
            call `prepare_file(h5f)` to set everything up!
        sort_by_module : bool
            set to False to avoid reordering by module
        apply_formatting : bool
            if false, no styling changes will be done to ax
    """

    assert "ana" in h5f.keypaths(), "`prepare_file(h5f)` before plotting!"

    log.info("Plotting Raster")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.set_rasterization_zorder(-1)

    for n_id in h5f["ana.neuron_ids"]:

        if sort_by_module:
            n_id_sorted = h5f["ana.mod_sort"](n_id)
        else:
            n_id_sorted = n_id

        m_id = h5f["data.neuron_module_id"][n_id]
        spikes = h5f["data.spiketimes"][n_id]

        ax.plot(
            spikes,
            n_id_sorted * np.ones(len(spikes)),
            "|",
            alpha=0.5,
            zorder=-2,
            color=h5f["ana.mod_colors"][m_id],
        )

    if apply_formatting:
        ax.margins(x=0, y=0)
        ax.set_ylabel("Raster")
        ax.set_xlabel("Time [seconds]")
        fig.tight_layout()

    return ax


def plot_module_rates(h5f, ax=None, apply_formatting=True, mark_bursts=True):

    assert "ana" in h5f.keypaths(), "`prepare_file(h5f)` before plotting!"

    log.info("Plotting Module Rates")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if not "ana.rates" in h5f.keypaths():
        ah.find_bursts_from_rates(h5f)

    dt = h5f["ana.rates.dt"]

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        pop_rate = h5f[f"ana.rates.module_level"][m_dc]
        # log.info(f"Threshold from SNR: {ah.get_threshold_via_signal_to_noise_ratio(pop_rate)}")
        log.info(f'CV {m_id}: {h5f[f"ana.rates.cv.module_level"][m_dc]:.3f}')
        mean_rate = np.nanmean(pop_rate)
        ax.plot(
            np.arange(0, len(pop_rate)) * dt,
            pop_rate,
            label=f"{m_id:d}: {mean_rate:.2f} Hz",
            color=h5f[f"ana.mod_colors"][m_id],
        )

    if mark_bursts:
        try:
            ax.axhline(
                y=h5f["ana.bursts.module_level"][0].rate_threshold, ls=":", color="gray"
            )
        except:
            pass

    leg = ax.legend(loc=1)

    if apply_formatting:
        leg.set_title("Module Rates")
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("#e4e5e6")
        leg.get_frame().set_alpha(0.95)

        ax.margins(x=0, y=0)
        ax.set_ylabel("Rates [Hz]")
        ax.set_xlabel("Time [seconds]")

        fig.tight_layout()

    return ax


def plot_system_rate(h5f, ax=None, apply_formatting=True):
    assert "ana" in h5f.keypaths(), "`prepare_file(h5f)` before plotting!"

    log.info("Plotting System Rate")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if not "ana.rates" in h5f.keypaths():
        ah.find_bursts_from_rates(h5f)

    dt = h5f["ana.rates.dt"]

    pop_rate = h5f["ana.rates.system_level"]
    mean_rate = np.nanmean(pop_rate)
    ax.plot(
        np.arange(0, len(pop_rate)) * dt,
        pop_rate,
        label=f"system: {mean_rate:.2f} Hz",
        color="black",
    )
    log.info(f'CV  : {h5f["ana.rates.cv.system_level"]:.3f}')

    leg = ax.legend(loc=1)

    if apply_formatting:
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("#e4e5e6")
        leg.get_frame().set_alpha(0.95)

        ax.margins(x=0, y=0)
        ax.set_ylabel("Rates [Hz]")
        ax.set_xlabel("Time [seconds]")

        fig.tight_layout()

    return ax


def plot_bursts_into_timeseries(h5f, ax=None, apply_formatting=True):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    pad = 3
    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        beg_times = h5f[f"ana.bursts.module_level.{m_dc}.beg_times"]
        end_times = h5f[f"ana.bursts.module_level.{m_dc}.end_times"]
        ax.plot(
            beg_times,
            np.ones(len(beg_times)) * (pad + 1 + m_id),
            marker="4",
            color=h5f["ana.mod_colors"][m_id],
            lw=0,
        )
        ax.plot(
            end_times,
            np.ones(len(end_times)) * (pad + 1 + m_id),
            marker="3",
            color=h5f["ana.mod_colors"][m_id],
            lw=0,
        )

    beg_times = h5f["ana.bursts.system_level.beg_times"]
    end_times = h5f["ana.bursts.system_level.end_times"]

    log.info(f"Found {len(beg_times)} system-wide bursts")

    ax.plot(beg_times, np.ones(len(beg_times)) * (pad), marker="4", color="black", lw=0)
    ax.plot(end_times, np.ones(len(end_times)) * (pad), marker="3", color="black", lw=0)

    ax.set_ylim(0, len(h5f["ana.mods"]) + 2 * pad)

    if apply_formatting:
        ax.margins(x=0, y=0)
        ax.set_ylabel("Bursts")
        ax.set_xlabel("Time [seconds]")
        ax.set_yticks([])
        fig.tight_layout()

    return ax


def plot_parameter_info(h5f, ax=None, apply_formatting=True, add=[]):

    log.info("Plotting Parameter Info")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if apply_formatting:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.margins(x=0, y=0)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title("Parameters")

    dat = []
    dat.append(["Connections k", h5f["meta.topology_k_inter"]])
    dat.append(["Stimulation", h5f["ana.stimulation_description"]])

    try:
        dat.append(["", ""])
        # dat.append(["Noise Rate [Hz]", h5f["meta.dynamics_rate"]])
        dat.append(["AMPA [mV]", h5f["meta.dynamics_jA"]])
        dat.append(["GABA [mV]", h5f["meta.dynamics_jG"]])
        dat.append(
            [
                "E/I",
                len(h5f["data.neuron_excitatiory_ids"])
                / len(h5f["data.neuron_inhibitory_ids"]),
            ]
        )
    except:
        pass

    if len(add) > 0:
        dat.append(["", ""])
        for el in add:
            assert len(el) == 2
            dat.append(el)

    left = ""
    right = ""
    for d in dat:
        left += f"{d[0]}\n"
        right += f"{d[1]}\n"
        if d[0] != "" and d[1] != "":
            log.info(f"{d[0]:>22}:    {d[1]}")

    # tables stretch rows of axis, so manually place some text instead
    # tab = ax.table(dat, loc="center", edges="open")
    # cells = tab.properties()["celld"]
    # for i in range(0, int(len(cells) / 2)):
    #     cells[i, 1]._loc = "left"

    # ax.plot([0,1], [0,1], ls=None)

    ax.text(
        0.46,
        0.9,
        left,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.54,
        0.9,
        right,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    if apply_formatting:
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()

    return ax


def plot_initiation_site(h5f, ax=None, apply_formatting=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_ylabel("Histogram")
    ax.set_xlabel("Initiation module")

    if not "ana.bursts" in h5f.keypaths():
        ah.find_bursts_from_rates(h5f)

    sequences = h5f["ana.bursts.system_level.module_sequences"]
    if len(sequences) == 0:
        return ax

    first_mod = np.ones(len(sequences), dtype=int) - 1
    for idx, seq in enumerate(sequences):
        first_mod[idx] = seq[0]

    # unique = np.sort(np.unique(first_mod))
    unique = np.array([0, 1, 2, 3])
    bins = (unique - 0.5).tolist()
    bins = bins + [bins[-1] + 1]

    N, bins, patches = ax.hist(first_mod, bins=bins)
    # assume they are ordered correctly
    try:
        for idx, patch in enumerate(patches):
            patches[idx].set_facecolor(h5f["ana.mod_colors"][idx])
    except Exception as e:
        log.debug(e)

    fig.tight_layout()

    return ax


# ------------------------------------------------------------------------------ #
# plots of pandas dataframes
# ------------------------------------------------------------------------------ #


def pd_sequence_length(df):

    df_res = None
    # prequery
    for query in tqdm(
        [
            "`Bridge weight` == 1 & `Connections` == 1 & `Number of inhibitory neurons` == 0",
            "`Bridge weight` == 1 & `Connections` == 2 & `Number of inhibitory neurons` == 0",
            "`Bridge weight` == 1 & `Connections` == 3 & `Number of inhibitory neurons` == 0",
            "`Bridge weight` == 1 & `Connections` == 5 & `Number of inhibitory neurons` == 0",
        ],
        desc="Queries",
        leave=False,
    ):
        # as hue value, we want to compare stimulation scenarios, need concat
        dfc = None
        for sq in [
            "`Stimulation` == 'Off'",
            "`Stimulation` == 'On (0)'",
            "`Stimulation` == 'On (0, 2)'",
            "`Stimulation` == 'On (0, 1, 2)'",
            "`Stimulation` == 'On (0, 1, 2, 3)'",
        ]:
            dfq = df.query(query + " & " + sq)
            if dfq.shape[0] == 0:
                log.debug(f"Skipping {sq}, no rows.")
                continue
            # calculate the probability of sequence length for every repetition
            dfq = ah.sequence_length_histogram_from_pd_df(
                dfq,
                keepcols=[
                    "Stimulation",
                    "Bridge weight",
                    "Connections",
                    "Number of inhibitory neurons",
                ],
            )
            dfc = pd.concat([dfc, dfq], ignore_index=True)
        try:
            plot_pd_boxplot(dfc, x="Sequence length", y="Probability")
        except Exception as e:
            log.error(e)

        # for returning everything
        df_res = pd.concat([df_res, dfc], ignore_index=True)

    return df_res


def pd_burst_duration(df):

    x = "Stimulation"
    y = "Duration"
    df = df.query("Connections > 0")
    # df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 2)'")
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 1"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 2"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 3"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 5"), x, y)

    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 1"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 2"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 3"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 5"), x, y)

    x = "Connections"
    # df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 2)'")
    df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 1, 2, 3)'")
    df2 = df2.query("`Number of inhibitory neurons` > 0")
    ax = plot_pd_violin(
        df2.query("`Bridge weight` == 1 & `Sequence length` == 1"), x, y
    )
    ax.set_title("L=1", loc="left")
    ax = plot_pd_violin(
        df2.query("`Bridge weight` == 1 & `Sequence length` == 2"), x, y
    )
    ax.set_title("L=2", loc="left")
    ax = plot_pd_violin(df2.query("`Bridge weight` == 1 & `Sequence length` > 1"), x, y)
    ax.set_title("L>1", loc="left")
    # plot_pd_violin(df2.query("`Bridge weight` == 0.75"), x, y)
    # plot_pd_violin(df2.query("`Bridge weight` == 0.5"), x, y)

    # df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 1, 2, 3)'")
    # plot_pd_violin(df2.query("`Bridge weight` == 1"), x, y)
    # plot_pd_violin(df2.query("`Bridge weight` == 0.75"), x, y)
    # plot_pd_violin(df2.query("`Bridge weight` == 0.5"), x, y)

    return df


def pd_burst_duration_change_under_stim(df):
    """
    df from `ah.batch_pd_bursts`
    """

    fig, ax = plt.subplots()
    ax.set_title("L=all, Inh, k=5, bw=1")

    # query so what remains is repetitions and the condition to compare
    df1 = df.query(
        "`Sequence length` >= 1 & `Number of inhibitory neurons` > 0 & `Connections` == 5 & `Bridge weight` == 1.0 & (Stimulation == 'Off' | Stimulation == 'On (0, 2)')"
    )
    # use the mean duration as the observable, to which contributions are exactly
    # one data point from one realization, and
    # compare between different `stimulation` conditions
    df2 = df1.groupby(["Stimulation", "Repetition"])["Duration"].mean().reset_index()

    # ax = plot_pd_violin(df2, x="Stimulation", y="Duration", split=False)
    # ax = plot_pd_boxplot(df2, x="Stimulation", y="Duration")
    palette = _stimulation_color_palette()
    order = [s for s in palette.keys() if s in df2["Stimulation"].unique()]
    sns.pointplot(
        data=df2,
        x="Stimulation",
        y="Duration",
        order=order,
        ax=ax,
        hue="Repetition",
        scale=0.5,
        color=".4",
    )
    sns.violinplot(
        data=df2,
        x="Stimulation",
        y="Duration",
        hue="Stimulation",
        split=True,
        order=order,
        ax=ax,
        palette=palette,
    )
    # sns.boxplot(data=df2, x="Stimulation", y="Duration", order=order, ax=ax, fliersize=0, palette=palette)
    # sns.swarmplot(data=df2, x="Stimulation", y="Duration", order=order, ax=ax, color=".4", size=4)
    ax.get_legend().set_visible(False)

    ax.set_ylim(0.0, 0.3)
    fig.tight_layout()

    return df2


def plot_pd_boxplot(
    df, x="Sequence length", y="Probability", ax=None, apply_formatting=True
):
    """
        Boxplot across stimulation conditions.
        `df` needs to be filtered already:
        ```
        data = df.loc[((df["Connections"] == k) & (df["Bridge weight"] == bw))]
        ```
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    palette = _stimulation_color_palette()

    order = None
    hue_order = None
    if x == "Stimulation":
        order = palette.keys()
    if x == "Connections":
        hue_order = []
        stim_of_df = df["Stimulation"].unique()
        for stim in palette.keys():
            if stim in stim_of_df:
                hue_order.append(stim)

    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue="Stimulation",
        order=order,
        hue_order=hue_order,
        fliersize=0.5,
        linewidth=0.5,
        ax=ax,
        palette=palette,
    )

    if apply_formatting:

        # if those columns are equal across rows, print it as title
        try:
            bw = df["Bridge weight"].to_numpy()
            if np.all(bw[0] == bw):
                ax.set_title(f"bw = {bw[0]}", loc="right")
        except:
            pass
        try:
            k = df["Connections"].to_numpy()
            if np.all(k[0] == k):
                ax.set_title(f"k = {k[0]}", loc="left")
        except:
            pass

        ax.legend()
        ax.get_legend().set_visible(False)
        ax.axhline(0, ls=":", color="black", lw=0.75, zorder=-1)
        ax.axhline(1, ls=":", color="black", lw=0.75, zorder=-1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.tick_params(axis="x", which="major", length=0)
        fig.tight_layout()

    ax.set_ylim(-0.01, 1.01)
    # ax.set_ylim(-.01,.2)

    return ax


def plot_pd_violin(df, x, y, ax=None, apply_formatting=True, split=True):
    """
        Violinplot across stimulation conditions.
        `df` needs to be filtered already:
        ```
        data = df.loc[((df["Connections"] == k) & (df["Bridge weight"] == bw))]
        ```
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # make a nice color map
    c0 = "#1f77b4"  # matplotlib.cm.get_cmap("tab10").colors[0] # blue
    c1 = "#ff7f0e"  # matplotlib.cm.get_cmap("tab10").colors[1] # orange

    palette = {
        "Off": c0,
        "On (0)": cc.alpha_to_solid_on_bg(base=c1, alpha=0.25, bg="white"),
        "On (0, 2)": cc.alpha_to_solid_on_bg(base=c1, alpha=0.50, bg="white"),
        "On (0, 1, 2)": cc.alpha_to_solid_on_bg(base=c1, alpha=0.75, bg="white"),
        "On (0, 1, 2, 3)": cc.alpha_to_solid_on_bg(base=c1, alpha=1.00, bg="white"),
    }

    order = None
    hue_order = None
    if x == "Stimulation":
        order = palette.keys()
    if x == "Connections":
        hue_order = []
        stim_of_df = df["Stimulation"].unique()
        for stim in palette.keys():
            if stim in stim_of_df:
                hue_order.append(stim)

    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue="Stimulation",
        hue_order=hue_order,
        order=order,
        size=0.3,
        edgecolor=(0, 0, 0, 0),
        linewidth=0.5,
        dodge=True,
        ax=ax,
        palette=palette,
    )

    sns.violinplot(
        data=df,
        x=x,
        y=y,
        hue="Stimulation",
        order=order,
        hue_order=hue_order,
        scale_hue=False,
        scale="area",
        split=split,
        inner="quartile",
        linewidth=0.5,
        ax=ax,
        palette=palette,
    )

    if apply_formatting:

        # if those columns are equal across rows, print it as title
        try:
            bw = df["Bridge weight"].to_numpy()
            if np.all(bw[0] == bw):
                ax.set_title(f"bw = {bw[0]}", loc="right")
        except:
            pass
        try:
            k = df["Connections"].to_numpy()
            if np.all(k[0] == k):
                ax.set_title(f"k = {k[0]}", loc="left")
        except:
            pass

        ax.legend()
        ax.get_legend().set_visible(False)
        ax.axhline(0, ls=":", color="black", lw=0.75, zorder=-1)
        ax.axhline(1, ls=":", color="black", lw=0.75, zorder=-1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.tick_params(axis="x", which="major", length=0)
        fig.tight_layout()

    ax.set_ylim(-0.01, 0.5)
    # ax.set_ylim(-.01,.2)

    return ax


def plot_pd_seqlen_vs_burst_duration(df, ax=None, apply_formatting=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sns.scatterplot(
        x="Sequence length", y="Duration", hue="Sequence length", data=df, ax=ax,
    )

    if apply_formatting:
        ax.legend()
        ax.get_legend().set_visible(False)
        fig.tight_layout()

    return ax


def _stimulation_color_palette():

    c0 = "#1f77b4"  # matplotlib.cm.get_cmap("tab10").colors[0] # blue
    c1 = "#ff7f0e"  # matplotlib.cm.get_cmap("tab10").colors[1] # orange

    palette = {
        "Off": c0,
        "On (0)": cc.alpha_to_solid_on_bg(base=c1, alpha=0.25, bg="white"),
        "On (0, 2)": cc.alpha_to_solid_on_bg(base=c1, alpha=0.50, bg="white"),
        "On (0, 1, 2)": cc.alpha_to_solid_on_bg(base=c1, alpha=0.75, bg="white"),
        "On (0, 1, 2, 3)": cc.alpha_to_solid_on_bg(base=c1, alpha=1.00, bg="white"),
    }

    return palette


# ------------------------------------------------------------------------------ #
# distributions
# ------------------------------------------------------------------------------ #


def plot_distribution_burst_duration(
    h5f, ax=None, apply_formatting=True, filenames=None
):

    if not "ana" in h5f.keypaths():
        ah.prepare_file(h5f)

    if not "ana.bursts" in h5f.keypaths():
        log.info("Finding Bursts from Rates")
        ah.find_bursts_from_rates(h5f)

    if filenames is None:
        log.info("Plotting Bursts")
        bursts = h5f["ana.bursts"]
    else:
        log.info("Gathering Bursts across provided filenames")
        # merge across files, and save the result
        if not "ana.ensemble" in h5f.keypaths():
            h5f["ana.ensemble"] = benedict()
            h5f["ana.ensemble.filenames"] = filenames
            ens = ah.batch_candidates_burst_times_and_isi(filenames, hot=True)
            h5f["ana.ensemble.bursts"] = ens["ana.bursts"]
            h5f["ana.ensemble.isi"] = ens["ana.isi"]
            del ens

        assert (
            h5f["ana.ensemble.filenames"] == filenames
        ), "`filenames` dont match. Using multiple ensembles is not implemented yet."
        bursts = h5f["ana.ensemble.bursts"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kwargs = {
        "ax": ax,
        "kde": False,
        # "binwidth": 1 / 1000,  # ms, beware the timestep of burst analysis (usually 2ms)
        # "binrange": (0.06, 0.12),
        "stat": "density",  # must use density when comparing across different binsizes!
        # 'multiple' : 'stack',
        "element": "poly",
    }

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        beg_times = np.array(bursts[f"module_level.{m_dc}.beg_times"])
        end_times = np.array(bursts[f"module_level.{m_dc}.end_times"])
        sns.histplot(
            data=end_times - beg_times,
            binwidth=1 / 1000,
            color=h5f["ana.mod_colors"][m_id],
            alpha=0.2,
            **kwargs,
            label=m_dc,
        )

    beg_times = np.array(bursts["system_level.beg_times"])
    end_times = np.array(bursts["system_level.end_times"])
    sns.histplot(
        data=end_times - beg_times,
        binwidth=1 / 1000,
        color="black",
        alpha=0,
        **kwargs,
        label="System-wide",
    )

    if apply_formatting:
        ax.set_xlabel(r"Burst duration $D$ (seconds)")
        ax.set_ylabel(r"Prob. density $P(D)$")
        ax.legend()
        fig.tight_layout()

    return ax


def plot_distribution_isi(
    h5f, ax=None, apply_formatting=True, log_binning=True, which="all", filenames=None,
):
    """
        Plot the inter spike intervals in h5f.

        # Parameters
        h5f : BetterDict
            if `h5f.ana.isi` is not set, it will be calculated (and added to the file)
        which : str
            `all`, `in_bursts` or `out_bursts`. which isi to focus on, default `all`.

        log_binning : bool, True
            set to false for linear bins

        filenames : str
            if `filenames` are provided, bursts an
    """
    assert which in ["all", "in_bursts", "out_bursts"]

    if h5f["ana.isi"] is None:
        log.info("Finding ISIS")
        assert which == "all", "`find_bursts` before plotting isis other than `all`"
        if not "ana.bursts" in h5f.keypaths():
            ah.find_bursts_from_rates(h5f)
        ah.find_isis(h5f)

    if filenames is None:
        log.info("Plotting ISI")
        isi = h5f["ana.isi"]
    else:
        log.info("Gathering ISI across provided filenames")
        # merge across files, and save the result
        if not "ana.ensemble" in h5f.keypaths():
            h5f["ana.ensemble"] = benedict()
            h5f["ana.ensemble.filenames"] = filenames
            ens = ah.batch_candidates_burst_times_and_isi(filenames, hot=True)
            h5f["ana.ensemble.bursts"] = ens["ana.bursts"]
            h5f["ana.ensemble.isi"] = ens["ana.isi"]
            del ens

        assert (
            h5f["ana.ensemble.filenames"] == filenames
        ), "`filenames` dont match. Using multiple ensembles is not implemented yet."
        isi = h5f["ana.ensemble.isi"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # we want to use the same bins across modules
    max_isi = 0
    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        assert len(isi[m_dc][which]) > 0, f"No isis found for '{which}'"
        mod_isi = isi[m_dc][which]
        this_max_isi = np.max(mod_isi)
        if this_max_isi > max_isi:
            max_isi = this_max_isi
        try:
            log.info(
                f'Mean ISI, in bursts, {m_dc}: {np.mean(isi[f"{m_id}.in_bursts"])*1000:.1f} (ms)'
            )
        except:
            pass

    log.info(f"Largest ISI: {max_isi:.3f} (seconds)")

    # do log binning manuall -> first transform to log scale, then apply the histogram
    # and (if desired) manually relabel.
    # Beware, probability is p(log10 isi) but if relabeling xaxis, we dont show
    # log isi on a linear scale (for readability). this might be confusing.
    kwargs = {
        "ax": ax,
        "kde": False,
        # "binwidth": 2.5 / 1000,  # ms
        # "binrange": (0.06, 0.12),
        "bins": np.linspace(-3, 3, num=200),
        # "stat": "density",
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "poly",
    }

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        sns.histplot(
            data=np.log10(isi[m_dc][which]),
            color=h5f["ana.mod_colors"][m_id],
            alpha=0.2,
            **kwargs,
            label=f"{m_dc} ({which})",
        )

    if apply_formatting:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(_ticklabels_lin_to_log10_power)
        )
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(_ticklocator_lin_to_log_minor())
        # maybe skip the relabeling and just set xlabel to "log10 ISI (log10 seconds)"
        ax.set_xlabel(r"Inter-spike Interval (seconds)")
        ax.set_ylabel(r"Prob. Density (log ISI)")
        ax.legend()
        fig.tight_layout()

    return ax


def plot_distribution_degree_k(h5f, ax=None, apply_formatting=True, filenames=None):
    """
        Plot the distribution of the in-and out degree for `h5f`.

        if `filenames` is provided, distribution data is accumulated from all files,
        and only styling/meta information is used from `h5f`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if filenames is None:
        k_in = h5f["data.neuron_k_in"][:]
        k_out = h5f["data.neuron_k_out"][:]
    else:
        log.info("Loading multiple files to plot degree distributions")
        k_in = h5.load(filenames, "/data/neuron_k_in")
        k_out = h5.load(filenames, "/data/neuron_k_out")
        k_in = np.concatenate(k_in).ravel()
        k_out = np.concatenate(k_out).ravel()

    maxbin = np.nanmax([k_in, k_out])
    br = np.arange(0, maxbin, 1)

    kwargs = {
        "ax": ax,
        "kde": True,
        "bins": br,
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "step",
        "alpha": 0.25,
    }

    sns.histplot(
        k_in,
        label=r"$k_{in}$",
        color=matplotlib.cm.get_cmap("tab10").colors[0],
        **kwargs,
    )
    sns.histplot(
        k_out,
        label=r"$k_{out}$",
        color=matplotlib.cm.get_cmap("tab10").colors[1],
        **kwargs,
    )

    if apply_formatting:
        ax.set_xlabel(f"Degree $k$ (per neuron)")
        ax.set_ylabel(f"Probability $p(k)$")
        ax.set_title("Degree distribution")
        ax.legend(loc="upper right")

        ax.text(
            0.95,
            0.7,
            f"median:\n" + r"$k_{in} \sim$" + f"{np.nanmedian(k_in):g}\n"
            r"$k_{out} \sim$"
            + f"{np.nanmedian(k_out):g}\n\n"
            + f"mean:\n"
            + r"$k_{in} \sim$"
            + f"{np.nanmean(k_in):g}\n"
            r"$k_{out} \sim$" + f"{np.nanmean(k_out):g}\n",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    return ax


def plot_distribution_axon_length(h5f, ax=None, apply_formatting=True, filenames=None):
    """
        Plot the distribution of the axon length for `h5f`.

        if `filenames` is provided, distribution data is accumulated from all files,
        and only styling/meta information is used from `h5f`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if filenames is None:
        alen = h5f["data.neuron_axon_length"][:]
        aete = h5f["data.neuron_axon_end_to_end_distance"][:]
    else:
        log.info("Loading multiple files for axon length distribution")
        alen = h5.load(filenames, "/data/neuron_axon_length")
        aete = h5.load(filenames, "/data/neuron_axon_end_to_end_distance")
        alen = np.concatenate(alen).ravel()
        aete = np.concatenate(aete).ravel()

    kwargs = {
        "ax": ax,
        "kde": True,
        "stat": "probability",
        "element": "step",
        "alpha": 0.25,
    }

    sns.histplot(
        alen, label="length", color=matplotlib.cm.get_cmap("tab10").colors[0], **kwargs
    )
    sns.histplot(
        aete,
        label="end to end",
        color=matplotlib.cm.get_cmap("tab10").colors[1],
        **kwargs,
    )

    if apply_formatting:
        ax.set_xlabel(f"Length $l\,[\mu m]$")
        ax.set_ylabel(f"Probability $p(l)$")
        ax.set_title("Axon length distribution")
        ax.legend()
        fig.tight_layout()

    return ax


def plot_distribution_dendritic_tree_size(
    h5f, ax=None, apply_formatting=True, filenames=None
):
    """
        Plot the distribution of the size of the dendritic tree for `h5f`.

        if `filenames` is provided, distribution data is accumulated from all files,
        and only styling/meta information is used from `h5f`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if filenames is None:
        n_R_d = h5f["data.neuron_radius_dendritic_tree"][:]
    else:
        log.info("Loading multiple files for distribution of dendritic tree size")
        n_R_d = h5.load(filenames, "/data/neuron_radius_dendritic_tree")
        n_R_d = np.concatenate(n_R_d).ravel()

    kwargs = {
        "ax": ax,
        "kde": True,
        "stat": "probability",
        "element": "step",
        "alpha": 0.25,
    }

    sns.histplot(n_R_d, label="dendritic radius", color="gray", **kwargs)

    if apply_formatting:
        ax.set_xlabel(f"Radius $r\,[\mu m]$")
        ax.set_ylabel(f"Probability $p(r)$")
        ax.set_title("Dendritic tree size")
        fig.tight_layout()

    return ax


# ------------------------------------------------------------------------------ #
# topology
# ------------------------------------------------------------------------------ #


def plot_axon_layout(h5f, ax=None, apply_formatting=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    _plot_soma(h5f, ax)
    _plot_axons(h5f, ax)

    if apply_formatting:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_aspect(1)
        ax.set_xlabel(f"Position $l\,[\mu m]$")
        fig.tight_layout()


def plot_connectivity_layout(h5f, ax=None, apply_formatting=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if "ana.networkx" in h5f.keypaths():
        G = h5f["ana.networkx.G"]
        pos = h5f["ana.networkx.pos"]
    else:
        # add nodes, generate positions in usable format
        G = nx.DiGraph()
        G.add_nodes_from(h5f["ana.neuron_ids"])
        pos = dict()
        for idx, n in enumerate(h5f["ana.neuron_ids"]):
            pos[n] = (h5f["data.neuron_pos_x"][idx], h5f["data.neuron_pos_y"][idx])

        # add edges
        G.add_edges_from(h5f["data.connectivity_matrix_sparse"][:])

        # add to h5f
        h5f["ana.networkx"] = benedict()
        h5f["ana.networkx.G"] = G
        h5f["ana.networkx.pos"] = pos

    log.debug("Drawing graph nodes")
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_size=0,  # we draw them with fixed size using _circles
        node_color="black",
        # with_labels=False,
    )
    log.debug("Drawing graph edges")
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color="black", arrows=False, width=0.1
    )
    log.debug("Drawing soma")
    _plot_soma(h5f, ax)

    if apply_formatting:
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect(1)
        fig.tight_layout()


def _plot_soma(h5f, ax, n_R_s=7.5):
    n_x = h5f["data.neuron_pos_x"][:]
    n_y = h5f["data.neuron_pos_y"][:]
    _circles(n_x, n_y, n_R_s, ax=ax, fc="white", ec="black", alpha=1, lw=0.25, zorder=4)
    # _circles(n_x, n_y, n_R_s, ax=ax, fc="white", ec="none", alpha=1, lw=0.5, zorder=4)
    # _circles(n_x, n_y, n_R_s, ax=ax, fc="none", ec="black", alpha=0.3, lw=0.5, zorder=5)


def _plot_axons(h5f, ax):
    # axon segments
    # zero-or-nan-padded 2d arrays
    seg_x = h5f["data.neuron_axon_segments_x"][:]
    seg_y = h5f["data.neuron_axon_segments_y"][:]
    seg_x = np.where(seg_x == 0, np.nan, seg_x)
    seg_y = np.where(seg_y == 0, np.nan, seg_y)

    # iterate over neurons to plot axons
    for n in range(len(seg_x)):
        m_id = h5f["data.neuron_module_id"][n]
        clr = h5f["ana.mod_colors"][m_id]
        ax.plot(seg_x[n], seg_y[n], color=clr, lw=0.35, zorder=0, alpha=0.5)


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def _ticklabels_lin_to_log10(x, pos):
    """
        converts ticks of manually logged data (lin ticks) to log ticks, as follows
         1 -> 10
         0 -> 1
        -1 -> 0.1

        # Example
        ```
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(_ticklabels_lin_to_log10_power)
        )
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(_ticklocator_lin_to_log_minor())
        ```
    """
    prec = int(np.ceil(-np.minimum(x, 0)))
    return "{{:.{:1d}f}}".format(prec).format(np.power(10.0, x))


def _ticklabels_lin_to_log10_power(x, pos, nicer=True, nice_range=[-1, 0, 1]):
    """
        converts ticks of manually logged data (lin ticks) to log ticks, as follows
         1 -> 10^1
         0 -> 10^0
        -1 -> 10^-1
    """
    if x.is_integer():
        # use easy to read formatter if exponents are close to zero
        if nicer and x in nice_range:
            return _ticklabels_lin_to_log10(x, pos)
        else:
            return r"$10^{{{:d}}}$".format(int(x))
    else:
        # return r"$10^{{{:f}}}$".format(x)
        return ""


def _ticklocator_lin_to_log_minor(vmin=-10, vmax=10, nbins=10):
    """
        get minor ticks on when manually converting lin to log
    """
    locs = []
    orders = int(np.ceil(vmax - vmin))
    for o in range(int(np.floor(vmin)), int(np.floor(vmax + 1)), 1):
        locs.extend([o + np.log10(x) for x in range(2, 10)])
    return matplotlib.ticker.FixedLocator(locs, nbins=nbins * orders)


# circles in data scale
# https://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker
def _circles(x, y, s, c="b", vmin=None, vmax=None, ax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None
    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection
