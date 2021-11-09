# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-11-08 17:51:24
# @Last Modified: 2021-11-09 19:20:04
# ------------------------------------------------------------------------------ #
# collect the functions to create figure panels here
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
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

import plot_helper as ph
import ana_helper as ah
import ndim_helper as nh
import colors as cc

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
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 6
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 150


colors = dict()
colors["pre"] = "#335c67"
colors["stim"] = "#e09f3e"
colors["post"] = "#9e2a2b"
colors["KCl_0mM"] = "gray"
colors["KCl_2mM"] = "gray"

simulation_coordinates = dict()
simulation_coordinates["jA"] = 45
simulation_coordinates["jG"] = 50
simulation_coordinates["tD"] = 8

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

# Fig 1
def experimental_raster_plots(
    bs_large=200 / 1000,  # width of the gaussian kernel for rate
    threshold_factor=10 / 100,  # fraction of max peak height for burst
):

    # define them here and use variables that are in outer scope
    def render_plots():
        h5f = ah.load_experimental_files(
            path_prefix=f"{path}/{experiment}/", condition=condition
        )

        ah.find_rates(h5f, bs_large=bs_large)
        threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
        ah.find_system_bursts_from_global_rate(
            h5f, rate_threshold=threshold, merge_threshold=0.1
        )

        fig, ax = plt.subplots()
        ph.plot_raster(h5f, ax, color=colors[c_str], clip_on=False)
        ph.plot_bursts_into_timeseries(
            h5f, ax, apply_formatting=False, style="fill_between", color=colors[c_str]
        )

        ax.set_ylim(-1, None)
        ax.set_xlabel("")
        if idx > 0:
            ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_title(c_str)

        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
        cc.set_size2(ax, 3, 1.5)
        ax.get_figure().savefig(
            f"./fig/paper/exp_raster_nozoom_{c_str}.pdf", dpi=300, transparent=True
        )

        fig, ax = plt.subplots()
        ph.plot_system_rate(
            h5f,
            ax,
            mark_burst_threshold=False,
            color=colors[c_str],
            apply_formatting=False,
            clip_on=True,
        )

        ax.margins(x=0, y=0)
        if "KCl" in condition:
            ax.set_ylim(0, 4.5)
        else:
            ax.set_ylim(0, 2.9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax.set_ylabel("Rates [Hz]")
        ax.set_xlabel("Time [seconds]")
        if idx > 0:
            ax.set_ylabel("")
        cc.set_size2(ax, 3, 0.75)
        ax.get_figure().savefig(
            f"./fig/paper/exp_rate_nozoom_{c_str}.pdf", dpi=300, transparent=True
        )

    # optogenetic
    path = "./dat/exp_in/1b"
    experiment = "210719_B"
    conditions = ["1_pre", "2_stim", "3_post"]

    for idx, condition in enumerate(conditions):
        c_str = condition[2:]
        render_plots()

    # chemical
    path = "./dat/exp_in/KCl_1b"
    experiment = "210720_B"
    conditions = ["1_KCl_0mM", "2_KCl_2mM"]

    for idx, condition in enumerate(conditions):
        c_str = condition[2:]
        render_plots()


def test(df):
    obs = "Fraction"

    fig, ax = plt.subplots()


# Fig 3
def ibi_simulation_vs_experiment(input_path):

    data = nh.load_ndim_h5f(input_path)
    obs = "sys_median_ibis"
    data[obs] = data[obs].sel(simulation_coordinates)

    num_reps = len(data[obs].coords["repetition"])
    dat_med = data[obs].mean(dim="repetition")
    dat_sem = data[obs].std(dim="repetition") / np.sqrt(num_reps)

    fig, ax = plt.subplots()
    x = dat_med.coords["rate"]
    y = dat_med
    yerr = dat_sem

    ax.errorbar(
        x=x,
        y=y,
        yerr=yerr,
        fmt="o",
        markersize=1.5,
        color="#333",
        elinewidth=1,
        capsize=0,
        zorder=2,
        label=f"simulation",
    )

    ax.plot(
        x, y, color=cc.alpha_to_solid_on_bg("#333", 0.3), zorder=1, label=f"simulation",
    )

    ax.plot([0, 72, 72], [40, 40, 0], ls=":", color=colors["pre"], zorder=0)
    ax.plot([0, 100, 100], [10, 10, 0], ls=":", color=colors["stim"], zorder=0)
    # ax.axhline(y=10, xmin=0, xmax=1, ls = ":", color="gray", zorder=0)
    # ax.axhline(y=40, xmin=0, xmax=1, ls = ":", color="gray", zorder=0)
    # ax.set_yscale("log")

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    ax.set_xlim(67.5, 112.5)
    ax.set_ylim(0, 60)

    if show_xlabel:
        ax.set_xlabel(r"Noise rate (Hz)")
    if show_ylabel:
        ax.set_ylabel("Inter-event-interval\n(median, in seconds)")
    if show_title:
        ax.set_title(r"Simulation IEI")
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        cc._legend_into_new_axes(ax)
    if use_compact_size:
        cc.set_size2(ax, 3.5, 2.5)

    ax.get_figure().savefig(
        f"./fig/paper/ibi_sim_vs_exp.pdf", dpi=300, transparent=True
    )

    return ax


def fraction_of_ibi(input_path, ax = None):

    data = nh.load_ndim_h5f(input_path)
    obs = "sys_mean_participating_fraction"
    data[obs] = data[obs].sel(simulation_coordinates)

    num_reps = len(data[obs].coords["repetition"])
    dat_med = data[obs].mean(dim="repetition")
    dat_sem = data[obs].std(dim="repetition") / np.sqrt(num_reps)

    x = dat_med.coords["rate"]
    y = dat_med
    yerr = dat_sem


    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.errorbar(
        x=x,
        y=y,
        yerr=yerr,
        fmt="o",
        markersize=1.5,
        color="red",
        elinewidth=1,
        capsize=0,
        zorder=2,
        label=f"simulation",
    )

    ax.plot(
        x, y, color=cc.alpha_to_solid_on_bg("red", 0.3), zorder=1, label=f"simulation",
    )

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))

    ax.set_xlim(67.5, 112.5)
    ax.set_ylim(0, 1)

    if show_xlabel:
        ax.set_xlabel(r"Noise rate (Hz)")
    if show_ylabel:
        ax.set_ylabel("Participating neurons\n(mean)")
    # if show_title:
    #     ax.set_title(r"")
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        cc._legend_into_new_axes(ax)
    if use_compact_size:
        cc.set_size2(ax, 3.5, 2)

    ax.get_figure().savefig(
        f"./fig/paper/participating_fraction.pdf", dpi=300, transparent=True
    )

    return ax


prev = None
x = None
nxt = None


def fraction_of_sequence_length(input_path, drop_zero_len=True):
    data = nh.load_ndim_h5f(input_path)
    dim1 = "rate"

    for obs in data.keys():
        data[obs].sel(simulation_coordinates)

    x = data["any_num_b"].coords[dim1]
    selects = np.where((x >= 67.5) & (x <= 112.5))

    fig, ax = plt.subplots()

    prev = np.zeros_like(x)
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
        )
        if seq_len != 0:
            ax.errorbar(
                x=x[selects],
                y=prev[selects] + nxt[selects],
                yerr=ratio_errs[selects],
                fmt="o",
                markersize=3,
                mfc=cc.alpha_to_solid_on_bg(clr, 0.2),
                elinewidth=0.5,
                capsize=2,
                label=f"{seq_len} module" if seq_len == 1 else f"{seq_len} modules",
                color=clr,
                clip_on=False,
            )

        # we coult try to place text in the areas
        if seq_len == 4:
            ycs = 6
            xcs = 1
            ax.text(
                x[selects][xcs],
                prev[selects][ycs] + (nxt[selects][ycs]) / 2,
                f"{seq_len} module" if seq_len == 1 else f"{seq_len} modules",
                color=clr,
                va="center",
            )

        prev += nxt

    fig.tight_layout()

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.set_xlim(67.5, 112.5)
    ax.set_ylim(0, 1)

    # ax.spines["left"].set_position(("outward", 5))
    # ax.spines["bottom"].set_position(("outward", 5))

    if show_xlabel:
        ax.set_xlabel(r"Noise rate (Hz)")
    if show_ylabel:
        ax.set_ylabel("Fraction of bursts\nspanning")
    # if show_title:
    #     ax.set_title(r"")
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        cc._legend_into_new_axes(ax)
    if use_compact_size:
        cc.set_size2(ax, 3.5, 2)

    ax.get_figure().savefig("./fig/paper/sim_fractions.pdf", dpi=300, transparent=True)


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def custom_violins(df, category, observable, ax=None, **violin_kwargs):

    violin_kwargs = violin_kwargs.copy()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    categories = df[category].unique()

    # we want half violins, so hack the data and abuse seaborns "hue" and "split"
    df["fake_hue"] = 0
    for cat in categories:
        dummy = pd.Series([np.nan] * len(df.columns), index=df.columns)
        dummy["fake_hue"] = 1
        dummy[category] = cat
        df = df.append(dummy, ignore_index=True)
    log.info(df)

    # lets use that seaborn looks up the `hue` variable as the key in palette dict,
    # maybe matching our global colors
    palette = colors.copy()
    light_palette = dict()
    for key in palette.keys():
        light_palette[key] = cc.alpha_to_solid_on_bg(palette[key], 0.5)

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

    for idx, cat in enumerate(categories):
        ax.collections[idx].set_color(light_palette[cat])
        ax.collections[idx].set_edgecolor(palette[cat])
        ax.collections[idx].set_linewidth(1.0)

        # custom error estimates
        # df_for_cat = df.query(f"`{category}` == '{cat}'")
        # try:
        #     mid, error = ah.pd_nested_bootstrap(
        #         df_for_cat,
        #         grouping_col="Experiment",
        #         obs=observable,
        #         num_boot=500,
        #         func=np.nanmedian,
        #         resample_group_col=True,
        #     )
        # except:
        #     log.warning("Nested bootstrap failed")
        #     # this may happen when category variable is not defined.
        #     mid, error = ah.pd_bootstrap(
        #         df_for_cat, obs=observable, num_boot=500, func=np.nanmedian
        #     )

        # _draw_error_stick(
        #     ax,
        #     center=idx,
        #     mid=mid,
        #     errors=[mid - error, mid + error],
        #     orientation="v",
        #     color=palette[cat],
        # )

    # swarms
    swarm_defaults = dict(
        size=1.2,
        palette = light_palette,
        dodge=True,
        order = categories,
        zorder=0,
    )
    sns.swarmplot(x=category, y=observable, data=df.sample(n=2500, replace=True, ignore_index=True), **swarm_defaults)

    ax.get_legend().set_visible(False)

    cc.set_size2(ax, 3, 4)

    return ax


def _draw_error_stick(
    ax, center, mid, errors, outliers=None, orientation="v", linewidth=1.5, **kwargs
):
    """
        Use this to draw errors likes seaborns nice error bars, but using our own
        error estimates.

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
    kwargs.setdefault(
        "transform",
        matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
    )

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
    if orientation == "h":
        ax.scatter(mid, center, s=np.square(linewidth * 2), **kwargs)
    else:
        ax.scatter(center, mid, s=np.square(linewidth * 2), **kwargs)
