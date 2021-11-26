# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-11-08 17:51:24
# @Last Modified: 2021-11-16 11:53:15
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

from benedict import benedict
from tqdm import tqdm

import plot_helper as ph
import ana_helper as ah
import ndim_helper as nh
import colors as cc
import hi5 as h5

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

simulation_coordinates = dict()
simulation_coordinates["jA"] = 45
simulation_coordinates["jG"] = 50
simulation_coordinates["tD"] = 20
simulation_coordinates["k_inter"] = 5

# we had one trial for single bond where we saw more bursts than everywhere else
remove_outlier = True

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

# Fig 1
def exp_raster_plots(
    bs_large=200 / 1000,  # width of the gaussian kernel for rate
    threshold_factor=10 / 100,  # fraction of max peak height for burst
):

    # define them here and use variables that are in outer scope
    def render_plots():
        global h5f
        h5f = ah.load_experimental_files(
            path_prefix=f"{path}/{experiment}/", condition=condition
        )

        ah.find_rates(h5f, bs_large=bs_large)
        threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
        ah.find_system_bursts_from_global_rate(
            h5f, rate_threshold=threshold, merge_threshold=0.1
        )

        fig, axes = plt.subplots(
            nrows=3, ncols=1, sharex=True, figsize=[3.5 / 2.54, 3 / 2.54]
        )

        # Raster
        # fig, ax = plt.subplots()
        ax = axes[1]
        ph.plot_raster(
            h5f,
            ax,
            base_color=colors[c_str],
            sort_by_module=True,
            neuron_id_as_y=False,
            clip_on=False,
            markersize=1.5,
            alpha=1,
        )
        # ph.plot_bursts_into_timeseries(
        #     h5f, ax, apply_formatting=False, style="fill_between", color=colors[c_str]
        # )

        ax.set_ylim(-5, None)
        ax.set_xlim(0, 540)
        ax.set_xlabel("")
        if idx >= 0:
            ax.set_ylabel("")
        # ax.set_title(c_str)
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_yticks([])
        # _time_scale_bar(ax=ax, x1=340, x2=520, y=-2, ylabel=-3.5, label="180 s")

        # ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        # ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
        ax.set_xticks([])

        # cc.set_size2(ax, 3, 0.75)
        # ax.get_figure().savefig(
        #     f"./fig/paper/exp_raster_nozoom_{c_str}.pdf",
        #     dpi=300,
        #     transparent=False,
        # )

        # Rates
        # fig, ax = plt.subplots()
        ax = axes[2]
        ph.plot_system_rate(
            h5f,
            ax,
            mark_burst_threshold=False,
            color=colors[c_str],
            apply_formatting=False,
            clip_on=False,
            lw=0.5,
        )

        ax.margins(x=0, y=0)
        ax.set_xlim(0, 540)
        if "KCl" in condition:
            ax.set_ylim(0, 4.5)
        else:
            ax.set_ylim(0, 3.0)
        # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        # ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        if "KCl" in condition:
            sns.despine(ax=ax, left=False, bottom=True, trim=True, offset=2)
        else:
            sns.despine(ax=ax, left=False, bottom=True, trim=False, offset=2)
        ax.set_xticks([])
        ax.set_ylabel("Rates [Hz]")
        # ax.set_xlabel("Time [seconds]")
        if idx >= 0:
            ax.set_ylabel("")

        if condition == "1_pre":
            _time_scale_bar(
                ax=ax, x1=340, x2=520, y=-0.5, ylabel=-1.0, label="180 s"
            )

        # cc.set_size(ax, 3, 0.75)
        # ax.get_figure().savefig(
        #     f"./fig/paper/exp_rate_nozoom_{c_str}.pdf",
        #     dpi=300,
        #     transparent=False,
        # )

        # fluorescence
        # fig, ax = plt.subplots()

        ax = axes[0]
        if "KCl" in condition:
            sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            n_per_mod = 4
            npm = int(len(h5f["ana.neuron_ids"]) / len(h5f["ana.mods"]))
            neurons_to_show = []
            for m in range(0, len(h5f["ana.mods"])):
                neurons_to_show.extend(np.arange(0, n_per_mod) + m * npm)

            # neurons_to_show = [1,2,6,9,11,12,17,19]

            # fig, ax = plt.subplots()
            ax = ph.plot_fluorescence_trace(
                h5f,
                ax=ax,
                neurons=np.array(neurons_to_show),
                lw=0.25,
                base_color=colors[c_str],
            )

            ax.set_xlim(0, 540)
            sns.despine(ax=ax, left=True, bottom=True, trim=True, offset=2)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_ylabel("Fluorescence")
            if idx >= 0:
                ax.set_ylabel("")

        # assert False

        # cc.set_size(ax, 3, 0.75)
        # fig.tight_layout()
        plt.subplots_adjust(hspace=0.001)
        ax.get_figure().savefig(
            f"./fig/paper/exp_combined_{c_str}.pdf",
            dpi=300,
            transparent=False,
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


def exp_chemical_vs_opto(observable="Mean Fraction", df="trials"):
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

# this becomes impossible to tweak further when using seaborn
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
            x_pos[l][c] = ldx * large_dx + cdx*small_dx
            x_pos_sticks[l][c] = ldx * large_dx + cdx*small_dx


    # opto
    for etype in categories:
        clr = colors["stim"] if etype == "exp" else colors["KCl_0mM"]
        for trial in dfs[etype]['Trial'].unique():
            df = dfs[etype].loc[dfs[etype]['Trial'] == trial]
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
                lw=.5,
                color=cc.alpha_to_solid_on_bg(clr, 0.2),
                # color=clr,
                # alpha = 0.3,
                label=trial,
                zorder=0,
                clip_on = False,
            )

    for etype in ["exp_chemical", "exp"]:
        clr = colors["stim"] if etype == "exp" else colors["KCl_0mM"]
        for stim in ["On", "Off"]:
            df = dfs[etype].query(f"Stimulation == '{stim}'")
            # sticklike error bar
            mid, error, percentiles = ah.pd_bootstrap(
                df,
                obs=observable,
                num_boot=500,
                func=np.nanmean,
                percentiles=[2.5, 50, 97.5],
            )
            df_max = np.nanmax(df[observable])
            df_min = np.nanmin(df[observable])

            _draw_error_stick(
                ax,
                center=x_pos_sticks[etype][stim],
                # mid=percentiles[1],
                # errors=[percentiles[0], percentiles[2]],
                mid=mid,
                errors=[mid-error, mid+error],
                outliers=[df_min, df_max],
                orientation="v",
                color=clr,
                zorder=2,
            )

    # ax.legend()
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-.25, 1.5)
    sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

    cc.set_size3(ax, 1.2, 2)

    ax.get_figure().savefig(f"./fig/paper/exp_chem_vs_opto_{observable}.pdf", dpi=300)

    return ax


def exp_sticks_across_layouts(observable="Functional Complexity"):

    layouts = ["1b", "3b", "merged"]
    conditions = ["pre", "stim", "post"]
    # we want the precalculated summary statistics of each trial
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
            x_pos[l][c] = cdx * large_dx + ldx*small_dx
            x_pos_sticks[l][c] = cdx * large_dx + ldx*small_dx

    # x_pos["1b"] = {"pre" : 0.0, "stim" : 1.0, "post" :  2.0 }
    # x_pos["3b"] = {"pre" : 0.25, "stim" : 1.25, "post"  :  2.25 }
    # x_pos["merged"] = {"pre" : 0.5, "stim" : 1.5, "post" :  2.5 }
    # x_pos_sticks["1b"] = {"pre" : 0.0, "stim" : 1.0, "post" :  2.0 }
    # x_pos_sticks["3b"] = {"pre" : 0.25, "stim" : 1.25, "post"  :  2.25 }
    # x_pos_sticks["merged"] = {"pre" : 0.5, "stim" : 1.5, "post" :  2.5 }


    # opto
    for edx, etype in enumerate(layouts):
        clr = f"C{edx}"
        for trial in dfs[etype]['Trial'].unique():
            df = dfs[etype].loc[dfs[etype]['Trial'] == trial]
            assert len(df) == len(layouts)
            x = []
            y = []
            for idx, row in df.iterrows():
                stim = row["Condition"]
                x.append(x_pos[etype][stim])
                y.append(row[observable])

            ax.plot(
                x,
                y,
                # marker="o",
                # markersize=1,
                lw=.5,
                color=cc.alpha_to_solid_on_bg(clr, 0.2),
                # color=clr,
                # alpha = 0.3,
                label=trial,
                zorder=0,
                clip_on = False,
            )

        for stim in conditions:
            df = dfs[etype].query(f"Condition == '{stim}'")
            # sticklike error bar
            mid, error, percentiles = ah.pd_bootstrap(
                df,
                obs=observable,
                num_boot=500,
                func=np.nanmean,
                percentiles=[2.5, 50, 97.5],
            )
            df_max = np.nanmax(df[observable])
            df_min = np.nanmin(df[observable])

            _draw_error_stick(
                ax,
                center=x_pos_sticks[etype][stim],
                # mid=percentiles[1],
                # errors=[percentiles[0], percentiles[2]],
                mid=mid,
                errors=[mid-error, mid+error],
                outliers=[df_min, df_max],
                orientation="v",
                color=clr,
                zorder=2,
            )

    # ax.legend()
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-.25, 3.5)
    sns.despine(ax=ax, bottom=True, left=False, trim=True, offset=5)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

    cc.set_size3(ax, 2.2, 2)

    # ax.get_figure().savefig(f"./fig/paper/exp_chem_vs_opto_{observable}.pdf", dpi=300)

    return ax


# Fig 2
def exp_violins_for_layouts():

    dfs = dict()
    dfs["single-bond"] = load_pd_hdf5("./dat/exp_out/1b.hdf5")
    dfs["triple-bond"] = load_pd_hdf5("./dat/exp_out/3b.hdf5")
    dfs["merged"] = load_pd_hdf5("./dat/exp_out/merged.hdf5")

    if remove_outlier:

        def query(df):
            if layout == "single-bond":
                return df.query("`Trial` != '210405_C'")
            else:
                return df

    else:

        def query(df):
            return df

    def apply_formatting(ax, ylim=True, trim=True):
        if ylim:
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        sns.despine(ax=ax, bottom=True, left=False, trim=trim, offset=5)
        ax.tick_params(bottom=False)
        # reuse xlabel for title
        # ax.set_xlabel(f"{layout}")
        ax.set_xlabel(f"")
        ax.set_ylabel(f"")
        ax.set_xticks([])
        cc.set_size2(ax, 3, 2.0)

    for layout in dfs.keys():
        log.info(f"")
        log.info(f"{layout}")
        ax = custom_violins(
            query(dfs[layout]["bursts"]),
            category="Condition",
            observable="Fraction",
            ylim=[0, 1],
            num_swarm_points=250,
            bw=0.2,
        )
        apply_formatting(ax)
        ax.get_figure().savefig(
            f"./fig/paper/exp_violins_fraction_{layout}.pdf", dpi=300
        )

    for layout in dfs.keys():
        log.info(f"")
        log.info(f"{layout}")
        ax = custom_violins(
            query(dfs[layout]["rij"]),
            category="Condition",
            observable="Correlation Coefficient",
            ylim=[0, 1],
            num_swarm_points=500,
            bw=0.2,
        )
        apply_formatting(ax)
        ax.get_figure().savefig(
            f"./fig/paper/exp_violins_rij_{layout}.pdf", dpi=300
        )

    for layout in dfs.keys():
        log.info(f"")
        log.info(f"{layout}")
        ax = custom_violins(
            # dfs[layout]["bursts"],
            # remove the single-bond outlier for ibis?
            query(dfs[layout]["bursts"]),
            category="Condition",
            observable="Inter-burst-interval",
            ylim=[0, 70],
            num_swarm_points=250,
            bw=0.2,
        )
        ax.set_ylim(0, 70)
        apply_formatting(ax, ylim=False, trim=False)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
        ax.get_figure().savefig(
            f"./fig/paper/exp_violins_ibi_{layout}.pdf", dpi=300
        )

    return ax


def exp_rij_for_layouts():
    dfs = dict()
    dfs["single-bond"] = load_pd_hdf5("./dat/exp_out/1b.hdf5")
    dfs["triple-bond"] = load_pd_hdf5("./dat/exp_out/3b.hdf5")
    dfs["merged"] = load_pd_hdf5("./dat/exp_out/merged.hdf5")

    if remove_outlier:

        def query(df):
            if layout == "single-bond":
                return df.query("`Trial` != '210405_C'")
            else:
                return df

    else:

        def query(df):
            return df

    for layout in dfs.keys():
        log.info(f"{layout}")
        pairings = None
        if layout == "merged":
            pairings = ["all"]
            # pairings = ["within_stim", "within_nonstim"]
        ax = custom_rij_scatter(
            query(dfs[layout]["rij_paired"]), pairings=pairings
        )

        cc.set_size3(ax, 3, 3)
        ax.get_figure().savefig(f"./fig/paper/exp_2drij_{layout}.pdf", dpi=300)

        ax = custom_rij_barplot(
            query(dfs[layout]["rij_paired"]), pairings=pairings
        )
        ax.set_ylim(0, 1)
        # ax.set_xlabel(layout)
        ax.set_xlabel("")
        ax.set_ylabel("")
        cc.set_size3(ax, 3, 1.5)
        ax.get_figure().savefig(
            f"./fig/paper/exp_rij_barplot_{layout}.pdf", dpi=300
        )

    return ax


# Fig 3
def sim_raster_plots(
    bs_large=20 / 1000,  # width of the gaussian kernel for rate
    threshold_factor=2.5 / 100,  # fraction of max peak height for burst
    naked=False,  # whether to strip frames
):

    rates = [80, 90]

    # exclude_nids_from_raster = np.delete(np.arange(0, 160), np.arange(0, 160, 8))
    exclude_nids_from_raster = []

    for rdx, rate in enumerate(rates):

        c_str = f"{rate} Hz"

        h5f = ah.prepare_file(
            f"./dat/inhibition_sweep_rate_160/dyn/stim=off_k=5_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate={rate:.1f}_rep=000.hdf5"
        )

        # ------------------------------------------------------------------------------ #
        # raster
        # ------------------------------------------------------------------------------ #

        fig, ax = plt.subplots()
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
        # ph.plot_bursts_into_timeseries(
        #     h5f, ax, apply_formatting=False, style="fill_between", zorder=-2
        # )

        ax.set_ylim(-1, None)
        ax.set_xlim(0, 540)
        ax.set_ylabel("")
        # ax.set_ylabel("Raster")
        ax.set_xlabel("")
        # ax.set_title(c_str, loc="left")
        ax.set_yticks([])
        if naked:
            ax.set_xticks([])

            sns.despine(ax=ax, left=True, bottom=True)

            if rdx == len(rates) - 1:
                _time_scale_bar(
                    ax=ax, x1=340, x2=520, y=-15, ylabel=-25, label="180 s"
                )
        else:
            sns.despine(ax=ax, left=False, right=False, bottom=False, top=False)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
            # ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks([])

        cc.set_size3(ax, 3, 1.0)

        # # coordinates are pretty much ignored after cc.set_size2
        # ax.text(-0.2, 0.5, c_str, va="top")

        ax.get_figure().savefig(
            f"./fig/paper/sim_raster_nozoom_{c_str}.pdf", dpi=300
        )

        # ------------------------------------------------------------------------------ #
        # rates
        # ------------------------------------------------------------------------------ #

        ah.find_rates(h5f, bs_large=bs_large)
        threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
        ah.find_system_bursts_from_global_rate(
            h5f, rate_threshold=threshold, merge_threshold=0.1
        )

        fig, ax = plt.subplots()
        ph.plot_system_rate(
            h5f,
            ax,
            mark_burst_threshold=False,
            color="#333",
            apply_formatting=False,
            clip_on=True,
            lw=0.8,
        )

        ax.margins(x=0, y=0)
        ax.set_xlim(0, 540)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
        # ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([])
        # ax.set_ylabel("Rates [Hz]")
        # ax.set_xlabel("Time [seconds]")

        if rate < 90:
            ax.set_ylim(0, 80)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(80))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(20))
            if naked:
                ax.set_xticks([])
                sns.despine(ax=ax, left=False, bottom=True, trim=True, offset=2)
            cc.set_size2(ax, 3, 0.75)
        elif rate == 90:
            ax.set_ylim(0, 40)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(20))
            if naked:
                ax.set_xticks([])
                sns.despine(ax=ax, left=False, bottom=True, trim=True, offset=2)
            cc.set_size3(ax, 3, 0.75 / 2)

        ax.get_figure().savefig(
            f"./fig/paper/sim_rate_nozoom_{c_str}.pdf",
            dpi=300,
            transparent=False,
        )

        # ------------------------------------------------------------------------------ #
        # adaptation
        # ------------------------------------------------------------------------------ #

        fig, ax = plt.subplots()
        ph.plot_state_variable(
            h5f, ax, variable="D", lw=0.8, apply_formatting=False
        )

        ax.margins(x=0, y=0)
        ax.set_xlim(0, 540)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(180))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))

        ax.set_ylim(0, 1)
        # ax.set_ylabel("Resources")
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        # ax.set_xlabel("Time [seconds]")
        if naked:
            ax.set_xticks([])
            sns.despine(ax=ax, left=False, bottom=True, trim=True, offset=2)

        cc.set_size3(ax, 3, 0.75)

        ax.get_figure().savefig(
            f"./fig/paper/sim_resources_nozoom_{c_str}.pdf",
            dpi=300,
            transparent=False,
        )

        h5.close_hot()


def sim_vs_exp_violins(**kwargs):
    dfs = dict()
    dfs["exp"] = load_pd_hdf5("./dat/exp_out/1b.hdf5")
    dfs["sim"] = load_pd_hdf5("./dat/sim_out/k=5.hdf5")

    for key in dfs["sim"].keys():
        dfs["sim"][key] = dfs["sim"][key].query(
            "`Condition` == '80 Hz' | `Condition` == '90 Hz'"
        )

    if remove_outlier:
        for key in dfs["exp"].keys():
            dfs["exp"][key] = dfs["exp"][key].query("`Trial` != '210405_C'")

    # burst size
    df = pd.concat(
        [
            dfs["exp"]["bursts"].query(
                "Condition == 'pre' | Condition == 'stim'"
            ),
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
    ax.get_figure().savefig(
        f"./fig/paper/sim_vs_exp_violins_fraction.pdf", dpi=300
    )

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
            dfs["exp"]["bursts"].query(
                "Condition == 'pre' | Condition == 'stim'"
            ),
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


def sim_vs_exp_ibi(input_path, ax=None, **kwargs):

    kwargs = kwargs.copy()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    data = nh.load_ndim_h5f(input_path)
    obs = "sys_median_any_ibis"
    data[obs] = data[obs].sel(simulation_coordinates)

    num_reps = len(data[obs].coords["repetition"])
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
        elinewidth=1,
        capsize=0,
        zorder=2,
        **kwargs,
    )

    kwargs.pop("label")
    kwargs["color"] = cc.alpha_to_solid_on_bg(kwargs["color"], 0.3)

    ax.plot(x[selects], y[selects], zorder=1, **kwargs)

    ax.plot(
        [0, 80, 80],
        [20, 20, 0],
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

    ax.set_xlim(65, 110)
    ax.set_ylim(0, 70)

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
    # if use_compact_size:
    cc.set_size2(ax, 3.5, 2.5)

    ax.get_figure().savefig(f"./fig/paper/ibi_sim_vs_exp.pdf", dpi=300)

    return ax


def controls_sim_vs_exp_ibi():
    fig, ax = plt.subplots()

    p1 = "./dat/the_last_one/ndim_jM=15_tD=8_t2.5_k20.hdf5"
    p2 = "./dat/the_last_one/ndim_jM=15_tD=8_t2.5_k20_remove_null.hdf5"

    sim_vs_exp_ibi(p1, ax=ax, color="#333", label="all")
    sim_vs_exp_ibi(p2, ax=ax, color="red", label="0 removed")

    return ax


def sim_participating_fraction(input_path, ax=None, **kwargs):

    kwargs = kwargs.copy()

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

    kwargs.setdefault("color", "#333")

    ax.errorbar(
        x=x,
        y=y,
        yerr=yerr,
        fmt="o",
        markersize=1.5,
        elinewidth=1,
        capsize=0,
        zorder=2,
        label=f"simulation",
        **kwargs,
    )

    kwargs["color"] = cc.alpha_to_solid_on_bg(kwargs["color"], 0.3)

    ax.plot(x, y, zorder=1, label=f"simulation", **kwargs)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))

    ax.set_xlim(65, 110)
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

    ax.get_figure().savefig(f"./fig/paper/participating_fraction.pdf", dpi=300)

    return ax


def sim_modules_participating_in_bursts(
    input_path, xlim=[65, 110], drop_zero_len=True
):
    global data, dim1, x, selects
    data = nh.load_ndim_h5f(input_path)
    dim1 = "rate"

    for obs in data.keys():
        data[obs] = data[obs].sel(simulation_coordinates)

    x = data["any_num_b"].coords[dim1]
    # for 92.5 Hz we only sampled k = 10, hence drop the point
    selects = np.where((x >= xlim[0]) & (x <= xlim[1]) & (x!=92.5) )
    # selects = np.ones_like(x, dtype=bool)

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
        if seq_len != 0 and seq_len != 1:
            ax.errorbar(
                x=x[selects],
                y=prev[selects] + nxt[selects],
                yerr=ratio_errs[selects],
                fmt="o",
                markersize=3,
                mfc=cc.alpha_to_solid_on_bg(clr, 0.2),
                elinewidth=0.5,
                capsize=2,
                label=f"{seq_len} module"
                if seq_len == 1
                else f"{seq_len} modules",
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

    fig.tight_layout()

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.set_xlim(xlim)
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

    ax.get_figure().savefig("./fig/paper/sim_fractions.pdf", dpi=300)


def sim_violins_for_all_k():

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
        cc.set_size2(ax, 4, 3.0)

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
        ax.get_figure().savefig(
            f"./fig/paper/sim_violins_fraction_{k}.pdf", dpi=300
        )

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
        ax.get_figure().savefig(
            f"./fig/paper/sim_violins_rij_{k}.pdf", dpi=300
        )

        # depletion
        try:
            log.debug("depletion")
            ax = custom_violins(
                df["drij"],
                category="Condition",
                observable="Depletion rij",
                palette=colors[k],
                ylim=[-.5, 1],
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
        ax.get_figure().savefig(
            f"./fig/paper/sim_violins_ibi_{k}.pdf", dpi=300
        )

    for k in ["k=1", "k=5", "k=10"]:
        log.info("")
        log.info(k)
        log.info("Loading data")
        df = load_pd_hdf5(f"./dat/sim_out/{k}.hdf5", keys=["bursts", "rij", "drij"])
        render_plots()



# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def load_pd_hdf5(input_path, keys=None):
    """
    return a dict of data frames from processed conditions

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

    if keys is None:
        keys = ["bursts", "isis", "rij", "rij_paired", "drij", "trials"]
    res = dict()
    for key in keys:
        try:
            res[key] = pd.read_hdf(input_path, f"/data/df_{key}")
        except:
            log.debug(f"/data/df_{key} not in {input_path}, skipping")

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

    log.info(f"Violins for {observable}")
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
        hist, bins = np.histogram(
            df_for_cat[observable], _unit_bins(ylim[0], ylim[1])
        )
        max_points = np.max([max_points, np.max(hist)])

    for idx, cat in enumerate(categories):
        ax.collections[idx].set_color(light_palette[cat])
        ax.collections[idx].set_edgecolor(palette[cat])
        ax.collections[idx].set_linewidth(1.0)

        # custom error estimates
        df_for_cat = sub_dfs[cat]
        log.info("bootstrapping")
        try:
            raise KeyError
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
            # this may happen when category variable is not defined.
            mid, error, percentiles = ah.pd_bootstrap(
                df_for_cat,
                obs=observable,
                num_boot=500,
                func=np.nanmedian,
                percentiles=[2.5, 50, 97.5],
            )

        log.info(
            f"{cat}: median {mid:.3g}, se {error:.3g}, percentiles:"
            f" {percentiles}"
        )

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

    return ax


def custom_pointplot(
    df, category, observable, hue="Trial", ax=None, **point_kwargs
):
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


def custom_rij_barplot(df, ax=None, pairings=None, recolor=True):
    """
    query the layout of df first!
    provide a rij_paired dataframe
    """
    if pairings is None:
        pairings = ["within_stim", "within_nonstim", "across"]
        # pairings = df["Pairing"].unique()

    conditions = ["pre", "stim"]
    df = df.query("Pairing == @pairings")
    df = df.query("Condition == @conditions")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

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
            palette=colors,
            errcolor=".0",
            edgecolor=".0",
            estimator=np.nanmedian,
            ci=95,
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


def custom_rij_scatter(df, ax=None, pairings=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if pairings is None:
        pairings = ["within_stim", "within_nonstim", "across"]
        # pairings = df["Pairing"].unique()

    kwargs = kwargs.copy()

    for pdx, pairing in enumerate(pairings):

        rijs_before = []
        rijs_after = []
        df_paired = df.query(f"Pairing == '{pairing}'")
        # print(pairing)

        # for each paring, make two long lists of rij: before stim and during stim.
        # add data from all experiments
        # rij may be np.nan if one of the neurons did not have any spikes.
        for trial in df_paired["Trial"].unique():
            df_trial = df_paired.query(f"`Trial` == '{trial}'")
            df_before = df_trial.query(f"`Condition` == 'pre'")
            df_after = df_trial.query(f"`Condition` == 'stim'")

            assert np.all(
                df_before["Pair ID"].to_numpy()
                == df_after["Pair ID"].to_numpy()
            )

            rijs_before.extend(df_before["Correlation Coefficient"].to_list())
            rijs_after.extend(df_after["Correlation Coefficient"].to_list())

        scatter_kwargs = kwargs.copy()
        try:
            scatter_kwargs.setdefault("color", colors[f"rij_{pairing}"])
        except:
            scatter_kwargs.setdefault("color", f"C{pdx}")

        scatter_kwargs["color"] = cc.alpha_to_solid_on_bg(
            scatter_kwargs["color"], 0.2
        )

        scatter_kwargs.setdefault("alpha", 1)
        scatter_kwargs.setdefault("label", pairing)
        scatter_kwargs.setdefault("zorder", 1)
        scatter_kwargs.setdefault("edgecolor", None)
        scatter_kwargs.setdefault("linewidths", 0.3)

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

        ax.scatter(
            rijs_before,
            rijs_after,
            **scatter_kwargs,
            # clip_on=False
        )

        for low_l in [0.5, 0.9, 0.95, 0.975]:
            sns.kdeplot(
                x=rijs_before,
                y=rijs_after,
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
    ax.set_xlabel("rij pre")
    ax.set_ylabel("rij stim")
    sns.despine(top=False, bottom=False, left=False, right=False)

    return ax


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
