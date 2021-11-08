# ------------------------------------------------------------------------------ #
# Create the raster plots for the overview figure
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import numbers
import matplotlib

import plot_helper as ph
import ana_helper as ah

matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 150

# width of the gaussian kernel for rate
bs_large = 200 / 1000
# fraction of max peak height for burst
threshold_factor = 10 / 100

colors = dict()
colors["pre"] = "#335c67"
colors["stim"] = "#e09f3e"
colors["post"] = "#9e2a2b"
colors["KCl_0mM"] = "gray"
colors["KCl_2mM"] = "gray"


def render_plots_using_global_vars():
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
    render_plots_using_global_vars()


# chemical
path = "./dat/exp_in/KCl_1b"
experiment = "210720_B"
conditions = ["1_KCl_0mM", "2_KCl_2mM"]

for idx, condition in enumerate(conditions):
    c_str = condition[2:]
    render_plots_using_global_vars()
