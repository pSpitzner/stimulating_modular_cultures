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


# ------------------------------------------------------------------------------ #
# old code snippets
# ------------------------------------------------------------------------------ #


# mpl.use("agg")

# Preliminaries for data
tf = 1800.0
data = mph.read_csv_and_format(
    "/Users/paul/Downloads/victor/22-02-25/modeldata/coup0.10-1/noise0.hdf5", "model"
)

# Burst detection
threshold = 1.0
find_system_bursts_from_module_bursts(data, threshold, 0.0)

# ---------- #
# Figure 1
# Time series with and without gates
# ---------- #

# Create figure and named axes
fig, axes = plt.subplots(nrows=3)
axd = {"rate": axes[0], "raster": axes[2], "reso": axes[1]}

# Plot series of activity and synaptic resources
mph.plot_series(data["df.model"], data["ana.mods"], axd["rate"], data["ana.mod_colors"])
axd["rate"].set_ylabel("Firing Rate")

resource_cols = [colname + "_res" for colname in data["ana.mods"]]
mph.plot_series(data["df.model"], resource_cols, axd["reso"], data["ana.mod_colors"])
axd["reso"].set_ylabel("Synp. Resources")

# Finally plot a raster
mph.plot_raster(data, axd["raster"])
axd["raster"].set_ylabel("Cluster ID")

# Set up all figure labels. Raster and series have different X axis
ticks = np.linspace(500, 700, 10)
for ax in axes[:2]:
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax.set_xlim(500, 700)

axes[1].set_xticks(ticks)
axes[1].set_xlabel("Time [s]")

ticks = np.linspace(0, tf, 10)
axes[2].set_xticks(ticks)
axes[2].set_xlabel("Time [s]")

# Save
fig.tight_layout()
fig.savefig("activity.pdf", bbox_inches="tight")
plt.close()
# plt.show()
gc.collect()

# -----


# ---------- #
# Figure 3
# Module contribution graph
# ---------- #

# Where are the model results and what X axes we used for them
# pathnoise = "modeldata/var_noise/noise{j}.csv"
pathnoise = "modeldata/coup0.10-1/noise{j}.hdf5"

noise_span = np.linspace(0.0, 1.0, 28)

# To store results
contributions = np.empty((noise_span.size, 4))

for j, noise in enumerate(noise_span):

    # Load data and find contribution
    data = mph.read_csv_and_format(pathnoise.format(j=j), "model")
    contrib = mph.module_contribution(data, "model", 1.0, area_min=0.7, roll_window=0.5)

    # Do the histogram of contribution values
    bins = [0.5, 1.5, 2.5, 3.5, 4.5]
    histo = np.histogram(contrib, bins=bins)[0]

    # Get the % of contribution for each case
    total = histo[::-1].cumsum()
    contributions[j, :] = total / total[-1]


# Make the actual graph
plt.figure()
clr = cc.cmap_cycle("cold", edge=False, N=4)[3]

# Plot markers and do the first fill
plt.fill_between(
    noise_span,
    0.0,
    contributions[:, 0],
    linewidth=0,
    color=cc.alpha_to_solid_on_bg(clr, 0.4),
)
plt.plot(noise_span, contributions[:, 0], marker="o", color=clr, ls="none", ms=3.0)

# Do the other three fills over the first one
for c in range(3):
    clr = cc.cmap_cycle("cold", edge=False, N=4)[3 - c - 1]
    plt.fill_between(
        noise_span,
        contributions[:, c],
        contributions[:, c + 1],
        linewidth=0,
        color=cc.alpha_to_solid_on_bg(clr, 0.4),
    )
    plt.plot(
        noise_span, contributions[:, c + 1], marker="o", ls="none", color=clr, ms=3.0
    )

# Labels and save
plt.xlabel("External rate")
plt.ylabel("Contribution")
plt.savefig("contribution_mesoscopic.pdf", bbox_inches="tight")
plt.close()
gc.collect()


# ---------- #
# Figure 3
# Contributions for different couplings
# ---------- #

couplings = np.array([0.1, 0.1, 0.3, 0.6])
# Where are the model results and what X axes we used for them
del pathnoise
# pathnoise = "modeldata/coup{c:.2f}/noise{j}.csv"
noise_span = np.linspace(0.0, 1.0, 30)

# To store results
# contributions = np.empty((noise_span.size, couplings.size))
# correlations = np.empty((noise_span.size, couplings.size))

events = np.empty((noise_span.size, couplings.size))
correlations = np.empty((noise_span.size, couplings.size))

functionlist = [mph.ps_f_correlation_coefficients, mph.ps_f_event_size]


for i, c in enumerate(couplings):
    _, results = mph.ps_process_data_from_folder(
        f"modeldata/coup{c:.2f}-1/", functionlist
    )

    # print(results)
    events[:, i] = results["ps_f_event_size"]
    correlations[:, i] = results["ps_f_correlation_coefficients"]

# Make the actual graph
savefigs = ["event size", "correlations"]

for j, y in enumerate([events, correlations]):
    plt.figure()

    # Plot markers and do the first fill
    for i, c in enumerate(couplings):
        clr = cc.cmap_cycle("cold", edge=False, N=5)[i + 1]
        plt.plot(
            noise_span,
            y[:, i],
            marker="o",
            color=clr,
            ls="none",
            ms=3.0,
            label=rf"$w_0 = {c}$",
        )

    # Labels and save
    plt.xlabel("External rate")
    plt.ylabel(f"{savefigs[j]}")
    plt.legend(loc="best")
    plt.savefig(f"{savefigs[j]}.pdf", bbox_inches="tight")
    plt.close()
    gc.collect()
