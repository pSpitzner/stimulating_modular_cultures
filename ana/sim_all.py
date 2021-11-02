# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-25 17:28:21
# @Last Modified: 2021-11-02 12:19:48
# ------------------------------------------------------------------------------ #
# Hard coded script to analyse experimental data
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import logging
import warnings
import functools
import itertools
import tempfile
import psutil
import re
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict
from tqdm import tqdm
from benedict import benedict

# from addict import Dict

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

import dask_helper as dh
import ana_helper as ah
import plot_helper as ph
import hi5 as h5
import colors as cc

input_path = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/inhibition_sweep_rate_160/dyn"

exclude_list = []

layout = "k=5"
conditions = ["76", "106"]

burstframes = []
isisframes = []
expframes = []
rijframes = []

for condition in conditions:
    for path in glob.glob(
        f"{input_path}/stim=off_k=5_jA=45.0_jG=50.0_jM=15.0_rate={condition}.0_rep=*.hdf5"
    ):
        experiment = os.path.basename(path)
        experiment = experiment.split("rep=")[-1].split(".")[0]

        print(f"\n{condition} rep {experiment}")

        h5f = ah.prepare_file(path)

        ah.find_rates(h5f, bs_large=20 / 1000)
        threshold = 0.025 * np.nanmax(h5f["ana.rates.system_level"])
        ah.find_system_bursts_from_global_rate(
            h5f, rate_threshold=threshold, merge_threshold=0.1, skip_sequences=False,
        )

        ah.find_participating_fraction_in_bursts(h5f)
        fracs = np.array(h5f["ana.bursts.system_level.participating_fraction"])

        blen = np.array(h5f["ana.bursts.system_level.end_times"]) - np.array(
            h5f["ana.bursts.system_level.beg_times"]
        )
        slen = np.array(
            [len(x) for x in h5f["ana.bursts.system_level.module_sequences"]]
        )
        olen = ah.find_onset_durations(h5f, return_res=True)

        df = pd.DataFrame(
            {
                "Duration": blen,
                "Sequence length": slen,
                "Fraction": fracs,
                "Onset duration": olen,
                "Condition": condition + " Hz",
                "Experiment": experiment,
                "Stimulation": "On" if float(condition) > 95 else "Off",
            }
        )

        burstframes.append(df)

        # ISI
        ah.find_isis(h5f)
        isis = []
        for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
            m_dc = h5f["ana.mods"][mdx]
            isis.extend(h5f[f"ana.isi.{m_dc}.all"])

        df = pd.DataFrame(
            {
                "ISI": isis,
                "Condition": condition + " Hz",
                "Experiment": experiment,
                "Stimulation": "On" if float(condition) > 95 else "Off",
            }
        )
        isisframes.append(df)

        # statistics across experiments, rij
        # NxN matrix
        rij = ah.find_rij(h5f, time_bin_size=200 / 1000)
        np.fill_diagonal(rij, np.nan)
        rij_flat = rij.flatten()

        df = pd.DataFrame(
            {
                "Correlation Coefficient": rij_flat,
                "Condition": condition + " Hz",
                "Experiment": experiment,
                "Stimulation": "On" if float(condition) > 95 else "Off",
            }
        )
        rijframes.append(df)

        fc = ah._functional_complexity(rij)
        mean_rij = np.nanmean(rij)

        # to compare to experiments, normalize the number of bursts to the 10 minutes
        # of experimental recordings
        print(f"found {len(blen)} bursts")

        df = pd.DataFrame(
            {
                "Num Bursts": [len(blen) / 6],
                "Mean Correlation": [mean_rij],
                "Functional Complexity": [fc],
                "Condition": condition + " Hz",
                "Experiment": experiment,
                "Stimulation": "On" if float(condition) > 95 else "Off",
            }
        )
        expframes.append(df)

        # if experiment == "210405_C" and "post" in condition:
        # assert False

        h5.close_hot()
        del h5f


df_bursts = pd.concat(burstframes, ignore_index=True)
df_isis = pd.concat(isisframes, ignore_index=True)
df_isis["logISI"] = df_isis.apply(lambda row: np.log10(row["ISI"]), axis=1)
df_rij = pd.concat(rijframes, ignore_index=True)
df_exp = pd.concat(expframes, ignore_index=True)

def to_hdf5(df_path):
    df_bursts.to_hdf(df_path, "/data/df_bursts", complevel=6)
    df_isis.to_hdf(df_path, "/data/df_isis", complevel=6)
    df_rij.to_hdf(df_path, "/data/df_rij", complevel=6)
    df_exp.to_hdf(df_path, "/data/df_exp", complevel=6)

def from_hdf5(df_path):
    global df_bursts
    global df_isis
    global df_rij
    global df_exp

    df_bursts = pd.read_hdf(df_path, "/data/df_bursts")
    df_isis = pd.read_hdf(df_path, "/data/df_isis")
    df_rij = pd.read_hdf(df_path, "/data/df_rij")
    df_exp = pd.read_hdf(df_path, "/data/df_exp")

# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #

# sns.set_theme(style="whitegrid", palette=None)
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.left"] = True
matplotlib.rcParams["axes.spines.bottom"] = True
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams["lines.solid_capstyle"] = "round"


# defaul bins when using histograms
def unit_bins(low=0, high=1, num_bins=20):
    bw = (high - low) / num_bins
    return np.arange(low, high + 0.1 * bw, bw)


violins = False


def plot_median_quartiles(ax, data, center, whiskers=False, linewidth=1.5, **kwargs):
    q25, q50, q75 = np.nanpercentile(data, [25, 50, 75])
    whisker_lim = 1.5 * (q75 - q25)
    h1 = np.min(data[data >= (q25 - whisker_lim)])
    h2 = np.max(data[data <= (q75 + whisker_lim)])

    kwargs.setdefault("color", "black")
    kwargs.setdefault("zorder", 3)
    kwargs.setdefault("transform", mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes))
    if whiskers:
        ax.plot(
            [h1, h2], [center, center], linewidth=linewidth, clip_on=False, **kwargs
        )
    ax.plot(
        [q25, q75], [center, center], linewidth=linewidth * 3, clip_on=False, **kwargs
    )

    kwargs["zorder"] += 1
    kwargs["edgecolor"] = kwargs["color"]
    kwargs["color"] = "white"
    ax.scatter(q50, center, s=np.square(linewidth * 2), clip_on=False, **kwargs)


def plot_mean_std(ax, data, center, linewidth=1.5, sem=False, **kwargs):
    mean = np.nanmean(data)
    std = np.nanstd(data)

    if True:
        std /= np.sqrt(len(data))


    kwargs.setdefault("color", "black")
    kwargs.setdefault("zorder", 3)
    kwargs.setdefault("transform", mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes))
    ax.plot(
        [mean - std, mean + std],
        [center, center],
        linewidth=linewidth * 3,
        clip_on=False,
        **kwargs,
    )

    kwargs["zorder"] += 1
    kwargs["edgecolor"] = kwargs["color"]
    kwargs["color"] = "white"
    ax.scatter(mean, center, s=np.square(linewidth * 2), clip_on=False, **kwargs)



def cat_hist_plot(df, category, obs, bins, **kwargs):
    cats = df[category].unique()

    fig, axes = plt.subplots(nrows=len(cats), sharex=True, sharey=True)

    for idx in range(0, len(cats)):
        df_local = df.query(f"`{category}` == '{cats[idx]}'")
        ax = axes[idx]

        sns.histplot(
            data=df_local,
            x=obs,
            ax=ax,
            kde=False,
            bins=bins,
            # stat = "density",
            stat="probability",
            element="step",
            color=f"C{idx}",
            clip_on=False,
            zorder=2,
            **kwargs,
        )

        plot_median_quartiles(
            ax,
            data=df_local[obs],
            center=-.05,
            color=cc.alpha_to_solid_on_bg(f"C{idx}", 0.75, "black"),
        )

        # plot_mean_std(
        #     ax,
        #     data=df_local[obs],
        #     center=-.15,
        #     color=cc.alpha_to_solid_on_bg(f"C{idx}", 0.75, "black"),
        # )


        # ax.set_title(f"{cats[idx]}")
        ax.text(
            0.01,
            0.8,
            f"{cats[idx]}",
            color = cc.alpha_to_solid_on_bg(f"C{idx}", 0.75, "black"),
            va="center",
            transform = ax.transAxes
        )

        ax.margins(x=0, y=0)

        if idx == 0:
            ax.set_title(f"{layout}")

        if idx < len(cats) - 1:
            ax.set_ylabel("")
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)

        else:
            ax.spines['bottom'].set_position(['outward', 8])

        ax.spines['left'].set_position(['outward', 4])


    fig.tight_layout()
    return fig


if not violins:
    # Bursts
    fig = cat_hist_plot(df_bursts, "Condition", "Duration", bins=unit_bins(0, 1, 20) )
    fig.get_axes()[0].set_xlim(0, 1)

    fig = cat_hist_plot(df_bursts, "Condition", "Fraction", bins=unit_bins(0, 1, 20) )
    fig.get_axes()[0].set_xlim(0, 1)

    fig = cat_hist_plot(df_bursts, "Condition", "Sequence length", bins=unit_bins(-.5, 4.5, 5) )
    fig.get_axes()[0].set_xlim(-.5, 4.5)
    fig.get_axes()[0].set_ylim(0, 1)

    fig = cat_hist_plot(df_bursts, "Condition", "Onset duration", bins=unit_bins(0, .5, 20) )
    fig.get_axes()[0].set_xlim(0, .5)

    # ISI
    fig = cat_hist_plot(df_isis, "Condition", "logISI", bins=unit_bins(-3, 3, 50) )
    ax = fig.get_axes()[0]
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(cc._ticklabels_lin_to_log10_power)
    )
    ax.set_xlim(-3, 3)

    # rij
    fig = cat_hist_plot(df_rij, "Condition", "Correlation Coefficient", bins=unit_bins(0, 1, 20))
    fig.get_axes()[0].set_xlim(0, 1)

else:

    # Bursts
    fig, ax = plt.subplots()
    sns.violinplot(x="Stimulation", y="Duration", data=df_bursts, **violin_kwargs)
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.violinplot(x="Stimulation", y="Fraction", data=df_bursts, **violin_kwargs)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.violinplot(x="Stimulation", y="Sequence length", data=df_bursts, **violin_kwargs)
    ax.set_ylim(-0.05, 4.05)
    fig.tight_layout()

    fig, ax = plt.subplots()
    sns.violinplot(x="Stimulation", y="Onset duration", data=df_bursts, **violin_kwargs)
    fig.tight_layout()

    # ISI
    fig, ax = plt.subplots()
    sns.violinplot(x="Stimulation", y="logISI", data=df_isis, **violin_kwargs)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(cc._ticklabels_lin_to_log10_power)
    )
    ax.set_ylim(-4, 3)
    fig.tight_layout()

    # rij
    fig, ax = plt.subplots()
    sns.violinplot(x="Stimulation", y="Correlation Coefficient", data=df_rij, **violin_kwargs)
    # sns.stripplot(
    #     x="Stimulation",
    #     y="Correlation Coefficient",
    #     hue="Condition",
    #     data=df_rij,
    #     jitter=0.375,
    #     size=1,
    #     zorder=1
    # )
    # ax.get_legend().set_visible(False)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()

# exp
# df_exp = pd.concat(expframes, ignore_index=True)
fig, ax = plt.subplots()
sns.pointplot(
    x="Stimulation", y="Num Bursts", hue="Experiment", palette="YlGnBu_d", data=df_exp,
)
ax.get_legend().set_visible(False)
cc._legend_into_new_axes(ax)
ax.set_ylim(-5, None)
fig.tight_layout()


fig, ax = plt.subplots()
sns.pointplot(
    x="Stimulation",
    y="Mean Correlation",
    hue="Experiment",
    palette="YlGnBu_d",
    data=df_exp,
)
ax.get_legend().set_visible(False)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()


fig, ax = plt.subplots()
sns.pointplot(
    x="Stimulation",
    y="Functional Complexity",
    hue="Experiment",
    palette="YlGnBu_d",
    data=df_exp,
)
ax.get_legend().set_visible(False)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
