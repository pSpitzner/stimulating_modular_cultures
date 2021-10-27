# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-25 17:28:21
# @Last Modified: 2021-10-27 11:41:18
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

# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
# import pandas as pd
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

input_path = "/Users/paul/mpi/simulation/brian_modular_cultures/data_for_jordi/2021-10-11/Experimental data - ML spike"
output_path = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/exp_out"

exclude_list = []

# layouts = ["1b", "3b", "merged"]
layouts = ["1b"]
conditions = ["1_spont_", "2_stim_", "3_post_"]

burstframes = []
isisframes = []
expframes = []
rijframes = []

for layout in layouts:
    for condition in conditions:
        for path in glob.glob(f"{input_path}/{layout}/*"):
            experiment = os.path.basename(path)

            if experiment in ["210405_C"]:
                continue

            h5f = ah.load_experimental_files(
                path_prefix=f"{path}/", condition=condition
            )

            ah.find_rates(h5f, bs_large=200 / 1000)
            threshold = 0.10 * np.nanmax(h5f["ana.rates.system_level"])
            ah.find_system_bursts_from_global_rate(
                h5f, rate_threshold=threshold, merge_threshold=0.1
            )

            # plot overview
            os.makedirs(f"{output_path}/{layout}/{experiment}", exist_ok=True)
            fig = ph.overview_dynamic(h5f)
            fig.savefig(f"{output_path}/{layout}/{experiment}/{condition}_overview.pdf")

            # get a nice zoom in
            try:
                beg = h5f["ana.bursts.system_level.beg_times"][0]
            except:
                beg = 0
            beg = np.fmax(0, beg - 15)
            fig.get_axes()[-2].set_xlim(beg, beg + 60)
            fig.savefig(f"{output_path}/{layout}/{experiment}/{condition}_zoom.pdf")

            ah.find_participating_fraction_in_bursts(h5f)
            fracs = np.array(h5f["ana.bursts.system_level.participating_fraction"])

            blen = np.array(h5f["ana.bursts.system_level.end_times"]) - np.array(
                h5f["ana.bursts.system_level.beg_times"]
            )
            slen = np.array(
                [len(x) for x in h5f["ana.bursts.system_level.module_sequences"]]
            )

            df = pd.DataFrame(
                {
                    "Duration": blen,
                    "Sequence length": slen,
                    "Fraction": fracs,
                    "Condition": condition[2:-1],
                    "Experiment": experiment,
                    "Stimulation": "On" if condition == "2_stim_" else "Off",
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
                    "Condition": condition[2:-1],
                    "Experiment": experiment,
                    "Stimulation": "On" if condition == "2_stim_" else "Off",
                }
            )
            isisframes.append(df)

            # statistics across experiments, rij
            # NxN matrix
            rij = ah.find_rij(h5f, time_bin_size=400 / 1000)
            np.fill_diagonal(rij, np.nan)
            rij_flat = rij.flatten()

            df = pd.DataFrame(
                {
                    "Correlation Coefficient": rij_flat,
                    "Condition": condition[2:-1],
                    "Experiment": experiment,
                    "Stimulation": "On" if condition == "2_stim_" else "Off",
                }
            )
            rijframes.append(df)

            fc = ah._functional_complexity(rij)
            mean_rij = np.nanmean(rij)

            df = pd.DataFrame(
                {
                    "Num Bursts": [len(blen)],
                    "Mean Correlation": [mean_rij],
                    "Functional Complexity": [fc],
                    "Condition": condition[2:-1],
                    "Experiment": experiment,
                    "Stimulation": "On" if condition == "2_stim_" else "Off",
                }
            )
            expframes.append(df)

            # if experiment == "210405_C" and "post" in condition:
            # assert False

            plt.close(fig)
            h5.close_hot()
            del h5f


# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #

sns.set_theme(style="whitegrid", palette=None)
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.left"] = False
matplotlib.rcParams["axes.spines.bottom"] = False


violin_kwargs = dict(scale_hue=True, cut=0,)

# Bursts
df_bursts = pd.concat(burstframes, ignore_index=True)
fig, ax = plt.subplots()
sns.violinplot(x="Condition", y="Duration", data=df_bursts, **violin_kwargs)
fig.tight_layout()

fig, ax = plt.subplots()
sns.violinplot(x="Condition", y="Fraction", data=df_bursts, **violin_kwargs)
fig.tight_layout()

# ISI
df_isis = pd.concat(isisframes, ignore_index=True)
df_isis["logISI"] = df_isis.apply(lambda row: np.log10(row["ISI"]), axis=1)
fig, ax = plt.subplots()
sns.violinplot(x="Condition", y="logISI", data=df_isis, **violin_kwargs)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(cc._ticklabels_lin_to_log10_power)
)
fig.tight_layout()

# rij
df_rij = pd.concat(rijframes, ignore_index=True)
fig, ax = plt.subplots()
sns.stripplot(
    x="Condition",
    y="Correlation Coefficient",
    hue="Condition",
    data=df_rij,
    jitter=0.25,
    size=1,
)
ax.get_legend().set_visible(False)
fig.tight_layout()

# exp
df_exp = pd.concat(expframes, ignore_index=True)
fig, ax = plt.subplots()
sns.pointplot(
    x="Condition", y="Num Bursts", hue="Experiment", palette="YlGnBu_d", data=df_exp,
)
ax.get_legend().set_visible(False)
cc._legend_into_new_axes(ax)
fig.tight_layout()


fig, ax = plt.subplots()
sns.pointplot(
    x="Condition",
    y="Mean Correlation",
    hue="Experiment",
    palette="YlGnBu_d",
    data=df_exp,
)
ax.get_legend().set_visible(False)
fig.tight_layout()


fig, ax = plt.subplots()
sns.pointplot(
    x="Condition",
    y="Functional Complexity",
    hue="Experiment",
    palette="YlGnBu_d",
    data=df_exp,
)
ax.get_legend().set_visible(False)
fig.tight_layout()
