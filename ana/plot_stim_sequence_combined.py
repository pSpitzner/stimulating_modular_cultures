# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-12-03 17:56:15
# @Last Modified: 2020-12-11 10:04:13
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import logging
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut
import logisi as logisi

input_path_off = (
    "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/dyn/2x2_fixed/*.hdf5"
)
input_path_on = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/stim/2x2_fixed/*.hdf5"


def process_candidates(
    input_path, Method, stim="Unknown", skip_0=False,
):

    columns = ["Seq. Length", "Probability", "total", "Stimulation", "skip_0", "Method"]
    df = pd.DataFrame(columns=columns)

    candidates = glob.glob(full_path(input_path))

    if stim == "Unknown":
        for text in input_path.split("/"):
            if "stim" in text:
                stim = "On"
            elif "dyn" in text:
                stim = "Off"

    for cdx, candidate in enumerate(tqdm(candidates, desc="Candidate files")):
        spikes = ut.h5_load(candidate, "/data/spiketimes", silent=True)
        mod_ids = ut.h5_load(candidate, "/data/neuron_module_id")

        if Method == "Logisi":
            sequence_function = sequences_from_logisi
        elif Method == "Rates":
            sequence_function = sequences_from_rates
        seqs = sequence_function(spikes, mod_ids)
        labels, probs, total = histogram_from_sequences(
            seqs, mods=[0, 1, 2, 3], skip_0=skip_0
        )

        for ldx, l in enumerate(labels):
            df = df.append(
                pd.DataFrame(
                    data=[[labels[ldx], probs[ldx], total, stim, skip_0, Method]],
                    columns=columns,
                ),
                ignore_index=True,
            )

    return df


def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


def sequences_from_rates(spikes, mod_ids):
    beg_times = []
    end_times = []
    for m in np.unique(mod_ids):
        selects = np.where(mod_ids == m)[0]
        beg_time, end_time = logisi.burst_detection_pop_rate(
            spikes[selects], bin_size=0.02, rate_threshold=15  # Hz
        )

        beg_time, end_time = logisi.merge_if_below_separation_threshold(
            beg_time, end_time, threshold=0.1  # seconds
        )

        beg_times.append(beg_time)
        end_times.append(end_time)

    all_begs, all_ends, all_seqs = logisi.system_burst_from_module_burst(
        beg_times, end_times, threshold=0.1
    )

    return all_seqs


def sequences_from_logisi(spikes, mod_ids):
    network_bursts, details = logisi.network_burst_detection(
        spikes, network_fraction=0.1
    )
    all_seqs = logisi.sequence_detection(network_bursts, details, mod_ids)

    return all_seqs["module_seq"]


def histogram_from_sequences(list_of_sequences, mods=[0, 1, 2, 3], skip_0=False):
    seq_labs, seq_hist = logisi.sequence_entropy(list_of_sequences, mods)
    if skip_0:
        for sdx, s in enumerate(seq_labs):
            if s[0] == 0:
                seq_hist[sdx] = 0

    seq_str_labs = np.array(logisi.sequence_labels_to_strings(seq_labs))
    seq_lens = np.zeros(len(seq_str_labs), dtype=np.int)
    seq_begs = np.zeros(len(seq_str_labs), dtype=np.int)
    for idx, s in enumerate(seq_str_labs):
        seq_lens[idx] = len(s)
        seq_begs[idx] = int(s[0])

    skip_empty = True
    if skip_empty:
        nz_idx = np.where(seq_hist != 0)[0]
    else:
        nz_idx = slice(None)

    # plot by seq length
    # get possible
    catalog = np.unique(seq_lens)
    len_hist = np.zeros(len(catalog))
    lookup = dict()
    for c in catalog:
        lookup[c] = np.where(catalog == c)[0][0]

    for sdx, s in enumerate(seq_hist):
        c = seq_lens[sdx]
        len_hist[lookup[c]] += s

    # assert np.sum(len_hist) == len(list_of_sequences), "sanity check"
    # total = len(list_of_sequences)
    total = np.sum(len_hist)

    return catalog, len_hist / total, total


# fmt:off

# df = pd.DataFrame()
# df = df.append(process_candidates(input_path_off, "Rates",  skip_0=False), ignore_index=True)
# df = df.append(process_candidates(input_path_off, "Logisi", skip_0=False), ignore_index=True)
# df = df.append(process_candidates(input_path_on,  "Rates",  skip_0=False), ignore_index=True)
# df = df.append(process_candidates(input_path_on,  "Logisi", skip_0=False), ignore_index=True)
# df = df.append(process_candidates(input_path_on,  "Rates",  skip_0=True), ignore_index=True)
# df = df.append(process_candidates(input_path_on,  "Logisi", skip_0=True), ignore_index=True)

# fmt:on

# df.to_hdf("/Users/paul/Desktop/pd.hdf5", "/data/df")
df = pd.read_hdf("/Users/paul/Desktop/pd.hdf5", "/data/df")

def plot(data, compare, title):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 6])
    # sns.violinplot(
    #     data=data,
    #     x="Seq. Length",
    #     y="Probability",
    #     hue=compare,
    #     scale_hue=True,
    #     scale='width',
    #     split=True,
    #     inner="quartile",
    #     linewidth=.5,
    #     ax=ax,
    #     palette={"Off": "C0", "On": "C1"}
    # )

    sns.boxplot(
        data=data,
        x="Seq. Length",
        y="Probability",
        hue=compare,
        fliersize=.5,
        linewidth=.5,
        ax=ax,
        palette={"Off": "C0", "On": "C1"}
    )

    ax.set_title(title)
    ax.get_legend().set_visible(False)
    ax.axhline(0, ls=':', color='black', lw=.75, zorder=-1)
    ax.axhline(1, ls=':', color='black', lw=.75, zorder=-1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.tick_params(axis="x", which="major", length=0)
    fig.tight_layout()


    ax.set_ylim(-.01, 1.01)
    # ax.set_ylim(-.01,.2)

    return fig, ax


plot(
    title="From Rates",
    compare="Stimulation",
    data=df.loc[((df["skip_0"] == False) & (df["Method"] == "Rates"))],
)

plot(
    title="From Logisi",
    compare="Stimulation",
    data=df.loc[((df["skip_0"] == False) & (df["Method"] == "Logisi"))],
)

plot(
    title="From Rates, ignoring 0",
    compare="Stimulation",
    data=df.loc[
        (
            (df["Method"] == "Rates")
            & (
                ((df["skip_0"] == True) & (df["Stimulation"] == "On"))
                | ((df["skip_0"] == False) & (df["Stimulation"] == "Off"))
            )
        )
    ],
)

plot(
    title="From Logisi, ignoring 0",
    compare="Stimulation",
    data=df.loc[
        (
            (df["Method"] == "Logisi")
            & (
                ((df["skip_0"] == True) & (df["Stimulation"] == "On"))
                | ((df["skip_0"] == False) & (df["Stimulation"] == "Off"))
            )
        )
    ]
)


# for i in plt.get_fignums():
#     plt.figure(i)
#     plt.savefig(f"/Users/paul/Desktop/figure_{i:d}_2.pdf", dpi=600)

