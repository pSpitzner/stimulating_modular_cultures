# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-12-03 17:56:15
# @Last Modified: 2021-02-15 16:33:16
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


k = 5
input_path = dict()
input_path[
    "off"
] = f"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/dyn/2x2_fixed/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k:d}_rep=*.hdf5"

for stim in ["0", "02", "012", "0123"]:
    input_path[
        stim
    ] = f"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jitter_{stim}/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k:d}_rep=*.hdf5"


super_long_seq_lists = dict()

def process_candidates(
    input_path, Method, stim="Unknown", skip_0=False,
):

    columns = ["Seq. Length", "Probability", "total", "Stimulation", "skip_0", "Method"]
    df = pd.DataFrame(columns=columns)

    candidates = glob.glob(full_path(input_path))

    for cdx, candidate in enumerate(tqdm(candidates, desc="Candidate files")):
        spikes = ut.h5_load(candidate, "/data/spiketimes", silent=True)
        mod_ids = ut.h5_load(candidate, "/data/neuron_module_id")

        if Method == "Logisi":
            sequence_function = sequences_from_logisi
        elif Method == "Rates":
            sequence_function = sequences_from_rates
        global seqs, super_long_seq_list
        seqs = sequence_function(spikes, mod_ids)
        super_long_seq_lists[stim] += seqs
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
            spikes[selects], bin_size=0.02, rate_threshold=15,  # Hz
            highres_bin_size = 0.002
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


df = pd.DataFrame()
for key in input_path.keys():
    super_long_seq_lists[key] = []
    path = input_path[key]
    df = df.append(
        process_candidates(path, "Rates", stim=key, skip_0=False), ignore_index=True
    )


df.to_hdf(
    "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/seqs_k={k:d}.hdf5",
    "/data/df",
)
# df = pd.read_hdf("/Users/paul/Desktop/pd.hdf5", "/data/df")


def plot(data, compare, title, **kwargs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[4, 2.5])
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
        fliersize=0.5,
        linewidth=0.5,
        ax=ax,
        **kwargs
        # palette={"Off": "C0", "On": "C1"},
    )

    ax.set_title(title)
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
    ax.legend()

    return fig, ax



def stim_clr(base_as_hex, alpha):
    import matplotlib.colors as mcolors
    def rgba_to_rgb(c, bg="white"):
        bg = mcolors.to_rgb(bg)
        alpha = c[-1]

        res = (
            (1 - alpha) * bg[0] + alpha * c[0],
            (1 - alpha) * bg[1] + alpha * c[1],
            (1 - alpha) * bg[2] + alpha * c[2],
        )
        return res
    global base
    base = list(mcolors.to_rgba(base_as_hex))
    base[3] = alpha
    return mcolors.to_hex(rgba_to_rgb(base))

c1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

palette={"off": "C0",
    "0": stim_clr(c1, 0.25),
    "02": stim_clr(c1, 0.5),
    "012": stim_clr(c1, 0.75),
    "0123": stim_clr(c1, 1),
}


fig, ax = plot(
    title="From Rates",
    compare="Stimulation",
    data=df.loc[((df["skip_0"] == False) & (df["Method"] == "Rates"))],
    palette=palette
)

ax.set_title(f"k={k}")
plt.show()

# for i in plt.get_fignums():
#     plt.figure(i)
#     plt.savefig(f"/Users/paul/Desktop/figure_{i:d}_2.pdf", dpi=600)
