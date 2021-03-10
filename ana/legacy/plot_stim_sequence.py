# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-12-03 17:56:15
# @Last Modified: 2020-12-04 13:06:47
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

# parse arguments
parser = argparse.ArgumentParser(description="Merge Multidm")
parser.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input path with *.hdf5 files",
    metavar="FILE",
)
args = parser.parse_args()


def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


# if a directory is provided as input, merge individual hdf5 files down
if os.path.isdir(args.input_path):
    candidates = glob.glob(full_path(args.input_path + "/*.hdf5"))
    print(f"{args.input_path} is a directory, using contained hdf5 files")
elif len(glob.glob(full_path(args.input_path))) <= 1:
    print(
        "Provide a directory with hdf5 files or wildcarded path as string: 'path/to/file_ptrn*.hdf5''"
    )
    exit()
else:
    candidates = glob.glob(full_path(args.input_path))
    print(f"{args.input_path} is a (list of) file")


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


def histogram_from_sequences(list_of_sequences, mods=[0, 1, 2, 3]):
    seq_labs, seq_hist = logisi.sequence_entropy(list_of_sequences, mods)
    seq_hist[0] = 0
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


# collect histograms
# from rate based burst detection
r_collection = dict()
r_collection["probs"] = [None] * len(candidates)
r_collection["labels"] = [None] * len(candidates)
r_collection["totals"] = [None] * len(candidates)

# from logisi burst detection
l_collection = dict()
l_collection["probs"] = [None] * len(candidates)
l_collection["labels"] = [None] * len(candidates)
l_collection["totals"] = [None] * len(candidates)

for cdx, candidate in enumerate(tqdm(candidates, desc="Candidate files")):
    spikes = ut.h5_load(candidate, "/data/spiketimes", silent=True)
    mod_ids = ut.h5_load(candidate, "/data/neuron_module_id")

    r_seqs = sequences_from_rates(spikes, mod_ids)
    r_labels, r_probs, r_totals = histogram_from_sequences(r_seqs)
    r_collection["probs"][cdx] = r_probs
    r_collection["labels"][cdx] = r_labels
    r_collection["totals"][cdx] = r_totals

    # l_seqs = sequences_from_logisi(spikes, mod_ids)
    # l_labels, l_probs, l_totals = histogram_from_sequences(l_seqs)
    # l_collection["probs"][cdx] = l_probs
    # l_collection["labels"][cdx] = l_labels
    # l_collection["totals"][cdx] = l_totals


# reshape to get full length sequences
r_probs = np.vstack(r_collection["probs"])
r_labels = r_collection["labels"][0]

l_probs = np.vstack(l_collection["probs"])
l_labels = l_collection["labels"][0]

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[6, 8])
sns.violinplot(
    data=pd.DataFrame(data=r_probs, columns=r_labels),
    ax=axes[0],
    palette="Spectral",
)
axes[0].set_xlabel("Sequence length")
axes[0].set_ylabel("Probability\n(from Rates algorithm)")
axes[0].set_ylim(1e-3, 1)
axes[0].set_title(" ")
axes[0].text(
    0.05,
    0.95,
    f"Num bursts: {np.nanmean(r_collection['totals']):.1f}",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
)

sns.violinplot(data=pd.DataFrame(data=l_probs, columns=l_labels), ax=axes[1],
    palette='Spectral')
axes[1].set_xlabel("Sequence length")
axes[1].set_ylabel("Probability\n(from Logisi)")
# axes[1].set_title("Logisi", loc='left')
axes[1].set_ylim(1e-3, 1)
axes[1].text(
    0.05,
    0.95,
    f"Num bursts: {np.nanmean(l_collection['totals']):.1f}",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
)


for text in args.input_path.split("/"):
    if "stim" in text:
        fig.suptitle("stimulation ON")
    if "dyn" in text:
        fig.suptitle("stimulation OFF")
fig.tight_layout()
