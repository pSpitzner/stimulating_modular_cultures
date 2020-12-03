# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-12-03 17:56:15
# @Last Modified: 2020-12-03 22:45:49
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
    # dont use seq_lens as is, this will just get the permutaitons of labels
    labels, counts = np.unique(seq_lens[nz_idx], return_counts=True)

    # make sure every histogram has the same labels
    fixed_labels = np.arange(1, len(mods)+1).astype(int)
    fixed_counts = np.zeros(len(fixed_labels)).astype(int)

    for ldx, val in enumerate(counts):
        fixed_counts[int(labels[ldx]-1)] = val

    return fixed_labels, fixed_counts


# collect histograms
rate_collection = dict()
rate_collection["counts"] = [None] * len(candidates)
rate_collection["labels"] = [None] * len(candidates)

logisi_collection = dict()
logisi_collection["counts"] = [None] * len(candidates)
logisi_collection["labels"] = [None] * len(candidates)

for cdx, candidate in enumerate(tqdm(candidates, desc="Candidate files")):
    spikes = ut.h5_load(candidate, "/data/spiketimes", silent=True)
    mod_ids = ut.h5_load(candidate, "/data/neuron_module_id")

    r_seqs = sequences_from_rates(spikes, mod_ids)
    r_labels, r_counts = histogram_from_sequences(r_seqs)
    rate_collection["counts"][cdx] = r_counts
    rate_collection["labels"][cdx] = r_labels

    l_seqs = sequences_from_logisi(spikes, mod_ids)
    l_labels, l_counts = histogram_from_sequences(l_seqs)
    logisi_collection["counts"][cdx] = l_counts
    logisi_collection["labels"][cdx] = l_labels


# reshape to get full length sequences
rate_counts = np.vstack(rate_collection["counts"])
rate_labels = rate_collection["labels"][0]

logisi_counts = np.vstack(logisi_collection["counts"])
logisi_labels = logisi_collection["labels"][0]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[12, 4])
sns.barplot(data=pd.DataFrame(data=rate_counts, columns=rate_labels), ax=axes[0])
axes[0].set_xlabel("Sequence length")
axes[0].set_ylabel("Occurrences")
axes[0].set_title("Rates")

sns.barplot(data=pd.DataFrame(data=logisi_counts, columns=logisi_labels), ax=axes[1])
axes[1].set_xlabel("Sequence length")
axes[1].set_ylabel("Occurrences")
axes[1].set_title("Logisi")
fig.suptitle(args.input_path)
