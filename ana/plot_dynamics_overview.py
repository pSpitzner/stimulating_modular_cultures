# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-17 13:43:10
# @Last Modified: 2020-12-03 09:51:15
# ------------------------------------------------------------------------------ #


import os
import sys
import glob
import h5py
import argparse
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from brian2.units import *

log = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut
import logisi as logisi

# interactive plotting
plt.ion()


parser = argparse.ArgumentParser(description="Dynamic Overview")
parser.add_argument("-i", dest="input_path", help="input path", metavar="FILE")
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
args = parser.parse_args()
file = args.input_path

assert len(glob.glob(args.input_path)) != 0, "Invalid input path"

# we want to plot spikes sorted by module, if they exists
try:
    num_n = int(ut.h5_load(args.input_path, "/meta/topology_num_neur"))
    # get the neurons sorted according to their modules
    mod_ids = ut.h5_load(args.input_path, "/data/neuron_module_id")
    mod_sorted = np.zeros(num_n, dtype=int)
    mods = np.sort(np.unique(mod_ids))
    if len(mods) == 1:
        raise NotImplementedError  # avoid resorting.
    temp = np.argsort(mod_ids)
    for i in range(0, num_n):
        mod_sorted[i] = np.argwhere(temp == i)

    mod_sort = lambda x: mod_sorted[x]
except:
    mod_sort = lambda x: x


# 2d array, dims: neuron_id x list of spiketime, padded at the end with zeros
spikes = ut.h5_load(args.input_path, "/data/spiketimes")
spikes = np.where(spikes == 0, np.nan, spikes)
sim_duration = ut.h5_load(args.input_path, "/meta/dynamics_simulation_duration")

try:
    stim_times = ut.h5_load(
        args.input_path, "/data/stimulation_times_as_list", raise_ex=True
    )
except:
    stim_times = None

plt.ion()
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 8))

mod_clrs = []
for m in mods:
    mod_clrs.append(f"C{m:d}")
if len(mod_clrs) == 1:
    mod_clrs[0] = "black"


log.info("Plotting raster")
ax[0].set_ylabel("Raster")
for n in range(0, spikes.shape[0]):
    ax[0].plot(
        spikes[n],
        mod_sort(n) * np.ones(len(spikes[n])),
        "|",
        alpha=0.1,
        color=mod_clrs[mod_ids[n]],
    )

# if stim_times is not None:
# ax[0].plot(stim_times[n], mod_sort(n), "|k", alpha=0.1)


# highlight one neuron in particular:
# sel = np.random.randint(0, num_n)
# ax[0].plot(spikes[sel], mod_sort(sel) ** np.ones(len(spikes[sel])), "|")

log.info("Calculating Population Activity")
ax[1].set_ylabel("ASDR")
bs = 1.0
pop_act = ut.population_activity(spikes, bin_size=bs)
ax[1].plot(np.arange(0, len(pop_act)) * bs, pop_act, color="gray")

log.info(f"ASDR (mean): {np.mean(pop_act):g}")
ax[1].text(
    0.95,
    0.95,
    f"ASDR (mean): {np.mean(pop_act):g}",
    transform=ax[1].transAxes,
    ha="right",
    va="top",
)


ax[2].set_ylabel("Rates")
# population rate from brian
# try:
#     pop_rate_brian = ut.h5_load(file, "/data/population_rate_smoothed")
#     y = pop_rate_brian[:,1]
#     x = pop_rate_brian[:,0] # in seconds, beware bs
#     ax[2].plot(x,y, color="gray", label='brian')
# except Exception as e:
#     log.info(e)

bs = 0.02
pop_rate = logisi.population_rate(spikes, bin_size=bs)
ax[2].plot(
    np.arange(0, len(pop_rate)) * bs, pop_rate / bs, color="darkgray", label=None
)

beg_times = []
end_times = []

for m in mods:
    selects = np.where(mod_ids == m)[0]
    pop_rate = logisi.population_rate(spikes[selects], bin_size=bs)
    mn = np.nanmean(pop_rate / bs)
    ax[2].plot(
        np.arange(0, len(pop_rate)) * bs,
        pop_rate / bs,
        label=f"{m:d}: ({mn:.2f} Hz)",
        color=mod_clrs[m],
    )
    beg_time, end_time = logisi.burst_detection_pop_rate(
        spikes[selects], bin_size=0.02, rate_threshold=15 # Hz
    )

    beg_time, end_time = logisi.merge_if_below_separation_threshold(
        beg_time, end_time, threshold=0.1 # seconds
    )

    beg_times.append(beg_time)
    end_times.append(end_time)

    # ax[2].axhline(y=100*np.nanmean(pop_rate / bs), alpha=.5, color=mod_clrs[m])
    ax[2].plot(
        beg_time, np.ones(len(beg_time)) * (20 + m), marker="4", color=mod_clrs[m], lw=0
    )
    ax[2].plot(
        end_time, np.ones(len(end_time)) * (20 + m), marker="3", color=mod_clrs[m], lw=0
    )

all_begs, all_ends, all_seqs = logisi.system_burst_from_module_burst(beg_times, end_times, threshold=0.1)

ax[2].plot(
    all_begs, np.ones(len(all_begs)) * (25), marker="4", color='black', lw=0
)
ax[2].plot(
    all_ends, np.ones(len(all_ends)) * (25), marker="3", color='black', lw=0
)

ax[2].legend(loc=1)
ax[2].axhline(y=15, ls=":", color="black")


log.info("Detecting Bursts")
ax[3].set_ylabel("Bursts")
ax[3].set_yticks([])
bursts, time_series, summed_series = ut.burst_times(
    spikes, bin_size=0.5, threshold=0.75, mark_only_onset=False, debug=True
)
ax[3].plot(bursts, np.ones(len(bursts)), "|", markersize=12)
ibis = ut.inter_burst_intervals(bursttimes=bursts)
log.info(f"sim duration: {sim_duration} [seconds]")
log.info(f"Num bursts: {len(bursts):g}")
log.info(f"IBI (mean): {np.mean(ibis):g} [seconds]")
log.info(f"IBI (median): {np.median(ibis):g} [seconds]")
ax[3].text(
    0.95,
    0.95,
    f"IBI simple mean: {np.mean(ibis):g}\nvar: {np.var(ibis):g}",
    transform=ax[3].transAxes,
    ha="right",
    va="top",
)

# some more meta data
for text in args.input_path.split("/"):
    # if "2x2" in text:
    fig.suptitle(text)
ga = ut.h5_load(args.input_path, "/meta/dynamics_gA")
rate = ut.h5_load(args.input_path, "/meta/dynamics_rate")
tD = ut.h5_load(args.input_path, "/meta/dynamics_tD")
ax[0].set_title(f"Ampa: {ga:.1f} mV", loc="left")
ax[0].set_title(f"Rate: {rate:.1f} Hz", loc="right")
if "2x2_fixed" in args.input_path:
    k_inter = ut.h5_load(args.input_path, "/meta/topology_k_inter")[0]
    ax[0].set_title(f"k: {k_inter:d}", loc="center")

ax[-1].set_xlabel("Time [seconds]")
ax[-1].set_xlim(0, sim_duration)


# ------------------------------------------------------------------------------ #
# logisi
# ------------------------------------------------------------------------------ #

spiketimes = ut.h5_load(file, "/data/spiketimes", silent=True)
network_bursts, details = logisi.network_burst_detection(
    spiketimes, network_fraction=0.1
)
# logisi.reformat_network_burst(network_bursts, details, file)
ibis = network_bursts["IBI"]
uniq = network_bursts["unique"]
neuron_bursts = details["t_beg"]
num_bursts = len(network_bursts["i_beg"])


res = dict()
print(f"mean ibi pasqu: {np.nanmean(ibis) if len(ibis) > 0 else np.inf}")
print(f"var ibi pasqu: {np.nanvar(ibis) if len(ibis) > 0 else np.inf}")
print(f"mean fraction: {np.nanmean(uniq) if len(uniq) > 0 else 0}")

ax[3].text(
    0.95,
    0.05,
    f"IBI logisi mean: {np.nanmean(ibis):g}\nvar: {np.nanvar(ibis):g}",
    transform=ax[3].transAxes,
    ha="right",
    va="bottom",
)
ax[3].text(
    0.05,
    0.05,
    f"unique neurons: {np.nanmean(uniq):.1f} / {num_n}",
    transform=ax[3].transAxes,
    ha="left",
    va="bottom",
)

# network burst begin
ax[3].plot(
    neuron_bursts[network_bursts["i_beg"]],
    0.98 * np.ones(len(network_bursts["i_beg"])),
    "|y",
    markersize=6,
)
# network burst median
ax[3].plot(
    network_bursts["t_med"],
    0.98 * np.ones(len(network_bursts["t_med"])),
    "|r",
    markersize=6,
)
# neuron bursts
ax[3].plot(neuron_bursts, 1.02 * np.ones(len(neuron_bursts)), "|g", markersize=6)

ax[3].set_ylim(0.9, 1.1)


# ------------------------------------------------------------------------------ #
# bursts on module level.
# ------------------------------------------------------------------------------ #

# spiketimes = ut.h5_load(file, "/data/spiketimes", silent=True)
# modtimes = []
# for m in range(0,4):
#     mod_sel = np.where(mod_ids == m)[0]
#     s = spiketimes[mod_sel, :].flatten()
#     s[s==0] = np.nan
#     s = np.sort(s)
#     s[np.isnan(s)] = 0
#     modtimes.append(s)

#     mod_bursts, _, _, _ = logisi.burst_detection_pasquale(s, cutoff=0.02)
#     mod_bursts = mod_bursts['t_med']
#     ax[0].plot(mod_bursts, 12 + m * 25 * np.ones(len(mod_bursts)), "|r", alpha=0.8, markersize=12, zorder=1)

# modtimes = np.vstack(modtimes)


# ------------------------------------------------------------------------------ #
# sequences
# ------------------------------------------------------------------------------ #

starters = details["neuron_ids"][network_bursts["i_beg"]]

# we could look at modules instead of individual neurons.
# try:
#     starters = mod_ids[starters]
# except:
#     pass


per_mod_hists = np.zeros(num_n, dtype=np.int)
per_mod_ranks = np.zeros(num_n, dtype=np.int)

# get histogram counts per on a per-module level
mod_rank = np.zeros(len(mods), dtype=np.int)
for m in mods:
    idx = np.where(mod_ids[starters] == m)[0]
    hist, edges = np.histogram(
        starters[idx], bins=np.arange(0, num_n + 1), density=False
    )
    edges = edges[:-1]
    # remove zero entries
    nzdx = np.where(hist > 0)[0]
    hist = hist[nzdx]
    edges = edges[nzdx]
    per_mod_hists[edges] = hist
    mod_rank[m] = np.sum(hist)

    # sort by largest amount
    jdx = np.flip(np.argsort(hist))
    # hist = hist[jdx]
    # edges = edges[jdx]
    per_mod_ranks[edges[jdx]] = np.arange(len(jdx))

get_mod_rank = lambda m: np.where(np.flip(np.argsort(mod_rank)) == m)[0][0]
get_mod_sum = lambda m: mod_rank[m]

# # order starters by id first
starters = np.sort(np.unique(starters))
# # then by histogram height
# starters = starters[ np.flip(np.argsort(per_mod_hists[starters])) ]
# # then by module id according to module dominance
# for m in mods:
#     idx = np.where(mod_ids[starters] == m)[0]
#     jdx = np.argsort(np.vectorize(get_mod_))
# starters = starters[ np.flip(np.argsort(mod_ids[starters])) ]


df = pd.DataFrame(
    data={
        "n_id": starters,
        "mod": mod_ids[starters],
        "mod_rank": np.vectorize(get_mod_rank)(mod_ids[starters]),
        "mod_sum": np.vectorize(get_mod_sum)(mod_ids[starters]),
        "per_mod_hists": per_mod_hists[starters],
        "per_mod_ranks": per_mod_ranks[starters],
    }
)

df = df.sort_values(by=["per_mod_hists", "mod_rank"], ascending=[False, True])


fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=[12, 4])

# plot histogram with seaborn
# sns.catplot(data=df, col="mod", y="per_mod_hists", x="per_mod_ranks", palette="rocket", ax=ax2, kind="bar", aspect=.8)
sns.barplot(
    data=df,
    x="n_id",
    order=df["n_id"],
    y="per_mod_hists",
    hue="mod",
    hue_order=mods,
    dodge=False,
    palette="tab10",
    ax=ax2[0],
)

sns.barplot(
    data=df,
    x="mod_rank",
    y="mod_sum",
    hue="mod",
    hue_order=mods,
    dodge=False,
    palette="tab10",
    ax=ax2[1],
)

# ax2.axhline(0, color="k", clip_on=False)
# ax2.set_xticklabels(edges[:lim])
# ax2.get_legend().set_visible(False)
# ax2[0].legend
# some more meta data
for text in args.input_path.split("/"):
    # if "2x2" in text:
    fig2.suptitle(text)
ax2[0].set_title(f"Ampa: {ga:.1f} mV", loc="left")
ax2[0].set_title(f"Rate: {rate:.1f} Hz", loc="right")
if "2x2_fixed" in args.input_path:
    k_inter = ut.h5_load(args.input_path, "/meta/topology_k_inter")[0]
    ax2[0].set_title(f"k: {k_inter:d}", loc="center")


# ------------------------------------------------------------------------------ #
# new, full sequence analysis
# ------------------------------------------------------------------------------ #

seqs = logisi.sequence_detection(network_bursts, details, mod_ids)

def plot_seqs_as_bars(list_of_sequences, ax, ax_merged):
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

    clrs = 4 * seq_lens[nz_idx] + seq_begs[nz_idx]
    clrs = mpl.cm.get_cmap("Spectral")(clrs / (5 * len(mods)))

    sns.barplot(
        x=seq_str_labs[nz_idx], y=seq_hist[nz_idx], dodge=False, palette=clrs, ax=ax,
    )
    ax.text(
        0.95,
        0.95,
        f"Num bursts: {np.sum(seq_hist[nz_idx]):d}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    # plot by seq length
    # dont use seq_lens as is, this will just get the permutaitons of labels
    unique, counts = np.unique(seq_lens[nz_idx], return_counts=True)
    clrs = 4 * unique + 1
    clrs = mpl.cm.get_cmap("Spectral")(clrs / (5 * len(mods)))

    sns.barplot(
        x=unique, y=counts, dodge=False, palette=clrs, ax=ax_merged,
    )
    ax_merged.text(
        0.95,
        0.95,
        f"Num bursts: {np.sum(seq_hist[nz_idx]):d}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )





fig3, ax3 = plt.subplots(nrows=2, ncols=1, figsize=[12, 6])
fig3b, ax3b = plt.subplots(nrows=2, ncols=1, figsize=[6, 6])
plot_seqs_as_bars(seqs["module_seq"], ax3[0], ax3b[0])
plot_seqs_as_bars(all_seqs, ax3[1], ax3b[1])

ax3[0].set_ylabel("From Logisi")
ax3[1].set_ylabel("From Rates")
ax3[0].set_title("Sequences")
ax3b[0].set_ylabel("From Logisi")
ax3b[1].set_ylabel("From Rates")
ax3b[0].set_title("Sequence Length")

for text in args.input_path.split("/"):
    # if "2x2" in text:
    fig3.suptitle(f"{text}")
    fig3b.suptitle(f"{text}")


