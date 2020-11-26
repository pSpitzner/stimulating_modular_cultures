# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-17 13:43:10
# @Last Modified: 2020-11-26 18:22:55
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
fig, ax = plt.subplots(3, 1, sharex=True)

log.info("Plotting raster")
ax[0].set_ylabel("Raster")
for n in range(0, spikes.shape[0]):
    ax[0].plot(spikes[n], mod_sort(n) * np.ones(len(spikes[n])), "|k", alpha=0.1)

# if stim_times is not None:
# ax[0].plot(stim_times[n], mod_sort(n), "|k", alpha=0.1)


# highlight one neuron in particular:
# sel = np.random.randint(0, num_n)
# ax[0].plot(spikes[sel], mod_sort(sel) ** np.ones(len(spikes[sel])), "|")

log.info("Calculating Population Activity")
ax[1].set_ylabel("ASDR")
bs = 1.0
pop_act = ut.population_activity(spikes, bin_size=bs)
ax[1].plot(np.arange(0, len(pop_act)) * bs, pop_act)

log.info(f"ASDR (mean): {np.mean(pop_act):g}")
ax[1].text(
    0.95,
    0.95,
    f"ASDR (mean): {np.mean(pop_act):g}",
    transform=ax[1].transAxes,
    ha="right",
    va="top",
)

# population rate from brian
try:
    pop_rate = ut.h5_load(file, "/data/population_rate_smoothed")
    y = pop_rate[:,1] / np.nanmax(pop_rate[:,1]) * np.nanmax(pop_act)
    x = pop_rate[:,0] # in seconds, beware bs
    ax[1].plot(x,y)
except Exception as e:
    log.info(e)




log.info("Detecting Bursts")
ax[2].set_ylabel("Bursts")
ax[2].set_yticks([])
bursts, time_series, summed_series = ut.burst_times(
    spikes, bin_size=0.5, threshold=0.75, mark_only_onset=False, debug=True
)
ax[2].plot(bursts, np.ones(len(bursts)), "|", markersize=12)
ibis = ut.inter_burst_intervals(bursttimes=bursts)
log.info(f"sim duration: {sim_duration} [seconds]")
log.info(f"Num bursts: {len(bursts):g}")
log.info(f"IBI (mean): {np.mean(ibis):g} [seconds]")
log.info(f"IBI (median): {np.median(ibis):g} [seconds]")
ax[2].text(
    0.95,
    0.95,
    f"IBI simple mean: {np.mean(ibis):g}\nvar: {np.var(ibis):g}",
    transform=ax[2].transAxes,
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
    spiketimes, network_fraction=0.25
)
# logisi.reformat_network_burst(network_bursts, details, file)
ibis = network_bursts["IBI"]
uniq = network_bursts["unique"]
neuron_bursts = details["t_beg"]
num_bursts = len(network_bursts["i_beg"])

seqs = logisi.sequence_detection(network_bursts, details, mod_ids)


res = dict()
print(f"mean ibi pasqu: {np.nanmean(ibis) if len(ibis) > 0 else np.inf}")
print(f"var ibi pasqu: {np.nanvar(ibis) if len(ibis) > 0 else np.inf}")
print(f"mean fraction: {np.nanmean(uniq) if len(uniq) > 0 else 0}")

ax[2].text(
    0.95,
    0.05,
    f"IBI logisi mean: {np.nanmean(ibis):g}\nvar: {np.nanvar(ibis):g}",
    transform=ax[2].transAxes,
    ha="right",
    va="bottom",
)
ax[2].text(
    0.05,
    0.05,
    f"unique neurons: {np.nanmean(uniq):.1f} / {num_n}",
    transform=ax[2].transAxes,
    ha="left",
    va="bottom",
)

# network burst begin
ax[2].plot(
    neuron_bursts[network_bursts["i_beg"]],
    0.98 * np.ones(len(network_bursts["i_beg"])),
    "|y",
    markersize=6,
)
# network burst median
ax[2].plot(
    network_bursts["t_med"],
    0.98 * np.ones(len(network_bursts["t_med"])),
    "|r",
    markersize=6,
)
# neuron bursts
ax[2].plot(neuron_bursts, 1.02 * np.ones(len(neuron_bursts)), "|g", markersize=6)

ax[2].set_ylim(0.9, 1.1)


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
