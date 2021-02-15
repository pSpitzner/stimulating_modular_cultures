# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-17 13:43:10
# @Last Modified: 2021-02-15 12:22:50
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

avg_spike_rate = len(np.where(np.isfinite(spikes))[0])/sim_duration/spikes.shape[0]
log.info(f"neuron spike rate ~ {avg_spike_rate:.2f} Hz")

try:
    stim_times = ut.h5_load(
        args.input_path, "/data/stimulation_times_as_list", raise_ex=True
    )
except:
    stim_times = None

plt.ion()
# fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(4, 7))

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
        alpha=0.5,
        color=mod_clrs[mod_ids[n]],
    )

# if stim_times is not None:
# ax[0].plot(stim_times[n], mod_sort(n), "|k", alpha=0.1)


# highlight one neuron in particular:
# sel = np.random.randint(0, num_n)
# ax[0].plot(spikes[sel], mod_sort(sel) ** np.ones(len(spikes[sel])), "|")


ax[1].set_ylabel("Rates")
# population rate from brian
# try:
#     pop_rate_brian = ut.h5_load(file, "/data/population_rate_smoothed")
#     y = pop_rate_brian[:,1]
#     x = pop_rate_brian[:,0] # in seconds, beware bs
#     ax[1].plot(x,y, color="gray", label='brian')
# except Exception as e:
#     log.info(e)

bs = 0.02
pop_rate = logisi.population_rate(spikes, bin_size=bs)
ax[1].plot(
    np.arange(0, len(pop_rate)) * bs, pop_rate / bs, color="darkgray", label=None
)

beg_times = []
end_times = []

# inconsistency, fix this!
bs = 0.002
for m in mods:
    selects = np.where(mod_ids == m)[0]
    pop_rate = logisi.population_rate(spikes[selects], bin_size=bs)
    pop_rate = logisi.smooth_rate(pop_rate, clock_dt=bs, width=0.02)
    mn = np.nanmean(pop_rate / bs)
    ax[1].plot(
        np.arange(0, len(pop_rate)) * bs,
        pop_rate / bs,
        label=f"{m:d}: ({mn:.2f} Hz)",
        color=mod_clrs[m],
    )
    beg_time, end_time = logisi.burst_detection_pop_rate(
        spikes[selects], bin_size=0.02, rate_threshold=15, # Hz
        highres_bin_size = bs,
    )

    beg_time, end_time = logisi.merge_if_below_separation_threshold(
        beg_time, end_time, threshold=0.1 # seconds
    )

    beg_times.append(beg_time)
    end_times.append(end_time)

    # ax[1].axhline(y=100*np.nanmean(pop_rate / bs), alpha=.5, color=mod_clrs[m])
    ax[1].plot(
        beg_time, np.ones(len(beg_time)) * (20 + m), marker="4", color=mod_clrs[m], lw=0
    )
    ax[1].plot(
        end_time, np.ones(len(end_time)) * (20 + m), marker="3", color=mod_clrs[m], lw=0
    )

all_begs, all_ends, all_seqs = logisi.system_burst_from_module_burst(beg_times, end_times, threshold=0.1)

ax[1].plot(
    all_begs, np.ones(len(all_begs)) * (25), marker="4", color='black', lw=0
)
ax[1].plot(
    all_ends, np.ones(len(all_ends)) * (25), marker="3", color='black', lw=0
)

leg = ax[1].legend(loc=1)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_facecolor("#F0F0F0")
leg.get_frame().set_alpha(0.95)
leg.set_title("Module Rates")
ax[1].axhline(y=15, ls=":", color="black")


log.info("Detecting Bursts")
ax[2].set_ylabel("Potential")
ax[3].set_ylabel("Homplast")
bursts, time_series, summed_series = ut.burst_times(
    spikes, bin_size=0.5, threshold=0.75, mark_only_onset=False, debug=True
)
# ax[3].plot(bursts, np.ones(len(bursts)), "|", markersize=12)
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
    if "stim" in text:
        fig.suptitle("stimulation ON")
    if "dyn" in text:
        fig.suptitle("stimulation OFF")

ga = ut.h5_load(args.input_path, "/meta/dynamics_gA")
rate = ut.h5_load(args.input_path, "/meta/dynamics_rate")
tD = ut.h5_load(args.input_path, "/meta/dynamics_tD")
ax[0].set_title(f"Ampa: {ga:.1f} mV", loc="left")
ax[0].set_title(f"Rate: {rate:.1f} Hz", loc="right")
if "2x2_fixed" in args.input_path:
    k_inter = ut.h5_load(args.input_path, "/meta/topology_k_inter")[0]
    ax[0].set_title(f"k: {k_inter:d}", loc="center")

ax[-1].set_xlabel("Time [seconds]")
# ax[-1].set_xlim(0, sim_duration)

# ------------------------------------------------------------------------------ #
# state variabels
# ------------------------------------------------------------------------------ #

try:
    t = ut.h5_load(args.input_path, '/data/state_vars_time')
    v = ut.h5_load(args.input_path, '/data/state_vars_v')
    H = ut.h5_load(args.input_path, '/data/state_vars_H')

    ax[-2].plot(t,v[0])
    ax[-1].plot(t,H[0])
except:
    log.info("Skipped state variables")



