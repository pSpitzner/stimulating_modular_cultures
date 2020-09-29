# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-17 13:43:10
# @Last Modified: 2020-09-29 21:44:46
# ------------------------------------------------------------------------------ #


import os
import sys
import glob
import h5py
import matplotlib
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    temp = np.argsort(mod_ids)
    for i in range(0, num_n):
        mod_sorted[i] = np.argwhere(temp == i)

    mod_sort = lambda x: mod_sorted[x]
except:
    mod_sort = lambda x: x


# 2d array, dims: neuron_id x list of spiketime, padded at the end with zeros
spikes = ut.h5_load(args.input_path, "/data/spiketimes")
spikes = np.where(spikes == 0, np.nan, spikes)
sim_duration = ut.h5_load(args.input_path, '/meta/dynamics_simulation_duration')

plt.ion()
fig, ax = plt.subplots(3, 1, sharex=True)

log.info("Plotting raster")
ax[0].set_ylabel("Raster")
for n in range(0, spikes.shape[0]):
    ax[0].plot(spikes[n], mod_sort(n) * np.ones(len(spikes[n])), "|k", alpha = 0.1)

# highlight one neuron in particular:
# sel = np.random.randint(0, num_n)
# ax[0].plot(spikes[sel], mod_sort(sel) ** np.ones(len(spikes[sel])), "|")

log.info("Calculating Population Activity")
ax[1].set_ylabel("ASDR")
pop_act = ut.population_activity(spikes, bin_size=1.0)
ax[1].plot(np.arange(0, len(pop_act)) * 1.0, pop_act)
log.info(f"ASDR (mean): {np.mean(pop_act):g}")
ax[1].text(.95, .95, f"ASDR (mean): {np.mean(pop_act):g}",
    transform=ax[1].transAxes, ha="right", va="top")

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
ax[2].text(.95, .95, f"IBI (mean): {np.mean(ibis):g}\nIBI (median): {np.median(ibis):g}",
    transform=ax[2].transAxes, ha="right", va="top")

# some more meta data
for text in args.input_path.split('/'):
    if '2x2' in text:
        fig.suptitle(text)
ga = ut.h5_load(args.input_path, '/meta/dynamics_gA')
rate = ut.h5_load(args.input_path, '/meta/dynamics_rate') * 1000
tD = ut.h5_load(args.input_path, '/meta/dynamics_tD')
ax[0].set_title(f"Ampa: {ga:.1f} mV", loc='left')
ax[0].set_title(f"Rate: {rate:.1f} Hz", loc='right')
ax[0].set_title(f"tD: {tD:.1f} s", loc='center')

ax[-1].set_xlabel("Time [seconds]")
ax[-1].set_xlim(0,sim_duration)

# plot ibi distribution
# fig, ax = plt.subplots()
# if len(bursts) > 2:
#     ibis = bursts[1:]-bursts[:-1]
#     # bins = np.arange(0, np.nanmax(ibis), 0.5)
#     bins = 10
#     sns.distplot(ibis, ax=ax, bins=bins, label="IBI", hist=True, kde=False)
# ax.set_xlabel("IBI [seconds]")
# ax.set_title(f"Ampa: {ga:.0f} mV", loc='left')
# ax.set_title(f"Rate: {rate:.0f} Hz", loc='right')
# ax.set_title(f"tD: {tD:.1f} s", loc='center')
# for text in args.input_path.split('/'):
#     if '2x2' in text:
#         fig.suptitle(text)

log.setLevel('DEBUG')
s = spikes[1]
s = s[np.isfinite(s)]
bursts, isi_low, hist, edges = logisi.burst_detection_pasquale(s)

per_neuron_bursts, thr, hist, hist_smooth, edges = logisi.network_burst_detection(spikes)


