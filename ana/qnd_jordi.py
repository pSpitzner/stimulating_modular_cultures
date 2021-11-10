# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-09 11:16:44
# @Last Modified: 2021-10-23 00:06:23
# ------------------------------------------------------------------------------ #
# All the plotting is in here.
#
# What's a good level of abstraction?
# * Basic routines that plot one thing or the other, directly from file.
# * target an mpl ax element with normal functions and
# * provide higher level ones that combine those to `overview` panels
# ------------------------------------------------------------------------------ #


# fmt: off
import os
import sys
import glob
import h5py
import argparse
import logging
import functools
import numpy as np

import matplotlib
# matplotlib.rcParams['font.sans-serif'] = "Arial"
# matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"]=8
matplotlib.rcParams["ytick.labelsize"]=8
matplotlib.rcParams["axes.titlesize"]= 8
matplotlib.rcParams["axes.labelsize"]= 8
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
# matplotlib.rcParams["axes.spines.right"] = False
# matplotlib.rcParams["axes.spines.top"] = False
# matplotlib.rcParams["axes.spines.left"] = False
# matplotlib.rcParams["axes.spines.bottom"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
    "#233954", "#ea5e48", "#1e7d72", "#f49546", "#e8bf58", # dark
    "#5886be", "#f3a093", "#53d8c9", "#f2da9c", "#f9c192", # light
    ]) # qualitative, somewhat color-blind friendly, in mpl words 'tab5'

import seaborn as sns

path_off = [
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_OFF_00.txt",
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_OFF_01.txt",
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_OFF_02.txt",
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_OFF_03.txt",
]

path_on = [
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_ON_00.txt",
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_ON_01.txt",
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_ON_02.txt",
"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_ON_03.txt",

]

res_off = []
for path in path_off:
    res_off.append(np.loadtxt(path, delimiter=",", skiprows=1))
res_off =  np.vstack(res_off)

res_on = []
for path in path_on:
    res_on.append(np.loadtxt(path, delimiter=",", skiprows=1))
res_on = np.vstack(res_on)

idx_on = np.where(res_on[:, 1] > 0.0)[0]
idx_off = np.where(res_off[:, 1] > 0.0)[0]

# fig, ax = plt.subplots()
# sns.histplot(res_off[idx_off, 1], bins=20, stat="density", label="off", ax=ax)
# sns.histplot(res_on[idx_on, 1], bins=20, stat="density", label="on", ax=ax)
# ax.set_xlim(0, 1)
# ax.set_title("size")
# ax.legend()

# fig, ax = plt.subplots()
# sns.histplot(res_off[idx_off, -1] - res_off[idx_off, -2], bins=20, stat="density", label="off", ax=ax)
# sns.histplot(res_on[idx_on, -1] - res_on[idx_on, -2], bins=20, stat="density", label="on", ax=ax)
# ax.set_title("duration")
# ax.legend()


g = sns.jointplot(x=res_off[idx_off, 1], y=res_off[idx_off, -1] - res_off[idx_off, -2],
    # marker="+",
    s=1.5,
    color="C0", xlim=(0,1), ylim=(0,5), marginal_kws=dict(bins=160, stat="probability"))
g.set_axis_labels("Size", "Duration")
g.fig.get_axes()[-1].set_title("Off")
g.fig.tight_layout()

g = sns.jointplot(x=res_on[idx_on, 1], y=res_on[idx_on, -1] - res_on[idx_on, -2],
    # marker="+",
    s=1.5,
    color="C1", xlim=(0,1), ylim=(0,5), marginal_kws=dict(bins=160, stat="probability"))
g.set_axis_labels("Size", "Duration")
g.fig.get_axes()[-1].set_title("On")
g.fig.tight_layout()




import plot_helper as ph

h5f = ph.ah.prepare_file("/Users/paul/mpi/simulation/brian_modular_cultures/data_for_jordi/2021-10-11/stim=off_rep=000.hdf5")
fig = ph.overview_dynamic(h5f, skip=["bursts"])

temp = np.loadtxt("/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jordi/prepped/GNA sizes ALL_OFF_00.txt",
    delimiter=",", skiprows=1)
beg_times = temp[:, -2]
end_times = temp[:, -1]

y_offset = np.random.uniform(-1, 1, size=len(beg_times))
ph._plot_bursts_into_timeseries(fig.get_axes()[3], beg_times, end_times, y_offset = y_offset)
