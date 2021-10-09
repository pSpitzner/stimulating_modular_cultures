import os
import sys
import glob
import h5py
import argparse
import numbers

import matplotlib

sys.path.append(os.path.abspath("/Users/paul/code/pyhelpers/"))


matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 150
from matplotlib.ticker import MultipleLocator, LogLocator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import hi5 as h5
from benedict import benedict

matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["figure.dpi"] = 150

# matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.bottom"] = False

import matplotlib.pyplot as plt


fig, ax1 = plt.subplots()
fig, ax2 = plt.subplots()

dat = benedict()
for sdx, strength in enumerate([30, 35, 40]):
    try:
        dat[strength] = h5.recursive_load(
            f"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/epsp_stim/{strength}.hdf5"
        )
        sns.histplot(
            data=dat[strength]["num_spikes"],
            ax=ax1,
            bins=np.arange(-0.5, 5.6, 1),
            color=f"C{sdx}",
            alpha=0.2,
            # kde=True,
            stat= "density",
            element="step",
            label=strength,
        )
        sns.histplot(
            data=dat[strength]["times_to_first"],
            ax=ax2,
            # binwidth=25,
            color=f"C{sdx}",
            alpha=0.2,
            kde=True,
            stat= "density",
            element="poly",
            label=strength,
        )
    except:
        pass

ax1.legend()
ax2.legend()
ax1.set_xlim(-.5, 5.5)
ax1.set_xlabel("Num spikes")
ax2.set_xlabel("Times to first")
ax1.get_figure().tight_layout()
ax2.get_figure().tight_layout()
