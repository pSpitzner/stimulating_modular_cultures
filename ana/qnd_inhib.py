# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-06-08 13:49:46
# @Last Modified: 2021-06-08 14:13:05
# ------------------------------------------------------------------------------ #


import os
import sys
import glob
import h5py
import argparse
import numbers

import matplotlib
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams['figure.dpi'] = 150

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import hi5 as h5

from mpl_toolkits.mplot3d import Axes3D

input_path = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/inhibition_test/parameter_sweep_3.hdf5"

base_colors = dict()
base_colors["num_bursts"] = cc.to_hex(f"black")
base_colors["num_b_1"] = cc.to_hex(f"C0")
base_colors["num_b_geq_4"] = cc.to_hex(f"C1")
base_colors["sys_rate_cv"] = cc.to_hex(f"C2")
base_colors["mean_rate"] = cc.to_hex(f"C3")

titles = dict()
titles["num_bursts"] = "Number of bursts"
titles["num_b_1"] = "Single-module bursts"
titles["num_b_geq_4"] = "System-wide bursts"
titles["sys_rate_cv"] = "Coefficient of variation"
titles["mean_rate"] = "Mean Rate"

for odx, obs_to_plot in enumerate(["num_b_1", "num_b_geq_4", "num_bursts", "sys_rate_cv", "mean_rate"]):

    fig, ax = plt.subplots()

    ax.set_title(titles[obs_to_plot])

    x = h5.load(input_path, "/data/axis_jG", silent=True)
    data_nd = h5.load(input_path, f"/data/{obs_to_plot}", silent=True)
    data_3d = np.nanmean(data_nd, axis=-1)

    which = h5.load(input_path, "/data/axis_jA", silent=True)
    for wdx in range(2, -1, -1):
        w = which[wdx]
        color = cc.alpha_to_solid_on_bg(base_colors[obs_to_plot], 1/(3-wdx))
        ax.plot(x, data_3d[wdx, :], label = f"jA = {w}", color = color)

    ax.set_xlabel("GABA strength jG (inhibition)")
    ax.legend()

    if "num_b" in obs_to_plot:
        ax.set_ylabel("# Bursts (in 1 hour)")
    elif "mean_rate" == obs_to_plot:
        ax.set_ylabel("Rate (Hz)")
    elif "sys_rate_cv" == obs_to_plot:
        ax.set_ylabel("CV")

    fig.tight_layout()

