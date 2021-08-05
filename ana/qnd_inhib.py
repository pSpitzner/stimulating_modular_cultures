# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-06-08 13:49:46
# @Last Modified: 2021-07-08 21:38:59
# ------------------------------------------------------------------------------ #
# from the E-vs-I sweep (ndim merge), plot various observables as a function
# of inhibitory current strength
# ------------------------------------------------------------------------------ #


import os
import sys
import glob
import h5py
import argparse
import numbers

import matplotlib

matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import hi5 as h5

from mpl_toolkits.mplot3d import Axes3D

input_path = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/inhibition_sweep_gaba_160/ndim.hdf5"
# input_path = "/Users/paul/mpi/simulation/alpha_synuclein/_latest/dat/inhib/gaba_sweep.hdf5"
# input_path = "/Users/paul/mpi/simulation/alpha_synuclein/_latest/dat/inhib_rate_sweep/rate_sweep.hdf5"
# input_path = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/inhibition_sweep_rate_100/ndim.hdf5"
# input_path = "/Users/paul/mpi/simulation/alpha_synuclein/_latest/dat/k_frac_sweep_1/k_frac_sweep.hdf5"
# input_path = "/Users/paul/mpi/simulation/alpha_synuclein/_latest/dat/esweep/esweep.hdf5"

x_axis = "/data/axis_jG"
# x_axis = "/data/axis_k_frac"
y_set = "/data/axis_jA"
# x_axis = "/data/axis_rate"
# y_set = "/data/axis_jG"

# y_set = "/data/axis_jG"
# x_axis = "/data/axis_jA"



base_colors = dict()
base_colors["num_bursts"] = cc.to_hex(f"black")
base_colors["num_b_1"] = cc.to_hex(f"C0")
base_colors["num_b_geq_4"] = cc.to_hex(f"C1")
base_colors["sys_rate_cv"] = cc.to_hex(f"C2")
base_colors["mean_rate"] = cc.to_hex(f"C3")
base_colors["blen_all"] = cc.to_hex(f"black")
base_colors["blen_1"] = cc.to_hex(f"C0")
base_colors["blen_4"] = cc.to_hex(f"C1")
base_colors["ibis_module"] = cc.to_hex(f"C0")
base_colors["ibis_system"] = cc.to_hex(f"C1")
base_colors["ibis_cv_module"] = cc.to_hex(f"C0")
base_colors["ibis_cv_system"] = cc.to_hex(f"C1")

titles = dict()
titles["num_bursts"] = "Number of bursts"
titles["num_b_1"] = "Single-module bursts"
titles["num_b_geq_4"] = "System-wide bursts"
titles["sys_rate_cv"] = "Coefficient of variation"
titles["mean_rate"] = "Mean rate"
titles["blen_all"] = "Burst duration"
titles["blen_1"] = "Single-module burst duration"
titles["blen_4"] = "System-wide burst duration"

for odx, obs_to_plot in enumerate(
    [
        "num_b_1",
        "num_b_geq_4",
        "num_bursts",
        "sys_rate_cv",
        "mean_rate",
        "blen_all",
        "blen_1",
        "blen_4",
        "ibis_module",
        "ibis_system",
        "ibis_cv_system",
        "ibis_cv_module",
        "functional_complexity",
    ]
):

    print(obs_to_plot)
    fig, ax = plt.subplots()

    try:
        ax.set_title(titles[obs_to_plot])
    except:
        ax.set_title(obs_to_plot)

    x = h5.load(input_path, x_axis, silent=True, keepdim=True)
    data_nd = h5.load(input_path, f"/data/{obs_to_plot}", silent=True, keepdim=True)
    data_3d = np.nanmean(data_nd, axis=-1)
    err_3d = np.nanstd(data_nd, axis=-1) / np.sqrt(data_nd.shape[-1])

    which = h5.load(input_path, y_set, silent=True, keepdim=True)
    # which = np.delete(which, np.where(which == 25))
    for wdx in range(len(which)):
        w = which[wdx]
        if w == 25: continue
        alpha = np.fmin(1, 1 / (len(which) - wdx ))
        try:
            color = cc.alpha_to_solid_on_bg(base_colors[obs_to_plot], alpha)
        except:
            color = cc.alpha_to_solid_on_bg("black", alpha)
        # ax.plot(x, data_3d[wdx, :], label = f"jA = {w}", color = color)
        # ax.plot(x, data_3d[wdx, :], label = f"jA = {w}", color = color)
        ax.errorbar(
            x=x,
            # y=data_3d[wdx, :] if not "num_b" in obs_to_plot else 3600 / data_3d[wdx, :],
            y=data_3d[wdx, :],
            yerr=err_3d[wdx, :],
            # y=data_3d[:,wdx],
            # yerr=err_3d[:,wdx],
            fmt="-",
            markersize=1,
            elinewidth=0.5,
            capsize=3,
            label=f'{y_set.replace("/data/axis_", "")} = {w}',
            color=color,
        )

    if "rate" in x_axis:
        ax.set_xlabel("Noise rate (Hz)")
    elif "jG" in x_axis:
        ax.set_xlabel("GABA strength jG (inhibition)")
    elif "jA" in x_axis:
        ax.set_xlabel("AMPA strength jA (excitation)")
    elif "k_frac" in x_axis:
        ax.set_xlabel("Fraction of Connections")
    ax.legend()

    if "num_b" in obs_to_plot:
        ax.set_ylabel("# Bursts (in 1 hour)")
    elif "mean_rate" == obs_to_plot:
        ax.set_ylabel("Rate (Hz)")
    elif "sys_rate_cv" == obs_to_plot:
        ax.set_ylabel("CV")
    elif "blen" in obs_to_plot:
        ax.set_ylabel("seconds")

    fig.tight_layout()
