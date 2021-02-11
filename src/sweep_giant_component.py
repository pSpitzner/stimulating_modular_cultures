# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-11 15:04:12
# @Last Modified: 2021-02-11 16:00:23
# ------------------------------------------------------------------------------ #
import h5py
import argparse
import os
import tempfile
import sys
import shutil
import numpy as np
import logging
import matplotlib.pyplot as plt
from brian2.units.allunits import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

topo_exe = "/Users/paul/mpi/simulation/modular_cultures/_latest/exe/orlandi_standalone"
num_n = 500
rho = 125

alpha_weighted = np.logspace(np.log10(1e-4), np.log10(1e0), base=10, num=33)
alpha_unweighted = np.logspace(np.log10(1e-4), np.log10(1e0), base=10, num=33)

reps = 5
n_weighted = np.ones(shape=(reps, len(alpha_weighted))) * np.nan
g_weighted = np.ones(shape=(reps, len(alpha_weighted))) * np.nan
k_weighted = np.ones(shape=(reps, len(alpha_weighted))) * np.nan
n_unweighted = np.ones(shape=(reps, len(alpha_unweighted))) * np.nan
g_unweighted = np.ones(shape=(reps, len(alpha_unweighted))) * np.nan
k_unweighted = np.ones(shape=(reps, len(alpha_unweighted))) * np.nan

for idx, a in enumerate(tqdm(alpha_unweighted)):
    for r in range(reps):
        os.system(
            f"{topo_exe} -o ~/Desktop/temp.hdf5 -N {num_n} -rho {rho} -s $RANDOM -a_weighted 0 -a {a} > /dev/null",
        )
        a_ij = ut.h5_load(
            "/Users/paul/Desktop/temp.hdf5", "/data/connectivity_matrix_sparse"
        )
        n, g = ut.components_from_connection_matrix(a_ij, num_n)
        g_unweighted[r, idx] = g
        n_unweighted[r, idx] = n
        k_unweighted[r, idx] = a_ij.shape[0]

for idx, a in enumerate(tqdm(alpha_unweighted)):
    for r in range(reps):
        os.system(
            f"{topo_exe} -o ~/Desktop/temp.hdf5 -N {num_n} -rho {rho} -s $RANDOM -a_weighted 1 -a {a} > /dev/null",
        )
        a_ij = ut.h5_load(
            "/Users/paul/Desktop/temp.hdf5", "/data/connectivity_matrix_sparse"
        )
        n, g = ut.components_from_connection_matrix(a_ij, num_n)
        g_weighted[r, idx] = g
        n_weighted[r, idx] = n
        k_weighted[r, idx] = a_ij.shape[0]

plt.ion()

fig, ax = plt.subplots()
ax.plot(alpha_unweighted, np.mean(g_unweighted, axis=0), label='unweighted')
ax.plot(alpha_weighted, np.mean(g_weighted, axis=0), label='weighted')
ax.set_xlabel(f"Connection probability $\\alpha$")
ax.set_ylabel(f"Fraction of largest component")
ax.set_xscale("log")

