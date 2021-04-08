# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-11 15:04:12
# @Last Modified: 2021-04-08 16:57:12
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

matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    "color",
    [
        "#233954",
        "#ea5e48",
        "#1e7d72",
        "#f49546",
        "#e8bf58",  # dark
        "#5886be",
        "#f3a093",
        "#53d8c9",
        "#f2da9c",
        "#f9c192",  # light
    ],
)  # qualitative, somewhat color-blind friendly, in mpl words 'tab5'

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

topo_exe = "/Users/paul/mpi/simulation/modular_cultures/_latest/exe/orlandi_standalone"
num_n = 500
rho = 125


ratio = 20
# alpha_weighted = [0.0125]
# alpha_unweighted = [0.0125 * ratio]
alpha_weighted = np.logspace(np.log10(1e-4), np.log10(1e0), base=10, num=50)
alpha_unweighted = np.logspace(np.log10(1e-4 * ratio), np.log10(1e0), base=10, num=50)


reps = 20
n_weighted = np.ones(shape=(reps, len(alpha_weighted))) * np.nan
g_weighted = np.ones(shape=(reps, len(alpha_weighted))) * np.nan
k_weighted = np.ones(shape=(reps, len(alpha_weighted))) * np.nan
n_unweighted = np.ones(shape=(reps, len(alpha_unweighted))) * np.nan
g_unweighted = np.ones(shape=(reps, len(alpha_unweighted))) * np.nan
k_unweighted = np.ones(shape=(reps, len(alpha_unweighted))) * np.nan


def components_from_connection_matrix(a_ij_sparse, num_n):
    """
        find the number of components and return the fraction of the larest one

        a_ij_sparse of shape [num_connections, 2]
        first column is "from neuron"
        second column is "to neuron"
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    try:
        ones = np.ones(a_ij_sparse.shape[0], dtype=np.int8)
        n_from = a_ij_sparse[:, 0]
        n_to = a_ij_sparse[:, 1]
    except:
        # no connections
        return 0, 0

    graph = csr_matrix((ones, (n_from, n_to)), shape=(num_n, num_n), dtype=np.int8)

    n_components, labels = connected_components(
        csgraph=graph, directed=True, connection="weak", return_labels=True
    )

    # find the largest component
    size = 0
    for l in np.unique(labels):
        s = len(np.where(labels == l)[0])
        if s > size:
            size = s

    return n_components, size / num_n

for idx, a in enumerate(tqdm(alpha_unweighted)):
    for r in range(reps):
        # path = f"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/alpha_syn_example/unweighted_aidx={idx}_rep={r}.hdf5"
        path = f"~/Desktop/temp.hdf5"
        os.system(
            f"{topo_exe} -o {path} -N {num_n} -rho {rho} -s $RANDOM -a_weighted 0 -a {a} > /dev/null",
        )
        a_ij = h5.load(
            path, "/data/connectivity_matrix_sparse"
        )
        n, g = components_from_connection_matrix(a_ij, num_n)
        g_unweighted[r, idx] = g
        n_unweighted[r, idx] = n
        k_unweighted[r, idx] = a_ij.shape[0]

for idx, a in enumerate(tqdm(alpha_weighted)):
    for r in range(reps):
        # path = f"/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/alpha_syn_example/weighted_aidx={a}_rep={r}.hdf5"
        path = f"~/Desktop/temp.hdf5"
        os.system(
            f"{topo_exe} -o {path} -N {num_n} -rho {rho} -s $RANDOM -a_weighted 1 -a {a}  > /dev/null",
        )
        a_ij = h5.load(
            path, "/data/connectivity_matrix_sparse"
        )
        n, g = components_from_connection_matrix(a_ij, num_n)
        g_weighted[r, idx] = g
        n_weighted[r, idx] = n
        k_weighted[r, idx] = a_ij.shape[0]

plt.ion()


def w2uw(x):
    return x * 20

def uw2w(x):
    return x / 20


# giant component
fig, wax = plt.subplots()
uwax = wax.secondary_xaxis('top', functions=(w2uw, uw2w))

wax.set_xlabel(f"Connection probability $\\alpha$ (weighted)")
lines = wax.plot(alpha_unweighted / ratio, np.mean(g_unweighted, axis=0), label="unweighted", color="C1")
clr = lines[0].get_color()
uwax.tick_params(axis='x', colors=clr, which='both')
uwax.xaxis.label.set_color(clr)

uwax.set_xlabel(f"Connection probability $\\alpha$ (unweighted)")
lines = wax.plot(alpha_weighted, np.mean(g_weighted, axis=0), label="weighted", color="C0")
clr = lines[0].get_color()
wax.tick_params(axis='x', colors=clr, which='both')
wax.xaxis.label.set_color(clr)
wax.axvline(0.0125, 0, 1, ls=":", color="gray")

wax.set_ylabel(f"Fraction of largest component")
wax.set_xscale("log")
wax.legend()
fig.tight_layout()



# in and out degree
fig, wax = plt.subplots()
uwax = wax.secondary_xaxis('top', functions=(w2uw, uw2w))

wax.set_xlabel(f"Connection probability $\\alpha$ (weighted)")
lines = wax.plot(alpha_unweighted / ratio, np.mean(k_unweighted, axis=0)/num_n, label="unweighted", color="C1")
clr = lines[0].get_color()
uwax.tick_params(axis='x', colors=clr, which='both')
uwax.xaxis.label.set_color(clr)


uwax.set_xlabel(f"Connection probability $\\alpha$ (unweighted)")
lines = wax.plot(alpha_weighted, np.mean(k_weighted, axis=0)/num_n, label="weighted", color="C0")
clr = lines[0].get_color()
wax.tick_params(axis='x', colors=clr, which='both')
wax.xaxis.label.set_color(clr)
ticks = wax.get_xticks()

wax.set_ylabel(f"Number of Connections")
wax.set_xscale("log")
wax.legend()
fig.tight_layout()
wax.axvline(0.0125, 0, 1, ls=":", color="gray")

# only until alpha=1
# wax.set_xticks(wax.get_xticks()[wax.get_xticks()<=1])
uwax.set_xticks(uwax.get_xticks()[uwax.get_xticks()<=1])
