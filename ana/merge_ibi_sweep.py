# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2020-08-31 09:05:41
#
# Scans the provided directory for .hdf5 files and checks if they have the right
# data to plot a 2d heatmap of ibi = f(gA, rate)
# uses ut.inter_burst_interval to get ibi from spiketimes
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

parser = argparse.ArgumentParser(description="Plot Ibi Sweep")
parser.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input path with *.hdf5 files",
    metavar="FILE",
)
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
args = parser.parse_args()


# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #


def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


# ------------------------------------------------------------------------------ #
# load, calcualte ibi, merge in a new hdf5 file
# ------------------------------------------------------------------------------ #


# if a directory is provided as input, merge individual hdf5 files down
if os.path.isdir(args.input_path):
    merge_path = full_path(args.input_path + "_ibi.hdf5")
    print(f"{args.input_path} is a directory. Merging (overwriting) to {merge_path}")

    candidates = glob.glob(full_path(args.input_path + "/*.hdf5"))
    l_ga = []
    l_rate = []
    l_valid = []

    # check what's in the files and create axes labels for 2d matrix
    for candidate in candidates:
        try:
            ga = ut.h5_load(candidate, "/meta/dynamics_gA", silent=True)
            rate = ut.h5_load(candidate, "/meta/dynamics_rate", silent=True)

            # check all observables are in the files
            # not anymore, now we calculate ibi from spiketimes
            # maybe check for existence later
            # _ = ut.h5_load(candidate, "/data/ibi", silent=True)

            if ga not in l_ga:
                l_ga.append(ga)
            if rate not in l_rate:
                l_rate.append(rate)
            l_valid.append(candidate)
        except Exception as e:
            print(f"incompatible file: {candidate}")

    l_ga = np.array(sorted(l_ga))
    l_rate = np.array(sorted(l_rate))

    # we might have repetitions
    num_rep = int(np.ceil(len(l_valid) / len(l_ga) / len(l_rate)))
    sampled = np.zeros(shape=(len(l_ga), len(l_rate)), dtype=int)

    # 3d: x, y, repetition
    heatmap = np.ones(shape=(len(l_ga), len(l_rate), num_rep)) * np.nan

    print(f"found ga: ", l_ga)
    print(f"   rates: ", l_rate)
    print(f"   repetitions: ", num_rep)

    for candidate in tqdm(l_valid):
        ga = ut.h5_load(candidate, "/meta/dynamics_gA", silent=True)
        rate = ut.h5_load(candidate, "/meta/dynamics_rate", silent=True)

        # transform to indices
        ga = np.where(l_ga == ga)[0][0]
        rate = np.where(l_rate == rate)[0][0]

        # consider repetitions for each data point and stack values
        rep = sampled[ga, rate]
        if rep <= num_rep:
            sampled[ga, rate] += 1

            # load spiketimes and calculate ibi
            spiketimes = ut.h5_load(candidate, "/data/spiketimes", silent=True)
            heatmap[ga, rate, rep] = ut.inter_burst_interval(
                spiketimes=spiketimes, simulation_duration=3600
            )
        else:
            print(
                f"Error: unexpected repetitions (already read {num_rep}): {candidate}"
            )

    if not np.all(sampled == sampled[0, 0]):
        print(
            f"repetitions vary across data points, from {np.min(sampled)} to {np.max(sampled)}"
        )

    # write to a new hdf5 file
    f_tar = h5py.File(merge_path, "w")

    # this seems broken due to integer division
    dset = f_tar.create_dataset("/data/axis_rate", data=l_rate)
    dset = f_tar.create_dataset("/data/axis_ga", data=l_ga)
    dset = f_tar.create_dataset("/data/num_samples", data=sampled)
    dset.attrs["description"] = "number of repetitions in /data/ibi, same shape"

    dset = f_tar.create_dataset("/data/ibi", data=heatmap)
    dset.attrs["description"] = "3d array with axis_ga x axis_rate x repetition"

    f_tar.close()

else:
    print("Provide a directory")
    exit()
