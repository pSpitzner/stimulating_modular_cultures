# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-05-10 16:19:18
# @Last Modified: 2021-05-22 20:14:42
# ------------------------------------------------------------------------------ #
# Use this in conjunction with a file resulting form `ndim_merge` to find
# parameter regions that match desired criteria.
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import numbers

import numpy as np
import hi5 as h5
from itertools import product


# ------------------------------------------------------------------------------ #
# functions to test
# ------------------------------------------------------------------------------ #

def num_bursts(h5f, coords):
    # we may still have repetitions, as last index
    nb = np.mean(h5f.data.num_bursts[coords])
    return nb > 50

# number of bursts with sequence length longer at least 2
def nb2(h5f, coords):
    # IBI wanted ~ 40s, 3600s sim duration, hence ~ 90 bursts
    nb = np.mean(h5f.data.num_b_geq_2[coords])
    return nb > 50 and nb < 150

def firing_rate(h5f, coords):
    r = np.mean(h5f.data.mean_rate[coords])
    return r > 0.05 and r < 0.15



d_crit = dict()
# d_crit["nb"] = num_bursts
d_crit["nb2"] = nb2
d_crit["rate"] = firing_rate

# match_criteria = np.any
match_criteria = np.all


# ------------------------------------------------------------------------------ #
# arg parse
# ------------------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="ndim_roi")
parser.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input path with *.hdf5 files",
    metavar="FILE",
)
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
parser.add_argument(
    "-c", "--center", dest="center_cmap_around", default=None, type=float,
)
parser.add_argument(
    "-a",
    "--annot",
    dest="enable_annotation",
    default=False,
    action="store_true",
)

args = parser.parse_args()

# get all observables that qualify (i.e. are not axis)
l_obs = h5.ls(args.input_path, "/data/")
l_obs = [obs for obs in l_obs if obs.find("axis_") != 0]
assert len(l_obs) > 0

# get all axis
l_axs = h5.load(args.input_path, "/meta/axis_overview", silent=True).astype("str")
assert len(l_axs) > 0

d_axs = {}
for ax in l_axs:
    d_axs[ax] = h5.load(args.input_path, f"/data/axis_{ax}", keepdim=True)


def print_cols(l_obs, pad):
    for o in l_obs:
        print(f"{o:^{pad}}", end='')

h5f = h5.recursive_load(args.input_path, hot=False, keepdim=True)

pad = 10
print_cols(d_axs.keys(), pad)
print("|", end="")
print_cols(d_crit.keys(), pad)
print(f'\n{"":-^{1+(len(d_axs.keys()) + len(d_crit.keys()))*pad}}\n', end='')

# we strongly rely on dicts being ordered, as of recent py versions
# iterate over all state space
for row in product(*d_axs.values()):
    # get the access coordinates
    index = ()
    for adx, ax_name in enumerate(d_axs.keys()):
        temp = row[adx]
        temp = np.where(d_axs[ax_name] == temp)[0][0]
        index += (temp,)

    # check all conditions for the coordinates
    passed = []
    for f in d_crit.values():
        passed.append(f(h5f, index))

    # only print if something passed
    if not match_criteria(passed):
        continue

    # build printout, col values
    cols = []
    for adx, ax_name in enumerate(d_axs.keys()):
        cols.append(d_axs[ax_name][index[adx]])
    print_cols(cols, pad)
    print("|", end="")

    # passed state
    cols = []
    for p in passed:
        if p:
            cols.append("✅")
        else:
            cols.append("❌")
    print_cols(cols, pad)
    print("\n", end="")








