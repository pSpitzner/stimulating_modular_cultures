# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-10-21 18:41:45
#
# plot a merged down, multidimensional hdf5 file (from individual simulations)
# and select which dims to show where
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
import ndim_helper as nh
import colors as cc
import palettable as pt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, Normalize


parser = argparse.ArgumentParser(description="ndim_plot")
parser.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input path with *.hdf5 files",
    metavar="FILE",
)


args = parser.parse_args()

# ------------------------------------------------------------------------------ #
# load data and select what to plot
# ------------------------------------------------------------------------------ #

data = nh.load_ndim_h5f(args.input_path)

# obs = nh.choose(data.keys(), "Choose an observable:", max=1, via_int=True)

# selected_cs = dict()
# for dim in data[obs].dims:
#     if dim == "repetition" or dim == "vector_observable":
#         continue
#     options = data[obs].coords[dim].to_numpy()
#     if len(options) == 1:
#         selected_cs[dim] = options[0]
#     else:
#         selected_cs[dim] = nh.choose(options, f"Choose a coordinates for '{dim}':")
#     print(f"Using '{dim}' = {selected_cs[dim]}")







