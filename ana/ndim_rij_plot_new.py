# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-16 11:54:20
# @Last Modified: 2021-10-20 13:59:08
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
import ndim_helper as nd
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

data = nd.load_ndim_h5f(args.input_path)

obs_to_plot = nd.choose(data.keys(), "Choose an observable:", min=1, max=1)






