# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-10-28 10:42:05
# @Last Modified: 2020-10-28 10:46:59
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import logging
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut
import logisi as logisi


# variables to span axes and how to get them from the hdf5 files
d_obs = dict()
d_obs["ga"] = "/meta/dynamics_gA"
d_obs["rate"] = "/meta/dynamics_rate"
d_obs["tD"] = "/meta/dynamics_tD"
d_obs["alpha"] = "/meta/topology_alpha"
d_obs["k_inter"] = "/meta/topology_k_inter"


parser = argparse.ArgumentParser(description="fix scatter brain please")
parser.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input path with *.hdf5 files",
    metavar="FILE",
)

args = parser.parse_args()

def full_path(path):
    return os.path.abspath(os.path.expanduser(path))


# if a directory is provided as input, merge individual hdf5 files down
if os.path.isdir(args.input_path):
    candidates = glob.glob(full_path(args.input_path + "/*.hdf5"))
    print(f"{args.input_path} is a directory, using contained hdf5 files")
elif len(glob.glob(full_path(args.input_path))) <= 1:
    print(
        "Provide a directory with hdf5 files or wildcarded path as string: 'path/to/file_ptrn*.hdf5''"
    )
    exit()
else:
    candidates = glob.glob(full_path(args.input_path))
    print(f"{args.input_path} is a (list of) file")

# which values occur across files
d_axes = dict()
for obs in d_obs.keys():
    d_axes[obs] = []



# check what's in the files and create axes labels for n-dim tensor
print(f"Adding data to {len(candidates)} files:")
l_valid = []
for candidate in tqdm(candidates):
    try:
        for obs in d_obs:
            temp = ut.h5_load(candidate, d_obs[obs], silent=True)

            # dirty workaround for missing metadata
            if obs == 'k_inter':
                temp = int(candidate[candidate.find('k=')+2])

                # write back
                f_tar = h5py.File(merge_path, "r+")
                dset = f_tar.create_dataset("/meta/topology_k_inter", data=temp)
                f_tar.close()



    except Exception as e:
        print(f"incompatible file: {candidate}")
