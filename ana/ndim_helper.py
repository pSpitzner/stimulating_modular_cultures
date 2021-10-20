# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-19 18:20:20
# @Last Modified: 2021-10-20 14:05:03
# ------------------------------------------------------------------------------ #


import os
import sys
import glob
import h5py
import re
import tempfile
import numbers
import numpy as np
import pandas as pd
import xarray as xr

import hi5 as h5
from addict import Dict
from benedict import benedict
from tqdm import tqdm
from itertools import permutations

import logging
import warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def load_ndim_h5f(filename, exclude_pattern=None):
    """
        Load an ndimensional hdf5 file as a dict of xarrays.
        every key of the dict is an observable and every dimension of the xarrays
        corresponds to one parameter that we investiagted (or repetitions)
    """
    h5f = h5.recursive_load(filename)

    res = dict()
    for obs in h5f["data"].keys():
        if exclude_pattern is not None and exclude_pattern in obs:
            continue
        if "axis_" in obs or "num_samples" in obs:
            # in the old format, axis labels were stored in data
            continue
        res[obs] = h5f_to_xarray(h5f, obs)

    return res


def h5f_to_xarray(h5f, obs):
    """
        load the specified observable as an xarray

        # Parameters
        h5f : benedict
            an open h5 file
        obs : str
            observable to load from `data.{obs}`

        # Returns
        res : xarray

    """

    assert obs in h5f["data"].keys()

    dims = [dim for dim in h5f["meta"]["axis_overview"].astype("str")]
    coords = dict()
    for dim in dims:
        try:
            dim_vals = h5f["meta"][f"axis_{dim}"]
        except:
            # old file format
            dim_vals = h5f["data"][f"axis_{dim}"]
        # we might need to cast scalars to arrays
        try:
            len(dim_vals)
        except:
            dim_vals = np.array([dim_vals])

        coords[dim] = dim_vals

    # ideally, we have a repetition axis, else, for older files, we need to infer
    if not "rep" in coords.keys():
        dims.append("rep")

    # if vector quantity, repetitions are not the last dimension. for the vector
    # observables, we do not have a known axis
    if obs[0:3] == "vec":
        dims.append("vector_observable")

    log.info(obs)

    return xr.DataArray(h5f["data"][f"{obs}"], dims=dims, coords=coords)


def choose(options, prompt=None, min=1, max=1, via_int=False, default="all"):

    if prompt is None:
        prompt = f"Choose an option:"

    if via_int:
        prompt += "\n"
        for idx, opt in enumerate(options):
            prompt += f"{idx:d} "
            prompt += f"{opt}\n"
    else:
        maxlen = np.max([len(str(opt)) for opt in options])
        if maxlen < 5:
            prompt += f"\n{options}\n"
        else:
            prompt += "\n"
            for idx, opt in enumerate(options):
                prompt += f"{opt}\n"

    options = np.array(options)

    while True:
        txt = input(prompt)
        if len(txt) == 0:
            if default == "all":
                selection = options.tolist()
            elif isinstance(default, int):
                selection = [options[default]]
            elif isinstance(default, list):
                selection = default
        else:
            txt = txt.split(" ")
            if len(txt) < min or len(txt) > max:
                continue

            if not via_int:
                try:
                    selection = [type(options[0])(t) for t in txt]
                except:
                    continue
            else:
                # we have some more checks if choosing via integers
                if np.any([not _isint(i) for i in txt]) or np.any(
                    [int(i) >= len(options) for i in txt]
                ):
                    continue

                selection = options[[int(t) for t in txt]].tolist()

        if (
            len(selection) >= min
            and len(selection) <= max
            and np.all([i in options for i in selection])
        ):
            print(f"Using {selection}")
            break
    return selection


def _isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def _isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
