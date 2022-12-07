# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-19 18:20:20
# @Last Modified: 2022-12-05 15:01:33
# ------------------------------------------------------------------------------ #
# This is essentially a poor-mans version of xarray that only exists for
# historic reasons. Unfortunately, some of our code still needs it.
# ------------------------------------------------------------------------------ #

import numpy as np
import pandas as pd
import xarray as xr
import logging

import bitsandbobs as bnb

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")


def custom_ndim_to_xr_dset(filename, exclude_pattern=None):
    """
    Fixing past mistakes:
    This is a wrapper that constructs a conventional xarray dataset from my
    custom hdf5 file (that uses a slightly different layout than the xarray/netcdf does)
    """
    h5f = bnb.hi5.recursive_load(filename)

    res = dict()
    for obs in h5f["data"].keys():
        if exclude_pattern is not None and exclude_pattern in obs:
            continue
        if "axis_" in obs or "num_samples" in obs:
            # in a previous version, I stored the axis labels in data
            continue

        res[obs] = h5f_to_xr_array(h5f, obs)

    # now we have a dict of xarrays, but only the dimensions of the scalars
    # will match. vector quantities need to be treated separately
    scalars = []
    vectors = dict()
    for key in res.keys():
        array = res[key]
        array.name = key
        if "vector_observable" in array.dims:
            vectors[key] = array
        else:
            scalars.append(array)

    # print(scalars)
    scalars = xr.merge(scalars)

    # memo to paul: think about how to combine histogram bin edges with values

    return scalars, vectors


def load_ndim_h5f(filename, exclude_pattern=None):
    """
        Load an ndimensional hdf5 file as a dict of xarrays.
        every key of the dict is an observable and every dimension of the xarrays
        corresponds to one parameter that we investiagted (or repetitions)

        # Returns
        res : dict or xarray dataset.
            in any case, accessing the data is done via `res["observable_name"]` to get an xarray.
    """

    # this may just be a saved xarray datset
    try:
        res = xr.load_dataset(filename)
        if len(res.variables) == 0:
            # likely not what we are after.
            raise ValueError
        return res
    except:
        # its probably my custom stuff
        h5f = bnb.hi5.recursive_load(filename)

        res = dict()
        for obs in h5f["data"].keys():
            if exclude_pattern is not None and exclude_pattern in obs:
                continue
            if "axis_" in obs or "num_samples" in obs:
                # in the old format, axis labels were stored in data
                continue
            res[obs] = h5f_to_xr_array(h5f, obs)

        return res


def h5f_to_xr_array(h5f, obs):
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
            if isinstance(dim_vals, (str, bytes)):
                # sometimes we get byte strings,
                # it will be easiert to have them decoded already
                try:
                    dim_vals = dim_vals.decode("utf-8")
                except:
                    pass
                raise TypeError
        except:
            dim_vals = np.array([dim_vals])

        coords[dim] = dim_vals

    # ideally, we have a repetition axis, else, for older files, we need to infer
    if not "repetition" in coords.keys():
        dims.append("repetition")

    # if vector quantity, repetitions are not the last dimension. for the vector
    # observables, we do not have a known axis
    if obs[0:3] == "vec":
        dims.append("vector_observable")

    log.debug(f"obs: {obs}")
    log.debug(f"dims: {dims}")
    log.debug(f"coords: {coords}")

    return xr.DataArray(h5f["data"][f"{obs}"], dims=dims, coords=coords)


def choose(
    options,
    prompt=None,
    min=1,
    max=np.inf,
    via_int=False,
    default="all",
    always_return_list=False,
):

    if prompt is None:
        prompt = f"Choose an option"
        if via_int:
            prompt += " (e.g. 0 1)"
        prompt += ":"

    if via_int:
        prompt += "\n"
        for idx, opt in enumerate(options):
            prompt += f"{idx:d} "
            prompt += f"{opt}\n"
    else:
        maxlen = np.max([len(str(opt)) for opt in options])
        if maxlen < 8:
            prompt += f" {options}\n"
        else:
            prompt += "\n"
            for idx, opt in enumerate(options):
                prompt += f"{opt}\n"

    if isinstance(options, np.ndarray):
        pass
    elif isinstance(options, list):
        options = np.array(options)
    else:
        options = np.array(list(options))

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
            log.debug(f"Using {selection}")
            break

    if len(selection) == 1 and not always_return_list:
        return selection[0]
    return selection

def choose_all_dims(data, skip = None):

    # select one non-histogram observable to select dimensions
    for obs in data.keys():
        if obs[0:4] != "vec_":
            break

    possible_dims = [
        dim
        for dim in data[obs].dims
        if (dim != "repetition" and dim != "vector_observable" and dim != skip)
    ]

    # limit the xarray for the remaining dims
    for dim in possible_dims:

        possible_cutplanes = data[obs].coords[dim].to_numpy()
        if len(possible_cutplanes) == 1:
            cutplane = possible_cutplanes[0]
        else:
            cutplane = choose(
                possible_cutplanes,
                f"Choose one value for '{dim}':",
                via_int=False,
                min=1,
                max=1,
            )

        for o in data.keys():
            data[o] = data[o].sel({dim : cutplane})

    return data

def load_and_choose_two_dims(filename, **kwargs):

    data = load_ndim_h5f(filename, **kwargs)

    # select one non-histogram observable to select dimensions
    for obs in data.keys():
        if obs[0:4] != "vec_":
            break

    possible_dims = [
        dim
        for dim in data[obs].dims
        if (dim != "repetition" and dim != "vector_observable")
    ]

    selected_dims = choose(possible_dims, "Choose two dims:", via_int=True, min=2, max=2)

    # optionally limit the coordinates of the second dim:
    dim = selected_dims[-1]
    possible_cutplanes = data[obs].coords[dim].to_numpy()
    if len(possible_cutplanes) == 1:
        cutplane = possible_cutplanes[0]
    else:
        cutplane = choose(
            possible_cutplanes,
            f"Choose at least one value for '{dim}', press enter for all:",
            via_int=True,
            min=1,
        )
    for o in data.keys():
        data[o] = data[o].sel({dim : cutplane})


    # let's already limit the xarray for the remaining dims
    for dim in possible_dims:

        if dim == "repetition" or dim == "vector_observable":
            continue
        if dim in selected_dims:
            continue

        possible_cutplanes = data[obs].coords[dim].to_numpy()
        if len(possible_cutplanes) == 1:
            cutplane = possible_cutplanes[0]
        else:
            cutplane = choose(
                possible_cutplanes,
                f"Choose one value for '{dim}':",
                via_int=False,
                min=1,
                max=1,
            )

        for o in data.keys():
            data[o] = data[o].sel({dim : cutplane})



    return data, selected_dims

    # selected_cs = dict()
    # for dim in data[obs].dims:
    #     if dim == "repetition" or dim == "vector_observable":
    #         continue
    #     options = data[obs].coords[dim].to_numpy()
    #     if len(options) == 1:
    #         selected_cs[dim] = options[0]
    #     else:
    #         selected_cs[dim] = choose(options, f"Choose coordinates for '{dim}':")
    #     print(f"Using '{dim}' = {selected_cs[dim]}")

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

# ------------------------------------------------------------------------------ #
# helpers to describe my custom variables
# ------------------------------------------------------------------------------ #

# map short observable labels to longer descriptions
def obs_labels(short):
    label = ""
    if "mean" in short:
        label += "Mean "
    elif "median" in short:
        label += "Median "

    if "ratio_num_b" in short:
        label += "Fraction of bursts"
    elif "num_b" in short:
        label += "Number of bursts"
    elif "blen" in short:
        label += "Burst duration"
    elif "rate_cv" in short:
        label += "CV of the rate"
    elif "ibis_cv" in short:
        label += "CV of the IBI"
    elif "rate_threshold" in short:
        label += "Rate Threshold"
    elif "rate" in short:
        label += "Rate"
    elif "ibis" in short:
        label += "Inter-Burst-Interval"
    elif "functional_complexity" in short:
        label += "Functional complexity"
    elif "participating_fraction_complexity" in short:
        label += "Fraction complexity"
    elif "participating_fraction" in short:
        label += "Fraction of neurons in bursts"
    elif "num_spikes_in_bursts" in short:
        label += "Spikes per neuron per burst"
    elif "correlation_coefficients" in short:
        label += "Correlation coefficients"
    else:
        return short

    # return label

    if "sys" in short:
        label += "\n(system-wide)"
    elif "any" in short:
        label += "\n(anywhere)"
    elif "mod" in short or "ratio" in short:
        label += "\n("
        label += short[-1]
        label += " modules)"

    return label

# map short dimension labels to longer descriptions
def dim_labels(short):
    if "rate" in short:
        return "Noise rate (Hz)"
    elif "jG" in short:
        return "GABA strength jG (inhibition)"
    elif "jA" in short:
        return "AMPA strength jA (excitation)"
    elif "jE" in short:
        return "External current strength jE"
    elif "k_frac" in short:
        return "Fraction of Connections"

