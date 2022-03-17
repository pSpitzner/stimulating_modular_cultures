# ------------------------------------------------------------------------------ #
# @Authors:       Victor Buendia Ruiz-Azuaga, F. Paul Spitzner
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #
# Analysis functions of the mesoscopic model.
# Most things have a similar counter part in ana_helper.py
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Cluster detection, very useful for avalanches
from scipy.ndimage import measurements

import hi5 as h5
from benedict import benedict
import ana_helper as ah

import logging
import warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings


# ------------------------------------------------------------------------------ #
# Prep files in the needed format
# ------------------------------------------------------------------------------ #


# ps: this was `read_csv_and_format`
def prepare_file(file_path):
    """
    Reads a csv or hdf5 and formats a Benedict with our data format
    Parameters:
    - file_path: string where the CSV or hdf5 file is located
    """

    # this checks file type and loads
    df = _load_if_path(file_path)

    # Create a dictionary and store the names of DF columns
    h5f = benedict()

    h5f["ana.mods"] = [f"mod_{m}" for m in range(0, 4)]
    h5f["ana.mod_ids"] = np.arange(4)
    h5f["data.neuron_module_id"] = np.arange(4)

    # ps: we need an index shift because I used a different 0-convetion for modules
    def mapper(col):
        regex = re.search("mod_(\d+)", col, re.IGNORECASE)
        if regex:
            mod_id = int(regex.group(1))
            return col.replace(f"mod_{mod_id}", f"mod_{mod_id-1}")
        else:
            return col

    df.rename(columns=mapper, inplace=True)

    # keep a copy of the original pandas data frame
    # ps: to simplify, I removed the dict_argument everywhere since we did not use it.
    h5f[f"data.raw_df"] = df

    # Re-copy in correct format to use Paul's helper functions for analysis
    clusters = h5f["ana.mods"]
    h5f[f"ana.rates.system_level"] = df[clusters].mean(axis=1)
    h5f[f"ana.rates.cv.system_level"] = (
        h5f[f"ana.rates.system_level"].std() / h5f[f"ana.rates.system_level"].mean()
    )
    h5f[f"data.state_vars_D"] = np.ones(shape=(4, len(h5f[f"ana.rates.system_level"])))

    for cdx, c in enumerate(clusters):
        h5f[f"ana.rates.module_level.{c}"] = df[c].to_numpy()
        h5f[f"ana.rates.cv.module_level.{c}"] = df[c].std() / df[c].mean()
        h5f[f"data.state_vars_D"][cdx, :] = df[f"{c}_res"].to_numpy()

    # Get dt
    h5f[f"ana.rates.dt"] = df["time"][1] - df["time"][0]
    h5f[f"data.state_vars_time"] = df["time"]

    # Prepare colors
    h5f["ana.mod_colors"] = [f"C{x}" for x in range(0, len(h5f["ana.mods"]))]

    return h5f


# ------------------------------------------------------------------------------ #
# Processing
# ------------------------------------------------------------------------------ #


def process_data_from_file(file_path, processing_functions, labels=None):
    """
    provide a path to a single file on which a set of processing functions are run

    # Parameters
    file_path : str
    process_functions : list of functions that take file path as input and give back
        a scalar
    labels : list of length `len(process_functions)`

    # Returns
    res : dict
        keys are the provided labels of if not provided, the function names
        values are the scalar results
    """
    y = dict()
    for fdx, f in enumerate(processing_functions):
        label = f.__name__ if labels is None else labels[fdx]
        y[label] = f(file_path)
    return y


# ps: currently not used, afaik
def process_data_from_folder(
    folder_path, processing_functions, file_extension=".hdf5"
):
    """
    similar to `process_data_from_file` but this uses all files in the provided folder and extracts the index from the file name.

    columns: | line | time | activity 1 | 2 | 3 | 4 | resources 1 | 2 | 3 | 4 |

    this gets applied on all files in the folder that end in `xx.csv` or `.hdf5`

    # Parameters
    folder_path : str, leading to data files
    processing_functions : list of functions, each takes one argument and
        returns a scalar

    # Returns
    x : 1d array with the indices of the files
    y : scalars computed with `processing_function`
    """

    # Get list of files
    candidates = glob.glob(
        os.path.abspath(os.path.expanduser(folder_path)) + f"/*{file_extension}"
    )

    # Initialize a dictionary based on functions we want to evaluate
    x = np.ones(len(candidates)) * np.nan
    y = dict()
    for f in processing_functions:
        y[f.__name__] = np.ones(len(candidates)) * np.nan

    # print(candidates, os.path.abspath(os.path.expanduser(folder_path)))
    for cdx, candidate in enumerate(candidates):

        # ~/path/to/noise17.csv
        # get the number used for file indexing
        fdx = re.search(f"(\d+){file_extension}$", candidate, re.IGNORECASE)
        if fdx:
            fdx = fdx.group(1)
        else:
            fdx = -1
            print(f"no file index found for {candidate}")
        x[cdx] = fdx

        # read victors format
        # columns: | line | time | activity 1 | 2 | 3 | 4 | resources 1 | 2 | 3 | 4 |
        # rows are first index, cols are second
        if ".csv" in file_extension:
            raw = pd.read_csv(candidate)
        elif ".hdf5" in file_extension:
            raw = pd.read_hdf(candidate, f"/dataframe")

        # Apply function to data
        for f in processing_functions:
            y[f.__name__][cdx] = f(raw)
        # print(cdx, raw, y)

        del raw

    # Sort all results
    idx = np.argsort(x)
    x = x[idx]
    for f in processing_functions:
        y[f.__name__] = y[f.__name__][idx]

    return x, y


# ------------------------------------------------------------------------------ #
# Analysis functions that provide scalars, for processing
# currently, most of these do not need a fully prepped h5f with our custom format
# but work directly on the `raw` pandas table that was loaded from file
# ------------------------------------------------------------------------------ #


def f_correlation_coefficients(raw, return_matrix=False):

    raw = _load_if_path(raw)

    act = raw[["mod_1", "mod_2", "mod_3", "mod_4"]].to_numpy()

    # plt.figure()
    # plt.plot(act[:,0])
    # plt.plot(act[:,1])
    # plt.show()

    rij = np.corrcoef(act.T)

    # "only interested in pair-wise interactions, discard diagonal entries rii"
    np.fill_diagonal(rij, np.nan)

    if return_matrix:
        return rij
    else:
        return np.nanmean(rij)


def f_functional_complexity(raw):
    raw = _load_if_path(raw)
    # we end up calculating rij twice, which might get slow.
    rij = f_correlation_coefficients(raw, return_matrix=True)

    return ah._functional_complexity(rij)


def f_event_size(raw):

    # this should be redundant with the prepping below
    raw = _load_if_path(raw)
    # lets reuse some of victors tricks
    h5f = prepare_file(raw)
    # `module_contribution` retuns a list of how many modules contributed
    # (round about the length of detected events)
    contrib = module_contribution(h5f, 1.0, area_min=0.7, roll_window=0.5)

    return np.nanmean(contrib)


# this guy breaks convention
def module_contribution(
    data,
    system_thres,
    roll_window=0.5,
    area_min=1.0,
    write_back_burst_times=True,
):
    """
    This function computes how many modules contributed to each system-wide burst using area overlap.
    Parameters
    - data: benedict containing all information (or raw dataframe, in which case we call `prepare_file(data)`)
    - system_thres: float, threshold to consider system-wide burst
    - roll_window: in seconds, width of rolling average kernel
    - area_min: how much area do we consider to see if a module contributed.
    """

    if isinstance(data, pd.DataFrame) or isinstance(data, str):
        h5f = prepare_file(data)
    else:
        h5f = data

    dt = h5f["ana.rates.dt"]
    n_roll = int(roll_window / dt)

    # Get total rate and module rate
    datacols = h5f["ana.mods"]
    smoothrate = (
        h5f[f"data.raw_df"][datacols].mean(axis=1).rolling(n_roll).mean().values
    )
    smoothcluster = h5f[f"data.raw_df"][datacols].rolling(n_roll).mean().values

    # Filter total rate and label with a different tag each burst
    filterrate = smoothrate >= system_thres
    features, num = measurements.label(filterrate)

    # Get where are these tags positioned
    unique, label_events = np.unique(features, return_index=True)
    del unique
    # print(features[1150:1370])
    # print(system_thres)
    # print(smoothrate[1150:1370])

    # paul: label events contains integer start times of bursts (rate above threshold)
    label_events = label_events[1:]  # (from 1 onwards, 0 means no cluster)
    if write_back_burst_times:
        # for now we only need to find cycles, so its okay to reuse times
        beg_times = label_events*dt
        end_times = label_events*dt
        beg_times = np.append(beg_times, np.inf)
        end_times = np.insert(end_times, 0, np.nan)
        h5f["ana.bursts.system_level.beg_times"] = beg_times
        h5f["ana.bursts.system_level.end_times"] = end_times


    # Now that we know where burst start we can go through them
    contribution = np.zeros(label_events.size - 1, dtype=int)
    for j in range(label_events.size - 1):
        burst = label_events[j]
        next_burst = label_events[j + 1]

        # Total area over the threshold given by this burst
        filter_positive = np.where(smoothrate[burst:next_burst] >= system_thres)[
            0
        ]  # Find where the module is above threshold inside this cluster
        filter_positive += burst  # Set the indices correctly, starting in threshold, to recover them later from full series

        filtered = smoothrate[filter_positive]
        # area_burst =  0.5*np.sum(filtered[1:] + filtered[:-1] - 2*system_thres)

        # Get contribution from each module
        area_contrib = np.empty(4)
        for c in range(4):
            filtered = smoothcluster[:, c][filter_positive]
            filtered = filtered[filtered >= system_thres]
            area_contrib[c] = 0.5 * np.sum(
                filtered[1:] + filtered[:-1]
            )  # - 2*system_thres)

        area_contrib /= np.sum(area_contrib)
        # print(4*area_contrib)
        contribution[j] = int((4 * area_contrib > area_min).sum())


    return contribution

# ------------------------------------------------------------------------------ #
# Others
# ------------------------------------------------------------------------------ #


def fourier_transform(
    h5f, t_sti_start=600.0, t_sti_end=1200.0, dataroll=10, avroll=5
):
    """
    This function performs a Fourier transform of the data provided.
    Parameters
    - h5f: system level rates we will be transformed to the frequency domain
    - t_sti_start: when does stimulation starts, in seconds
    - t_sti_ends: when does stimulation ends, in seconds
    - dataroll: to smooth the data with rolling average
    - avroll: to smooth the result with rolling average
    """

    dt = h5f["ana.rates.dt"]
    av = h5f["ana.rates.system_level"].rolling(dataroll).mean()

    # Set up dt and compute indices where estimulation starts and ends
    bin_sti_start, bin_sti_end = int(t_sti_start / dt), int(t_sti_end / dt)

    # Perform and normalise transform
    data_fourier = np.fft.fft(av[bin_sti_start:bin_sti_end])
    data_fourier /= data_fourier[0]

    # Convert it to a handy datadframe to make rolling average
    pandasfourier = pd.DataFrame(data_fourier)
    pandasfourier = np.abs(pandasfourier)
    pandasfourier = pandasfourier.rolling(avroll).mean().values

    # Get X axis
    freq = np.fft.fftfreq(bin_sti_end - bin_sti_start, d=dt)

    # Transformation is symmetric, so get the right half we are set
    half = freq.size // 2

    return (freq[:half], pandasfourier[:half])


def _load_if_path(raw):
    """
    Wrapper that loads the file (`raw`) to pandas data frame if it is not loaded, yet
    """
    if not isinstance(raw, str):
        return raw

    if ".csv" in raw:
        return pd.read_csv(raw)
    elif ".hdf5" in raw:
        return pd.read_hdf(raw, f"/dataframe")
