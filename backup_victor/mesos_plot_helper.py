# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

# from os import read
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glob
import os
import re

# from tqdm import tqdm
# import xarray as xr

from scipy.ndimage import measurements  # Cluster detection, very useful for avalanches

from benedict import benedict
from ana_helper import find_system_bursts_from_module_bursts, _functional_complexity


# ----------------------------------- #
#    Read a post-processed CSV (model)
# ----------------------------------- #


def read_csv_and_format(path_2_file, dict_name, sep=","):
    """
    Reads a csv and formats a Benedict with our data
    Parameters:
    - path_2_file: string where the CSV file is located
    - dict_name: string, to store whole pandas dataframe inside
    """

    if isinstance(path_2_file, pd.DataFrame):
        df = path_2_file
    else:
        # read raw data
        if ".csv" in path_2_file:
            df = pd.read_csv(path_2_file, sep=sep)
        elif ".hdf5" in path_2_file:
            df = pd.read_hdf(path_2_file, f"/dataframe")

    # Create a dictionary and store the names of DF columns
    data = benedict()

    data["ana.mods"] = [f"mod_{m}" for m in range(0, 4)]
    data["ana.mod_ids"] = np.arange(4)
    data["data.neuron_module_id"] = np.arange(4)

    # paul: we need an index shift because I used a different 0-convetion for modules
    def mapper(col):
        regex = re.search("mod_(\d+)", col, re.IGNORECASE)
        if regex:
            mod_id = int(regex.group(1))
            return col.replace(f"mod_{mod_id}", f"mod_{mod_id-1}")
        else:
            return col

    df.rename(columns=mapper, inplace=True)
    data[f"df.{dict_name}"] = df

    # Re-copy in correct format to use Paul's helper functions for analysis
    clusters = data["ana.mods"]
    data[f"ana.rates.system_level"] = df[clusters].mean(axis=1)
    data[f"ana.rates.cv.system_level"] = (
        data[f"ana.rates.system_level"].std() / data[f"ana.rates.system_level"].mean()
    )
    data[f"data.state_vars_D"] = np.ones(shape=(4, len(data[f"ana.rates.system_level"])))

    for cdx, c in enumerate(clusters):
        data[f"ana.rates.module_level.{c}"] = df[c].to_numpy()
        data[f"ana.rates.cv.module_level.{c}"] = df[c].std() / df[c].mean()
        data[f"data.state_vars_D"][cdx, :] = df[f"{c}_res"].to_numpy()

    # Get dt
    data[f"ana.rates.dt"] = df["time"][1] - df["time"][0]
    data[f"data.state_vars_time"] = df["time"]

    # Prepare colors
    data["ana.mod_colors"] = [f"C{x}" for x in range(0, len(data["ana.mods"]))]

    return data


# ----------------------------------- #
#   Data analysis
# ----------------------------------- #


def module_contribution(
    data,
    dict_name,
    system_thres,
    roll_window=0.5,
    area_min=1.0,
    write_back_burst_times=True,
):
    """
    This function computes how many modules contributed to each system-wide burst using area overlap.
    Parameters
    - data: benedict containing all information
    - dict_name: string, to store whole pandas dataframe inside
    - system_thres: float, threshold to consider system-wide burst
    - roll_window: in seconds, width of rolling average kernel
    - area_min: how much area do we consider to see if a module contributed.
    """

    dt = data["ana.rates.dt"]
    n_roll = int(roll_window / dt)

    # Get total rate and module rate
    datacols = data["ana.mods"]
    smoothrate = (
        data[f"df.{dict_name}"][datacols].mean(axis=1).rolling(n_roll).mean().values
    )
    smoothcluster = data[f"df.{dict_name}"][datacols].rolling(n_roll).mean().values

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
        beg_times = label_events*data["ana.rates.dt"]
        end_times = label_events*data["ana.rates.dt"]
        beg_times = np.append(beg_times, np.inf)
        end_times = np.insert(end_times, 0, np.nan)
        data["ana.bursts.system_level.beg_times"] = beg_times
        data["ana.bursts.system_level.end_times"] = end_times


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


def fourier_transform(
    inpt_data, t_sti_start=600.0, t_sti_end=1200.0, dataroll=10, avroll=5
):
    """
    This function performs a Fourier transform of the data provided.
    Parameters
    - inpt_data: the data we will transform to the frequency domain
    - t_sti_start: when does stimulation starts, in seconds
    - t_sti_ends: when does stimulation ends, in seconds
    - dataroll: to smooth the data with rolling average
    - avroll: to smooth the result with rolling average
    """

    dt = inpt_data["ana.rates.dt"]
    av = inpt_data["ana.rates.system_level"].rolling(dataroll).mean()

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


# ----------------------------------- #
#   Backbone plotting functions
# ----------------------------------- #


def plot_series(
    data, series_col_names, ax, module_colors, labels=None, roll_window=0.5, lw=0.5
):
    """
    Plot smoothed activity in the selected axes from our chosen experiment
    Parameters:
    - data: pandas DataFrame, to be plotted
    - series_col_names: names of the columns that will be plotted
    - ax: the axes where we want to plot
    - roll_window: in seconds, width of rolling average kernel
    """

    # Get some data from the series
    dt = data["time"][1] - data["time"][0]
    n_roll = int(roll_window / dt)

    # Plot each module first
    for j, c in enumerate(series_col_names):
        av = data[c].rolling(n_roll).mean()
        ax.plot(data["time"], av, color=module_colors[j], lw=lw)

    # Plot average
    av = data[series_col_names].rolling(n_roll).mean().mean(axis=1)
    ax.plot(data["time"], av, color="black", lw=lw)


def plot_raster(datadict, ax):
    """
    Plot rasterplot in the selected axes from our chosen experiment
    Parameters:
    - data: our benedict data structure
    - ax: the axes where we want to plot
    """

    # Check the info we need for raster is already available
    assert "ana.bursts" in datadict.keypaths(), "run `find_bursts_from_rates` first"

    # Initialise the state for each cluster
    events = []
    for m_dc in datadict["ana.mods"]:
        events.append(datadict[f"ana.bursts.module_level.{m_dc}.beg_times"])

    # Run plot
    ax.eventplot(
        events, color=datadict["ana.mod_colors"], linelengths=0.95, linewidths=0.4
    )


# ----------------------------------- #
#    Read raw fluorescency data
# ----------------------------------- #


def load_data(path_2_data, tf=600.0):
    """
    Load all the relevant data, returning the corresponding BeneDict.
    To be probably integrated with Paul's rutines
    Parameters:
    - path_2_data: folder where the fluorescency data is
    - tf: time contained in each file, in seconds
    """

    data = benedict()  # Init empty dict

    # Load data from each different experiment
    for experiment in ["A", "B", "C"]:

        table = []  # To be filled later with a DataFrame

        # Each experiment is recorded in three different parts, spont+stim+spont
        for part in [1, 2, 3]:
            # Get filepath
            is_stim = "stim" if part == 2 else "spont"
            path = f"{path_2_data}/{experiment}1_{part}_{is_stim}_traces_smoothed_everything.csv"

            # Read file path and set time correctly
            partdata = pd.read_csv(path)
            partdata["time (s)"] += (part - 1) * tf
            table.append(partdata)  # Append dataframe

        # Gerate the whole pandas dataframe now that all files were read
        table = pd.concat(table, ignore_index=True)
        table = table.rename(
            columns={
                "time (s)": "time",
                " 1": "mod_1",
                " 2": "mod_2",
                " 3": "mod_3",
                " 4": "mod_4",
            }
        )

        # Add read to our dictionary
        data[f"experiment.{experiment}"] = table
        data["ana.mods"] = [f"mod_{m}" for m in range(1, 5)]
        data["ana.mod_ids"] = range(1, 5)

    return data


def format_experiment(datadict, experiment, dict_name):
    """
    This function takes the result of a single dataset from load_data and returns it in a good format for helper functions
    Parameters:
    - datadict: result from load_experiment
    - experiment: String A,B, or C, the chosen experiment.
    - dict_name: string, to store whole pandas dataframe inside
    """

    clusters = datadict["ana.mods"]

    data = benedict()
    data["ana.mods"] = clusters
    data["ana.mod_ids"] = range(1, 5)
    data[f"df.{dict_name}"] = datadict[f"experiment.{experiment}"]

    # Re-copy in correct format to use Paul's helper functions for analysis
    for c in clusters:
        data[f"ana.rates.module_level.{c}"] = data[f"df.{dict_name}"][c]
    data[f"ana.rates.system_level"] = data[f"df.{dict_name}"][clusters].mean(axis=1)

    # Get dt
    data[f"ana.rates.dt"] = (
        data[f"df.{dict_name}"]["time"][1] - data[f"df.{dict_name}"]["time"][0]
    )

    return data


# ------------------------------------------------------------------------------ #
# Pauls plot functions
# ------------------------------------------------------------------------------ #


def ps_process_data_from_folder(
    folder_path, processing_functions, file_extension=".hdf5"
):
    """
    Provide a list of `processing_functions`, where each function takes the
    loaded raw data in Victor's format and returns a skaler.

    columns: | line | time | activity 1 | 2 | 3 | 4 | resources 1 | 2 | 3 | 4 |

    this gets applied on all files in the folder that end in `xx.csv`

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


def ps_process_data_from_file(file_path, processing_functions, labels=None):
    y = dict()
    for fdx, f in enumerate(processing_functions):
        label = f.__name__ if labels is None else labels[fdx]
        y[label] = f(file_path)
    return y


def _ps_load_if_path(raw):
    if not isinstance(raw, str):
        return raw

    if ".csv" in raw:
        return pd.read_csv(raw)
    elif ".hdf5" in raw:
        return pd.read_hdf(raw, f"/dataframe")


def ps_f_correlation_coefficients(raw, return_matrix=False):

    raw = _ps_load_if_path(raw)

    # if we have numpy
    # time = raw[:, 1]
    # act = raw[:, 2:6]
    # res = raw[:, 6:10]

    # for pandas df
    # time = raw['time'].to_numpy()
    act = raw[["mod_1", "mod_2", "mod_3", "mod_4"]].to_numpy()
    # res = raw[['mod_1_res', 'mod_2_res', 'mod_3_res', 'mod_4_res']].to_numpy()

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


def ps_f_functional_complexity(raw):
    raw = _ps_load_if_path(raw)
    # we end up calculating rij twice, which might get slow.
    rij = ps_f_correlation_coefficients(raw, return_matrix=True)

    return _functional_complexity(rij)


def ps_f_event_size(raw):

    # this should be redundant with the prepping below
    raw = _ps_load_if_path(raw)
    # lets reuse some of victors tricks
    data = read_csv_and_format(raw, "model")
    # `module_contribution` retuns a list of how many modules contributed
    # (round about the length of detected events)
    contrib = module_contribution(data, "model", 1.0, area_min=0.7, roll_window=0.5)

    return np.nanmean(contrib)
