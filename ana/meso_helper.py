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
import functools
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
# I/O
# ------------------------------------------------------------------------------ #


# ps: this was `read_csv_and_format`
def prepare_file(file_path):
    """
    Reads a csv or hdf5 and formats a Benedict with our data format
    Parameters:
    - file_path: string where the CSV or hdf5 file is located
    """

    # this checks file type and loads that raw pandas data frame
    # also does module index shift, if needed, so we start with `mod_0`
    df = _load_if_path(file_path)

    # Create a dictionary and store the names of DF columns
    h5f = benedict()

    # to get many of pauls analysis working, we can pretend every module has
    # only one neuron
    h5f["data.neuron_module_id"] = np.arange(4)
    h5f["ana.mods"] = [f"mod_{m}" for m in range(0, 4)]
    h5f["ana.mod_ids"] = np.arange(4)
    h5f["ana.neuron_ids"] = np.arange(4)

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


def write_xr_dset_to_hdf5(dset, output_path, **kwargs):
    """
    Wrapper to write xarray Dataset to disk as hdf5 with Pauls preferred default
    arguments (mainly compression with zlib for all variables)

    to write a DataArray, you can cast beforehand e.g.
    `my_array.to_dataset(name="data")`

    to load back
    `dset = xr.load_dataset("/path/to/file.hdf5")`

    # Parameters
    kwargs : dict, passed to `xr.Dataset.to_netcdf()`
    """
    # enable compression
    encoding = {d: {"zlib": True, "complevel": 9} for d in dset.data_vars}

    dset.to_netcdf(
        output_path, format="NETCDF4", engine="h5netcdf", encoding=encoding, **kwargs
    )


def _load_if_path(raw):
    """
    Wrapper that loads the file (`raw`) to pandas data frame if it is not loaded, yet
    """

    if isinstance(raw, pd.DataFrame):
        return raw
    else:
        assert isinstance(raw, str), "provide a loaded pandas dataframe or file path"

    if ".csv" in raw:
        df = pd.read_csv(raw)
    elif ".hdf5" in raw:
        df = pd.read_hdf(raw, f"/dataframe")

    # we may need an index shift for older data,
    # because my analysis assumes modules start at index 0, not 1
    if "mod_0" not in df.columns:

        def mapper(col):
            regex = re.search("mod_(\d+)", col, re.IGNORECASE)
            if regex:
                mod_id = int(regex.group(1))
                return col.replace(f"mod_{mod_id}", f"mod_{mod_id-1}")
            else:
                return col

        df.rename(columns=mapper, inplace=True)

    return df


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

    # Returns
    res : dict
        keys are the labels/observables (possibly more than one per function)
        values are the scalar results
    """
    y = dict()
    for fdx, f in enumerate(processing_functions):
        res = f(file_path)
        for key in res.keys():
            y[key] = res[key]
    return y


# ps: a semi-minor todo is to use meta data and more meaningful filenames
def process_data_from_folder(input_folder):
    """
    Calls `process_data_from_file` on all the files in the given folder and returns
    an xarray dataset

    The structure we get by default from `run/meso_launcher` is
    ```
    input_folder
        coup0.10-0
            noise0.hdf5
            noise1.hdf5
        coup0.10-1
            ...
    ```
    0.10 is the coupling value, -0 and 1 are the repetitions
    and the integer after noise is the noise value

    # Returns
    dset : xarray DataSet

    # Example
    ```
    import meso_helper as mh
    dset = mh.process_data_from_folder("./dat/meso_in/")
    mh.write_xr_dset_to_hdf5(dset, "./dat/meso_out/analysed.hdf5")
    ```
    """

    global dset
    global results
    global candidate
    # collect the folders and files
    # the `/**/` for all subdirectories is only supported by python >= 3.5
    candidates = glob.glob(
        os.path.abspath(os.path.expanduser(input_folder + "/**/*.hdf5"))
    )

    couplings = []
    noises = []
    reps = []

    # fetch the coordinates we may find in our folders, in order to create a 3d xarray
    for cdx, candidate in enumerate(candidates):
        regex = re.search("coup(\d+.\d+)-(\d+)/noise(\d+).hdf5", candidate, re.IGNORECASE)
        coupling = float(regex.group(1))
        rep = int(regex.group(2))
        noise = int(regex.group(3))

        if coupling not in couplings:
            couplings.append(coupling)
        if noise not in noises:
            noises.append(noise)
        if rep not in reps:
            reps.append(rep)

    log.info(f"found {len(couplings)} couplings, {len(noises)} noises, {len(reps)} reps")

    # create coords that work with xarrays
    coords = dict()
    coords["repetition"] = np.sort(reps)
    coords["coupling"] = np.sort(couplings)
    coords["noise"] = np.sort(noises)

    # lets have an xarray dataset that has an array for every scalar observable
    dset = xr.Dataset(coords=coords)

    # assigning default variables makes it easer to parallelise via dask, if needed later
    f = functools.partial(
        process_data_from_file,
        processing_functions=[f_correlation_coefficients, f_event_size_and_friends],
    )

    for candidate in tqdm(candidates, desc="analysing files"):
        regex = re.search("coup(\d+.\d+)-(\d+)/noise(\d+).hdf5", candidate, re.IGNORECASE)
        coupling = float(regex.group(1))
        rep = int(regex.group(2))
        noise = int(regex.group(3))
        cs = dict(noise=noise, coupling=coupling, repetition=rep)

        results = f(candidate)
        try:
            observables = list(results.keys())
            assert len(observables) != 0
            dset[observables[0]]
        except KeyError:
            # initialize all data arrays, we count on the observables being the same
            # for all analyzed files.
            for obs in results.keys():
                dset[obs] = xr.DataArray(np.nan, coords=dset.coords)

        for obs in results.keys():
            dset[obs].loc[cs] = results[obs]

    return dset


# ps: old version currently not used, afaik
def _process_data_from_folder(folder_path, processing_functions, file_extension=".hdf5"):
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

    # log.info(candidates, os.path.abspath(os.path.expanduser(folder_path)))
    for cdx, candidate in enumerate(candidates):

        # ~/path/to/noise17.csv
        # get the number used for file indexing
        fdx = re.search(f"(\d+){file_extension}$", candidate, re.IGNORECASE)
        if fdx:
            fdx = fdx.group(1)
        else:
            fdx = -1
            log.error(f"no file index found for {candidate}")
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
        # log.info(cdx, raw, y)

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

    act = raw[["mod_0", "mod_1", "mod_2", "mod_3"]].to_numpy()

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
        return dict(correlation_coefficient=np.nanmean(rij))


def f_functional_complexity(raw):
    raw = _load_if_path(raw)
    # we end up calculating rij twice, which might get slow.
    rij = f_correlation_coefficients(raw, return_matrix=True)

    return dict(functional_complexity=ah._functional_complexity(rij))


def f_event_size_and_friends(raw):
    # this should be redundant with the prepping below
    raw = _load_if_path(raw)
    # lets reuse some of victors tricks
    h5f = prepare_file(raw)
    find_system_bursts_and_module_contributions2(h5f)

    res = dict()
    res["event_size"] = np.nanmean(h5f["ana.bursts.event_sizes"])

    # here we stick with convention that is not great but has not been refactored.
    # number of bursts with different number of modules contributing
    slens = h5f["ana.bursts.system_level.module_sequences"]
    res["any_num_b"] = len(slens)
    res["mod_num_b_0"] = len([x for x in slens if len(x) == 0])
    res["mod_num_b_1"] = len([x for x in slens if len(x) == 1])
    res["mod_num_b_2"] = len([x for x in slens if len(x) == 2])
    res["mod_num_b_3"] = len([x for x in slens if len(x) == 3])
    res["mod_num_b_4"] = len([x for x in slens if len(x) == 4])

    return res



# this guy breaks convention
# before was called `module_contribution`
# I adapted this a bit to not return anything, but rather add the info to the h5f
# structure. This makes it more similar to what we have in the `ana_helper.py`
def find_system_bursts_and_module_contributions(
    h5f,
    system_thres=1.0,
    roll_window=0.5,
    area_min=0.7,
):
    """
    Detect burst start- and end-times using Vicotrs area overlap method.
    From those, we find which module contributes to which burst.
    Modifies h5f in place and add a couple of entries

    # Parameters:
    h5f : benedict
        containing all information (or raw dataframe, in which case we call `prepare_file(h5f)`)
    system_thres : float,
        threshold to consider system-wide burst
    roll_window : float,
        in seconds, width of rolling average kernel
    area_min : flaot,
        how much area do we consider to see if a module contributed.

    # Returns
    h5f : benedict
        with additional / overwritten entires:
        - h5f["ana.bursts.system_level.beg_times"]
        - h5f["ana.bursts.system_level.end_times"]
        - h5f["ana.bursts.system_level.module_sequences"]
        - h5f["data.spiketimes"]
        - h5f["ana.bursts.areas"]
        - h5f["ana.bursts.contributions"]
    """

    if isinstance(h5f, pd.DataFrame) or isinstance(h5f, str):
        h5f = prepare_file(h5f)

    dt = h5f["ana.rates.dt"]
    n_roll = int(roll_window / dt)

    # Get total rate and module rate
    datacols = h5f["ana.mods"]
    # we could do this from the already calculated system-level rate, right?
    smoothrate = h5f[f"data.raw_df"][datacols].mean(axis=1).rolling(n_roll).mean().values
    smoothcluster = h5f[f"data.raw_df"][datacols].rolling(n_roll).mean().values

    # Filter total rate and label with a different tag each burst
    filterrate = smoothrate >= system_thres
    features, num = measurements.label(filterrate)

    # Get where are these tags positioned
    unique, label_events = np.unique(features, return_index=True)
    del unique

    # Burst times
    # -----------
    # paul: label events contains integer start times of bursts (rate above threshold)
    label_events = label_events[1:]  # (from 1 onwards, 0 means no cluster)

    # indices of burst begin-times and end-times
    beg_idx = label_events
    end_idx = list()

    # now get the end times
    for i in range(0, len(beg_idx)):
        burst = beg_idx[i]
        try:
            next_burst = beg_idx[i + 1]
        except:
            next_burst = len(smoothrate) - 1
        burst_duration = len(np.where(smoothrate[burst:next_burst] >= system_thres)[0])
        end_idx.append(burst + burst_duration)
    end_idx = np.array(end_idx)

    beg_times = beg_idx * dt
    end_times = end_idx * dt
    h5f["ana.bursts.system_level.beg_times"] = beg_times.tolist()
    h5f["ana.bursts.system_level.end_times"] = end_times.tolist()

    # Contributions / Sequences
    # -------------------------
    contribution = list()
    sequences = list()
    areas = list()
    # victors hack: use the burst beg_times as spike times.
    # if we save them as spike times, we can use the raster plot from plot_helper.py
    # spiketimes are a 2d array, nan-padded
    spiketimes = np.ones(shape=(4, len(beg_times))) * np.nan

    for bdx in range(0, len(beg_idx)):
        beg = beg_idx[bdx]
        end = end_idx[bdx]

        # Get contribution from each module
        area_contrib = np.empty(4)
        for c in range(4):
            filtered = smoothcluster[:, c][beg:end]
            filtered = filtered[filtered >= system_thres]
            area_contrib[c] = 0.5 * np.sum(
                filtered[1:] + filtered[:-1]
            )  # - 2*system_thres)

        area_contrib /= np.sum(area_contrib)
        areas.append(4 * area_contrib)

        # this gives us the number of contributing modules
        contribution.append(int((4 * area_contrib > area_min).sum()))

        # Sequences are not ordered but they are also used by some of
        # pauls functions to find out which  module contributed
        # to which bursts
        seq = tuple(np.where(4 * area_contrib > area_min)[0])
        sequences.append(seq)

        for mod_id in range(0, 4):
            if mod_id in seq:
                spiketimes[mod_id, bdx] = beg_times[bdx]

    h5f["ana.bursts.system_level.module_sequences"] = sequences
    h5f["data.spiketimes"] = np.sort(spiketimes)

    # these are so far unused but may come in handy
    h5f["ana.bursts.areas"] = areas
    h5f["ana.bursts.contributions"] = contribution

    return h5f


def find_system_bursts_and_module_contributions2(h5f, threshold_factor=0.1):
    """
    We can use the same pipeline as for experiments and microscopic simulations to find burst begin and end times.

    What remains is the event size and module contribution.

    # Parameters
    threshold_factor : at how many percent of max system rate to threshold
        (0.1 in exp., 0.025 for microscopic sim)
    """

    threshold = threshold_factor * np.nanmax(h5f["ana.rates.system_level"])
    ah.find_system_bursts_from_global_rate(
        h5f,
        skip_sequences=True,
        rate_threshold=threshold,
        merge_threshold=0.1,
    )

    dt = h5f["ana.rates.dt"]
    sys_rate = h5f["ana.rates.system_level"]
    beg_times = np.array(h5f["ana.bursts.system_level.beg_times"])
    end_times = np.array(h5f["ana.bursts.system_level.end_times"])
    beg_idx = (beg_times / dt).astype(int)
    end_idx = (end_times / dt).astype(int)

    contributions = list()
    sequences = list()
    areas = list()
    event_sizes = list()

    for bdx in range(0, len(beg_idx)):
        beg = beg_idx[bdx]
        end = end_idx[bdx]

        mod_areas = np.zeros(4)
        for mod_id in range(0, 4):
            rate = h5f[f"ana.rates.module_level.mod_{mod_id}"]
            mod_areas[mod_id] = np.sum(rate[beg:end])
        mod_areas /= np.sum(mod_areas)
        areas.append(mod_areas)

        # Sequences are not ordered but they are also used by some of
        # pauls functions to find out which  module contributed
        # to which bursts
        seq = tuple(np.where(mod_areas > 0.10)[0])
        sequences.append(seq)
        contributions.append(len(seq))

        event_sizes.append(np.sum(sys_rate[beg:end]))

    h5f["ana.bursts.areas"] = areas
    h5f["ana.bursts.system_level.module_sequences"] = sequences
    h5f["ana.bursts.contributions"] = contributions
    h5f["ana.bursts.event_sizes"] = event_sizes


# ------------------------------------------------------------------------------ #
# Others
# ------------------------------------------------------------------------------ #


def fourier_transform(h5f, t_sti_start=600.0, t_sti_end=1200.0, dataroll=10, avroll=5):
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
