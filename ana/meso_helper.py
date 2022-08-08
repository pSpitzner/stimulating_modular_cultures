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
import itertools
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import logging
import warnings
from tqdm import tqdm
from benedict import benedict
import matplotlib.pyplot as plt

# Cluster detection, very useful for avalanches
from scipy.ndimage import measurements
# nullclines
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.signal import find_peaks

# out tools
import bitsandbobs as bnb
import ana_helper as ah

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")
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

    # try meta data
    try:
        h5f["meta.coupling"] = bnb.hi5.load(file_path, "/meta/coupling")
        h5f["meta.noise"] = bnb.hi5.load(file_path, "/meta/noise")
        h5f["meta.rep"] = bnb.hi5.load(file_path, "/meta/rep")
        h5f["meta.gating_mechanism"] = bnb.hi5.load(file_path, "/meta/gating_mechanism")
    except:
        pass

    # we may want to plot the gates, they are saved native to hdf5, not part of
    # the dataframe
    h5f["data.gate_history"] = bnb.hi5.load(file_path, "/data/gate_history")

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
    kwargs : dict, passed to `xr.Dataset.to_netcdf()`. Noteworthy:
    group : str, hdf5 group where to place the dataset.
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
        coupling, noise, rep = _coords_from_file(candidate)

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
        coupling, noise, rep = _coords_from_file(candidate)

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


def _coords_from_file(candidate):
    # helper to get simulation coordinates from a file
    try:
        if isinstance(candidate, str):
            # get values from meta data of hdf5 on disk
            coupling = bnb.hi5.load(candidate, "/meta/coupling")
            noise = bnb.hi5.load(candidate, "/meta/noise")
            rep = bnb.hi5.load(candidate, "/meta/rep")
        else:
            # already loaded as benedict
            coupling = candidate["meta.coupling"]
            noise = candidate["meta.noise"]
            rep = candidate["meta.rep"]
    except:
        # get values from regex, this gives noise values as integers, only.
        regex = re.search("coup(\d+.\d+)-(\d+)/noise(\d+).hdf5", candidate, re.IGNORECASE)
        coupling = float(regex.group(1))
        noise = int(regex.group(3))
        rep = int(regex.group(2))

    return coupling, noise, rep


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

    # correlation coefficients of resources are pretty similar to what you get from rates.
    # rsrcs = raw[["mod_0_res",  "mod_1_res",  "mod_2_res",  "mod_3_res"]].to_numpy()
    # rsrcs = np.corrcoef(rsrcs.T)
    # np.fill_diagonal(rsrcs, np.nan)

    if return_matrix:
        return rij
    else:
        return dict(
            mean_correlation_coefficient=np.nanmean(rij),
            median_correlation_coefficient=np.nanmedian(rij),
        )


def f_functional_complexity(raw):
    raw = _load_if_path(raw)
    # we end up calculating rij twice, which might get slow.
    rij = f_correlation_coefficients(raw, return_matrix=True)

    return dict(functional_complexity=ah._functional_complexity(rij))


def f_event_size_and_friends(raw):
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
        merge_threshold=0.5,
    )

    ah.sequences_from_area(h5f)


# ------------------------------------------------------------------------------ #
# Plotting and nullcline solving
# ------------------------------------------------------------------------------ #


def _plot_nullclines_for_input_array(ext_inpt_array=None, tolerance=1e-3, **kwargs):
    """
    Plots nullclines of the mesoscopic model for the selected set of parameters.

    #Parameters
    - ext_inpt_array : N-dim ndarray
        If not None (default) generates a figure with several subplots following the same shape as ext_inpt_array.
    - tolerance : float, optional
        Tolerance to detect when the two branches of the saddle node merge. Default is 1e-3
    - kwargs : dict
        Any argument that can be passed to the mesoscopic model simulation

    #Returns
    - fig : matplotlib figure
        Returns a figure with the phase space and nullclines
    """

    if ext_inpt_array == None:
        ext_str = kwargs.get("ext_str", 0.0)
        fig = plt.figure()
        plot_nullcline(fig.gca(), ext_str, **kwargs)
        return fig
    else:

        # A bit of input checking, to accept also python lists, which is handy for few values
        if isinstance(ext_inpt_array, list):
            ext_inpt_array = np.array(ext_inpt_array)

        # Do also check we are using a simple N-dim array before proceeding, and inform the user
        if len(ext_inpt_array.shape) == 1:
            fig, axes = plt.subplots(ncols=ext_inpt_array.size)
            for ext_str, ax in zip(ext_inpt_array, axes):
                plot_nullcline(ax, ext_str, **kwargs)
        else:
            raise ValueError("ext_input_array must be N-dim.")

        return fig


def plot_nullcline(ax=None, tolerance=1e-3, ode_coords=None, **kwargs):
    """
    Does the actual computation of the nullclines using the indicated control parameter and plots them into the selected axis

    #Parameters
    - ax : matplotlib axis
        The axis where this figure will be drawn
    - ext_str : float
        External input, as the control parameter
    - tolerance : float, optional
        Tolerance to detect when the two branches of the saddle node merge. Default is 1e-3
    - ode_coords : array, optional
        times for which to evaluate.
    - kwargs : dict
        Any argument that can be passed to the mesoscopic model simulation and it is related to single module.
    """

    # Lets not hard-code the model, rather import what we can.
    # remember that all this assumes we are calling from the base directory of the repo.
    sys.path.append("./src")
    from mesoscopic_model import default_pars, transfer_function, single_module_odes

    # pars will be set to defaults, but everything provided here via kwargs is overwritten
    pars = default_pars.copy()
    for key, value in kwargs.items():
        assert key in default_pars.keys(), f"unknown kwarg for mesoscopic model: '{key}'"
        pars[key] = value

    if ax is None:
        fig, ax = plt.subplots(figsize=[6.4, 6.4])
    else:
        fig = ax.get_figure()

    # Get the nullcline for resources, which is quite easy
    r_values = np.linspace(0.01, pars["max_rsrc"] + 1, 1000)
    r_nullc = (
        (pars["max_rsrc"] / r_values - 1.0) * pars["tau_discharge"] / pars["tau_charge"]
    )

    # --- Now let's go for the real deal: activity nullcline

    # Auxiliary shortcut to pre-compute this constant
    # which is used in sigmoid transfer function
    aux_thrsig = np.exp(pars["k_inpt"] * pars["thrs_inpt"])

    # Function defined by dx/dt = 0 that we want to solve,
    # it is transcendental so we will have to go for numerical
    def x_eq(x, r, aux_thrsig, **pars):

        xdot_1 = pars["tau_rate"] * x  # Spontaneous decay to small firing
        xdot_2 = transfer_function(
            r * (x + pars["ext_str"]),
            pars["gain_inpt"],
            pars["k_inpt"],
            pars["thrs_inpt"],
            aux_thrsig,
        )
        return -xdot_1 + xdot_2

    # the line below sets every key in pars as default kwargs,
    # so we avoid having to pass them all as order arguments to fsolve.
    x_eq_with_kwargs = functools.partial(x_eq, aux_thrsig=aux_thrsig, **pars)

    # Method of root finding: we know that for large enough resources system presents an oscillatory bifurcation type-I excitable, so we'll have a saddle node
    # Thus, use two different initial conditions to get all the branch
    # Strategy: we know that for very large resources we have stable up-state near x=gain and unstable in down. So start here and then go
    # backwards, using the latest found solution as initial seed for the next one, to make sure we stay in the stable manifold of the next solution (continuation algorithm)
    x0_1 = pars["gain_inpt"]
    x0_2 = 1.0

    x_nullc_up, x_nullc_dw = [], []  # Empty list that will have our saddle
    is_saddle_found = False  # Integrate both branches till we find saddle

    # Initial value of r and change to get all r_values
    index = -1

    while not is_saddle_found:
        # Get parameters and solutions with different IV
        r = r_values[index]
        arglist = (r,)
        x_nullc_up.append(fsolve(x_eq_with_kwargs, x0_1, args=arglist))
        x_nullc_dw.append(fsolve(x_eq_with_kwargs, x0_2, args=arglist))

        # Use latest found for next. In case of second solution, it can decay to 0 (always stable)
        # so perturb it a little
        x0_1, x0_2 = x_nullc_up[-1], x_nullc_dw[-1] + 0.2

        # Check termination condition
        is_saddle_found = np.abs(x_nullc_up[-1] - x_nullc_dw[-1]) < tolerance
        index -= 1

    start_sn = (
        index + 1
    )  # Store how many points we needed to get to saddle node, needed to plot lower branch

    # Then finish the reamining branch after the saddle, till we find the absorbing state
    while index >= -r_values.size:
        r = r_values[index]
        arglist = (r,)
        x_nullc_up.append(fsolve(x_eq_with_kwargs, x0_1, args=arglist))
        x0_1 = x_nullc_up[-1]
        index -= 1

    # Put the arrays in correct order, from low r to high r
    x_nullc_up = np.array(x_nullc_up[::-1])
    x_nullc_dw = np.array(x_nullc_dw[::-1])

    # --- Now let us get a trajectory to illustrate what the deterministic single module does

    # the line below sets every key in pars as default kwargs
    ode_with_kwargs = functools.partial(single_module_odes, **pars)

    if ode_coords is None:
        # Being excitable, spike is very fast and the other is slow,
        # so sample differently to ensure continuity
        ode_coords = np.concatenate(
            (np.linspace(0, 20, 1000), np.linspace(20.01, 10000, 10000))
        )

    # Initial conditions [rate, rsrc] so that we will have an excitable trajectory:
    # full of resources and small kick
    y0 = np.array([0.5, pars["max_rsrc"]])

    traj = odeint(ode_with_kwargs, y0, ode_coords)

    # --- Plot the graph

    # Nullcline for r is easy
    # ax.plot(r_values, r_nullc, color="green")

    # Absorbing state is stable from beginning till the start of the bifurcation
    # If external input is 0, whole x=0 is stable, if not, only a portion of it
    # if ext_str == 0.0:
    #     ax.axhline(0.0, color="red")
    # else:
    #     ax.plot((r_values[0], r_values[start_sn]), (0.0, 0.0), color="red")

    # up branch pre- and post- bifurcation
    # ax.plot(r_values[start_sn:], x_nullc_up[start_sn:], color="red")
    # ax.plot(r_values[:start_sn], x_nullc_up[:start_sn], color="red", ls="--")
    # down branch
    # ax.plot(r_values[start_sn:], x_nullc_dw, color="red", ls="--")

    # Trajectory
    ax.plot(traj[:, 1], traj[:, 0], color="black", clip_on=False, zorder=4)

    return ax


def plot_flow_field(
    ax=None,
    time_points=None,
    rsrc_max=1.5,
    rsrc_min=0,
    rate_max=17,
    rate_min=0,
    plot_kwargs=dict(),
    **kwargs,
):
    """
    Plot a flow field of the mesoscopic model at provided parameters.

    Integrates the ode of the meso model from a range of initial conditions,
    for a short time. Each IC creates a different trajectory.

    # Paremters
    **kwargs : dict of parameters passed to meso model
    """

    sys.path.append("./src")
    from mesoscopic_model import default_pars, single_module_odes

    # this should look the same everytime we do it.
    np.random.seed(815)

    # pars will be set to defaults, but everything provided here via kwargs is overwritten
    pars = default_pars.copy()
    for key, value in kwargs.items():
        assert key in default_pars.keys(), f"unknown kwarg for mesoscopic model: '{key}'"
        pars[key] = value

    # the line below sets every key in pars as default kwargs
    ode_with_kwargs = functools.partial(single_module_odes, **pars)

    if time_points is None:
        time_points = np.linspace(0, 0.1, 1000)

    # if rsrc_0 is None:
    #     rsrc_0 = np.arange(0, 2, 0.25)
    # if rate_0 is None:
    #     rate_0 = np.arange(0, 10, 0.5)

    # initial_conditions = itertools.product(rate_0, rsrc_0)

    num_trajectories = 1000
    rate_0 = np.random.uniform(low=rate_min, high=rate_max, size=num_trajectories)
    rsrc_0 = np.random.uniform(low=rsrc_min, high=rsrc_max, size=num_trajectories)
    d_rate = rate_max - rate_min
    d_rsrc = rsrc_max - rsrc_min

    initial_conditions = zip(rate_0, rsrc_0)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    trajects = []
    traject_lens = []
    # y0 is a 2d tuple passed to single_module_ode
    for y0 in tqdm(initial_conditions, desc="Flow field"):
        traj = odeint(ode_with_kwargs, y0, time_points)
        trajects.append(traj)

        traject_lens.append(
            np.sum(
                (np.diff(traj[:, 1]) / d_rsrc) ** 2 + (np.diff(traj[:, 0]) / d_rate) ** 2
            )
        )

    # maybe we want to color code this later
    traject_lens = np.array(traject_lens)
    traject_lens = np.log10(traject_lens)
    traject_lens -= np.nanmin(traject_lens) * 1.5
    traject_lens /= np.nanmax(traject_lens)
    traject_lens = np.clip(traject_lens, 0, 1)

    plot_kwargs = plot_kwargs.copy()
    plot_kwargs.setdefault("alpha", 0.3)
    plot_kwargs.setdefault("zorder", 0)
    plot_kwargs.setdefault("lw", 0.2)
    plot_kwargs.setdefault("clip_on", False)
    plot_kwargs.setdefault("color", "black")

    for tdx, traj in enumerate(trajects):
        ax.plot(
            traj[:, 1],
            traj[:, 0],
            # color=plt.cm.Spectral(traject_lens[tdx]),
            # alpha=1.0,
            **plot_kwargs,
        )

    return ax


def plot_h_diagram(ax, tf=1e5, dt=0.01, ode_coords=None, **kwargs):
    """
    Does the actual computation of the nullclines using the indicated control parameter and plots them into the selected axis

    #Parameters
    - ax : matplotlib axis
        The axis where this figure will be drawn
    - kwargs : dict
        Any argument that can be passed to the mesoscopic model simulation and it is related to single module.
    """

    # Lets not hard-code the model, rather import what we can.
    # remember that all this assumes we are calling from the base directory of the repo.
    sys.path.append("./src")
    from mesoscopic_model import default_pars, transfer_function

    # pars will be set to defaults, but everything provided here via kwargs is overwritten
    pars = default_pars.copy()
    for key, value in kwargs.items():
        assert key in default_pars.keys(), f"unknown kwarg for mesoscopic model: '{key}'"
        pars[key] = value

    # Auxiliary shortcut to pre-compute this constant
    # which is used in sigmoid transfer function
    aux_thrsig = np.exp(pars["k_inpt"] * pars["thrs_inpt"])

    # @victor: this is the same ode as above, right?
    # can we place them in `mesoscopic_model.py`?
    # with the pars hack above, this should be neat.
    def module_ODE(vars, t, *args):
        x, r = vars
        max_rsrc = args[0]
        tc, td = args[1:3]
        tau_rate = args[3]
        basefiring = args[4]
        ext_str = args[5]
        k, thr, gain = args[6:9]
        aux_thrsig = args[9]

        xdot_1 = (1 / tau_rate) * (x - basefiring)
        xdot_2 = transfer_function(r * (x + ext_str), gain, k, thr, aux_thrsig)
        rdot = (max_rsrc - r) / tc - r * x / td

        return np.array([-xdot_1 + xdot_2, rdot])

    # Initial conditions are such that we will have an excitable trajectory: full of resources and small kick
    x0 = np.array([0.25, max_rsrc])

    # External inputs to check. We will increase resolution near the transition
    h_space = np.concatenate(
        (np.linspace(0, 0.18, 5), np.linspace(0.18, 0.22, 50), np.linspace(0.22, 0.3, 15))
    )
    frequency = np.zeros(h_space.size) * np.nan

    npoints = int(tf / dt)

    if ode_coords is None:
        # Being excitable, spike is very fast and the other is slow, so sample differently to ensure continuity
        ode_coords = np.linspace(0, tf, npoints)

    # Simulate the deterministic model for increasing external inputs
    for j, ext_str in enumerate(h_space):
        traj = odeint(
            module_ODE,
            x0,
            ode_coords,
            args=(
                max_rsrc,
                tc,
                td,
                tau_rate,
                basefiring,
                ext_str,
                k,
                thres,
                gain,
                aux_thrsig,
            ),
        )
        peaks, _ = find_peaks(traj[:, 0], distance=10, height=7)
        frequency[j] = (
            peaks.size / tf
        )  # Estimation of spiking frequency as number of peaks detected vs time waited

    ax.plot(h_space, frequency)

    return ax


def get_stationary_solutions(input_range, time_points=None, **kwargs):
    """
    Use scipy numeric solver to integrate `single_module_odes` for long times.

    This yields the fixed points

    # Parameters
    input_range : array of floats
        for which input strength to find fixed points
    **kwargs
        other mesoscopic model parameters

    # Returns
    rate, rsrc : 1d arrays
        of length `input_range` corresponding to the stationary rate, and resources.
    """

    sys.path.append("./src")
    from mesoscopic_model import default_pars, single_module_odes

    # pars will be set to defaults, but everything provided here via kwargs is overwritten
    pars = default_pars.copy()
    for key, value in kwargs.items():
        assert key in default_pars.keys(), f"unknown kwarg for mesoscopic model: '{key}'"
        pars[key] = value

    pars.pop("ext_str")

    if time_points is None:
        # lets assume this is t->infty
        time_points = np.linspace(0, 10000, 100000)

    y0 = (0.1, pars["max_rsrc"])

    rates = []
    rsrcs = []

    # y0 is a 2d tuple passed to single_module_ode
    for h in tqdm(input_range):

        # the line below sets every key in pars as default kwargs
        ode_with_kwargs = functools.partial(single_module_odes, ext_str=h, **pars)

        traj = odeint(ode_with_kwargs, y0, time_points)
        # use the last time point as solution
        rate = traj[-1, 0]
        rsrc = traj[-1, 1]
        rates.append(rate)
        rsrcs.append(rsrc)

        # keep track of how much stuff changed over the last few time steps
        rate_converged = 1 - traj[-1000, 0] / rate
        rsrc_converged = 1 - traj[-1000, 1] / rsrc

        log.debug(
            f"h={h:.3f}, rate change {rate_converged:.2e}, resource change"
            f" {rsrc_converged:.2e}"
        )

    return np.array(rates), np.array(rsrcs)


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
