# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-03-10 13:23:16
# @Last Modified: 2021-11-12 21:50:11
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

# from hi5 import BetterDict
import hi5 as h5
from addict import Dict
from benedict import benedict
from tqdm import tqdm
from itertools import permutations

import logging
import warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

try:
    from numba import jit, prange

    # raise ImportError
    # let's not print this 256 times on import when using 256 threads :P
    # log.info("Using numba for parallelizable functions")

    try:
        from numba.typed import List
    except:
        # older numba versions dont have this
        def List(*args):
            return list(*args)

    # silence deprications
    try:
        from numba.core.errors import (
            NumbaDeprecationWarning,
            NumbaPendingDeprecationWarning,
        )

        warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
        warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
    except:
        pass

except ImportError:
    log.info("Numba not available, skipping compilation")
    # replace numba functions if numba not available:
    # we only use jit and prange
    # helper needed for decorators with kwargs
    def parametrized(dec):
        def layer(*args, **kwargs):
            def repl(f):
                return dec(f, *args, **kwargs)

            return repl

        return layer

    @parametrized
    def jit(func, **kwargs):
        return func

    def prange(*args):
        return range(*args)

    def List(*args):
        return list(*args)


# ------------------------------------------------------------------------------ #
# high level functions
# ------------------------------------------------------------------------------ #


def prepare_file(
    h5f, mod_colors="auto", hot=True, skip=None,
):
    """
        modifies h5f in place! (not on disk, only in RAM)

        # Parameters
        h5f           : file path as str or existing h5f benedict
        mod_colors    : "auto" or False (all back) or list of colors
        hot           : wether to load data to ram (true) or fetch as needed (false)
        skip          : lists of names of datasets of the h5file that are excluded,
                        if `h5f` is a path

        # adds the following attributes:
        h5f["ana.mod_sort"]   : function that maps from neuron_id to sorted id, by module
        h5f["ana.mods"]       : list of unique module ids
        h5f["ana.mod_colors"] : list of colors associated with each module
        h5f["ana.neuron_ids"] : array of neurons, if we speciefied a sensor, this will
                             only contain the recorded ones.
    """

    log.debug("Preparing File")

    if isinstance(h5f, str):
        h5f = h5.recursive_load(h5f, hot=hot, skip=skip, dtype=benedict)

    h5f["ana"] = benedict()
    num_n = h5f["meta.topology_num_neur"]
    # if we had a sensor, many neurons were not recorded and cannot be analyzed.
    if "meta.topology_n_within_sensor" in h5f.keypaths():
        num_n = h5f["meta.topology_n_within_sensor"]

    # ------------------------------------------------------------------------------ #
    # mod sorting
    # ------------------------------------------------------------------------------ #
    try:
        # get the neurons sorted according to their modules
        mod_sorted = np.zeros(num_n, dtype=int)
        mod_ids = h5f["data.neuron_module_id"][:]
        mods = np.sort(np.unique(mod_ids))
        if len(mods) == 1:
            raise NotImplementedError  # avoid resorting.
        temp = np.argsort(mod_ids)
        for n_id in range(0, num_n):
            mod_sorted[n_id] = np.argwhere(temp == n_id)

        h5f["ana.mods"] = [f"mod_{m}" for m in mods]
        h5f["ana.mod_ids"] = mods
        h5f["ana.mod_sort"] = lambda x: mod_sorted[x]
    except Exception as e:
        log.debug(e)
        h5f["ana.mods"] = ["mod_0"]
        h5f["ana.mod_ids"] = [0]
        h5f["ana.mod_sort"] = lambda x: x

    # ------------------------------------------------------------------------------ #
    # assign colors to modules so we can use them in every plot consistently
    # ------------------------------------------------------------------------------ #
    if mod_colors is False:
        h5f["ana.mod_colors"] = ["black"] * len(h5f["ana.mods"])
    elif mod_colors == "auto":
        h5f["ana.mod_colors"] = [f"C{x}" for x in range(0, len(h5f["ana.mods"]))]
    else:
        assert isinstance(mod_colors, list)
        assert len(mod_colors) == len(h5f["ana.mods"])
        h5f["ana.mod_colors"] = mod_colors

    # ------------------------------------------------------------------------------ #
    # spikes
    # ------------------------------------------------------------------------------ #

    # maybe change this to exclude neurons that did not spike
    # neuron_ids = np.unique(spikes[:, 0]).astype(int, copy=False)
    neuron_ids = np.arange(0, num_n, dtype=int)
    h5f["ana.neuron_ids"] = neuron_ids

    # make sure that the 2d_spikes representation is nan-padded, requires loading!
    try:
        spikes = h5f["data.spiketimes"][:]
        if spikes is None:
            raise ValueError
        spikes[spikes == 0] = np.nan
        h5f["data.spiketimes"] = spikes
    except:
        log.info("No spikes in file, plotting and analysing dynamics will not work.")

    # # now we need to load things. [:] loads to ram and makes everything else faster
    # # convert spikes in the convenient nested (2d) format, first dim neuron,
    # # then ndarrays of time stamps in seconds
    # spikes = h5f["data.spiketimes_as_list"][:]
    # spikes_2d = []
    # for n_id in neuron_ids:
    #     idx = np.where(spikes[:, 0] == n_id)[0]
    #     spikes_2d.append(spikes[idx, 1])
    # # the outer array is essentially a list but with fancy indexing.
    # # this is a bit counter-intuitive
    # h5f["ana.spikes_2d"] = np.array(spikes_2d, dtype=object)

    # Stimulation description
    stim_str = "Unknown"
    if not "data.stimulation_times_as_list" in h5f.keypaths():
        stim_str = "Off"
    else:
        try:
            stim_neurons = np.unique(
                h5f["data.stimulation_times_as_list"][:, 0]
            ).astype(int)
            stim_mods = np.unique(h5f["data.neuron_module_id"][stim_neurons])
            stim_str = f"On {str(tuple(stim_mods)).replace(',)', ')')}"
        except Exception as e:
            log.warning(e)
            stim_str = f"Error"
    h5f["ana.stimulation_description"] = stim_str

    # Guess the repetition from filename, convention: `foo/bar_parameters_rep=09.hdf5`
    try:
        fname = str(h5f["uname.original_file_path"].decode("UTF-8"))
        rep = re.search("(?<=rep=)(\d+)", fname)[0]  # we only use the first match
        h5f["ana.repetition"] = int(rep)
    except Exception as e:
        log.debug(e)
        h5f["ana.repetition"] = -1

    return h5f



def load_experimental_files(path_prefix, condition="1_pre_"):
    """
        helper to import experimental csv files from jordi into a compatible
        h5f

        # Parameters
        path_prefix: str
        condition: str

        # Returns
        h5f: benedict with our needed strucuter
    """

    # assert os.path.isdir(path_prefix)

    h5f = benedict()

    # ROIs as neuron centers
    rois = np.loadtxt(f"{path_prefix}/RoiSet_Cartesian.txt", delimiter=",", skiprows=1)

    h5f["data.neuron_pos_x"] = rois[:, 1].copy()
    h5f["data.neuron_pos_y"] = rois[:, 2].copy()

    # add some more stuff that is usually already in the meta data
    num_n = len(h5f["data.neuron_pos_x"])
    h5f["meta.topology_num_neur"] = num_n

    # in this format, we have a 50 ms timestep, the column is the neuron id
    # and the row is whether a neuron fired in this time step.
    spikes_as_sparse = np.loadtxt(
        f"{path_prefix}{condition}/raster.csv", delimiter=",", skiprows=0
    )

    # drop first 60 seconds due to artifacts at the beginning of the recording
    spikes_as_sparse = spikes_as_sparse[1200:, :]

    spikes_as_list = _spikes_as_sparse_to_spikes_as_list(spikes_as_sparse, dt=50 / 1000)
    h5f["data.spiketimes_as_list"] = spikes_as_list
    h5f["data.spiketimes"] = _spikes_as_list_to_spikes_2d(spikes_as_list, num_n=num_n)

    # approximate module ids from position
    # 0 lower left, 1 upper left, 2 lower right, 3 upper right
    h5f["data.neuron_module_id"] = np.ones(num_n, dtype=int) * -1
    if "merged" in path_prefix:
        lim = 200
    else:
        lim = 300
    for nid in range(0, num_n):
        x = h5f["data.neuron_pos_x"][nid]
        y = h5f["data.neuron_pos_y"][nid]
        if x < lim and y < lim:
            h5f["data.neuron_module_id"][nid] = 0
        elif x < lim and y >= lim:
            h5f["data.neuron_module_id"][nid] = 1
        elif x >= lim and y < lim:
            h5f["data.neuron_module_id"][nid] = 2
        elif x >= lim and y >= lim:
            h5f["data.neuron_module_id"][nid] = 3

    h5f["meta.dynamics_simulation_duration"] = 540.0

    try:
        # fluorescence traces
        fl_traces = np.loadtxt(
            f"{path_prefix}{condition}/Results.csv", delimiter=",", skiprows=1
        )

        # for each neuron, we have 4 columns, and want to use the 2nd one, "mean"
        # first col is time index, then we start counting neurons
        fl_idx = np.arange(0, num_n, dtype="int")*4 + 2
        fl_traces = fl_traces[:, fl_idx]

        # drop first 60 seconds due to artifacts at the beginning of the recording
        fl_traces = fl_traces[1200:, :]

        h5f["data.neuron_fluorescence_trace"] = fl_traces.copy().T
        h5f["data.neuron_fluorescence_timestep"] = 50 / 1000
        # h5f["pd.fl"] = pd.read_csv(f"{path_prefix}{condition}/Results.csv")

    except Exception as e:
        log.exception(f"{path_prefix} {condition}")
        log.exception(e)

    return prepare_file(h5f)


# depricating, do this more explicitly
def __find_bursts_from_rates(
    h5f,
    rate_threshold=None,  # default 7.5 Hz
    merge_threshold=0.1,  # seconds, merge bursts if separated by less than this
    system_bursts_from_modules=True,
    write_to_h5f=True,
    return_res=False,
):
    """
        Based on module-level firing rates, find bursting events.

        returns two benedicts, `bursts` and `rates`,
        modifies `h5f`

        # Parameters
        h5f : benedict, with our usual structure
        bs_large : float, seconds, time bin size to smooth over (gaussian kernel)
        bs_small : float, seconds, small bin size at which the rate is sampled
        rate_threshold : float, Hz, above which we start detecting bursts
        merge_threshold : float, seconds merge bursts if separated by less than this
        system_bursts_from_modules : bool, whether the system-level bursts are
            "stitched together" from individual modules or detected independently
            from the system wide rate, using `rate_threshold`


        Note on smoothing: previously, we time-binned the activity on the module level
        and convolve this series with a gaussian kernel to smooth.
        Current, precise way is to convolve the spike-train of each neuron with
        the kernel (thus, keeping the high precision of each spike time).
    """

    assert h5f["ana"] is not None, "`prepare_file(h5f)` first!"
    assert write_to_h5f or return_res

    if rate_threshold is None:
        rate_threshold = 7.5

    spikes = h5f["data.spiketimes"]

    bursts = benedict()
    rates = benedict()
    rates["dt"] = bs_small

    beg_times = []  # lists of length num_modules
    end_times = []

    try:
        duration = h5f["meta.dynamics_simulation_duration"]
    except KeyError:
        duration = np.nanmax(h5f["data.spiketimes"]) + 15

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        # for keys, use readable description of the module
        m_dc = h5f["ana.mods"][mdx]
        selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
        # if sensor is specified, we might get more neuron_module_id than were recorded
        selects = selects[np.isin(selects, h5f["ana.neuron_ids"])]
        pop_rate = population_rate_exact_smoothing(
            spikes[selects], bin_size=bs_small, smooth_width=bs_large, length=duration,
        )

        beg_time, end_time = burst_detection_pop_rate(
            rate=pop_rate, bin_size=bs_small, rate_threshold=rate_threshold,
        )

        if len(beg_time) > 0:
            beg_time, end_time = merge_if_below_separation_threshold(
                beg_time, end_time, threshold=merge_threshold
            )

        beg_times.append(beg_time)
        end_times.append(end_time)

        rates[f"module_level.{m_dc}"] = pop_rate
        rates[f"cv.module_level.{m_dc}"] = np.nanstd(pop_rate) / np.nanmean(pop_rate)
        bursts[f"module_level.{m_dc}.beg_times"] = beg_time.copy()
        bursts[f"module_level.{m_dc}.end_times"] = end_time.copy()
        bursts[f"module_level.{m_dc}.rate_threshold"] = rate_threshold

    pop_rate = population_rate_exact_smoothing(
        spikes[:], bin_size=bs_small, smooth_width=bs_large, length=duration,
    )
    rates["system_level"] = pop_rate
    rates["cv.system_level"] = np.nanstd(pop_rate) / np.nanmean(pop_rate)

    if system_bursts_from_modules:
        sys_begs, sys_ends, sys_seqs = system_burst_from_module_burst(
            beg_times, end_times, threshold=merge_threshold,
        )
    else:
        sys_begs, sys_ends = burst_detection_pop_rate(
            rate=pop_rate, bin_size=bs_small, rate_threshold=rate_threshold,
        )
        # in this case, we dont get sequences for free. lets check the first spike
        # in each modules
        sys_seqs = []
        for idx in range(0, len(sys_begs)):
            beg = sys_begs[idx]
            end = sys_ends[idx]
            firsts = np.ones(len(h5f["ana.mods"])) * np.nan
            for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
                m_dc = h5f["ana.mods"][mdx]
                selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
                selects = selects[np.isin(selects, h5f["ana.neuron_ids"])]
                s = spikes[selects]
                s = s[(s >= beg) & (s <= end)]
                if len(s) > 0:
                    firsts[mdx] = np.nanmin(s)
            num_valid = len(firsts[np.isfinite(firsts)])
            mdx_order = np.argsort(firsts)[0:num_valid]
            seq = tuple(np.array(h5f["ana.mod_ids"])[mdx_order])
            sys_seqs.append(seq)

    bursts["system_level.beg_times"] = sys_begs.copy()
    bursts["system_level.end_times"] = sys_ends.copy()
    bursts["system_level.module_sequences"] = sys_seqs

    if write_to_h5f:
        # if isinstance(h5f["ana.bursts"], benedict):
        # if isinstance(h5f["ana.rates"], benedict):
        try:
            h5f["ana.bursts"].clear()
        except Exception as e:
            log.debug(e)
        try:
            h5f["ana.rates"].clear()
        except Exception as e:
            log.debug(e)
        # so, overwriting keys with dicts (nesting) can cause memory leaks.
        # to avoid this, call .clear() before assigning the new dict
        # testwise I made this the default for setting keys of benedict
        h5f["ana.bursts"] = bursts
        h5f["ana.rates"] = rates

    if return_res:
        return bursts, rates


def find_rates(
    h5f,
    bs_large=0.02,  # seconds, time bin size to smooth over (gaussian kernel)
    bs_small=0.0005,  # seconds, small bin size
    write_to_h5f=True,
    return_res=False,
):
    """
        Uses `population_rate_exact_smoothing` to find global system rate
        and the rates within modules

        Note on smoothing: previously, we time-binned the activity on the module level
        and convolve this series with a gaussian kernel to smooth.
        Current, precise way is to convolve the spike-train of each neuron with
        the kernel (thus, keeping the high precision of each spike time).
    """
    assert h5f["ana"] is not None, "`prepare_file(h5f)` first!"
    assert write_to_h5f or return_res

    spikes = h5f["data.spiketimes"]

    rates = benedict()
    rates["dt"] = bs_small

    beg_times = []  # lists of length num_modules
    end_times = []

    try:
        duration = h5f["meta.dynamics_simulation_duration"]
    except KeyError:
        duration = np.nanmax(h5f["data.spiketimes"]) + 15

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
        selects = selects[np.isin(selects, h5f["ana.neuron_ids"])]
        pop_rate = population_rate_exact_smoothing(
            spikes[selects], bin_size=bs_small, smooth_width=bs_large, length=duration,
        )

        rates[f"module_level.{m_dc}"] = pop_rate
        rates[f"cv.module_level.{m_dc}"] = np.nanstd(pop_rate) / np.nanmean(pop_rate)

    pop_rate = population_rate_exact_smoothing(
        spikes[:], bin_size=bs_small, smooth_width=bs_large, length=duration,
    )
    rates["system_level"] = pop_rate
    rates["cv.system_level"] = np.nanstd(pop_rate) / np.nanmean(pop_rate)

    if write_to_h5f:
        try:
            h5f["ana.rates"].clear()
        except Exception as e:
            log.debug(e)
        # so, overwriting keys with dicts (nesting) can cause memory leaks.
        # to avoid this, call .clear() before assigning the new dict
        # testwise I made this the default for setting keys of benedict
        h5f["ana.rates"] = rates

    if return_res:
        return rates


def find_system_bursts_from_module_bursts(
    h5f,
    rate_threshold,  # Hz
    merge_threshold,  # seconds, merge bursts if separated by less than this
    write_to_h5f=True,
    return_res=False,
):
    """
        Based on module-level firing rates, find bursting events.

        optionally returns `bursts`
        optionally modifies `h5f`

        # Parameters
        h5f : benedict, with our usual structure
        rate_threshold : float in Hz, above which we start detecting bursts
        merge_threshold : float, seconds merge bursts if separated by less than this

        # Adds to h5f if `write_to_h5f`:
            h5f["ana.bursts.module_level.{m_dc}.beg_times"]
            h5f["ana.bursts.module_level.{m_dc}.end_times"]
            h5f["ana.bursts.module_level.{m_dc}.rate_threshold"]
            h5f["ana.system_level.beg_times"]
            h5f["ana.system_level.end_times"]
            h5f["ana.system_level.module_sequences"]

    """

    rate_dt = h5f["ana.rates.dt"]
    bursts = benedict()
    beg_times = []
    end_times = []

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        rate = h5f[f"ana.rates.module_level.{m_dc}"]
        beg_time, end_time = burst_detection_pop_rate(
            rate=rate, bin_size=rate_dt, rate_threshold=rate_threshold,
        )
        beg_time, end_time = merge_if_below_separation_threshold(
            beg_time, end_time, threshold=merge_threshold
        )
        beg_times.append(beg_time)
        end_times.append(end_time)

        bursts[f"module_level.{m_dc}.beg_times"] = beg_time.copy()
        bursts[f"module_level.{m_dc}.end_times"] = end_time.copy()
        bursts[f"module_level.{m_dc}.rate_threshold"] = rate_threshold

    sys_begs, sys_ends, sys_seqs = system_burst_from_module_burst(
        beg_times, end_times, threshold=merge_threshold,
    )

    bursts["system_level.beg_times"] = sys_begs.copy()
    bursts["system_level.end_times"] = sys_ends.copy()
    bursts["system_level.module_sequences"] = sys_seqs
    bursts["system_level.algorithm"] = "from_module_bursts"

    if write_to_h5f:
        try:
            h5f["ana.bursts"].clear()
        except Exception as e:
            log.debug(e)
        h5f["ana.bursts"] = bursts

    if return_res:
        return bursts


def find_system_bursts_from_global_rate(
    h5f,
    rate_threshold,  # Hz
    merge_threshold,  # seconds, merge bursts if separated by less than this
    write_to_h5f=True,
    return_res=False,
    skip_sequences=False,
    **sequence_kwargs,
):
    """
        Find global bursting events only based on the merged down rate.
        To get sequences, uses `sequences_from_module_contribution` and
        passes `sequence_kwargs`.
        Per default, to count a module as "contributing" to a sequence,
        at least _20%_ of the neurons of the module (but at least one neuron) have to
        contribute at least _1_ spike

        optionally returns `bursts`
        optionally modifies `h5f`

        Note: does not provide the
        h5f["ana.bursts.module_level.{m_dc}"]

        # Parameters
        h5f : benedict, with our usual structure
        rate_threshold : float, Hz, above which we start detecting bursts
        merge_threshold : float, seconds merge bursts if separated by less than this

        # Adds to h5f if `write_to_h5f`:
            h5f["ana.system_level.beg_times"]
            h5f["ana.system_level.end_times"]
    """

    bursts = benedict()
    rate = h5f[f"ana.rates.system_level"]
    rate_dt = h5f["ana.rates.dt"]

    beg_times, end_times = burst_detection_pop_rate(
        rate=rate, bin_size=rate_dt, rate_threshold=rate_threshold,
    )
    beg_times, end_times = merge_if_below_separation_threshold(
        beg_times, end_times, threshold=merge_threshold
    )


    if skip_sequences:
        sys_seqs = [tuple()] * len(beg_times)
    else:
        sequence_kwargs.setdefault("min_spikes", 1)
        # 20% of a modules neurons
        npm = h5f["meta.topology_num_neur"] / len(h5f["ana.mod_ids"])
        min_neurons = np.nanmax([1, int(0.20 * npm)])
        sequence_kwargs.setdefault("min_neurons", min_neurons)
        sys_seqs = sequences_from_module_contribution(
            h5f, beg_times, end_times, **sequence_kwargs
        )

    bursts["beg_times"] = beg_times.copy()
    bursts["end_times"] = end_times.copy()
    bursts["module_sequences"] = sys_seqs
    bursts["rate_threshold"] = rate_threshold
    bursts["algorithm"] = "from_global_rate"

    if write_to_h5f:
        try:
            h5f["ana.bursts.system_level"].clear()
        except Exception as e:
            log.debug(e)
        h5f["ana.bursts.system_level"] = bursts

    if return_res:
        return bursts


def find_isis(h5f, write_to_h5f=True, return_res=False):
    """
        What are the the inter-spike-intervals within and out of bursts?
    """

    assert write_to_h5f or return_res

    isi = benedict()

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
        selects = selects[np.isin(selects, h5f["ana.neuron_ids"])]  # sensor
        spikes_2d = h5f["data.spiketimes"][selects]
        try:
            b = h5f[f"ana.bursts.module_level.{m_dc}.beg_times"]
            e = h5f[f"ana.bursts.module_level.{m_dc}.end_times"]
        except:
            if mdx == 0:
                log.debug("Module bursts were not detected before searching ISI.")
            b = None
            e = None

        ll_isi = _inter_spike_intervals(spikes_2d, beg_times=b, end_times=e,)
        isi[m_dc] = ll_isi

    if write_to_h5f:
        h5f["ana.isi"] = isi

    if return_res:
        return isi


def find_ibis(h5f, write_to_h5f=True, return_res=False):
    """
        What are the the inter-burst-intervals? End-of-burst to start-of-burst
    """
    assert write_to_h5f or return_res

    ibi = benedict()
    # ibi["module_level"] = benedict()
    # ibi["system_level"] = benedict()

    l_ibi_across_mods = []
    for mdx, m_id in enumerate(h5f["ana.mods"]):
        m_dc = h5f["ana.mods"][mdx]
        try:
            b = np.array(h5f[f"ana.bursts.module_level.{m_dc}.beg_times"])
            e = np.array(h5f[f"ana.bursts.module_level.{m_dc}.end_times"])
        except Exception as e:
            log.debug("Module-level bursts were not detected before searching IBI.")
            break
            # raise e

        if len(b) < 2:
            l_ibi = np.array([])
        else:
            l_ibi = b[1:] - e[:-1]

        l_ibi = l_ibi.tolist()
        ibi[f"module_level.{m_dc}"] = l_ibi
        l_ibi_across_mods.extend(l_ibi)

    # and again for system-wide, no matter how many modules involved
    try:
        b = np.array(h5f["ana.bursts.system_level.beg_times"])
        e = np.array(h5f["ana.bursts.system_level.end_times"])
    except Exception as e:
        log.error("System-level bursts were not detected before searching IBI.")
        raise e

    if len(b) < 2:
        l_ibi = np.array([])
    else:
        l_ibi = e[1:] - b[:-1]

    ibi["system_level.any_module"] = l_ibi.tolist()
    ibi["system_level.cv_any_module"] = np.nanstd(l_ibi) / np.nanmean(l_ibi)
    ibi["system_level.cv_across_modules"] = np.nanstd(l_ibi_across_mods) / np.nanmean(
        l_ibi_across_mods
    )

    # we are also interested in system-wide bursts that included all modules
    try:
        # l = np.vectorize(len)(h5f["ana.bursts.system_level.module_sequences"][1:])
        # deprication warning -.-
        slen = [len(seq) for seq in h5f["ana.bursts.system_level.module_sequences"]]
        idx = np.where(np.array(slen) >= len(h5f["ana.mods"]))
        b = b[idx]
        e = e[idx]
        if len(b) < 2:
            l_ibi = np.array([])
        else:
            l_ibi = e[1:] - b[:-1]
        ibi["system_level.all_modules"] = l_ibi.tolist()
        ibi["system_level.cv_all_modules"] = np.nanstd(l_ibi) / np.nanmean(l_ibi)
    except Exception as e:
        log.info("Failed to find system-wide ibis (where all modules are involved)")
        log.info(e)

    if write_to_h5f:
        try:
            h5f["ana.ibi"].clear()
        except:
            pass
        h5f["ana.ibi"] = ibi

    if return_res:
        return ibi


def find_participating_fraction_in_bursts(h5f, write_to_h5f=True, return_res=False):
    """
        Once we have found bursts, check what is the fraction of neurons participating
        in every burst, and the total number of spikes.

        adds `participating_fraction` to the bursts: fraction of unique neurons fired
        a spike in the burst (relative to total number of neurons in module / system)
        adds `num_spikes_in_bursts`: how many spikes per contributing neuron
    """

    assert "ana.bursts" in h5f.keypaths(), "run `find_bursts_from_rates` first"
    assert write_to_h5f or return_res

    spikes = h5f["data.spiketimes"]
    bursts = h5f["ana.bursts"]
    if not write_to_h5f:
        bursts = bursts.clone()

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
        selects = selects[np.isin(selects, h5f["ana.neuron_ids"])]  # sensor
        try:
            bt = bursts[f"module_level.{m_dc}.beg_times"]
            et = bursts[f"module_level.{m_dc}.end_times"]
        except KeyError:
            log.debug("No module bursts for participating fraction. Skipping")
            break
        fraction = np.zeros(len(bt))
        num_spks = np.zeros(len(bt))
        for bdx in range(0, len(bt)):
            n_ids = np.where(
                (bt[bdx] <= spikes[selects]) & (spikes[selects] <= et[bdx])
            )[0]
            n_unk = len(np.unique(n_ids))
            fraction[bdx] = n_unk / len(selects)
            num_spks[bdx] = len(n_ids) / np.fmax(n_unk, 1)
        bursts[f"module_level.{m_dc}.participating_fraction"] = fraction.tolist()
        bursts[f"module_level.{m_dc}.num_spikes_in_bursts"] = num_spks.tolist()

    # system level
    selects = h5f["ana.neuron_ids"]
    bt = bursts["system_level.beg_times"]
    et = bursts["system_level.end_times"]
    fraction = np.zeros(len(bt))
    num_spks = np.zeros(len(bt))
    for bdx in range(0, len(bt)):
        n_ids = np.where((bt[bdx] <= spikes[selects]) & (spikes[selects] <= et[bdx]))[0]
        n_unk = len(np.unique(n_ids))
        fraction[bdx] = n_unk / len(selects)
        num_spks[bdx] = len(n_ids) / np.fmax(n_unk, 1)
    bursts["system_level.participating_fraction"] = fraction.tolist()
    bursts["system_level.num_spikes_in_bursts"] = num_spks.tolist()

    if return_res:
        return bursts


def find_onset_durations(h5f, write_to_h5f=True, return_res=False):
    """
        Similar to the duration of a burst (start time to end time),
        we can ask how long did it take from activating the first
        to activating the last module
    """

    assert "ana.bursts.system_level" in h5f.keypaths()

    spikes = h5f["data.spiketimes"]
    beg_times = h5f["ana.bursts.system_level.beg_times"]
    end_times = h5f["ana.bursts.system_level.end_times"]

    onset_durations = []
    for idx in range(0, len(beg_times)):
        beg = beg_times[idx]
        end = end_times[idx]
        nids, tidx = np.where((spikes >= beg) & (spikes <= end))
        onsets = []
        for nid in np.unique(nids):
            ndx = np.where(nids == nid)[0]
            times = spikes[nid, tidx[ndx]]
            onsets.append(np.nanmin(times))
        if len(onsets) == 0:
            # in rare situations when the threshold is _barely_ crossed,
            # we might detect burst boundaries but all spikes
            # are out of the detected interval, due to gaussian smoothing
            onset_durations.append(np.nan)
        else:
            onset_durations.append(np.nanmax(onsets) - np.nanmin(onsets))

    if write_to_h5f:
        h5f["ana.bursts.system_level.onset_durations"] = onset_durations

    if return_res:
        return onset_durations


def find_rij(h5f=None, which="neurons", time_bin_size=200 / 1000):
    """
    # Paramters
    which : str, "neurons", "modules", "depletion",
        if "modules", mod rates have to be in h5f
        if "depletion" time_bin_size is ignored and native time resolution is used
    time_bin_size : float, if "neurons" selected this is the bin size for
        `binned_spike_count` in seconds

    # Returns
    rij : 2d array, correlation coefficients, with nans along the diagonal

    # Note
    rij may contain np.nan if a neuron did not have any spikes.
    """
    assert which in ["neurons", "modules", "depletion"]

    if which == "neurons":
        if time_bin_size is None:
            time_bin_size = 40 / 1000  # default 40 ms

        # if sensor is specified, we might get more neuron_module_id than were recorded
        # selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
        # selects = selects[np.isin(selects, h5f["ana.neuron_ids"])]
        selects = h5f["ana.neuron_ids"]
        spikes = h5f["data.spiketimes"][selects, :]

        series = binned_spike_count(spikes, time_bin_size)

    elif which == "modules":
        assert "ana.rates.module_level" in h5f.keypaths(), "`find_rates` first"
        num_steps = len(h5f["ana.rates.module_level.mod_0"])
        series = np.zeros(shape=(4, num_steps))
        for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
            m_dc = h5f["ana.mods"][mdx]
            series[mdx, :] = h5f[f"ana.rates.module_level.{m_dc}"][:]

    elif which == "depletion":
        series = h5f["data.state_vars_D"]

    rij = np.corrcoef(series)
    return rij


def find_rij_pairs(h5f, rij=None, pairing="within_modules", **kwargs):
    """
        get a flat list of the rij values for pairs of neurons matching a criterium,
        e.g. all neuron pairs within the same module.

        # Parameters
        rij : 2d array, if already calculated, skip rij computation
        pairing : str,
            "within_modules" : only neuron pairs within the same module
            "across_modules" : pairs spanning modules
            "within_group_02" : instead of modules, use all pairs in this group of modules
            "across_groups_02_13" : compare across the two sepcified groups
        kwargs : passed to `find_rij()`
    """

    if rij is None:
        rij = find_rij(h5f, which="neurons", time_bin_size=40 / 1000)

    if "group" in pairing:

        mods_a = [int(mod) for mod in pairing.split("_")[-1]]
        if "across" in pairing:
            mods_b = [int(mod) for mod in pairing.split("_")[-2]]

    res = []
    n = len(h5f["data.neuron_module_id"][:])
    for i in range(0, n):
        for j in range(0, n):
            if j >= i:
                # skip upper diagonal
                continue

            if pairing == "all":
                res.append(rij[i, j])
            elif pairing == "within_modules":
                if h5f["data.neuron_module_id"][i] == h5f["data.neuron_module_id"][j]:
                    res.append(rij[i, j])

            elif pairing == "across_modules":
                if h5f["data.neuron_module_id"][i] != h5f["data.neuron_module_id"][j]:
                    res.append(rij[i, j])

            elif "within_group" in pairing:
                if (
                    h5f["data.neuron_module_id"][i] in mods_a
                    and h5f["data.neuron_module_id"][j] in mods_a
                ):
                    res.append(rij[i, j])

            elif "across_groups" in pairing:
                if (
                    h5f["data.neuron_module_id"][i] in mods_a
                    and h5f["data.neuron_module_id"][j] in mods_b
                ) or (
                    h5f["data.neuron_module_id"][i] in mods_b
                    and h5f["data.neuron_module_id"][j] in mods_a
                ):
                    res.append(rij[i, j])

    return res


# todo: implement convention write_to_h5f
def find_functional_complexity(
    h5f,
    rij=None,
    write_to_h5f=True,
    return_res=False,
    num_bins=20,
    bins=None,
    **kwargs,
):
    """
    Find functional complexity, either from module rates or neurons.

    # Paramters
    rij : if you already comuted the rij matrix, you can provide it.
    num_bins : int, number of bins for the correlation coefficients (m)
    bins :     use these bins to pass to np.histogram and ignore `num_bins`
    kwargs passed to `find_rij`


    # Returns
    C : float, functional complexity
    rij : 2d array, correlation coefficients, with nans along the diagonal
    """

    if rij is None:
        rij = find_rij(h5f, **kwargs)

    if return_res:
        return _functional_complexity(rij, num_bins, bins), rij


def find_state_variable(h5f, variable, write_to_h5f=True, return_res=False):
    """
        from the neuron-level state variable,
        calculate module-level state variables and some properties
    """

    assert f"data.state_vars_{variable}" in h5f.keypaths()
    assert f"data.state_vars_time" in h5f.keypaths()

    states = benedict()
    states["time"] = h5f["data.state_vars_time"][:]
    stat_vals = h5f[f"data.state_vars_{variable}"]

    # merge down neurons to modules
    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]

        selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
        selects = selects[np.isin(selects, h5f["ana.neuron_ids"])]

        states[f"module_level.{m_dc}"] = np.nanmean(stat_vals[selects, :], axis=0)

    # system-wide
    states["system_level"] = np.nanmean(stat_vals, axis=0)

    if write_to_h5f:
        try:
            h5f["ana.states"].clear()
        except Exception as e:
            log.debug(e)
        # so, overwriting keys with dicts (nesting) can cause memory leaks.
        # to avoid this, call .clear() before assigning the new dict
        # testwise I made this the default for setting keys of benedict
        h5f["ana.states"] = states

    if return_res:
        return states


def find_rij_within_across(h5f):
    """
        We already have a nice way to get the rij in `find_functional_complexity`.
        Lets use that and do the sorting according to modules afterwards.
    """

    # this gives us rij as a matrix, 2d np array
    _, rij = find_functional_complexity(h5f, which="neurons", return_res=True)

    # 40 ^ 2 * 4 / 2 + 40 * 120 * 4 / 2

    n = len(h5f["data.neuron_module_id"][:])
    # n_mods = len(h5f["ana.mod_ids"])
    # n_per_mod = int(n / n_mods)
    # n_within = int(n_per_mod * n_per_mod * n_mods / 2)
    # n_across = int(n_per_mod * (n - n_per_mod) * n_mods / 2)

    rij_within = []
    rij_across = []

    adx = 0
    wdx = 0
    for i in range(0, n):
        for j in range(0, n):
            if j >= i:
                # skip upper diagonal
                continue

            if h5f["data.neuron_module_id"][i] == h5f["data.neuron_module_id"][j]:
                rij_within.append(rij[i, j])
            else:
                rij_across.append(rij[i, j])

    return rij, rij_within, rij_across


# ------------------------------------------------------------------------------ #
# batch processing across realization
# ------------------------------------------------------------------------------ #


def batch_across_filenames(
    filenames, ana_function, merge="append", res=None, parallel=True
):
    """
        function to generalize some analysis parts across multiple files.

        Takes a list of `filenames` and applies `ana_function` on every one of them.

        # Parameters
        filenames : str or list of strings, wildcards will be globbed
        ana_function : function that needs to return
            * the result
            * for consistency, a check value that should be the same for every filename
            * the filename
        merge : default "append"  or function.
            if not "append", merge(res, this_res) is called to combine results across
            files, otherwise we append to a list
        res : per default (None) we create a list and append to it. when `merge=np.add`
            we want to provide res to be an empty zero array, e.g. `np.array([0])`

        # Returns
        res : list or dtype that was provided to `res`
        check : the value of the consistency check that was provided by the first file


    """

    if isinstance(filenames, str):
        filenames = [filenames]

    candidates = [glob.glob(f) for f in filenames]
    candidates = [item for sublist in candidates for item in sublist]  # flatten

    assert len(candidates) > 0, f"Are the filenames correct?"

    if res is None:
        # default, just append to a list
        res = list()

    check = None

    if not parallel or len(candidates) == 1:
        for idx in range(0, len(candidates)):
            this_candidate = candidates[idx]
            this_res, this_check, _ = ana_function(this_candidate)

            if idx == 0:
                check = this_check
            if this_res is None or not np.all(check == this_check):
                log.warning(f"file seems invalid, skipping: {this_candidate}")
            else:
                if merge == "append":
                    res.append(this_res)
                else:
                    merge(res, this_res)

    else:
        import dask_helper as dh

        dh.init_dask()

        # import functools
        # set arguments for variables needed by every worker
        # f = functools.partial(ana_function, candidates)

        # dispatch, reading in parallel may be faster
        futures = dh.client.map(ana_function, candidates)

        # next, we gather
        for idx, future in enumerate(
            tqdm(dh.as_completed(futures), total=len(futures), desc="Files")
        ):
            this_res, this_check, this_candidate = future.result()

            if idx == 0:
                check = this_check
            if this_res is None or not np.all(check == this_check):
                log.warning(f"file seems invalid, skipping: {this_candidate}")
            else:
                if merge == "append":
                    res.append(this_res)
                else:
                    merge(res, this_res)

        dh.close()

    return res, check


def batch_pd_sequence_length_probabilities(list_of_filenames):
    """
        Create a pandas data frame (long form, every row corresponds to one sequence
        length that occured - usually, 4 rows per realization).
        Remaining columns include meta data, conditions etc.
        This can be directly used for box plotting in seaborn.

        # Parameters:
        list_of_filenames: list of str,
            each str will be globbed and process
    """

    if isinstance(list_of_filenames, str):
        list_of_filenames = [list_of_filenames]

    columns = [
        "Sequence length",
        "Probability",
        "Total",  # Number of entries contributing to probability
        "Stimulation",
        "Connections",
        "Bridge weight",
    ]
    df = pd.DataFrame(columns=columns)

    candidates = [glob.glob(f) for f in list_of_filenames]
    candidates = [item for sublist in candidates for item in sublist]  # flatten

    assert len(candidates) > 0, f"Are the filenames correct?"

    for candidate in tqdm(candidates, desc="Seq. length for files"):
        h5f = prepare_file(candidate, hot=True)
        # this adds the required entry for sequences
        find_bursts_from_rates(h5f)
        labels, probs, total = sequence_length_histogram_from_list(
            list_of_sequences=h5f["ana.bursts.system_level.module_sequences"],
            mods=h5f["ana.mods"],
        )

        # fetch meta data for every row
        stim = h5f["ana.stimulation_description"]
        bridge_weight = h5f["meta.dynamics_bridge_weight"]
        if bridge_weight is None:
            bridge_weight = 1.0
        num_connections = h5f["meta.topology_k_inter"]

        for ldx, l in enumerate(labels):
            df = df.append(
                pd.DataFrame(
                    data=[
                        [
                            labels[ldx],
                            probs[ldx],
                            total,
                            stim,
                            num_connections,
                            bridge_weight,
                        ]
                    ],
                    columns=columns,
                ),
                ignore_index=True,
            )

    return df


def batch_pd_bursts(
    load_from_disk=False, list_of_filenames=None, df_path=None, client=None
):
    """
        Create a pandas data frame (long form, every row corresponds to one burst.
        Remaining columns include meta data, conditions etc.
        This can be directly used for box plotting in seaborn.

        # Parameters:
        list_of_filenames: list of str,
            each str will be globbed and process
    """
    if list_of_filenames is None:
        list_of_filenames = [
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/inhibition/dyn/*rep=*.hdf5",
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/bridge_weights/dyn/*rep=*.hdf5",
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/dyn/2x2_fixed/*.hdf5",
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jitter_0/*.hdf5",
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jitter_02/*.hdf5",
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jitter_012/*.hdf5",
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/jitter_0123/*.hdf5",
        ]
    elif isinstance(list_of_filenames, str):
        list_of_filenames = [list_of_filenames]

    if df_path is None and not load_from_disk:
        df_path = f"{tempfile.gettempdir()}/ana_helper/bursts_dataframe.hdf5"
        log.warning(f"No `df_path` set, writing to {df_path}")

    if load_from_disk:
        try:
            df = pd.read_hdf(df_path, "/data/df")
            return df
        except Exception as e:
            log.info("Could not load from disk, (re-)processing data")
            log.debug(e)

    candidates = [glob.glob(f) for f in list_of_filenames]
    candidates = [item for sublist in candidates for item in sublist]  # flatten

    assert len(candidates) > 0, f"Are the filenames correct?"

    if client is None:
        res = []
        for candidate in tqdm(candidates, desc="Burst duration for files"):
            res.append(_dfs_from_realization(candidate))
    else:
        # use dask to do this in parallel
        from dask.distributed import as_completed

        futures = client.map(_dfs_from_realization, candidates)
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Burst duration for files (using dask)",
        ):
            pass
        res = client.gather(futures)

    res = pd.concat(res, ignore_index=True)

    try:
        os.makedirs(df_path, exist_ok=True)
        res.to_hdf(df_path, "/data/df")
    except Exception as e:
        log.debug(e)

    return res


def _dfs_from_realization(candidate):
    columns = [
        "Duration",
        "Sequence length",
        "Fraction of neurons",
        "First module",
        "Stimulation",
        "Connections",
        "Bridge weight",
        "Number of inhibitory neurons",
        "Repetition",
    ]
    df = pd.DataFrame(columns=columns)
    try:
        h5f = prepare_file(candidate, hot=False)
    except Exception as e:
        log.error(e)
        log.error(f"Skipping candidate {candidate}")
        # continue
        return df

    # fetch meta data for every repetition (applied to multiple rows)
    stim = h5f["ana.stimulation_description"]
    rep = h5f["ana.repetition"]
    bridge_weight = h5f["meta.dynamics_bridge_weight"]
    if bridge_weight is None:
        bridge_weight = 1.0
    num_connections = h5f["meta.topology_k_inter"]
    try:
        if "data.neuron_inhibitory_ids" in h5f.keypaths():
            num_inhibitory = len(h5f["data.neuron_inhibitory_ids"][:])
    except Exception as e:
        log.debug(e)
        # maybe its a single number, instead of a list
        if isinstance(h5f["data.neuron_inhibitory_ids"], numbers.Number):
            num_inhibitory = 1
        else:
            num_inhibitory = 0

    # do the analysis, entries are directly added to the h5f
    # for the system with inhibition, we might need a lower threshold (Hz)
    if num_inhibitory > 0:
        find_bursts_from_rates(h5f, rate_threshold=7.5)
    else:
        find_bursts_from_rates(h5f, rate_threshold=7.5)

    find_participating_fraction_in_bursts(h5f)

    data = h5f["ana.bursts.system_level"]
    bt = h5f["ana.bursts.system_level.beg_times"]
    et = h5f["ana.bursts.system_level.end_times"]
    ms = h5f["ana.bursts.system_level.module_sequences"]
    pf = h5f["ana.bursts.system_level.participating_fraction"]
    for idx in range(0, len(bt)):
        duration = et[idx] - bt[idx]
        seq_len = len(ms[idx])
        first_mod = ms[idx][0]
        fraction = pf[idx]
        df = df.append(
            pd.DataFrame(
                data=[
                    [
                        duration,
                        seq_len,
                        pf,
                        first_mod,
                        stim,
                        num_connections,
                        bridge_weight,
                        num_inhibitory,
                        rep,
                    ]
                ],
                columns=columns,
            ),
            ignore_index=True,
        )

    return df


def batch_candidates_burst_times_and_isi(input_path, hot=False):
    """
        get the burst times based on rate for every module and merge it down, so that
        we have ensemble average statistics
    """

    candidates = glob.glob(input_path)

    assert len(candidates) > 0, "Is the input_path correct?"

    res = None
    mods = None

    for cdx, candidate in enumerate(
        tqdm(candidates, desc="Bursts and ISIs for files", leave=False)
    ):
        h5f = h5.recursive_load(candidate, hot=hot, dtype=benedict)
        prepare_file(h5f)
        find_bursts_from_rates(h5f)
        find_isis(h5f)
        find_ibis(h5f)

        this_burst = h5f["ana.bursts"]
        this_isi = h5f["ana.isi"]
        this_ibi = h5f["ana.ibi"]

        if cdx == 0:
            res = h5f
            mods = h5f["ana.mods"]
            continue

        # todo: consistency checks
        # lets at least check that the modules are consistent across candidates.
        assert np.all(h5f["ana.mods"] == mods), "Modules differ between files"

        # copy over system level burst
        b = res["ana.bursts.system_level"]
        b["beg_times"].extend(this_burst["system_level.beg_times"])
        b["end_times"].extend(this_burst["system_level.end_times"])
        b["module_sequences"].extend(this_burst["system_level.module_sequences"])

        # copy over system-level ibi
        res["ana.ibi.system_level"].extend(this_ibi["system_level"])

        for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
            m_dc = h5f["ana.mods"][mdx]
            # copy over module level bursts
            b = res[f"ana.bursts.module_level.{m_dc}"]
            b["beg_times"].extend(this_burst[f"module_level.{m_dc}.beg_times"])
            b["end_times"].extend(this_burst[f"module_level.{m_dc}.end_times"])

            # and isis
            i = res[f"ana.isi.{m_dc}"]
            for var in ["all", "in_bursts", "out_bursts"]:
                i[var].extend(this_isi[m_dc][var])

            # and ibis
            res[f"ana.ibi.module_level.{m_dc}"].extend(this_ibi[f"module_level.{m_dc}"])

        if hot:
            # only close the last file (which we opened), and let's hope no other file
            # was opened in the meantime
            # h5.close_hot(which=-1)
            try:
                h5.close_hot(h5f["h5.filename"])
            except:
                log.debug("Failed to close file")

    return res


def batch_isi_across_conditions():

    stat = benedict()
    conds = _conditions()
    for k in tqdm(conds.varnames, desc="k values", position=0, leave=False):
        stat[k] = benedict()
        for stim in tqdm(
            conds[k].varnames, desc="stimulation targets", position=1, leave=False
        ):
            h5f = process_candidates_burst_times_and_isi(conds[k][stim])
            # preprocess so that plot functions wont do it again.
            # todo: make api consistent
            h5f["ana.ensemble"] = benedict()
            h5f["ana.ensemble.filenames"] = conds[k][stim]
            h5f["ana.ensemble.bursts"] = h5f["ana.bursts"]
            h5f["ana.ensemble.isi"] = h5f["ana.isi"]

            logging.getLogger("plot_helper").setLevel("WARNING")
            fig = ph.overview_burst_duration_and_isi(h5f, filenames=conds[k][stim])
            logging.getLogger("plot_helper").setLevel("INFO")

            fig.savefig(
                f"/Users/paul/mpi/simulation/brian_modular_cultures/_figures/isis/{k}_{stim}.pdf",
                dpi=300,
            )
            with open(
                f"/Users/paul/mpi/simulation/brian_modular_cultures/_figures/isis/pkl/{k}_{stim}.pkl",
                "wb",
            ) as fid:
                pickle.dump(fig, fid)
            plt.close(fig)
            del fig

            # lets print some statistics
            stat[k][stim] = benedict()
            for m in h5f["ana.ensemble.isi.varnames"]:
                try:
                    stat[k][stim][m] = np.mean(h5f[f"ana.ensemble.isi.{m}.in_bursts"])
                except Exception as e:
                    log.debug(e)

            del h5f
            h5.close_hot()

    return stat


def batch_conditions():
    # fmt:off
    path_base = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/"
    stim = benedict()
    for k in [0,1,2,3,5]:
        stim[k] = benedict()
        stim[k].off = f"{path_base}/dyn/2x2_fixed/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k}_rep=*.hdf5"
        for s in ["0", "02", "012", "0123"]:
            stim[k][s] = f"{path_base}/jitter_{s}/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k}_rep=*.hdf5"

    return stim
    # fmt:on


# ------------------------------------------------------------------------------ #
# lower level
# ------------------------------------------------------------------------------ #


def _spikes_as_sparse_to_spikes_as_list(spikes_as_sparse, dt):
    """
        # Parameters
        spikes_as_sparse : 2d array with shape (num_timesteps, num_neurons)
            columns correspond to neurons, rows to time bins of size dt.
            each row has a 0 (no spike in that time bin) or 1 (if spiked).

        # Returns
        spikes_as_list : 2d array with shape: (num_spikes, 2)
            first column is the neuron id, second column the spike time
    """

    times, neuron_ids = np.where(spikes_as_sparse)

    return np.array([neuron_ids, times * dt]).T


def _spikes_as_list_to_spikes_2d(spikes_as_list, num_n=None):
    """
        convert a list of spiketimes to the 2d matrix representation
        if num_n is not None, we will use 0 to num_n neurons

        # Parameters
        spikes_as_list : 2d array with shape: (num_spikes, 2)
            first column is the neuron id, second column the spike time

        # Returns
        spikes_nan_padded : 2d array
            with shape (num_neurons, max_number_spikes_for_single_neuron)
    """

    unique, counts = np.unique(spikes_as_list[:, 0], return_counts=True)

    if num_n is None:
        num_n = int(np.max(unique)) + 1

    assert np.all(unique >= 0)

    max_num_spikes = np.max(counts)

    spikes_2d = np.ones(shape=(num_n, max_num_spikes)) * np.nan

    for nid in unique.astype(int):
        selected = spikes_as_list[spikes_as_list[:, 0] == nid][:, 1]
        spikes_2d[nid, 0 : len(selected)] = selected

    return spikes_2d


# turns out this is faster without numba
def _inter_spike_intervals(spikes_2d, beg_times=None, end_times=None):
    """
        Returns a dict with lists of initer spike intverals and the matching CVs.
        Has the folliwing keys:
            all,
            in_bursts,
            out_bursts,
            cv_all,
            cv_in_bursts,
            cv_out_bursts,
    """

    isis_all = []
    isis_in = []
    isis_out = []
    cvs_all = []
    cvs_in = []
    cvs_out = []

    if beg_times is None or end_times is None:
        num_bursts = 0
    else:
        num_bursts = len(beg_times)

    for n in range(spikes_2d.shape[0]):
        spikes = spikes_2d[n]
        spikes = spikes[~np.isnan(spikes)]
        diffs = np.diff(spikes)
        isis_all.extend(diffs)
        if len(diffs) > 0:
            cvs_all.append(np.std(diffs) / np.mean(diffs))

        # check on burst level
        for idx in range(0, num_bursts):
            b = beg_times[idx]
            e = end_times[idx]

            spikes = spikes_2d[n]
            spikes = spikes[spikes >= b]
            spikes = spikes[spikes <= e]
            diffs = np.diff(spikes)
            isis_in.extend(diffs)
            if len(diffs) > 0:
                cvs_in.append(np.std(diffs) / np.mean(diffs))

            e = beg_times[idx]
            if idx > 0:
                b = end_times[idx - 1]
            else:
                # before first burst
                b = 0
            spikes = spikes_2d[n]
            spikes = spikes[spikes >= b]
            spikes = spikes[spikes <= e]
            diffs = np.diff(spikes)
            isis_out.extend(spikes)
            if len(diffs) > 0:
                cvs_out.append(np.std(diffs) / np.mean(diffs))

        # after last burst
        e = np.inf
        if num_bursts > 0:
            b = end_times[-1]
        else:
            b = 0
        spikes = spikes_2d[n]
        spikes = spikes[spikes >= b]
        spikes = spikes[spikes <= e]
        diffs = np.diff(spikes)
        isis_out.extend(spikes)
        if len(diffs) > 0:
            cvs_out.append(np.std(diffs) / np.mean(diffs))

        cv_all = np.mean(cvs_all)
        cv_in_bursts = np.mean(cvs_in)
        cv_out_bursts = np.mean(cvs_out)

    return benedict(
        all=isis_all,
        in_bursts=isis_in,
        out_bursts=isis_out,
        cv_all=cv_all,
        cv_in_bursts=cv_in_bursts,
        cv_out_bursts=cv_out_bursts,
    )


def _functional_complexity(rij, num_bins=20, bins=None):
    """
    Uses np corrcoef on series to get correlation coefficients and calculate
    gorkas functional complexity measure (Zamora-Lopez et al 2016)

    # Paramters
    num_bins : number of bins for the histogram (m)
    rij : 2d array, correlation coefficeints, (or a 1d array, already flattened)

    # Returns
    C : float, functional complexity
    """

    try:
        # "only interested in pair-wise interactions, discard diagonal entries rii"
        np.fill_diagonal(rij, np.nan)
    except:
        pass

    flat = rij.flatten()
    flat = flat[~np.isnan(flat)]

    if bins is None:
        bw = 1.0 / num_bins
        bins = np.arange(0, 1 + 0.1 * bw, bw)

    prob, _ = np.histogram(flat, bins=bins)
    prob = prob / np.sum(prob)

    # Zamora-Lopez et al 2016
    def fc(prob, m):
        c = 1 - np.sum(np.fabs(prob - 1 / m)) * m / 2 / (m - 1)
        return c

    return fc(prob, num_bins)


@jit(nopython=True, parallel=True, fastmath=False, cache=True)
def binned_spike_count(spiketimes, bin_size, length=None):
    """
        Similar to `population_rate`, but we get a number of spike counts, per neuron
        as needed for e.g. cross-correlations.

        Parameters
        ----------
        spiketimes :
            np array with first dim neurons, second dim spiketimes. nan-padded
        bin_size :
            float, in units of spiketimes
        length :
            duration of output trains, in units of spiketimes. Default: None,
            uses last spiketime

        Returns
        -------
        counts : 2d array
            time series of the counted number of spikes per bin,
            one row for each neuron, in steps of bin_size
    """

    num_n = spiketimes.shape[0]

    if length is not None:
        num_bins = int(np.ceil(length / bin_size))
    else:
        t_min = 0.0
        t_max = np.nanmax(spiketimes)
        num_bins = int(np.ceil((t_max - t_min) / bin_size))

    counts = np.zeros(shape=(num_n, num_bins))

    for n_id in range(0, num_n):
        train = spiketimes[n_id]
        for t in train:
            if not np.isfinite(t):
                break
            t_idx = int(t / bin_size)
            counts[n_id, t_idx] += 1

    return counts


@jit(nopython=True, parallel=True, fastmath=False, cache=True)
def population_rate(spiketimes, bin_size, length=None):
    """
        Calculate the activity across the whole population. naive binning,
        no sliding window.
        normalization by bin_size is still required.

        Parameters
        ----------
        spiketimes :
            np array with first dim neurons, second dim spiketimes. nan-padded
        bin_size :
            float, in units of spiketimes
        length :
            duration of output, in units of spiketimes. Default: None,
            uses last spiketime

        Returns
        -------
        rate : 1d array
            time series of the rate in number of spikes per bin,
            normalized per-neuron, in steps of bin_size
    """

    num_n = spiketimes.shape[0]

    # target array
    if length is not None:
        num_bins = int(np.ceil(length / bin_size))
    else:
        t_min = 0.0
        t_max = np.nanmax(spiketimes)
        num_bins = int(np.ceil((t_max - t_min) / bin_size))

    rate = np.zeros(num_bins)

    for n_id in range(0, num_n):
        train = spiketimes[n_id]
        for t in train:
            if not np.isfinite(t):
                break
            t_idx = int(t / bin_size)
            rate[t_idx] += 1

    rate = rate / num_n

    return rate


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def population_rate_exact_smoothing(spiketimes, bin_size, smooth_width, length=None):
    """
        Applies a gaussian kernel to every spike time, and keeps full precision of the
        peak. Time binning is only used for the result. This should be a bit more
        precise than e.g. `smooth_rate(population_rate(...))`.

        Provide all arguments in seconds to get resulting `rate` in Hz

        # Parameters
        spiketimes :  2d nan-padded ndarray, neurons x spiketimes
        bin_size   : size of the time step, in same units as `spiketimes`
        smooth_width : standard deviation (width) of the gaussian kernel
        length : float or None, how long the resulting time series should be (same
                   unit as `spiketimes` and `bin_size`)

        #Returns
        rate : 1d array
            time series of the rate in 1/(unit of `spiketimes`), normalized per-neuron,
            in steps of `bin_size`.
            if `spiketimes` are provided in seconds, rates is in Hz. normalization by
            `bin_size` is NOT required.
    """

    sigma = smooth_width

    if length is not None:
        num_bins = int(np.ceil(length / bin_size))
    else:
        t_min = 0.0
        t_max = np.nanmax(spiketimes) + 4 * sigma
        num_bins = int(np.ceil((t_max - t_min) / bin_size))

    res = np.zeros(num_bins)

    # precompute factors
    norm = sigma * np.sqrt(2 * np.pi)

    for n in prange(spiketimes.shape[0]):
        for mu in spiketimes[n, :]:
            if np.isnan(mu):
                # nan-checks such as np.nan do not work with fastmath=False in numba jit.
                # np.nanmax seems to work, though
                # https://github.com/numba/numba/issues/2919
                break

            # only consider bins 4 sigma around the spike time
            # bin_beg = int((mu - 0 * sigma) / bin_size)
            bin_beg = int((mu - 4 * sigma) / bin_size)
            bin_end = int((mu + 4 * sigma) / bin_size)

            for b in range(bin_beg, bin_end + 1):
                if b >= num_bins:
                    break
                x = b * bin_size
                res[b] += np.exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma)) / norm

    return res / spiketimes.shape[0]


def burst_detection_pop_rate(
    rate, bin_size, rate_threshold, extend=False, return_series=False,
):
    """
        given a population `rate` with `bin_size` define a burst when exceeding a
        `rate_threshold`

        # Returns
        beg_time, end_time : list
    """

    # work on index level of the `above` array. equivalent to time in steps of bin_size
    above = np.where(rate >= rate_threshold)[0]
    sep = np.where(np.diff(above) > 1)[0]

    if len(above > 0):
        sep = np.insert(sep, 0, -1)  # add the very left edge, will become 0
        sep = np.insert(sep, len(sep), len(above) - 1)  # add the very right edge

    # sep+1 gives begin index
    beg = sep[:-1] + 1
    end = sep[1:]

    # back to time level
    beg_time = above[beg] * bin_size
    end_time = above[end] * bin_size

    if extend:
        beg_time -= bin_size / 2
        end_time += bin_size / 2

    # causes numba deprecation warning. in the future this will need to be a typed list
    # List(foo) will solve this but introduces a new type for everything. not great.
    beg_time = beg_time.tolist()
    end_time = end_time.tolist()

    # workaround
    # from numba.typed import List

    # beg_time = List(beg_time)
    # end_time = List(end_time)
    # if len(beg_time) == 0:
    #     beg_time.append(1.0)
    #     end_time.append(1.0)
    #     del beg_time[0]
    #     del end_time[0]

    if not return_series:
        return beg_time, end_time
    else:
        series = np.zeros(len(rate), dtype=np.int)
        series[above] = 1
        series = series.tolist()
        return beg_time, end_time, series


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def merge_if_below_separation_threshold(beg_time, end_time, threshold):

    """
        If the `beg_time` of a new burst is closer than `threshold`
        to the `end_time` of a previous burst, merge them.

        Parameters
        ----------
        beg_time, end_time : 1d np array
            sorted, same length!

        threshold : float
            in units used in beg_time and end_time

        Returns
        -------
        beg_time, end_time : list
    """
    beg_res = []
    end_res = []

    if len(beg_time) == 0:
        return beg_res, end_res

    # skip a new burst beginning if within threshold of the previous ending
    skip = False
    beg = 0
    end = 0

    for idx in range(0, len(beg_time) - 1):

        if end_time[idx] >= end:
            end = end_time[idx]

        if not skip:
            beg = beg_time[idx]

        if beg_time[idx + 1] - threshold <= end:
            skip = True
            continue
        else:
            skip = False
            beg_res.append(beg)
            end_res.append(end)

    # last burst is either okay (skip=False) or we use the currently open one (skip=True)
    if skip:
        beg_res.append(beg)
    else:
        beg_res.append(beg_time[-1])
    end_res.append(end_time[-1])

    return beg_res, end_res


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def arg_merge_if_below_separation_threshold(beg_time, end_time, threshold):
    # same as above just that here we return the indices that would provide
    # the merged arrays. (like e.g. np.argsort)

    beg_res = []
    end_res = []

    if len(beg_time) == 0:
        return beg_res, end_res

    # skip a new burst beginning if within threshold of the previous ending
    skip = False
    beg = 0
    end = 0

    for idx in range(0, len(beg_time) - 1):

        if end_time[idx] >= end_time[end]:
            end = idx

        if not skip:
            beg = idx

        if beg_time[idx + 1] - threshold <= end_time[end]:
            skip = True
            continue
        else:
            skip = False
            beg_res.append(beg)
            end_res.append(end)

    # last burst is either okay (skip=False) or we use the currently open one (skip=True)
    if skip:
        beg_res.append(beg)
    else:
        beg_res.append(len(beg_time) - 1)
    end_res.append(len(end_time) - 1)

    # return np.array(beg_res, dtype=np.int32), np.array(end_res, dtype=np.int32)
    return beg_res, end_res


def system_burst_from_module_burst(beg_times, end_times, threshold, modules=None):
    """
        From the burst times (begin, end) of the modules, calculate
        the system-wide bursts.


        Parameters
        ----------
        beg_times, end_times: list of 1d np array
            for every module, the lists should have one 1d np array

        threshold : float
            bursts that are less apart than this will be merged (across modules)

        modules : list
            the module label that corresponds to the first dim of beg_times

        Returns
        -------
        all_begs, all_ends :  1d np arrays
            containing the start and end times of system-wide bursts.

        sequences : list of tuples
            that correspond to the sequence of module activation for a
            particular burst
    """

    if modules is None:
        modules = np.arange(len(beg_times))

    # construct a list of module ids to match beg_times
    mods = []
    for mdx, _ in enumerate(beg_times):
        m = modules[mdx]
        mods.append(np.array([m] * len(beg_times[mdx])))

    # stack everything together in one long flat list
    all_begs = np.hstack(beg_times)
    all_ends = np.hstack(end_times)
    all_mods = np.hstack(mods)

    # if no burst at all, skip the rest
    if len(all_begs) == 0:
        return [], [], []

    # sort consistently by begin time
    idx = np.argsort(all_begs)
    all_begs = all_begs[idx]
    all_ends = all_ends[idx]
    all_mods = all_mods[idx]

    # get indices that will yield merged system wide bursts
    idx_begs, idx_ends = arg_merge_if_below_separation_threshold(
        all_begs, all_ends, threshold
    )

    sequences = []

    # do sequence sorting next, for every burst
    for pos, idx in enumerate(idx_begs):
        # first module, save sequence as tuple
        seq = (all_mods[idx],)

        if pos < len(idx_begs) - 1:
            # if more bursts detected, we need to finish before it starts
            this_end_time = all_begs[idx_begs[pos + 1]]
        else:
            this_end_time = np.inf

        # first time occurences of follower
        jdx = idx + 1
        while jdx < len(all_ends) and all_ends[jdx] < this_end_time:
            # get module id of bursts that were in the system burst, add to sequence
            m = all_mods[jdx]
            if m not in seq:
                seq += (m,)
            if len(seq) == len(modules):
                break
            jdx += 1

        # add this particular sequence
        sequences.append(seq)

    return all_begs[idx_begs].tolist(), all_ends[idx_ends].tolist(), sequences


def smooth_rate(rate, clock_dt, window="gaussian", width=None):
    """
        Return a smooth version of the population rate.
        Taken from brian2 population rate monitor
        https://brian2.readthedocs.io/en/2.0rc/_modules/brian2/monitors/ratemonitor.html#PopulationRateMonitor

        Parameters
        ----------
        rate : ndarray
            in hz
        clock_dt : time in seconds that entries in rate are apart (sampling frequency)
        window : str, ndarray
            The window to use for smoothing. Can be a string to chose a
            predefined window(``'flat'`` for a rectangular, and ``'gaussian'``
            for a Gaussian-shaped window). In this case the width of the window
            is determined by the ``width`` argument. Note that for the Gaussian
            window, the ``width`` parameter specifies the standard deviation of
            the Gaussian, the width of the actual window is ``4*width + dt``
            (rounded to the nearest dt). For the flat window, the width is
            rounded to the nearest odd multiple of dt to avoid shifting the rate
            in time.
            Alternatively, an arbitrary window can be given as a numpy array
            (with an odd number of elements). In this case, the width in units
            of time depends on the ``dt`` of the simulation, and no ``width``
            argument can be specified. The given window will be automatically
            normalized to a sum of 1.
        width : `Quantity`, optional
            The width of the ``window`` in seconds (for a predefined window).

        Returns
        -------
        rate : ndarrayy
            The population rate in Hz, smoothed with the given window. Note that
            the rates are smoothed and not re-binned, i.e. the length of the
            returned array is the same as the length of the ``rate`` attribute
            and can be plotted against the `PopulationRateMonitor` 's ``t``
            attribute.
    """
    if width is None and isinstance(window, str):
        raise TypeError("Need a width when using a predefined window.")
    if width is not None and not isinstance(window, str):
        raise TypeError("Can only specify a width for a predefined window")

    if isinstance(window, str):
        if window == "gaussian":
            width_dt = int(np.round(2 * width / clock_dt))
            # Rounding only for the size of the window, not for the standard
            # deviation of the Gaussian
            window = np.exp(
                -np.arange(-width_dt, width_dt + 1) ** 2
                * 1.0
                / (2 * (width / clock_dt) ** 2)
            )
        elif window == "flat":
            width_dt = int(width / 2 / clock_dt) * 2 + 1
            used_width = width_dt * clock_dt
            if abs(used_width - width) > 1e-6 * clock_dt:
                log.info(
                    "width adjusted from %s to %s" % (width, used_width),
                    "adjusted_width",
                    once=True,
                )
            window = np.ones(width_dt)
        else:
            raise NotImplementedError('Unknown pre-defined window "%s"' % window)
    else:
        try:
            window = np.asarray(window)
        except TypeError:
            raise TypeError("Cannot use a window of type %s" % type(window))
        if window.ndim != 1:
            raise TypeError("The provided window has to be " "one-dimensional.")
        if len(window) % 2 != 1:
            raise TypeError("The window has to have an odd number of " "values.")
    return np.convolve(rate, window * 1.0 / sum(window), mode="same")


def get_threshold_from_snr_between_bursts(
    rate, rate_dt, merge_threshold, std_offset=3, itermax=1000
):
    """
        Find an ideal threshold `theta = mean + std_offset*std`,
        where mean and std are evaluated inbetween bursts.

        Iterates burst detection at a given threshold with lowering the threshold.
    """

    # init
    mean_orig = np.nanmean(rate)
    std_orig = np.nanstd(rate)
    theta_orig = mean_orig + std_offset * std_orig

    mean_old = mean_orig
    std_old = std_orig
    theta_old = theta_orig

    for iteration in range(0, 1000):
        beg_times, end_times = burst_detection_pop_rate(
            rate=rate, bin_size=rate_dt, rate_threshold=theta_old,
        )
        beg_times, end_times = merge_if_below_separation_threshold(
            beg_times, end_times, threshold=merge_threshold
        )
        beg_times = np.array(beg_times) / rate_dt
        end_times = np.array(end_times) / rate_dt

        mask = np.ones(len(rate), dtype=bool)
        for idx in range(0, len(beg_times)):
            beg = int(beg_times[idx])
            end = int(end_times[idx])
            mask[beg:end] = False

        mean_new = np.nanmean(rate[mask])
        std_new = np.nanstd(rate[mask])
        theta_new = mean_new + std_offset * std_new
        print(theta_new, mean_new, std_new, np.sum(mask) / len(rate))
        # print(f"{theta_new:.2g} Hz, {}")

        theta_old = theta_new


def get_threshold_via_signal_to_noise_ratio(time_series, snr=5, iterations=1):

    mean_orig = np.nanmean(time_series)
    mean = mean_orig
    std = np.nanstd(time_series)

    idx = None
    for i in range(0, iterations):
        idx = np.where(time_series <= mean + snr * std)[0]
        mean = np.nanmean(time_series[idx])
        std = np.nanstd(time_series[idx])

    return mean + snr * std


def get_threshold_from_logisi_distribution(list_of_isi, area_fraction=0.3):
    bins = np.linspace(-3, 3, num=200)
    hist, edges = np.histogram(np.log10(list_of_isi), bins=bins)
    hist = hist / np.sum(hist)

    log.info(np.sum(hist))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(edges[0:-1], hist)
    area = 0
    for idx in range(0, len(edges)):
        area += hist[idx]
        edge = edges[idx]
        if area > area_fraction:
            ax.axvline(edge, 0, 1, color="gray")
            return 1 / pow(10, edge)


# ------------------------------------------------------------------------------ #
# pandas data frame operations
# ------------------------------------------------------------------------------ #

def pd_bootstrap(
    df, obs, sample_size=None, num_boot=500, func=np.nanmean,
    percentiles=None
):
    """
        bootstrap across all rows of a dataframe to get the mean across
        many samples and standard error of this estimate.
        query the dataframe first to filter for the right conditions.

        # Parameters:
        obs : str, the column to estimate for
        sample_size: int or None, default (None) for samples that are as large
            as the original dataframe (number of rows)
        num_boot : int, how many bootstrap samples to generate
        func : function, default np.nanmean is used to calculate the estimate
            for each sample
        percentiles : list of floats
            the percentiles to return. default is [2.5, 50, 97.5]

        # Returns:
        mean : mean across all drawn bootstrap samples
        std : std

    """

    if sample_size is None:
        sample_size = np.fmin(len(df), 10_000)

    if percentiles is None:
        percentiles = [2.5, 50, 97.5]


    # drop nans, i.e. for ibis we have one nan-row at the end of every burst
    df = df.query(f"`{obs}` == `{obs}`")

    resampled_estimates = []
    for idx in range(0, num_boot):
        sample_df = df.sample(
            n=sample_size, replace=True, ignore_index=True
        )

        resampled_estimates.append(
            func(sample_df[obs])
        )

    mean = np.mean(resampled_estimates)
    std = np.std(resampled_estimates, ddof=1)
    q = np.percentile(resampled_estimates, percentiles)

    return mean, std, q

def pd_nested_bootstrap(df, grouping_col, obs, num_boot=500, func=np.nanmean, resample_group_col = False,
    percentiles=None):

    """
        bootstrap across rows of a dataframe to get the mean across
        many samples and standard error of this estimate.

        uses `grouping_col` to filter the dataframe into subframes and permute
        in those groups, see below.

        # Parameters:
        obs : str, the column to estimate for
        grouping_col : str,
            bootstrap samples are generated independantly for every unique entry
            of this column.
        resample_group_col : bool, default False.
            Per default, we draw for each "experiment" in `grouping_col` as many
            rows as in the original frame, and create one large list of rows from
            all those experiments. on this list, the bs estimator is calculated.
            When True, we also draw with replacement the experiments. This should
            yield the most conservative error estimate.
        num_boot : int, how many bootstrap samples to generate
        func : function, default np.nanmean is used to calculate the estimate
            for each sample

        # Returns:
        mean : mean across all drawn bootstrap samples
        std : std
    """

    if percentiles is None:
        percentiles = [2.5, 50, 97.5]

    candidates = df[grouping_col].unique()

    resampled_estimates = []

    sub_dfs = dict()
    for candidate in candidates:
        sub_df = df.query(f"`{grouping_col}` == '{candidate}'")
        # this is a hacky way to remove rows where the observable is nan,
        # such as could be for inter-burst-intervals at the end of the experiment
        sub_df = sub_df.query(f"`{obs}` == `{obs}`")
        sub_dfs[candidate] = sub_df

    for idx in tqdm(range(0, num_boot), desc="Bootstrapping dataframe", leave=False):
        merged_obs = []

        if resample_group_col:
            candidates_resampled = np.random.choice(candidates, size=len(candidates), replace=True)
        else:
            candidates_resampled = candidates

        for candidate in candidates_resampled:
            sub_df = sub_dfs[candidate]
            sample_size = np.fmin(len(sub_df), 10_000)

            log.debug(f"{candidate}: {sample_size} entries for {obs}")

            # make sure to use different seeds
            sample_df = sub_df.sample(
                n=sample_size, replace=True, ignore_index=True
            )
            merged_obs.extend(sample_df[obs])

        estimate=func(merged_obs)
        resampled_estimates.append(estimate)

    log.debug(resampled_estimates)

    mean = np.mean(resampled_estimates)
    sem = np.std(resampled_estimates, ddof=1)
    q = np.percentile(resampled_estimates, percentiles)

    return mean, sem, q



# ------------------------------------------------------------------------------ #
# sequences
# ------------------------------------------------------------------------------ #


def remove_bursts_with_sequence_length_null(h5f):
    """
        modifies h5f!
        redo ibi detection, afterwards!
    """

    assert "ana.bursts.system_level" in h5f.keypaths()

    slens = np.array([len(s) for s in h5f["ana.bursts.system_level.module_sequences"]])
    num_old = len(h5f["ana.bursts.system_level.module_sequences"])
    idx_todel = np.where(slens == 0)[0]
    for idx in sorted(idx_todel, reverse=True):
        del h5f["ana.bursts.system_level.module_sequences"][idx]
        del h5f["ana.bursts.system_level.beg_times"][idx]
        del h5f["ana.bursts.system_level.end_times"][idx]

    num_new = len(h5f["ana.bursts.system_level.module_sequences"])
    log.debug(
        f"deleted {num_old - num_new} out of {num_old} bursts, due to sequence length 0"
    )


def sequences_from_module_contribution(
    h5f, sys_begs, sys_ends, min_spikes=1, min_neurons=1
):
    """
        Another way to detect whether a module was involved in a burst is to check
        how many spikes were contributed per neuron.

        # Parameters
        sys_begs, sys_ends: list or array
            system-wide begin/end times
        min_spikes: number
            at least this many spikes must occur in a module during the system-wide
            burst time to qualify as "contributing"
        min_neurons: number
            at least this many neurons from a module must fire at least `min_spikes`
            during the system-wide burst time to qualify as "contributing"

        # Returns
        sequences : list of tuples
            Note that we might get empty tuples, if no module met our requirements
    """
    spikes = h5f["data.spiketimes"]

    selects = [[]] * len(h5f["ana.mod_ids"])
    for m_id in h5f["ana.mod_ids"]:
        s = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]
        selects[m_id] = s[np.isin(s, h5f["ana.neuron_ids"])]

    sys_seqs = []
    for idx in range(0, len(sys_begs)):
        beg = sys_begs[idx]
        end = sys_ends[idx]
        firsts = np.ones(len(h5f["ana.mods"])) * np.nan

        # faster? simpler?
        for m_id in h5f["ana.mod_ids"]:
            nids = []
            first_times = []
            for nid in selects[m_id]:
                s = spikes[nid, :]
                s = s[(s >= beg) & (s <= end)]
                if len(s) >= min_spikes:
                    nids.append(nid)
                    first_times.append(s[0])
            if len(nids) >= min_neurons:
                firsts[m_id] = np.nanmin(first_times)
        num_valid = len(firsts[np.isfinite(firsts)])
        mdx_order = np.argsort(firsts)[0:num_valid]
        seq = tuple(np.array(h5f["ana.mod_ids"])[mdx_order])
        sys_seqs.append(seq)

    return sys_seqs

def sequence_histogram(ids, sequences=None):

    """
        create a histogram for every occuring sequences.
        If no `sequences` are provided, only gives the possible combinations (based on
        `ids`) that could have occured.
        sequnces should be a list of tuples (or arrays?)
    """

    labels = []
    for r in range(0, len(ids)):
        labels += list(permutations(ids, r + 1))

    if sequences is None:
        return labels

    histogram = np.zeros(len(labels), dtype=np.int64)

    for s in sequences:
        # get the right label, cast arrays to tuples for comparison
        idx = labels.index(tuple(s))
        histogram[idx] += 1

    return labels, histogram


def sequence_labels_to_strings(labels):
    # helper to convert the labels from list of tuples to list of strings

    res = []
    for l in labels:
        s = str(l)
        s = s.replace("(", "")
        s = s.replace(")", "")
        s = s.replace(",", "")
        s = s.replace(" ", "")
        res.append(s)

    return res


def sequence_conditional_probabilities(
    sequences, mod_ids=[0, 1, 2, 3], only_first=True
):
    """
        calculate the conditional probabilites, e.g.
        p(mod2 bursts | mod 1bursted)
        excludes jumps over other modules!

        # Parameters
        sequences : list of tuples of ints
        mod_ids : list of ints, from 0 to number modules
    """

    num_mods = mod_ids[-1] + 1
    prob_counts = np.zeros(shape=(num_mods, num_mods))
    self_counts = np.zeros(shape=(num_mods))

    for seq in sequences:
        for idx in range(len(seq)):
            if only_first and idx == 1:
                break
            i = seq[idx]
            self_counts[i] += 1
            # we will still need to deal with sequences that run in cricles
            if idx < len(seq) - 1:
                j = seq[idx + 1]
                prob_counts[i][j] += 1

    for idx, norm in enumerate(self_counts):
        prob_counts[idx, :] = prob_counts[idx, :] / norm

    if True:
        for idx in range(num_mods):
            for jdx in range(num_mods):
                p = prob_counts[idx][jdx]
                print(f"p({idx} -> {jdx}) = {p:.2f}")
            print(f"total:      {self_counts[idx]}\n")

    return prob_counts, self_counts


def sequence_length_histogram_from_list(list_of_sequences, mods=[0, 1, 2, 3]):
    """
        Provide a list of sequences as tuples and get a histogram of sequence lengths.

        # Returns
        catalog : nd array with the sequences
        probs : nd array with probabilities of each sequence (normalized counts)
        total : int, number of total histogram entries
    """
    seq_labs, seq_hist = sequence_histogram(mods, list_of_sequences)

    seq_str_labs = np.array(sequence_labels_to_strings(seq_labs))
    seq_lens = np.zeros(len(seq_str_labs), dtype=np.int)
    seq_begs = np.zeros(len(seq_str_labs), dtype=np.int)
    for idx, s in enumerate(seq_str_labs):
        seq_lens[idx] = len(s)
        seq_begs[idx] = int(s[0])

    skip_empty = True
    if skip_empty:
        nz_idx = np.where(seq_hist != 0)[0]
    else:
        nz_idx = slice(None)

    catalog = np.unique(seq_lens)
    len_hist = np.zeros(len(catalog))
    lookup = benedict()
    for c in catalog:
        lookup[c] = np.where(catalog == c)[0][0]

    for sdx, s in enumerate(seq_hist):
        c = seq_lens[sdx]
        len_hist[lookup[c]] += s

    # assert np.sum(len_hist) == len(list_of_sequences), "sanity check"
    # total = len(list_of_sequences)
    total = np.sum(len_hist)

    return catalog, len_hist / total, total


def sequence_length_histogram_from_pd_df(df, keepcols=[]):
    """
        Provide a data frame where each row is a burst, and that has a column
        named `Sequence length` and `Repetition`.
        All remaining columns are not considered, make sure to `query` correctly!

        # Retruns
        A pandas data frame (long form) where every row corresponds to one sequence-
        length that occured, ~4 rows per realization.
    """
    assert isinstance(keepcols, list)

    # sanity check
    for col in df.columns:
        if col not in ["Duration", "Sequence length", "First module", "Repetition"]:
            num_vals = len(df[col].unique())
            if num_vals != 1:
                log.warning(f"Column '{col}' has {num_vals} different values.")
            else:
                log.debug(f"Column '{col}' has {num_vals} different values.")

    all_seqs = df["Sequence length"].to_numpy()
    all_reps = df["Repetition"].to_numpy()

    unique_reps = np.sort(np.unique(all_reps))
    unique_seqs = np.sort(np.unique(all_seqs))

    default_seqs = np.array([1, 2, 3, 4])
    if len(unique_seqs) != len(default_seqs) or np.any(unique_seqs != default_seqs):
        log.warning("unexpected sequence length encountered.")

    bins = (unique_seqs - 0.5).tolist()
    bins = bins + [bins[-1] + 1]

    columns = [
        "Sequence length",
        "Probability",
        "Occurences",
        "Total",  # Number of entries contributing to probability
        *keepcols,
        "Repetition",
    ]
    df_out = pd.DataFrame(columns=columns)

    # iterate over every repetition
    for rep_id in tqdm(unique_reps, desc="Repetitions", leave=False):
        idx = np.where(all_reps == rep_id)[0]
        hist, _ = np.histogram(all_seqs[idx], bins=bins)
        total = np.sum(hist)
        probs = hist / total

        keepcol_entries = []
        for col in keepcols:
            if not _pd_are_row_entries_the_same(df[col].iloc[idx]):
                log.warning(
                    "Filter (query) before calling this! If entries missmatch for one repetition, this will give a nonsensical histogram."
                )
            val = df[col].iloc[idx[0]]
            keepcol_entries.append(val)

        # and add one row for every sequence length
        for ldx, l in enumerate(unique_seqs):
            df_out = df_out.append(
                pd.DataFrame(
                    data=[
                        [
                            unique_seqs[ldx],
                            probs[ldx],
                            hist[ldx],
                            total,
                            *keepcol_entries,
                            rep_id,
                        ]
                    ],
                    columns=columns,
                ),
                ignore_index=True,
            )

    return df_out


# def pd_mean_across_repetitions(df, match=[]):
#     """
#         Keep everything as is but calulate the mean across repetitions
#     """
#     df_out = pd.DataFrame(columns=df.columns)


def _pd_are_row_entries_the_same(df):
    assert len(df.shape) == 1
    arr = df.to_numpy()
    if np.all(arr[0] == arr):
        return True
    else:
        return False
