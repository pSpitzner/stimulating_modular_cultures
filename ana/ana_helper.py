# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-03-10 13:23:16
# @Last Modified: 2021-03-11 14:05:50
# ------------------------------------------------------------------------------ #


import os
import sys
import glob
import h5py
import numpy as np

import hi5 as h5
from hi5 import BetterDict

import logging
import warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from numba import jit, prange
    from numba.typed import List

    # raise ImportError
    log.info("Using numba for parallelizable functions")

    # silence deprications
    from numba.core.errors import (
        NumbaDeprecationWarning,
        NumbaPendingDeprecationWarning,
    )

    warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
    warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
except ImportError:
    log.info("Numba not available, skipping parallelization")
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


def prepare_file(h5f, mod_colors="auto"):
    """
        modifies h5f in place! (not on disk, only in RAM)

        # adds the following attributes:
        h5f.ana.mod_sort   : function that maps from neuron_id to sorted id, by module
        h5f.ana.mods       : list of unique module ids
        h5f.ana.mod_colors : list of colors associated with each module
        h5f.ana.neuron_ids
    """

    log.debug("Preparing File")

    if isinstance(h5f, str):
        h5f = h5.recursive_load(h5f, hot=False)

    h5f.ana = BetterDict()
    num_n = h5f.meta.topology_num_neur

    # ------------------------------------------------------------------------------ #
    # mod sorting
    # ------------------------------------------------------------------------------ #
    try:
        # get the neurons sorted according to their modules
        mod_sorted = np.zeros(num_n, dtype=int)
        mod_ids = h5f.data.neuron_module_id[:]
        mods = np.sort(np.unique(mod_ids))
        if len(mods) == 1:
            raise NotImplementedError  # avoid resorting.
        temp = np.argsort(mod_ids)
        for n_id in range(0, num_n):
            mod_sorted[n_id] = np.argwhere(temp == n_id)

        h5f.ana.mods = mods
        h5f.ana.mod_sort = lambda x: mod_sorted[x]
    except:
        h5f.ana.mods = [0]
        h5f.ana.mod_sort = lambda x: x

    # ------------------------------------------------------------------------------ #
    # assign colors to modules so we can use them in every plot consistently
    # ------------------------------------------------------------------------------ #
    if mod_colors is False:
        h5f.ana.mod_colors = ["black"] * len(h5f.ana.mods)
    elif mod_colors is "auto":
        h5f.ana.mod_colors = [f"C{x}" for x in range(0, len(h5f.ana.mods))]
    else:
        assert isinstance(mod_colors, list)
        assert len(mod_colors) == len(h5f.ana.mods)
        h5f.ana.mod_colors = mod_colors

    # ------------------------------------------------------------------------------ #
    # spikes
    # ------------------------------------------------------------------------------ #

    # maybe change this to exclude neurons that did not spike
    # neuron_ids = np.unique(spikes[:, 0]).astype(int, copy=False)
    neuron_ids = np.arange(0, h5f.meta.topology_num_neur, dtype=int)
    h5f.ana.neuron_ids = neuron_ids

    # make sure that the 2d_spikes representation is nan-padded, requires loading!
    spikes = h5f.data.spiketimes[:]
    spikes[spikes == 0] = np.nan
    h5f.data.spiketimes = spikes

    # # now we need to load things. [:] loads to ram and makes everything else faster
    # # convert spikes in the convenient nested (2d) format, first dim neuron,
    # # then ndarrays of time stamps in seconds
    # spikes = h5f.data.spiketimes_as_list[:]
    # spikes_2d = []
    # for n_id in neuron_ids:
    #     idx = np.where(spikes[:, 0] == n_id)[0]
    #     spikes_2d.append(spikes[idx, 1])
    # # the outer array is essentially a list but with fancy indexing.
    # # this is a bit counter-intuitive
    # h5f.ana.spikes_2d = np.array(spikes_2d, dtype=object)

    return h5f


def find_bursts_from_rates(
    h5f,
    bs_large=0.02,
    bs_small=0.002,  # seconds, small bin size
    rate_threshold=15,  # Hz
    merge_threshold=0.1,  # seconds, merge bursts if separated by less than this
):
    """
        Based on module-level firing rates, find bursting events.

        returns two BetterDicts, `bursts` and `rates`,
        does not modify `h5f`
    """

    assert h5f.ana is not None, "`prepare_file(h5f)` first!"

    spikes = h5f.data.spiketimes

    bursts = BetterDict()
    bursts.module_level = BetterDict()
    rates = BetterDict()
    rates.dt = bs_small
    rates.module_level = BetterDict()

    beg_times = []  # lists of length num_modules
    end_times = []

    for m_id in h5f.ana.mods:
        selects = np.where(h5f.data.neuron_module_id[:] == m_id)[0]
        pop_rate = population_rate(spikes[selects], bin_size=bs_small)
        pop_rate = smooth_rate(pop_rate, clock_dt=bs_small, width=bs_large)
        pop_rate = pop_rate / bs_small

        beg_time, end_time = burst_detection_pop_rate(
            rate=pop_rate, bin_size=bs_small, rate_threshold=rate_threshold,
        )

        beg_time, end_time = merge_if_below_separation_threshold(
            beg_time, end_time, threshold=merge_threshold
        )

        beg_times.append(beg_time)
        end_times.append(end_time)

        rates.module_level[m_id] = pop_rate
        bursts.module_level[m_id] = BetterDict()
        bursts.module_level[m_id].beg_times = beg_time.copy()
        bursts.module_level[m_id].end_times = end_time.copy()
        bursts.module_level[m_id].rate_threshold = rate_threshold

    pop_rate = population_rate(spikes[:], bin_size=bs_small)
    pop_rate = smooth_rate(pop_rate, clock_dt=bs_small, width=bs_large)
    pop_rate = pop_rate / bs_small
    rates.system_level = pop_rate

    all_begs, all_ends, all_seqs = system_burst_from_module_burst(
        beg_times, end_times, threshold=merge_threshold,
    )

    bursts.system_level = BetterDict()
    bursts.system_level.beg_times = all_begs.copy()
    bursts.system_level.end_times = all_ends.copy()
    bursts.system_level.module_sequences = all_seqs

    return bursts, rates


def find_isis_from_bursts(h5f, bursts=None):
    """
        What are the the inter-spike-intervals within and out of bursts?
    """
    if bursts is None:
        assert h5f.ana.bursts is not None
        bursts = h5f.ana.bursts

    # get isi naively np.diff -> avoid boundary effects
    # do it for spike_train and beg_time, end_time
    # list of arrays
    # do this on the

    isi = BetterDict()

    for m_id in h5f.ana.mods:
        selects = np.where(h5f.data.neuron_module_id[:] == m_id)[0]
        spikes_2d = h5f.data.spiketimes[selects]
        isis_all, isis_in, isis_out = __inter_spike_intervals(
            spikes_2d,
            beg_times=bursts.module_level[m_id].beg_times,
            end_times=bursts.module_level[m_id].end_times,
        )
        isi[m_id] = BetterDict()
        isi[m_id].all = isis_all
        isi[m_id].in_bursts = isis_in
        isi[m_id].out_bursts = isis_out

    return isi


# ------------------------------------------------------------------------------ #
# lower level
# ------------------------------------------------------------------------------ #

# turns out this is faster without numba
def __inter_spike_intervals(spikes_2d, beg_times=None, end_times=None):
    """
        Returns a list of all interspike intervals, merged down for all neurons in
        `spikes_2d`.
        If `beg_times` and `end_times` are passed, returns two more lists
        with the isis inside and out of bursts.
    """
    isis_all = []
    isis_in_burst = []
    isis_out_burst = []
    if beg_times is None or end_times is None:
        num_bursts = 0
    else:
        num_bursts = len(beg_times)

    for n in range(spikes_2d.shape[0]):
        diffs = np.diff(spikes_2d[n])
        isis_all.extend(diffs[~np.isnan(diffs)])

        # check on burst level
        for idx in range(0, num_bursts):
            b = beg_times[idx]
            e = end_times[idx]

            spikes = spikes_2d[n]
            spikes = spikes[spikes >= b]
            spikes = spikes[spikes <= e]
            isis_in_burst.extend(np.diff(spikes))

            e = beg_times[idx]
            if idx > 0:
                b = end_times[idx - 1]
            else:
                # before first burst
                b = 0
            spikes = spikes_2d[n]
            spikes = spikes[spikes >= b]
            spikes = spikes[spikes <= e]
            isis_out_burst.extend(np.diff(spikes))

        # after last burst
        e = np.inf
        if num_bursts > 0:
            b = end_times[-1]
        else:
            b = 0
        spikes = spikes_2d[n]
        spikes = spikes[spikes >= b]
        spikes = spikes[spikes <= e]
        isis_out_burst.extend(np.diff(spikes))

    if beg_times is None or end_times is None:
        return isis_all
    else:
        return isis_all, isis_in_burst, isis_out_burst


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def population_rate(spiketimes, bin_size):
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

        Returns
        -------
        rate : 1d array
            time series of the rate in number of spikes per bin,
            normalized per-neuron, in steps of bin_size
    """

    num_n = spiketimes.shape[0]

    # target array
    t_max = np.nanmax(spiketimes)
    t_index_max = int(np.ceil(t_max / bin_size) + 1)

    rate = np.zeros(t_index_max)

    for n_id in range(0, num_n):
        train = spiketimes[n_id]
        for t in train:
            if not np.isfinite(t):
                break
            t_idx = int(t / bin_size)
            rate[t_idx] += 1

    rate = rate / num_n

    return rate


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

    if not return_series:
        return beg_time, end_time
    else:
        series = np.zeros(len(rate), dtype=np.int)
        series[above] = 1
        series = series.tolist()
        return beg_time, end_time, series


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
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

    # skip a new burst beginning if within threshold of the previous ending
    skip = False

    for idx in range(0, len(beg_time) - 1):

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


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def arg_merge_if_below_separation_threshold(beg_time, end_time, threshold):
    # same as above just that here we return the indices that would provide
    # the merged arrays. (like e.g. np.argsort)

    beg_res = []
    end_res = []

    # skip a new burst beginning if within threshold of the previous ending
    skip = False

    for idx in range(0, len(beg_time) - 1):

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
        Description

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

    # stack everything together in one lone flat list
    all_begs = np.hstack(beg_times)
    all_ends = np.hstack(end_times)
    all_mods = np.hstack(mods)

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

    # do sequence sorting next
    for pos, idx in enumerate(idx_begs):
        # first module, save sequence as tuple
        seq = (all_mods[idx],)

        # first time occurences of follower
        jdx = idx + 1
        while jdx <= idx_ends[pos]:
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
                logger.info(
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
