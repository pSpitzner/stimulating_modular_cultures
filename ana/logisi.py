# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-09-28 10:36:48
# @Last Modified: 2020-11-30 20:12:41
# ------------------------------------------------------------------------------ #
# My implementation of the logISI historgram burst detection algorithm
# by Pasuqale et al.
#
# Adapted from R-code https://github.com/ellesec/burstanalysis
# Original Algorithm: DOI 10.1007/s10827-009-0175-1
# Comparison by Ellese et al: DOI 10.1152/jn.00093.2016
#
# Only detects the bursts within each Channel / ROI / Unit. Thus, if we want to
# include short network bursts (less than 3 spikes per neuron),
# we need sth else.
#
# ToDo: Network-burst detection.
# ------------------------------------------------------------------------------ #

import sys
import os
import logging
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks
from tqdm import tqdm
from itertools import permutations

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

log = logging.getLogger(__name__)
# log.setLevel("DEBUG")

try:
    from numba import jit, prange

    # raise ImportError
    log.info("Using numba for parallelizable functions")
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


# ------------------------------------------------------------------------------ #
# population rate based burst detection
# ------------------------------------------------------------------------------ #


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


# @jit(nopython=True, parallel=True, fastmath=True, cache=True)
def burst_detection_pop_rate(
    spiketimes, bin_size=0.02, rate_threshold=15, extend=False
):
    """
        find the population rate and define a burst for exceeding a threshold

        Parameters
        ----------
        par: type
            parameter description

        Returns
        -------
        something: of_type
    """

    num_n = spiketimes.shape[0]

    rate = population_rate(spiketimes, bin_size=bin_size) / bin_size

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

    return beg_time, end_time


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
        beg_time, end_time
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

    return np.array(beg_res), np.array(end_res)


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

    return np.array(beg_res, dtype=np.int32), np.array(end_res, dtype=np.int32)


def system_burst_from_module_burst_broken(beg_times, end_times, threshold):

    # number modules
    num_m = len(beg_times)

    # system_wide burst begin and end times
    beg_sys = []
    end_sys = []

    # keep track which is the latest considered burst in each module
    b_ids = np.zeros(num_m, dtype=np.int)

    done = False
    while not done:
        first_id = np.argmin(beg_times[:, b_ids])
        # end = -np.inf

        # find the first module burst
        for m in range(0, num_m):
            b = beg_times[m, b_ids[m]]
            if b <= beg:
                beg = b

            if e > end:
                end = e


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
        something: of_type
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

    # do sequence sorting next
    # ...

    return all_begs[idx_begs], all_ends[idx_ends]


# ------------------------------------------------------------------------------ #
# Logisi method
# ------------------------------------------------------------------------------ #

# value to return if no bursts found.
no_bursts = dict()
for key in [
    "i_beg",
    "i_end",
    "t_beg",
    "t_end",
    "t_med",
    "IBI",
    "blen",
    "durn",
    "mean_isis",
]:
    no_bursts[key] = np.array([]).astype(int)


def burst_detection_pasquale(spike_train, cutoff=0.1):
    """
        burst detection for single neuron/channel/roi spike trains.

        this implements the main method described in DOI 10.1007/s10827-009-0175-1
        but only on a single spiketrain, not on a network level

        Parameters
        ----------
        spike_train: 1d array
            time stamps of the neurons' spikes, in seconds

        cutoff: float
            threshold for the isi, in seconds.
            Burst will be required to have a lower isi than this cutoff.

        Returns
        -------
        bursts: dict
            with the following keys, each with a 1d array of len = number bursts:
            * i_beg: the index in the train of the first spike in the burst
            * i_end: the index of the last spike of the burst
            * med: the time of the median spike within the burst
            * IBI: time from last spike of the prev. burst to first spike of this burst
            * blen: number of spikes counting into the burst
            * durn: duration of the burst, first spike to last spike
            * mean_isis: the mean inter spike interval within the burst

        isi_low: float
            the threshold where we separate between inter- and intra-burst intervals.

        hist: array
            the histogram with all inter-spike-interval counts

        edges: array
            the bin edges of the isi histogram (from np.histogram)
    """

    if len(spike_train) > 3:
        # Calculates threshold as isi_low
        isi_low, hist, hist_smooth, edges = logisi_break_calc(spike_train, cutoff)
        log.debug(f"isi_low {isi_low}")
        if isi_low is None or isi_low >= 1:
            # If no value for isi_low found, or isi_low above 1 second, find bursts using threshold equal to cutoff (default 100ms)
            result = logisi_find_burst(
                spike_train, min_ibi=0, min_durn=0, min_spikes=3, isi_low=cutoff
            )

        elif isi_low < 0:
            result = no_bursts

        elif isi_low > cutoff and isi_low < 1:
            # If isi_low >cutoff, find bursts using threshold equal to cutoff (default 100ms)
            bursts = logisi_find_burst(
                spike_train, min_ibi=isi_low, min_durn=0, min_spikes=3, isi_low=cutoff
            )
            if bursts is not None:
                # If bursts have been found, add burst related spikes using threshold of isi_low
                brs = logisi_find_burst(
                    spike_train, min_ibi=0, min_durn=0, min_spikes=3, isi_low=isi_low
                )
                result = add_brs(bursts, brs, spike_train)
            else:
                result = bursts

        else:
            # If isi_low<cutoff, find bursts using a threshold equal to isi_low
            result = logisi_find_burst(
                spike_train, min_ibi=0, min_durn=0, min_spikes=3, isi_low=isi_low
            )

    else:
        return [no_bursts] + [None] * 3

    return result, isi_low, hist, edges


# Function to find cutoff threshold.
def find_thresh(h, h_edges, ISITh=100.0, void_th=0.7, peak_kwargs=None):
    log.debug("find_thresh")
    if peak_kwargs is None:
        peak_kwargs = {"width": 1}
    log.debug(f"kwargs: {peak_kwargs}")
    # this is an ugly workaround because find_peaks is not detecting maxima at edges
    h = np.insert(h, 0, -np.inf)
    h_edges = np.insert(h_edges, 0, -np.inf)
    peak_idx, peak_props = find_peaks(h, height=0, **peak_kwargs)
    # peak_idx = peakutils.peak.indexes(h, thres=0.0, min_dist=3)
    num_peaks = len(peak_idx)
    peak_pos = h_edges[peak_idx]
    peak_heights = peak_props["peak_heights"]
    # peak_heights = h[peak_idx]
    # find peak position of the intra-burst-isi, if below the specified threshold
    if (peak_pos < ISITh).any():
        last_idx = np.where(peak_pos < ISITh)[0][-1]
        intra_idx = np.argmax(peak_heights[0:last_idx]) if last_idx > 0 else 0
        intra_height = peak_heights[intra_idx]
    else:
        return -1000.0

    y1 = intra_height
    x1 = peak_idx[intra_idx]
    log.debug(f"peak_idx {peak_idx}")
    log.debug(f"peak_heights {peak_heights}")

    log.debug(f"x1: {intra_idx}")
    log.debug(f"y1: {intra_height}")
    log.debug(f"last_idx: {last_idx}")
    num_peaks_after_burst = num_peaks - intra_idx

    if num_peaks_after_burst == 0:
        return None
    else:
        x_2s = peak_idx[intra_idx + 1 :]
        y_2s = peak_heights[intra_idx + 1 :]
        # x_2s = np.delete(x_2s, [3,6])
        # y_2s = np.delete(y_2s, [3,6])

        log.debug(f"h {h}")
        log.debug(f"x1 {x1}")
        log.debug(f"x_2s {x_2s}")
        log.debug(f"num_peaks {num_peaks}")

        if len(x_2s) == 0:
            return None

        f = lambda x: np.amin(h[x1:x])
        ymins = np.vectorize(f)(x_2s)

        f = lambda x: np.argmin(h[x1:x])
        # log.debug(h[x1 : x_2s[0]])
        # log.debug(h[x1 : x_2s[-1]])
        xmins = np.vectorize(f)(x_2s) + x1

        void_pars = 1 - (ymins / np.sqrt(y1 * y_2s))

        log.debug(f"ymins: {ymins}")
        log.debug(f"xmins: {xmins}")
        log.debug(f"void_pars: {void_pars}")

    try:
        void_idx = np.where(void_pars >= void_th)[0][0]
        log.debug(f"void idx: {void_idx}")
    except:
        void_idx = None

    if void_idx is None:
        return None
    else:
        log.debug(f"ISImax: {h_edges[xmins[void_idx]]}")
        return h_edges[xmins[void_idx]]


debug_h = None
debug_s = None
debug_e = None
debug_spikes = None

# Calculates cutoff for burst detection
def logisi_break_calc(st, cutoff, void_th=0.7, peak_kwargs=None):

    global debug_h
    global debug_s
    global debug_e
    global debug_isi
    global debug_spikes

    isi = np.diff(st) * 1000.0
    debug_spikes = st
    isi = isi[isi >= 1]
    max_isi = np.ceil(np.log10(np.max(isi)))
    br = np.logspace(0, max_isi, int(10 * max_isi))
    hist, edges = np.histogram(isi, bins=br, density=True)
    hist_smooth = lowess(
        endog=hist,
        exog=np.arange(len(hist)),
        frac=0.05,
        is_sorted=True,
        return_sorted=False,
    )
    debug_h = hist
    debug_s = hist_smooth
    debug_e = edges / 1000.0

    # log.debug(hist_smooth)
    thr = find_thresh(
        h=hist_smooth,
        h_edges=edges,
        ISITh=cutoff * 1000.0,
        void_th=void_th,
        peak_kwargs=peak_kwargs,
    )
    log.debug(f"thr: {thr}")
    if not thr is None:
        thr = thr / 1000.0

    return thr, hist, hist_smooth, edges / 1000.0


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _add_brs_inner(a_beg, a_end, b_beg, b_end):
    def is_between(x, a, b):
        return (x >= a) & (x <= b)

    num_a = len(a_beg)
    num_b = len(b_beg)

    res_beg = np.zeros(num_a, dtype=np.int32)
    res_end = np.zeros(num_a, dtype=np.int32)

    for i in prange(num_a):
        for j in range(num_b):
            if is_between(a_beg[i], b_beg[j], b_end[j]) or is_between(
                a_end[i], b_beg[j], b_end[j]
            ):
                res_beg[i] = np.fmin(a_beg[i], b_beg[j])
                res_end[i] = np.fmax(a_end[i], b_end[j])
                break
            else:
                res_beg[i] = a_beg[i]
                res_end[i] = a_end[i]

            if b_end[j] > a_end[i]:
                break

    return res_beg, res_end


###Function to add burst related spikes to edges of bursts
def add_brs(bursts, brs, spikes):

    num_bursts = len(bursts["i_beg"])
    num_brs = len(brs["i_beg"])

    res_beg, res_end = _add_brs_inner(
        bursts["i_beg"], bursts["i_end"], brs["i_beg"], brs["i_end"]
    )

    burst_adj = {
        "i_beg": res_beg,
        "i_end": res_end,
    }

    diff_begs = np.diff(burst_adj["i_beg"])
    diff_ends = np.diff(burst_adj["i_end"])

    rejects = np.array([])
    rejects = np.concatenate((rejects, np.where(diff_begs == 0)[0]))
    rejects = np.concatenate((rejects, np.where(diff_ends == 0)[0] + 1))
    rejects = np.sort(np.unique(rejects))

    if len(rejects) > 0:
        for key in burst_adj.keys():
            np.delete(burst_adj[key], rejects)

    burst_adj["blen"] = burst_adj["i_end"] - burst_adj["i_beg"] + 1
    burst_adj["durn"] = spikes[burst_adj["i_end"]] - spikes[burst_adj["i_beg"]]
    burst_adj["mean_isis"] = burst_adj["durn"] / (burst_adj["blen"] - 1)

    ibis = spikes[burst_adj["i_beg"][1:]] - spikes[burst_adj["i_end"][:-1]]
    ibis = np.insert(ibis, 0, np.nan)
    burst_adj["IBI"] = ibis

    # get the median time of the burst
    f = lambda b: np.median(spikes[burst_adj["i_beg"][b] : burst_adj["i_end"][b]])
    burst_adj["t_med"] = (
        np.vectorize(f)(np.arange(len(burst_adj["i_beg"])))
        if len(burst_adj["i_beg"]) > 0
        else np.array([])
    )

    # get the time stamps of i_beg and i_end
    burst_adj["t_beg"] = spikes[burst_adj["i_beg"]]
    burst_adj["t_end"] = spikes[burst_adj["i_end"]]

    return burst_adj


##Function for finding bursts, taken from sjemea
def logisi_find_burst(spikes, min_ibi, min_durn, min_spikes, isi_low, neuron_ids=None):
    """
        Find bursts, taken from sjemea

        Parameters
        ----------
        spikes: array
            containing the spike times

        min_ibi : float
            minimal interval to separate two bursts, otherwise will be merged

        min_durn, min_spikes: floats
            minimal burst-duration and number of spikes required to count as burst

        isi_low: float
            threshold for isi, isis below this value are taken to be in a burst.

        neuron_ids : np.array
            used when detecting network bursts. this array should contain the id of
            the neuron that caused the spike/burst contained in the passed 'spike' array.
            default None.
            If not None, the min_spikes requirement checks for unique contributions.

        Returns
        -------
        bursts: dict with various keys, containing arrays with the burst details.
        if 'neuron_ids' is provided in addition to the default keys, also 'unique'
        is present. this contains the number of unique neurons that were involved
        in a network burst.

    """

    ## For one spike train, find the burst using log isi method.
    ## e.g.
    ## find.bursts(s$spikes[[5]])
    ## init.
    ##

    nspikes = len(spikes)

    ## Create a temp array for the storage of the bursts.  Assume that
    ## it will not be longer than Nspikes/2 since we need at least two
    ## spikes to be in a burst.

    max_bursts = np.floor(nspikes / 2)

    bursts = []

    # current burst number
    burst = 0

    ## Phase 1 -- burst detection. Each interspike interval of the data
    ## is compared with the threshold THRE. If the interval is greater
    ## than the threshold value, it can not be part of a burst; if the
    ## interval is smaller or equal to the threhold, the interval may be
    ## part of a burst.

    ## last_end is the time of the last spike in the previous burst.
    ## This is used to calculate the IBI.
    ## For the first burst, this is no previous IBI
    last_end = np.nan

    eps = 1e-10
    n = 1
    in_burst = False

    log.debug("logisi_find_burst:")
    log.debug(f"nspikes {nspikes}")
    log.debug(f"isi_low {isi_low}")

    while n < nspikes - 1:
        next_isi = spikes[n] - spikes[n - 1]
        if in_burst:
            if next_isi - isi_low > eps:
                ## end of burst
                end = n - 1
                in_burst = False

                ibi = spikes[beg] - last_end
                last_end = spikes[end]
                res = np.array([beg, end, ibi])
                burst = burst + 1

                assert burst <= max_bursts
                bursts.append(res)

        else:
            ## not yet in burst.
            if next_isi - isi_low <= eps:
                ## Found the start of a new burst.
                beg = n - 1
                in_burst = True

        n = n + 1

    ## At the end of the burst, check if we were in a burst when the
    ## train finished.
    if in_burst:
        end = nspikes - 1
        ibi = spikes[beg] - last_end
        res = np.array([beg, end, ibi])
        burst = burst + 1
        assert burst <= max_bursts
        bursts.append(res)

    ## Check if any bursts were found.
    if burst > 0:
        # convert bursts into a dictionary of 1d arrays
        bursts = np.array(bursts)
        bursts = {
            "i_beg": bursts[:, 0].astype(int),
            "i_end": bursts[:, 1].astype(int),
            "IBI": bursts[:, 2],
        }

    else:
        ## no bursts were found, so return an empty structure.
        return no_bursts

    log.debug("End of phase 1")
    log.debug(f"num bursts before rejections: {len(bursts['i_beg'])}")
    # log.debug(bursts)

    ## Phase 2 -- merging of bursts.    Here we see if any pair of bursts
    ## have an IBI less than min_ibi; if so, we then merge the bursts.
    ## We specifically need to check when say three bursts are merged
    ## into one.

    ibis = bursts["IBI"]
    rejects = np.array([])

    if (ibis < min_ibi).any():
        ## Merge bursts efficiently.    Work backwards through the list, and
        ## then delete the merged lines afterwards.    This works when we
        ## have say 3+ consecutive bursts that merge into one.

        # remove these later
        merge_bursts = np.where(ibis < min_ibi)[0]
        log.debug(f"ibis < min_ibi: {len(merge_bursts)}")
        rejects = merge_bursts

        for burst in reversed(merge_bursts):
            bursts["i_end"][burst - 1] = bursts["i_end"][burst]
            # bursts["i_end"][burst] = np.nan  # not needed, but helpful.

    log.debug("End of phase 2\n")
    # log.debug(bursts)

    ## Phase 3 -- remove small bursts: less than min duration (MIN_DURN), or
    ## having too few spikes (less than MIN_SPIKES).
    ## In this phase we have the possibility of deleting all spikes.

    ## BLEN = number of spikes in a burst.
    ## DURN = duration of burst.
    bursts["blen"] = bursts["i_end"] - bursts["i_beg"] + 1
    # log.debug(bursts["i_beg"])
    # log.debug(bursts["i_end"])
    bursts["durn"] = spikes[bursts["i_end"]] - spikes[bursts["i_beg"]]
    bursts["mean_isis"] = bursts["durn"] / (bursts["blen"] - 1)

    rejects = np.concatenate((rejects, np.where(bursts["durn"] < min_durn)[0]))
    if neuron_ids is None:
        rejects = np.concatenate((rejects, np.where(bursts["blen"] < min_spikes)[0]))
    else:
        unique = lambda b: len(
            np.unique(neuron_ids[bursts["i_beg"][b] : bursts["i_end"][b]])
        )
        bursts["unique"] = (
            np.vectorize(unique)(np.arange(len(bursts["i_beg"])))
            if len(bursts["i_beg"]) > 0
            else np.array([])
        )
        rejects = np.concatenate((rejects, np.where(bursts["unique"] < min_spikes)[0]))

    rejects = np.sort(np.unique(rejects)).astype(int)

    if len(rejects) > 0:
        for key in bursts.keys():
            bursts[key] = np.delete(bursts[key], rejects)

    log.debug(f"num bursts after rejections: {len(bursts['i_beg'])}")

    if len(bursts["i_beg"]) == 0:
        pass
        ## All the bursts were removed during phase 3.
        # bursts = no_bursts
    else:
        ## Recompute IBI (only needed if phase 3 deleted some cells).
        if len(bursts["i_beg"]) > 1:
            ibis = spikes[bursts["i_beg"][1:]] - spikes[bursts["i_end"][:-1]]
            ibis = np.insert(ibis, 0, np.nan)
        else:
            ibis = np.array([np.nan])

        bursts["IBI"] = ibis
        assert len(bursts["IBI"]) == len(bursts["i_beg"])

    # get the median time of the burst
    f = lambda b: np.median(spikes[bursts["i_beg"][b] : bursts["i_end"][b]])
    bursts["t_med"] = (
        np.vectorize(f)(np.arange(len(bursts["i_beg"])))
        if len(bursts["i_beg"]) > 0
        else np.array([])
    )

    # get the time stamps of i_beg and i_end
    bursts["t_beg"] = spikes[bursts["i_beg"]]
    bursts["t_end"] = spikes[bursts["i_end"]]

    ## End -- return burst structure.
    log.debug("logisi_find_burst done")
    return bursts


# ------------------------------------------------------------------------------ #
# network burst detection
# ------------------------------------------------------------------------------ #


def network_burst_detection(spiketimes, network_fraction=0.75, sort_by="i_beg"):
    """
        Detection of network bursts using the logisi method by pasquale et al.
        The log-histogram trick is applied two times, once on the per-neuron-
        level and then on the network level.

        Parameters
        ----------
        spiketimes : 2d np array
            nan-padded spiketimes. first dim neurons, second dim spiketimes

        network_fraction : float
            the fraction of unique neurons that need to be bursting (on the
            single-neuron level) in order to detect a network burst

        sort_by : str
            "i_beg", "t_med", or "i_end"; default is "i_beg"
            what criterion to sort neuron-level bursts by for the network
            burst detection. only impacts the sequence of contributing
            neurons in bursts, not the bursts themselves.


        Returns
        -------
        network_bursts: dict
            containing the network bursts

        details: dict
            containing details such as the neuron-level bursts, neuron id
            corresponding to a burst etc.
    """
    assert sort_by in ["i_beg", "t_med", "i_end"]
    num_n = spiketimes.shape[0]

    # flat list of all bursts that occured on the single-neuron level
    # beginning, median, end times
    med_times = []
    beg_times = []
    end_times = []
    # which neuron did burst
    neuron_ids = []

    for n in tqdm(range(num_n), leave=None):
        train = spiketimes[n]
        train = train[np.isfinite(train)]
        train = train[np.nonzero(train)]
        bursts, _, _, _ = burst_detection_pasquale(train)
        neuron_ids += [n] * len(bursts["t_med"])
        med_times += bursts["t_med"].tolist()
        # in the burst strucutre, beg and end are indices, convert to times.
        beg_times += train[bursts["i_beg"]].tolist()
        end_times += train[bursts["i_end"]].tolist()

    neuron_ids = np.array(neuron_ids)
    med_times = np.array(med_times)
    beg_times = np.array(beg_times)
    end_times = np.array(end_times)

    details = dict()
    details["neuron_ids"] = np.array([])
    details["med_times"] = np.array([])
    details["beg_times"] = np.array([])
    details["end_times"] = np.array([])

    if len(med_times) == 0:
        return no_bursts, details

    # sort neuron-level bursts according to burst time, depending on user choice
    if sort_by == "i_beg":
        burst_times = beg_times
    elif sort_by == "i_end":
        burst_times = end_times
    elif sort_by == "t_med":
        burst_times = med_times

    idx = np.argsort(burst_times)
    burst_times = burst_times[idx]
    details["neuron_ids"] = neuron_ids[idx]
    details["t_med"] = med_times[idx]
    details["t_beg"] = beg_times[idx]
    details["t_end"] = end_times[idx]

    try:
        # we need slightly different parameters on the network level.
        thr, hist, hist_smooth, edges = logisi_break_calc(
            burst_times, cutoff=0.2, void_th=0, peak_kwargs={}
        )
    except Exception as e:
        log.debug(f"logisi_break_calc: {e}")
        return no_bursts, details

    # log.setLevel("DEBUG")
    try:
        nb = logisi_find_burst(
            spikes=burst_times,
            min_ibi=0.25,
            min_durn=0,
            min_spikes=int(network_fraction * num_n),
            isi_low=thr,
            neuron_ids=details["neuron_ids"],
        )
    except Exception as e:
        log.info(f"find_burst: {e}")
        nb = no_bursts

    return nb, details


def sequence_detection(network_bursts, details, mod_ids):
    """
        Find in which sequence modules were activated during each network burst.

        Parameters
        ----------
        network_bursts, details : dict
            pass as obtained from network_burst_detection()

        mod_ids : array like
            which module every neuron is in. mod_ids[neuron_id] should give the mod id.
            no boundary check is done, neuron_ids need to be below len(mod_ids)

        Returns
        -------
        sequences : dict
            * 'neuron_seq', the full sequence of neurons of the network burst
            * 'module_seq', sequence of module activation. (shorter if not all activated)
            * 'first_n_from_m', which neuron was first in each module from 'module_seq'
    """

    # get a sequence for every network burst
    m_seqs = []
    n_seqs = []
    i_seqs = []

    # go through all network bursts
    for b in range(0, len(network_bursts["i_beg"])):
        # get indices of bounds of network bursts
        beg = network_bursts["i_beg"][b]
        end = network_bursts["i_end"][b]

        # sequence of neurons
        n_seq = details["neuron_ids"][beg:end]
        n_seqs.append(n_seq)

        # sequence of modules, use np.unique and preserve order
        m_seq = mod_ids[n_seq]
        _, idx = np.unique(m_seq, return_index=True)
        m_seqs.append(m_seq[np.sort(idx)])
        i_seqs.append(n_seq[np.sort(idx)])  # which neuron was first, per-module

    res = dict()
    res["neuron_seq"] = n_seqs
    res["module_seq"] = m_seqs
    res["first_n_from_m"] = i_seqs

    return res


def sequence_entropy(sequences, ids):

    # create a list of all possible sequences.
    # each sequence is a tuple
    labels = []
    for r in range(0, len(ids)):
        labels += list(permutations(ids, r + 1))

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


def reformat_network_burst(network_bursts, details, write_to_file=None):
    """
        Format network bursts as one np array. optionally, write to hdf5 file.

        Pass network_bursts and details from network_burst_detection()!
    """

    description = """
        network bursts, based on the logisi method by pasuqale DOI 10.1007/s10827-009-0175-1
        2d array, each row is a network burst, 6 columns (col 3-6 will need integer casting):
        0 - network-burst time: begin
        1 - network-burst time: median
        2 - network-burst time: end
        3 - id of first neuron to spike
        4 - id of last neuron to spike
        5 - numer unique neurons involved in the burst
    """
    num_bursts = len(network_bursts["i_beg"])
    dat = np.ones(shape=(num_bursts, 6)) * np.nan

    dat[:, 0] = network_bursts["t_beg"]  # details["beg_times"][network_bursts["i_beg"]]
    dat[:, 1] = network_bursts["t_med"]  # urgh, this is so inconsistent.
    dat[:, 2] = network_bursts["t_end"]  # details["end_times"][network_bursts["i_end"]]
    dat[:, 3] = details["neuron_ids"][network_bursts["i_beg"]]
    dat[:, 4] = details["neuron_ids"][network_bursts["i_end"]]
    dat[:, 5] = network_bursts["unique"]

    if write_to_file is not None:
        log.info("writing network bursts to {write_to_file}")
        try:
            f_tar = h5py.File(write_to_file, "r+")
            try:
                dset = f_tar.create_dataset("/data/network_bursts_logisi", data=dat)
            except RuntimeError:
                dset = f_tar["/data/network_bursts_logisi"]
                dset[...] = dat
            dset.attrs["description"] = description
            f_tar.close()
        except Exception as e:
            log.info(e)

    return dat
