# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-21 11:11:40
# @Last Modified: 2020-10-08 12:12:11
# ------------------------------------------------------------------------------ #
# Helper functions that are needed in various other scripts
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def h5_load(filenames, dsetname, raise_ex=False, silent=False):
    def load(filename, dsetname, raise_ex):
        try:
            file = h5py.File(filename, "r")
            try:
                res = file[dsetname][:]
            except ValueError:
                res = file[dsetname][()]
            file.close()
            return res
        except Exception as e:
            if not silent:
                print(f"failed to load {dsetname} from {filename}")
            if raise_ex:
                raise e
            else:
                return np.nan

    files = glob.glob(filenames)
    res = []
    for f in files:
        res.append(load(f, dsetname, raise_ex))

    if len(files) == 1:
        return res[0]
    else:
        return res


def h5_ls(filename, dsetname="/"):
    """
        list the keys in a dsetname

        Parameters
        ----------
        filename: path to h5file
        dsetname: which dset to list

        Returns
        -------
        list: containing the contained keys as strings
    """
    try:
        file = h5py.File(filename, "r")
        try:
            res = list(file[dsetname].keys())
        except Exception as e:
            res = []
        file.close()
    except Exception as e:
        res = []

    return res


# helper function to convert a list of time stamps
# into a (binned) time series of activity
def bin_spike_times(spike_times, bin_size, pad_right=0):
    """
        Parameters
        ----------
        spike_times : 1d list of spike times for one neuron

        bin_size : float, same time unit as spike_times

        pad_right: attach some empty bins to the end of the series

        Returns
        -------
        1darray with the number of spikes per time bin

    """
    if len(spike_times) == 0:
        return np.array([])
    last_spike = spike_times[-1]
    # add one extra bin to be sure to avoid out of range errors
    num_bins = int(np.ceil(last_spike / bin_size) + 1)
    num_bins += int(pad_right)
    res = np.zeros(num_bins)
    for spike_time in spike_times:
        if not np.isfinite(spike_time):
            break
        target_bin = int(np.floor(spike_time / bin_size))
        res[target_bin] = res[target_bin] + 1
    return res


def burst_times_convolve(
    spks_m,
    win_size=None,
    bin_size=None,
    threshold=0.75,
    mark_only_onset=False,
    debug=False,
):
    # Let's detect bursts
    # more complicated implementation with a convolution. dont use.

    if win_size is None:
        win_size = 250 * ms
    if bin_size is None:
        bin_size = 50 * ms

    trains = spks_m.spike_trains()
    num_n = len(trains.keys())
    binned_trains = []
    tmax = 0
    # iterate over neurons
    for t in trains.keys():
        temp = bin_spike_times(
            spike_times=trains[t], bin_size=bin_size, pad_right=2 * win_size / bin_size
        )
        temp = np.clip(temp, 0, 1)
        if len(temp) > tmax:
            tmax = len(temp)
        binned_trains.append(temp)
    assert len(binned_trains) == num_n
    time_series = np.zeros(shape=(num_n, tmax))  # in steps of size bin_size

    # flat window for convolution
    width_dt = int(np.round(2 * win_size / bin_size))
    # window = np.ones(width_dt)

    # gaussian for convolution
    window = np.exp(-(((np.arange(width_dt) - width_dt / 2) / 0.75) ** 2) / 2)

    for n, t in enumerate(binned_trains):
        temp = np.convolve(t, window, mode="same")
        temp = np.clip(temp, 0, 1)
        assert len(t) == len(temp)
        time_series[n, 0 : len(temp)] = temp

    summed_series = np.sum(time_series, axis=0)
    bursting_bins = np.where(summed_series >= threshold * num_n)[0]

    if mark_only_onset:
        x = bursting_bins
        bursts = np.delete(x, [np.where(x[:-1] == x[1:] - 1)[0] + 1]) * bin_size
    else:
        bursts = bursting_bins

    if not debug:
        return bursts
    else:
        return bursts, time_series, summed_series, window


def burst_times(
    spiketimes, bin_size, threshold=0.75, mark_only_onset=False, debug=False
):
    """
        "Network bursts provided a quick insight into the collective dynamics in cultures
        and were defined as those activity episodes in which more than 75% of the neurons
        fired simultaneously for more than 500 ms."

        Parameters
        ----------
        spiketimes: 2d array
            nan padded spiketimes. first dim are neurons, second time
            spiketrain per neuron

        bin_size: float
            bin size in units of spiketimes, usually seconds. default 0.5

        mark_only_onset: bool
            whether to drop all consecutive burstin bins and only give the first one
            default false.

        debug: bool
            also return intermediate results


    """
    # assumes 2d spiketimes, np.nan padded with dim 1 the neuron id

    # replace zero padding with nans
    spiketimes = np.where(spiketimes == 0, np.nan, spiketimes)

    num_n = spiketimes.shape[0]
    bin_did_spike = []
    t_index_max = 0
    # iterate over neurons
    for n_id in range(0, num_n):
        spikes_per_bin = bin_spike_times(
            spike_times=spiketimes[n_id][np.isfinite(spiketimes[n_id])],
            bin_size=bin_size,
        )
        if len(spikes_per_bin) > t_index_max:
            t_index_max = len(spikes_per_bin)
        # did it burst yes/no?
        spikes_per_bin = np.clip(spikes_per_bin, 0, 1)
        bin_did_spike.append(spikes_per_bin)
    assert len(bin_did_spike) == num_n

    # make dimension consistent, arrays in bin_did_spike have different length
    # in steps of size bin_size
    time_series = np.zeros(shape=(num_n, t_index_max))
    for n_id, series in enumerate(bin_did_spike):
        time_series[n_id, 0 : len(series)] = series

    # 0 to num_n neurons spiking per bin
    summed_series = np.sum(time_series, axis=0)
    bursting_bins = np.where(summed_series >= threshold * num_n)[0]

    if mark_only_onset:
        x = bursting_bins
        bursts = np.delete(x, [np.where(x[:-1] == x[1:] - 1)[0] + 1]) * bin_size
    else:
        bursts = bursting_bins * bin_size

    # align burst time to the center of the bin
    bursts = bursts + bin_size / 2

    if not debug:
        return bursts
    else:
        return bursts, time_series, summed_series


def mutual_information(spiketimes, bin_size=0.050, embedding=10):

    log.debug("mututal_information()")
    binary_spikes = spikes_as_binned_binary_array(spiketimes, bin_size)
    pop_act = np.sum(binary_spikes, axis=0)
    num_n = binary_spikes.shape[0]
    num_t = binary_spikes.shape[1]

    # binarize the population activity
    pop_act = pop_act > np.median(pop_act)

    # MI for ---- on x
    # for every bin 0|1
    # |-------------|x|
    # |---- "y" ----|x|
    # then slide window:
    #   |-------------|x|

    # pattern, binary array to int. use int as the label of the pattern
    pattern_to_int = lambda b: b.dot(1 << np.arange(b.size)[::-1])

    # apply this on a sliding window for the whole binarized pop act
    f = lambda y: pattern_to_int(pop_act[y : y + embedding])
    pop_act_ints = np.vectorize(f)(np.arange(0, len(pop_act) - embedding))

    # indep. of neuron
    p_y = np.histogram(pop_act_ints, bins=np.max(pop_act_ints), density=True)[0]

    # dep. of neuron
    p_x = np.empty(shape=(num_n, 2))
    p_x[:, 1] = np.mean(binary_spikes, axis=1)
    p_x[:, 0] = 1 - p_x[:, 1]

    p_x_y = np.zeros(shape=(num_n, 2, 2**embedding))

    for t in range(0, num_t - embedding):
        patt
        p_x_y[0] += pop_act_ints[t]

    # H(Y), H(X)
    h_y = -np.sum(p_y * np.log2(p_y)) # scalar
    h_x = -np.sum(p_x * np.log2(p_x), axis=1) # size: num_n

    # H(Y|X) = -\sum p(x) \sum p(y|x) log p(y|x)
    # h_y_given_x -np.sum( )

    # I(X;Y) = H(Y) - H(Y|X)


def spikes_as_binned_binary_array(spiketimes, bin_size=0.050):
    """
        convert the nan-padded spikelist into a 2d array, binned with 0 or 1
        if spikes occured

        Parameters
        ----------
        spiketimes: 2d array
            nan-padded array of spiketimes, first dim are neurons

        bin_size: float
            time bin size, in same units as spiketimes. assuming seconds, defualt 50ms

        Returns
        -------
        binary_array: 2d np array with binned spikes
    """
    num_n = spiketimes.shape[0]

    # target array
    t_max = np.nanmax(spiketimes)
    t_index_max = int(np.ceil(t_max / bin_size))

    binned_ts = np.zeros(shape=(num_n, t_index_max), dtype=bool)

    for n_id in range(0, num_n):
        train = spiketimes[n_id]
        for t in train:
            if not np.isfinite(t):
                break
            t_idx = int(t / bin_size)
            binned_ts[n_id][t_idx] = True

    return binned_ts


def population_activity(spiketimes, bin_size):
    """
        calculate the activity across the whole population. naive binning,
        no sliding window.

        can be used to get the ASDR (Array wide spike detection rate)
        if bin_size is set to one second.

        spiketimes: np array with first dim neurons, second dim spiketimes. nan-padded
        bin_size: float, in units of spiketimes
    """

    num_n = spiketimes.shape[0]

    # target array
    t_max = np.nanmax(spiketimes)
    t_index_max = int(np.ceil(t_max / bin_size))

    population_activity = np.zeros(t_index_max)

    for n_id in range(0, num_n):
        train = spiketimes[n_id]
        for t in train:
            if not np.isfinite(t):
                break
            t_idx = int(t / bin_size)
            population_activity[t_idx] += 1

    return population_activity


def inter_burst_intervals(spiketimes=None, bursttimes=None):
    """
        calculate inter burst interval

        Parameters
        ----------
        spiketimes: 2d array or None
            zero-or-nan padded spiketimes. first dim are neurons, second time
            spiketrain per neuron

        bursttimes: 1d array or None
            if burst times were alread calculated they can be used to skip calculation

        Returns
        -------
        ibis: array of all ibi that were obsereved


    """

    if spiketimes is not None:
        assert bursttimes is None

    if bursttimes is None:
        bursttimes = burst_times(spiketimes, bin_size=0.5, threshold=0.75)

    if len(bursttimes) < 2:
        return np.array([])

    return bursttimes[1:] - bursttimes[:-1]


def spikes_as_matrix_to_spikes_as_list(spikes_as_matrix):
    """
        Parameters
        ----------
        spikes_as_matrix: 2d np array
            a zero- or nan-padded 2d array where first index is the neuron id
            and second index yields the spiketimes

        Returns
        -------
        spikes_as_list: 2d np array
            first column contains the neuron id
            second column contains the spiketime
    """

    spikes_as_list = []
    neurons_as_list = []

    num_neurons = spikes_as_matrix.shape[0]
    for n in range(num_neurons):
        train = spikes_as_matrix[n]
        # filter out zeros or nans
        train = train[train != 0]
        train = list(train[np.isfinite(train)])
        neuron_ids = [n] * len(train)

        neurons_as_list += neuron_ids
        spikes_as_list += train

    result = np.zeros(shape=(2, len(spikes_as_list)))
    result[0, :] = neurons_as_list
    result[1, :] = spikes_as_list

    return result
