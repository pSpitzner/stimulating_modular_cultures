# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-21 11:11:40
# @Last Modified: 2020-07-21 12:27:47
# ------------------------------------------------------------------------------ #
# Helper functions that are needed in various other scripts
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import numpy as np


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


def population_activity(spiketimes, bin_size):
    """
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


def inter_burst_interval(simulation_duration, spiketimes=None, burst_times=None):
    """
        calculate inter burst interval

        Parameters
        ----------
        spiketimes: 2d array
            zero-or-nan padded spiketimes. first dim are neurons, second time
            spiketrain per neuron

        simulation_duration: float or None
            in same time unit as spiketimes

        burst_times: 1d array or None
            if burst times were alread calculated they can be used to skip calculation


    """

    if spiketimes is not None:
        assert burst_times is None

    if burst_times is None:
        # replace zero padding with nans
        spiketimes = np.where(spiketimes == 0, np.nan, spiketimes)
        burst_times = burst_times(spiketimes, bin_size=0.5, threshold=0.75)

    try:
        ibi = simulation_duration / len(burst_times)
    except ZeroDivisionError:
        ibi = np.inf

    return ibi


