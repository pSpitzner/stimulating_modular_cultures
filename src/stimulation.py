# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-05 10:30:17
# @Last Modified: 2021-06-07 17:15:55
# ------------------------------------------------------------------------------ #
# Create additional spikes that model stimulation
#
# mimic the stimulation from experiments.
#     * 400ms time windows (interval)
#     * 10 candidate neurons across two modules, here we want 5 n in one module
#     * per time window p=0.4 for every candidate
#     * instead of spiking at window onset, draw a random time within the window
# ------------------------------------------------------------------------------ #

# %%
import os
import sys
import h5py
import numpy as np
from brian2.units import *
import logging

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------ #
# model parameters
# ------------------------------------------------------------------------------ #


def stimulation_pattern(interval, duration, target_modules, mod_ids, min_dt=0):
    """
        Generate times and indices (of neurons) that are stimulated.
        We induce on average one extra spike in the `interval`,
        per candidate neuron. We have 5 candidates per target module.

        # Parameters
            interval       : time, the time window of stimulation (with brian time unit)
            duration       : time, total duration
            target_modules : list, which modules to target
            mod_ids        : ndarry, id of the module each neuron of the system is in
            min_dt         : time, subsequent spikes of same neurons are ensured to
                            be separated by at least `min_dt`. set to brians timestep

        # Returns
            indices
            times
    """

    log.info(f"Setting up stimulation {interval} at module {target_modules}")

    candidates = _draw_candidates(
        mod_ids=mod_ids, n_per_mod=5, mod_targets=target_modules,
    )
    candidates = np.array(candidates)
    jitter = True

    global s_indxs
    global seps_for_c

    tries = 0
    while True:
        tries += 1
        s_indxs = []
        s_times = []
        for step in range(1, int(duration / interval) - 1):
            t = step * interval / second
            n_targets = _draw_pattern_from_candidates(
                candidates=candidates, p_per_candidate=0.4
            ).tolist()
            if False:
                # in the past we had completely determinstic spiking every 400ms
                s_indxs += n_targets
                s_times += [t] * len(n_targets)
            else:
                # instead of a fixed spike time, add poisson drive to every target
                num_segments = 100.0
                dt = interval / num_segments / second
                p = 1.0 / num_segments
                for n in n_targets:
                    for t_segment in np.arange(t, t + interval / second, dt):
                        if np.random.uniform(0, 1) < p:
                            s_indxs += [n]
                            s_times += [
                                np.random.uniform(low=t_segment, high=t_segment + dt)
                            ]

        # case to numpy array
        s_times = np.array(s_times)
        s_indxs = np.array(s_indxs)

        # make sure we did not draw two consecutive spikes that are smaller than
        # the simulation timestep
        smallest_sep_is_long_enough = True
        for c in candidates:
            seps_for_c = np.diff(s_times[s_indxs == c])
            if np.nanmin(seps_for_c) < min_dt / second:
                smallest_sep_is_long_enough = False
                break
        if smallest_sep_is_long_enough:
            break
        # if we exceed tries, throw an exception
        assert (
            tries < 1000
        ), f"could not generate times that are separated by {min_dt} in 1000 attempts"


    # sort by spike time
    idx = np.argsort(s_times)
    s_times = s_times[idx] * second
    s_indxs = s_indxs[idx]

    return s_indxs, s_times


def _draw_candidates(mod_ids, n_per_mod=5, mod_targets=[0]):
    """
        produces a random selection of neurons (candidates) drawn from specified
        modules

        Parameters
        ----------
        mod_ids : 1d array
            ids of the modules. index is the id of the neuron, the value at the
            index is the id of the module

        n_per_mod : int
            number of candidates to draw per module

        mod_targets : list of int
            which modules to pick candidates from

        Returns
        -------
        neuron_ids: list of int
            the neuron ids to target
    """

    num_in_mod = []
    offset = [0]
    arg_sorted = np.argsort(mod_ids)
    for mdx in range(0, np.nanmax(mod_targets) + 1):
        N = np.sum(mod_ids == mdx)
        offset.append(offset[-1] + N)
        num_in_mod.append(N)

    # draw a random neuron in the target module
    rand_in_mod = lambda m: arg_sorted[offset[m] + np.random.randint(0, num_in_mod[m])]

    res = []
    for mdx in mod_targets:
        assert mdx in mod_ids
        temp = []
        for n in range(0, n_per_mod):
            tar = rand_in_mod(mdx)
            while tar in temp:
                tar = rand_in_mod(mdx)
            temp.append(tar)
        res += temp

    return res


def _draw_pattern_from_candidates(candidates, p_per_candidate):
    """
        create a random pattern from given candidates.

        Parameters
        ----------
        candidates: list of int
            the ids of the candidates

        Returns
        -------
        activate: list of int
            the ids of the candidates to target
    """

    num_c = len(candidates)
    idx = np.random.random_sample(size=num_c)  # [0.0, 1.0)
    return candidates[idx < p_per_candidate]


def _time_random(t_start, t_end, num_n):
    """
        create `num_n` random times that lie within a time window
        from `t_start` to `t_end`

        Returns
        -------
        times: 1d np array of size `num_n`
    """

    return np.random.uniform(low=t_start, high=t_end, size=num_n)  # [low, high)


def stimulation_at_rate_with_cv(rate, cv, t_end, min_dt=0*second, t_start=0*second,):
    """
        For a single neuron, generate spike times that have a desired cv.

        Returns a 1d array with spike times in brian unit seconds.
    """
    tries = 0
    while True:
        tries += 1

        mean_isi = 1 / rate
        size = int(1.05*(t_end - t_start) / mean_isi)
        mu = mean_isi / second
        std = cv * (mu)

        # we want lognormal, so isis do not get negative
        mu_log = np.log(mu*mu/np.sqrt(mu*mu + std*std))
        std_log = np.sqrt(np.log(1+std*std/(mu*mu)))

        isis = np.random.lognormal(mean=mu_log, sigma=std_log, size=size) * second
        spike_times = np.cumsum(isis)
        isis = isis[spike_times <= t_end]
        spike_times = np.sort(spike_times[spike_times <= t_end])

        if np.nanmin(isis) >= min_dt:
            mean = np.mean(isis)
            cv = np.std(isis) / mean
            log.debug(f"{len(spike_times)} spike times with rate {1 / mean:.3f} Hz and cv {cv:.3f}")
            return spike_times

        # if we exceed tries, throw an exception
        assert (
            tries < 1000
        ), f"could not generate times that are separated by {min_dt} in 1000 attempts"

# ------------------------------------------------------------------------------ #
# explorative helpers
# ------------------------------------------------------------------------------ #


def _random_windows(interval, duration, p=0.4):
    num_intervals = int(duration / interval)

    res = []
    for i in range(1, num_intervals + 1):
        if np.random.uniform(0.0, 1.0) > p:
            res.append(i * interval)

    return np.array(res)
