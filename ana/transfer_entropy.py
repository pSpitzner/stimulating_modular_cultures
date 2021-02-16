# %%
import os
import sys
import glob
import h5py
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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


def spiketimes_to_act_mat(spiketimes, bin_size=1):
    """
    Create activity matrix from spike times.

    Parameters
    ----------

    """

    N = len(spiketimes)
    last_spike = np.nanmax(spiketimes)
    num_bins = int(np.ceil(last_spike / bin_size)) + 1
    act_mat = np.zeros((N, num_bins), dtype=np.int)

    for n in range(0, N):
        for this_time in spiketimes[n][spiketimes[n] != 0]:
            b = int(np.floor(this_time / bin_size))
            act_mat[n, b] += 1

    return act_mat


def transfer_entropy_jonas(act_mat):
    """
    returns the transfer entropy matrix TE[i,j] = TE from neuron j to i in basis 2.
    Runtime scales with num_neurons**2 and the algorithm only iterates over empty time bins.
    :param act_mat: activity matrix, 1st dimension neuron number, 2nd dimension timebins.
    :return: TE
    """
    num_neurons, num_timesteps = act_mat.shape
    IFT = 1  # has to be 0 or 1, whether to begin the first bin of the source neuron at the prediction time bin
    markov_order_destNeur = (
        2  # at least 1, the number of bins of the destination neuron
    )
    markov_order_sourceNeur = 2  # at least 1, the number of bins of the source neuron
    max_steps_past = max(markov_order_destNeur, markov_order_sourceNeur - IFT)

    ###make matrices with same dim as act_mat which contains for the destination timestep the pattern coded as number
    patterns_destNeur_mat = np.zeros((num_neurons, num_timesteps), dtype="int8")
    patterns_sourceNeur_mat = np.zeros((num_neurons, num_timesteps), dtype="int8")
    patterns_destNeur_mat[:, 1:] = act_mat[:, :-1]
    for step in range(2, markov_order_destNeur + 1):
        patterns_destNeur_mat[:, step:] += (
            act_mat[:, :-step] * 2 ** (step - 1)
        ).astype("int8")
    patterns_sourceNeur_mat[:, 1 - IFT :] = act_mat[:, : act_mat.shape[1] - 1 + IFT]
    for step in range(2 - IFT, markov_order_sourceNeur + 1 - IFT):
        patterns_sourceNeur_mat[:, step:] += (
            act_mat[:, :-step] * 2 ** (step - 1 + IFT)
        ).astype("int8")

    ###calculate the probability of each pattern P[i_destNeur, i_sourceNeur, act_destNeur, patterndest, patternsource]
    P = np.zeros(
        (
            num_neurons,
            num_neurons,
            2,
            2 ** markov_order_destNeur,
            2 ** markov_order_sourceNeur,
        )
    )
    # only iterate over the timesteps where patterns are not 0 (for optimization).
    time_nonzero = (
        np.nonzero(
            np.sum(act_mat + patterns_sourceNeur_mat + patterns_sourceNeur_mat, axis=0)[
                max_steps_past:
            ]
        )[0]
        + max_steps_past
    )
    neurons_arange = np.arange(num_neurons)
    for i_time in time_nonzero:
        for neur in neurons_arange:
            source_neurons = ~(neurons_arange == neur)
            P[
                neur,
                source_neurons,
                act_mat[neur, i_time],
                patterns_destNeur_mat[neur, i_time],
                patterns_sourceNeur_mat[source_neurons, i_time],
            ] += 1
    # normalize
    P /= num_timesteps - max_steps_past
    # calculate the probabilty of the pattern 0,0,0 using the property that all probabilities must sum up to one
    P[:, :, 0, 0, 0] = 1 - np.sum(P, axis=(2, 3, 4)) + P[:, :, 0, 0, 0]

    ###Calculate the transfer entropy from P (in bits)
    TE = -np.sum(
        np.sum(P, axis=4)
        * np.nan_to_num(
            np.log2(
                np.sum(P, axis=4)[:, :, :, :] / np.sum(P, axis=(2, 4))[:, :, None, :]
            )
        ),
        axis=(2, 3),
    ) + np.sum(
        P * np.nan_to_num(np.log2(P / np.sum(P, axis=2)[:, :, None, :, :])),
        axis=(2, 3, 4),
    )
    return TE


# memory hog
def transfer_entropy_paul(act_mat, skip_zeros=True):
    """
        # returns the transfer entropy matrix TE[i,j] = TE from neuron j to i in basis 2.
        # Runtime scales with num_neurons**2

        # Parameters
            act_mat: activity matrix, 1st dimension neuron number, 2nd dimension timebins.
    """

    # num neurons, num time steps
    N, T = act_mat.shape

    # instantaneous feedback term, include predicted time bin, 1 to include, 0 not
    IFT = 1

    # markov order, number of time bins to consider for source and target pattern
    src_mo = 2
    tar_mo = 2

    # maximum number of past time steps needed for time step
    past_steps = max(tar_mo, src_mo - IFT)

    # eg. 2 states for spikes 1 or no spike 0
    # num states for now only 2
    states = np.unique(act_mat)
    assert len(states) == 2

    # hmm, this is not ideal. we should probably check act_mat to have consecutive values for states?
    S = int(np.nanmax(act_mat) + 1)

    # high dimensional probability
    # [target index, source index, target state now, target pattern past, source pattern past]
    P = np.zeros(shape=(N, N, S, S ** tar_mo, S ** src_mo))

    # convert timeseries into a sequence of patterns for every neuron
    # keep full duration so one can iterate over all timesteps
    global time_nonzero, src_pat_seq, tar_pat_seq
    src_pat_seq = np.zeros(shape=(N, T), dtype="int8")
    tar_pat_seq = np.zeros(shape=(N, T), dtype="int8")

    # all neurons, all timesteps minus pattern length
    # code patterns as a number: S**markov_order possible values for each pattern
    print("finding patterns ...")

    # target
    # build pattern, start with now -1 bin. we have no past for the zeroth bin.
    dt = 1
    tar_pat_seq[:, dt:] += act_mat[:, :T-dt] * S ** (0)

    # go further into the past and add to pattern
    for pat, dt in enumerate(range(2, tar_mo+1)):
        tar_pat_seq[:, dt:-tar_mo] += act_mat[:, : -tar_mo - dt] * S ** (pat+1)

    # repeat for source pattern, here we might want to use the present bin as
    # the most recent one to contribute to the pattern (IFT = 1)
    dt = 0 + IFT
    src_pat_seq[:, dt:] += act_mat[:, :T-dt] * S ** (0)
    for pat, dt in enumerate(range(1+IFT, src_mo+1-IFT)):
        src_pat_seq[:, dt:-src_mo] += act_mat[:, : -src_mo - dt] * S ** (pat+1)

    # check with jonas: the offset or starting point of the pattern labeling
    # is different than in javiers code. not sure which version is right,
    # at the moment my version has a shift that jonas' does not.

    # print("act")
    # print(act_mat)
    # print("tar_pat_seq")
    # print(tar_pat_seq.shape)
    # print("src_pat_seq")
    # print(src_pat_seq.shape)

    if skip_zeros:
        # find indices where we do not have the pattern that corresponds to 0
        time_nonzero = np.nonzero(np.sum(act_mat + tar_pat_seq + src_pat_seq, axis=0))[
            0
        ]
    print(len(time_nonzero))

    print("finding probabilities ...")
    neurons = np.arange(0, N)
    norm = 0
    z = 0
    for t_step in tqdm(range(past_steps, T)):
        for tar in neurons:
            # bool array, true for all indices but tar
            src = ~(neurons == tar)
            P[
                tar,
                src,
                act_mat[tar, t_step],
                tar_pat_seq[tar, t_step],
                src_pat_seq[src, t_step],
            ] += 1
            norm += N

    # the normalization diff is from T-past_steps in jonas code not considering
    # that we make more than one entry per timestep. we make N*N entries!

    # normalization seems to only change the absolute value of the TE but all
    # probabilities need to be <= 1

    # print(P[0,1])
    # P /= (T-past_steps)
    P /= norm
    print("norm", norm, (T - past_steps))

    # 0 target index
    # 1 source index
    # 2 target state now
    # 3 target pattern past
    # 4 source pattern past

    # calculate the transfer entropy from P (in bits)
    print("summing up ...")
    # fmt: off
    TE = -np.sum(
        np.sum(P, axis=4) * np.nan_to_num(np.log2(
            np.sum(P, axis=4)[:, :, :, :]
            / np.sum(P, axis=(2, 4))[:, :, None, :])),
        axis=(2, 3),
    )

    TE = TE \
        + np.sum(P * np.nan_to_num(np.log2(
            P / np.sum(P, axis=2)[:, :, None, :, :])),
        axis=(2, 3, 4),
    )

    # fmt: on

    return TE


# mtx = spiketimes_to_act_mat(spiketimes, 100)
# plt.matshow(mtx)


def burst_times_to_te_time_series(beg_times, end_times, bin_size, length=None):
    """
        Create an activity matrix (as needed for TE estiamtion) for module-level
        burst times.

        # Paramters
            beg_times : list of arrays, times of burst beginning, 1 array per module, in seconds
            end_times : list of arrays, end times
            bin_size : time step to use for the timeseries, in seconds
            length   : series duration, number of bins of bin_size
    """
    if length is None:
        length = 0
        for train in end_times:
            l = np.max(train)
            if l > length:
                length = l
        length = np.ceil(length / bin_size) + 1
    length = int(np.ceil(length))

    num_mods = len(beg_times)
    res = np.zeros(shape=(num_mods, length), dtype=np.int8)

    for mod in range(num_mods):
        for tdx in range(len(beg_times[mod])):
            end = int(end_times[mod][tdx] / bin_size)
            beg = int(beg_times[mod][tdx] / bin_size)
            res[mod][beg:end] = 1

    return res


act_mat = burst_times_to_te_time_series(
    beg_times, end_times, bin_size=0.05, length=sim_duration / 0.05
)
te = transfer_entropy_paul(act_mat)
