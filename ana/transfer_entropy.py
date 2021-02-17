# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-17 18:52:17
# @Last Modified: 2021-02-17 18:59:50
# ------------------------------------------------------------------------------ #
# helper to calculate transfer entropy, on module level
# ------------------------------------------------------------------------------ #

# %%

import logging
import warnings

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

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


def transfer_entropy(act_mat, skip_zeros=True, use_numba=True):
    """
        # returns the transfer entropy matrix TE[i,j] = TE from neuron j to i in basis 2.
        # Runtime scales with number of neurons ** 2

        # Parameters
            act_mat : 2d np array,
                activity matrix
                1st dimension: `N` neuron number, 2nd dimension: `T` timebins.
            skip_zeros : bool
                whether to skip 0 timebins, gives a speedup, especially for
                trains that are mostly empty
            use_numba : bool
                set to false to avoid using numba parallelisation


        # Returns
            TE : 2d np array
                 size `N` * `N`
    """
    global nz, tar_pat_seq, src_pat_seq
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

    # convert timeseries into a sequence of patterns for every neuron
    # keep full duration so one can iterate over all timesteps
    src_pat_seq = np.zeros(shape=(N, T), dtype="int8")
    tar_pat_seq = np.zeros(shape=(N, T), dtype="int8")

    # all neurons, all timesteps minus pattern length
    # code patterns as a number: S**markov_order possible values for each pattern
    log.debug("finding patterns ...")

    # target
    # build pattern, start with now -1 bin. we have no past for the zeroth bin.
    dt = 1
    tar_pat_seq[:, dt:] += act_mat[:, : T - dt] * S ** (0)

    # go further into the past and add to pattern
    for pat, dt in enumerate(range(2, tar_mo + 1)):
        tar_pat_seq[:, dt:-tar_mo] += act_mat[:, : -tar_mo - dt] * S ** (pat + 1)

    # repeat for source pattern, here we might want to use the present bin as
    # the most recent one to contribute to the pattern (IFT = 1)
    dt = 2 - IFT
    src_pat_seq[:, dt:] += act_mat[:, : T - dt] * S ** (0)
    for pat, dt in enumerate(range(2 - IFT, src_mo + 1 - IFT)):
        src_pat_seq[:, dt:-src_mo] += act_mat[:, : -src_mo - dt] * S ** (pat + 1)

    log.debug("finding probabilities ...")

    if skip_zeros:
        # find times where we have meaningful patterns (not everything 0)
        # exclude first few bins, there we do not have patterns
        nz = np.sum(act_mat + tar_pat_seq + src_pat_seq, axis=0)
        nz = np.nonzero(nz)[0]
        nz = nz[nz >= past_steps]
        log.debug(f"non-zero time indices: {len(nz)}")
        t_steps_to_consider = nz
    else:
        t_steps_to_consider = np.arange(past_steps, T)

    norm = 0
    neurons = np.arange(0, N)

    # high dimensional probability
    # [target index, source index, target state now, target pattern past, source pattern past]
    P = np.zeros(shape=(N, N, S, S ** tar_mo, S ** src_mo), dtype=np.uint64)

    if not use_numba:
        for t_step in tqdm(t_steps_to_consider):
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
                norm += N - 1
    else:
        P = _prob_add_inner(P, act_mat, tar_pat_seq, src_pat_seq, t_steps_to_consider)
        norm = np.sum(P)

    if skip_zeros:
        # fill up probability of skipped timesteps
        num_skipped_steps = (T - past_steps) - len(t_steps_to_consider)
        for tar in neurons:
            for src in neurons:
                if src != tar:
                    P[tar, src, 0, 0, 0] += num_skipped_steps
                    norm += num_skipped_steps

    # log.debug(f"norm {np.sum(P)} {norm}")
    # log.debug(f"steps {len(t_steps_to_consider)} {num_skipped_steps} {(T - past_steps)}")

    # normalization without excluding steps should be:
    # assert norm == (T - past_steps) * N * (N - 1)
    P = P.astype(np.float64) / norm

    # 0 target index
    # 1 source index
    # 2 target state now
    # 3 target pattern past
    # 4 source pattern past

    # calculate the transfer entropy from P (in bits)
    log.debug("summing up ...")
    # fmt: off
    with warnings.catch_warnings(): # silence numpy
        warnings.simplefilter("ignore")

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


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _prob_add_inner(P, act_mat, tar_pat_seq, src_pat_seq, t_steps_to_consider):
    P = np.zeros(shape=P.shape, dtype=np.uint64)
    neurons = np.arange(0, act_mat.shape[0])
    for tar in neurons:
        for src in neurons:
            if src != tar:
                # do not prange this!
                for tdx in range(len(t_steps_to_consider)):
                    t_step = t_steps_to_consider[tdx]
                    P[
                        tar,
                        src,
                        act_mat[tar, t_step],
                        tar_pat_seq[tar, t_step],
                        src_pat_seq[src, t_step],
                    ] += 1.0
    return P


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

