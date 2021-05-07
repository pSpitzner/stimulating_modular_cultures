# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-03-10 13:23:16
# @Last Modified: 2021-05-07 14:09:57
# ------------------------------------------------------------------------------ #


import os
import sys
import glob
import h5py
import re
import numbers
import numpy as np
import pandas as pd

import hi5 as h5
from hi5 import BetterDict
from tqdm import tqdm
from itertools import permutations

import logging
import warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from numba import jit, prange
    # raise ImportError
    log.info("Using numba for parallelizable functions")

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


def prepare_file(h5f, mod_colors="auto", hot=True):
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
        h5f = h5.recursive_load(h5f, hot=hot)

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
    elif mod_colors == "auto":
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
    try:
        spikes = h5f.data.spiketimes[:]
        if spikes is None:
            raise ValueError
        spikes[spikes == 0] = np.nan
        h5f.data.spiketimes = spikes
    except:
        log.info("No spikes in file, plotting and analysing dynamics will not work.")

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

    # Stimulation description
    stim_str = "Unknown"
    if h5f.data.stimulation_times_as_list is None:
        stim_str = "Off"
    else:
        try:
            stim_neurons = np.unique(h5f.data.stimulation_times_as_list[:, 0]).astype(
                int
            )
            stim_mods = np.unique(h5f.data.neuron_module_id[stim_neurons])
            stim_str = f"On {str(tuple(stim_mods)).replace(',)', ')')}"
        except:
            stim_str = f"Error"
    h5f.ana.stimulation_description = stim_str

    # Guess the repetition from filename, convention: `foo/bar_parameters_rep=09.hdf5`
    try:
        fname = str(h5f.uname.original_file_path.decode("UTF-8"))
        rep = re.search("(?<=rep=)(\d+)", fname)[0]  # we only use the first match
        h5f.ana.repetition = int(rep)
    except Exception as e:
        log.debug(e)
        h5f.ana.repetition = -1

    return h5f


def find_bursts_from_rates(
    h5f,
    bs_large=0.02,  # seconds, time bin size to smooth over (gaussian kernel)
    bs_small=0.0005,  # seconds, small bin size
    rate_threshold=7.5,  # Hz
    merge_threshold=0.1,  # seconds, merge bursts if separated by less than this
    write_to_h5f=True,
):
    """
        Based on module-level firing rates, find bursting events.

        returns two BetterDicts, `bursts` and `rates`,
        modifies `h5f`

        Note on smoothing: at the moment, we time-bin the activity on the module level
        and convolute this series with a gaussian kernel to smooth.
        More precise way would be to convolute the spike-train of each neuron with
        the kernel (thus, keeping the high precision of each spike time).
    """

    assert h5f.ana is not None, "`prepare_file(h5f)` first!"

    spikes = h5f.data.spiketimes

    bursts = BetterDict()
    bursts.module_level = BetterDict()
    rates = BetterDict()
    rates.dt = bs_small
    rates.module_level = BetterDict()
    rates.system_level = None # just create the dict entry at the nice position
    rates.cv = BetterDict()
    rates.cv.module_level = BetterDict()
    rates.cv.system_level = None

    beg_times = []  # lists of length num_modules
    end_times = []

    for m_id in h5f.ana.mods:
        selects = np.where(h5f.data.neuron_module_id[:] == m_id)[0]
        pop_rate = population_rate_exact_smoothing(
            spikes[selects],
            bin_size=bs_small,
            smooth_width=bs_large,
            length=h5f.meta.dynamics_simulation_duration,
        )
        # pop_rate = population_rate(
        #     spikes[selects],
        #     bin_size=bs_small,
        #     length=h5f.meta.dynamics_simulation_duration,
        # )
        # pop_rate = smooth_rate(pop_rate, clock_dt=bs_small, width=bs_large)
        # pop_rate = pop_rate / bs_small

        beg_time, end_time = burst_detection_pop_rate(
            rate=pop_rate, bin_size=bs_small, rate_threshold=rate_threshold,
        )

        if len(beg_time) > 0:
            beg_time, end_time = merge_if_below_separation_threshold(
                beg_time, end_time, threshold=merge_threshold
            )

        beg_times.append(beg_time)
        end_times.append(end_time)

        rates.module_level[m_id] = pop_rate
        rates.cv.module_level[m_id] = np.nanstd(pop_rate) / np.nanmean(pop_rate)
        bursts.module_level[m_id] = BetterDict()
        bursts.module_level[m_id].beg_times = beg_time.copy()
        bursts.module_level[m_id].end_times = end_time.copy()
        bursts.module_level[m_id].rate_threshold = rate_threshold

    pop_rate = population_rate_exact_smoothing(
        spikes[:],
        bin_size=bs_small,
        smooth_width=bs_large,
        length=h5f.meta.dynamics_simulation_duration,
    )
    rates.system_level = pop_rate
    rates.cv.system_level = np.nanstd(pop_rate) / np.nanmean(pop_rate)

    all_begs, all_ends, all_seqs = system_burst_from_module_burst(
        beg_times, end_times, threshold=merge_threshold,
    )

    bursts.system_level = BetterDict()
    bursts.system_level.beg_times = all_begs.copy()
    bursts.system_level.end_times = all_ends.copy()
    bursts.system_level.module_sequences = all_seqs

    if write_to_h5f:
        if isinstance(h5f.ana.bursts, BetterDict):
            h5f.ana.bursts.clear()
        if isinstance(h5f.ana.rates, BetterDict):
            h5f.ana.rates.clear()
        # so, overwriting keys with dicts (nesting) can cause memory leaks.
        # to avoid this, call .clear() before assigning the new dict
        # testwise I made this the default for setting keys of BetterDict
        h5f.ana.bursts = bursts
        h5f.ana.rates = rates

    return bursts, rates


def find_isis(h5f, write_to_h5f=True):
    """
        What are the the inter-spike-intervals within and out of bursts?
    """

    isi = BetterDict()

    for idx, m_id in enumerate(h5f.ana.mods):
        selects = np.where(h5f.data.neuron_module_id[:] == m_id)[0]
        spikes_2d = h5f.data.spiketimes[selects]
        try:
            b = h5f.ana.bursts.module_level[m_id].beg_times
            e = h5f.ana.bursts.module_level[m_id].end_times
        except:
            if idx == 0:
                log.info(
                    "Bursts were not detected before searching ISI. Try `find_bursts_from_rates()`"
                )
            b = None
            e = None

        ll_isi = _inter_spike_intervals(spikes_2d, beg_times=b, end_times=e,)
        isi[m_id] = ll_isi

    if write_to_h5f:
        h5f.ana.isi = isi

    return isi


# ------------------------------------------------------------------------------ #
# batch processing across realization
# ------------------------------------------------------------------------------ #


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
            list_of_sequences=h5f.ana.bursts.system_level.module_sequences,
            mods=h5f.ana.mods,
        )

        # fetch meta data for every row
        stim = h5f.ana.stimulation_description
        bridge_weight = h5f.meta.dynamics_bridge_weight
        if bridge_weight is None:
            bridge_weight = 1.0
        num_connections = h5f.meta.topology_k_inter

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


def batch_pd_bursts(load_from_disk=True, list_of_filenames=None, df_path=None):
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

    if df_path is None:
        df_path = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/inhibition/pd/bursts.hdf5"

    if load_from_disk:
        try:
            df = pd.read_hdf(df_path, "/data/df")
            return df
        except Exception as e:
            log.info("Could not load from disk, (re-)processing data")
            log.debug(e)

    columns = [
        "Duration",
        "Sequence length",
        "First module",
        "Stimulation",
        "Connections",
        "Bridge weight",
        "Number of inhibitory neurons",
        "Repetition",
    ]
    df = pd.DataFrame(columns=columns)

    candidates = [glob.glob(f) for f in list_of_filenames]
    candidates = [item for sublist in candidates for item in sublist]  # flatten

    assert len(candidates) > 0, f"Are the filenames correct?"

    for candidate in tqdm(candidates, desc="Burst duration for files"):
        h5f = prepare_file(candidate, hot=False)

        # fetch meta data for every repetition (applied to multiple rows)
        stim = h5f.ana.stimulation_description
        rep = h5f.ana.repetition
        bridge_weight = h5f.meta.dynamics_bridge_weight
        if bridge_weight is None:
            bridge_weight = 1.0
        num_connections = h5f.meta.topology_k_inter
        try:
            num_inhibitory = len(h5f.data.neuron_inhibitory_ids[:])
        except Exception as e:
            log.debug(e)
            # maybe its a single number, instead of a list
            if isinstance(h5f.data.neuron_inhibitory_ids, numbers.Number):
                num_inhibitory = 1
            else:
                num_inhibitory = 0

        # do the analysis, entries are directly added to the h5f
        # for the system with inhibition, we might need a lower threshold (Hz)
        if num_inhibitory > 0:
            find_bursts_from_rates(h5f, rate_threshold=7.5)
        else:
            find_bursts_from_rates(h5f, rate_threshold=7.5)

        data = h5f.ana.bursts.system_level
        for idx in range(0, len(data.beg_times)):
            duration = data.end_times[idx] - data.beg_times[idx]
            seq_len = len(data.module_sequences[idx])
            first_mod = data.module_sequences[idx][0]
            df = df.append(
                pd.DataFrame(
                    data=[
                        [
                            duration,
                            seq_len,
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

    try:
        df.to_hdf(df_path, "/data/df")
    except Exception as e:
        log.debug(e)

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
        h5f = h5.recursive_load(candidate, hot=hot)
        prepare_file(h5f)
        find_bursts_from_rates(h5f)
        find_isis(h5f)

        this_burst = h5f.ana.bursts
        this_isi = h5f.ana.isi

        if cdx == 0:
            res = h5f
            mods = h5f.ana.mods

        # todo: consistency checks
        # lets at least check that the modules are consistent across candidates.
        assert np.all(h5f.ana.mods == mods), "Modules differ between files"

        # copy over system level burst
        b = res.ana.bursts.system_level
        b.beg_times.extend(this_burst.system_level.beg_times)
        b.end_times.extend(this_burst.system_level.end_times)
        b.module_sequences.extend(this_burst.system_level.module_sequences)
        for m_id in h5f.ana.mods:
            # copy over module level bursts
            b = res.ana.bursts.module_level[m_id]
            b.beg_times.extend(this_burst.module_level[m_id].beg_times)
            b.end_times.extend(this_burst.module_level[m_id].end_times)

            # and isis
            i = res.ana.isi[m_id]
            for var in ["all", "in_bursts", "out_bursts"]:
                i[var].extend(this_isi[m_id][var])

        if hot:
            # only close the last file (which we opened), and let's hope no other file
            # was opened in the meantime
            # h5.close_hot(which=-1)
            try:
                h5f.h5_file.close()
            except:
                log.debug("Failed to close file")

    return res


def batch_isi_across_conditions():

    stat = BetterDict()
    conds = _conditions()
    for k in tqdm(conds.varnames, desc="k values", position=0, leave=False):
        stat[k] = BetterDict()
        for stim in tqdm(
            conds[k].varnames, desc="stimulation targets", position=1, leave=False
        ):
            h5f = process_candidates_burst_times_and_isi(conds[k][stim])
            # preprocess so that plot functions wont do it again.
            # todo: make api consistent
            h5f.ana.ensemble = BetterDict()
            h5f.ana.ensemble.filenames = conds[k][stim]
            h5f.ana.ensemble.bursts = h5f.ana.bursts
            h5f.ana.ensemble.isi = h5f.ana.isi

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
            stat[k][stim] = BetterDict()
            for m in h5f.ana.ensemble.isi.varnames:
                try:
                    stat[k][stim][m] = np.mean(h5f.ana.ensemble.isi[m].in_bursts)
                except Exception as e:
                    log.debug(e)

            del h5f
            h5.close_hot()

    return stat


def batch_conditions():
    # fmt:off
    path_base = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/"
    stim = BetterDict()
    for k in [0,1,2,3,5]:
        stim[k] = BetterDict()
        stim[k].off = f"{path_base}/dyn/2x2_fixed/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k}_rep=*.hdf5"
        for s in ["0", "02", "012", "0123"]:
            stim[k][s] = f"{path_base}/jitter_{s}/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k}_rep=*.hdf5"

    return stim
    # fmt:on


# ------------------------------------------------------------------------------ #
# lower level
# ------------------------------------------------------------------------------ #

# turns out this is faster without numba
def _inter_spike_intervals(spikes_2d, beg_times=None, end_times=None):
    """
        Returns three list, thre first one contains all interspike intervals,
        merged down for all neurons in `spikes_2d`.
        If `beg_times` and `end_times` are passed, returns two more lists
        with the isis inside and out of bursts. otherwise, they are empty.
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

    return BetterDict(
        all=isis_all,
        in_bursts=isis_in,
        out_bursts=isis_out,
        cv_all=cv_all,
        cv_in_bursts=cv_in_bursts,
        cv_out_bursts=cv_out_bursts,
    )


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
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


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
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

    if len(beg_time) == 0:
        return beg_res, end_res

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
# sequences
# ------------------------------------------------------------------------------ #


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
    lookup = dict()
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
