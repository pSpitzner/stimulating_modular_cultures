# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-02-20 09:35:48
# @Last Modified: 2020-11-26 15:31:51
# ------------------------------------------------------------------------------ #
# Dynamics described in Orlandi et al. 2013, DOI: 10.1038/nphys2686
# Loads topology from hdf5 or csv and runs the simulations in brian.
#
# Trying to find parameters that create inter-burst-intervals as in
# Yamamoto et al. 2018, DOI: 10.1126/sciadv.aau4914
#
# 15 Bursts in 20 min; ibi ~ 80s for merged cultures
# 10 Bursts in 20 min; ibi ~ 120s for single-bond cultures
#
# in science advance we have 15+-4 and 9+-3 bursts in 20 min!
# 46 | 11
#
# Also, F/I curves would be nice.
#
# We want to set/get more biologically plausible values in the modules:
# ampa in vivo <= 1mV, in vitro <= 10mV
# Bursts do not occur more frequent than every (5-)10s,
# with around 10 spikes per neuron
# ------------------------------------------------------------------------------ #

import h5py
import argparse
import os
import tempfile
import sys
import shutil
import numpy as np
from brian2 import *

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

# we want to run this on a cluster, assign a custom cache directory to each thread
# putting this into a user-directory that gets backed-up turns out to be a bad idea.
# but shared dirs across users may cause trouble due to access-rights, defaults: 755
# cache_dir = os.path.expanduser(f"~/.cython/brian-pid-{os.getpid()}")
cache_dir = f"{tempfile.gettempdir()}/cython/brian-pid-{os.getpid()}"
prefs.codegen.runtime.cython.cache_dir = cache_dir
prefs.codegen.runtime.cython.multiprocess_safe = False

# Log level needs to be set in ~/.brian/user_preferences to work for all steps
prefs.logging.console_log_level = "INFO"

# we want enforce simulation with c
prefs.codegen.target = "cython"

from shutil import copy2

# ------------------------------------------------------------------------------ #
# model parameters
# ------------------------------------------------------------------------------ #

# fmt: off
# membrane potentials
vr = -60 * mV  # resting potential, neuron relaxes towards this without stimulation
vt = -45 * mV  # threshold potential
vp =  35 * mV  # peak potential, after vt is passed, rapid growth towards this
vc = -50 * mV  # reset potential

# soma
tc = 50 * ms  # time scale of membrane potential
ta = 50 * ms  # time scale of inhibitory current u

k = 0.5 / mV  # resistance over capacity(?), rescaled
b = 0.5       # sensitivity to sub-threshold fluctuations
d =  50 * mV  # after-spike reset of inhibitory current u

# synapse
tD =   1 * second  # characteristic recovery time, between 0.5 and 20 seconds
tA =  10 * ms      # decay time of post-synaptic current (AMPA current decay time)
gA =  50 * mV      # AMPA current strength, between 10 - 50 mV
                   # 170.612 value in javiers neurondyn
                   # this needs to scale with tc/tA

# noise
beta = 0.8         # D = beta*D after spike, to reduce efficacy, beta < 1
rate = 30 * Hz     # rate for the poisson input (shot-noise), between 10 - 50 Hz
gm =  25 * mV      # shot noise (minis) strength, between 10 - 50 mV
                   # (sum of minis arriving at target neuron)
gs = 300 * mV * mV * ms * ms  # white noise strength, via xi = dt**.5 * randn()


# stimulation
stim_interval = 400 * ms

# fmt:on

# integration step size
# this turns out to be quite crucial for synchonization:
# when too large (brian defaul 0.1ms) this forces sth like an integer cast at
# some point and may promote synchronized firing. (spike times are not precise enough)
# heuristically: do not go below 0.05 ms, better 0.01ms
defaultclock.dt = 0.05 * ms

# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def h5_load(filename, dsetname, raise_ex=True):
    try:
        file = h5py.File(filename, "r")
        res = file[dsetname][:]
        file.close()
        return res
    except Exception as e:
        print(f"failed to load {dsetname} from {filename}")
        if raise_ex:
            raise e
        else:
            return np.nan


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), "ok", ms=10)
    plot(ones(Nt), arange(Nt), "ok", ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], mod_sort([i, j]), "-k", lw=0.1)
    xticks([0, 1], ["Source", "Target"])
    ylabel("Neuron index")
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(mod_sort(S.i), mod_sort(S.j), "ok")
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel("Source neuron index")
    ylabel("Target neuron index")


def toy_topology(k_intra=8, k_inter=10):
    n_per_mod = 25
    # k_intra = 12 # per neuron
    # k_inter = 10 # per module
    mod_size = 200
    mods = []
    for mdx in range(0, 4):
        mod = dict()
        mod["pos_x"] = np.random.rand(n_per_mod) * mod_size
        mod["pos_y"] = np.random.rand(n_per_mod) * mod_size
        mod["n_id"] = np.arange(0, n_per_mod) + mdx * n_per_mod
        mods.append(mod)
    mods[1]["pos_x"] += 400
    mods[2]["pos_y"] += 400
    mods[3]["pos_x"] += 400
    mods[3]["pos_y"] += 400
    num_n = len(mods) * n_per_mod
    a_ij = np.zeros(shape=(num_n, num_n), dtype=int)
    counted_intra = 0
    counted_inter = 0
    # within modules
    for mdx in range(0, len(mods)):
        for idx in range(0, n_per_mod):
            i = mods[mdx]["n_id"][idx]
            k = 0
            while k < k_intra:
                j = i
                while j == i:
                    jdx = np.random.randint(0, n_per_mod)
                    j = mods[mdx]["n_id"][jdx]
                    if a_ij[i, j] == 1:
                        j = i
                k += 1
                counted_intra += 1
                a_ij[i, j] = 1
    # between modules
    for mod_pair in [[0, 1], [0, 2], [2, 3], [1, 3]]:
        mod_a = mods[mod_pair[0]]
        mod_b = mods[mod_pair[1]]
        k = 0
        while k < k_inter:
            foo = np.random.randint(0, n_per_mod)
            bar = np.random.randint(0, n_per_mod)
            if k < k_inter / 2:
                i = mod_a["n_id"][foo]
                j = mod_b["n_id"][bar]
            else:
                i = mod_b["n_id"][foo]
                j = mod_a["n_id"][bar]
            if a_ij[i, j] == 0:
                k += 1
                counted_inter += 1
                a_ij[i, j] = 1

    print(f"Connections between modules: {counted_inter}")
    print(f"Connections within modules:  {counted_intra}")
    print(f"Ratio:                       {counted_inter/counted_intra*100 :.2f}%")

    return mods, a_ij


# helper function to convert a list of time stamps
# into a (binned) time series of activity
def bin_spike_times(spike_times, bin_size, pad_right=0):
    if len(spike_times) == 0:
        return np.array([])
    last_spike = spike_times[-1]
    num_bins = int(np.ceil(last_spike / bin_size))
    num_bins += int(pad_right)
    res = np.zeros(num_bins)
    for spike_time in spike_times:
        target_bin = int(np.floor(spike_time / bin_size))
        res[target_bin] = res[target_bin] + 1
    return res


def burst_times(spks_m, debug=False):
    # Let's detect bursts
    # "Network bursts provided a quick insight into the collective dynamics in cultures
    # and were defined as those activity episodes in which more than 75% of the neurons
    # fired simultaneously for more than 500 ms."

    win_size = 250 * ms
    bin_size = 50 * ms

    trains = spks_m.spike_trains()
    num_n = len(trains.keys())
    binned_trains = []
    tmax = 0
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
    bursting_bins = np.where(summed_series >= 0.75 * num_n)[0]

    x = bursting_bins
    bursts = np.delete(x, [np.where(x[:-1] == x[1:] - 1)[0] + 1]) * bin_size

    if not debug:
        return bursts
    else:
        return bursts, time_series, summed_series, window


# ------------------------------------------------------------------------------ #
# stimulation
# recreate the stimulation used by hideaki.
#     * 400ms time windows
#     * 10 candidate neurons across two modules
#     * per time window p=0.4 for every candidate
# ------------------------------------------------------------------------------ #


def stimulation_pattern_candidates(candidates, p_per_candidate=0.4):
    """
        create a random pattern from given candidates.
        the stimulation_pattern_random() can be used once to get candidates.

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


def stimulation_pattern_random(n_per_mod=5, mod_targets=[0]):
    """
        produces a random pattern for one time window

        Parameters
        ----------
        n_per_mod : int
            number of candidates per module

        mod_targets : list of int
            which targets to pick candidates

        Returns
        -------
        neuron_ids: list of int
            the neuron ids to target
    """
    global mod_ids

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


# ------------------------------------------------------------------------------ #
# command line arguments
# ------------------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Brian")
parser.add_argument("-i", dest="input_path", help="input path", metavar="FILE")
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
parser.add_argument("-s", dest="seed", default=117, help="rng", type=int)
parser.add_argument("-N", dest="num_n", default=-1, type=int)
parser.add_argument(
    "-d", dest="duration", default=20 * 60, help="in seconds", type=float
)
parser.add_argument(
    "-equil",
    "--equilibrate",
    dest="equil_duration",
    default=2 * 60,
    help="in seconds",
    type=float,
)
parser.add_argument(
    "-stim",
    "--stimulate",
    dest="enable_stimulation",
    default=False,
    action="store_true",
)
parser.add_argument("-gA", dest="gA", default=gA / mV, help="in mV", type=float)
parser.add_argument("-gm", dest="gm", default=gm / mV, help="in mV", type=float)
parser.add_argument("-r", dest="r", default=rate / Hz, help="in Hz", type=float)
parser.add_argument(
    "-tD", dest="tD", default=tD / second, help="in seconds", type=float
)
args = parser.parse_args()

numpy.random.seed(args.seed)
gA = args.gA * mV
gm = args.gm * mV
tD = args.tD * second
rate = args.r * Hz

print(f'#{"":#^75}#\n#{"running dynamics in brian":^75}#\n#{"":#^75}#')
print(f"input topology: ", args.input_path)
print(f"seed: ", args.seed)
print(f"gA: ", gA)
print(f"gm: ", gm)
print(f"tD: ", tD)
print(f"rate: ", rate)
print(f"stimulation: ", args.enable_stimulation)

try:
    # load from hdf5
    print("Loading connectivity from hdf5 ... ")
    try:
        a_ij = None
        a_ij_sparse = h5_load(args.input_path, "/data/connectivity_matrix_sparse")
        a_ij_sparse = a_ij_sparse.astype(int, copy=False)  # brian doesnt like uints
    except:
        a_ij_sparse = None
        a_ij = h5_load(args.input_path, "/data/connectivity_matrix")
    num_n = int(h5_load(args.input_path, "/meta/topology_num_neur"))
    # get the neurons sorted according to their modules
    mod_ids = h5_load(args.input_path, "/data/neuron_module_id")
    mod_sorted = np.zeros(num_n, dtype=int)
    temp = np.argsort(mod_ids)
    for idx in range(0, num_n):
        mod_sorted[idx] = np.argwhere(temp == idx)

    mod_sort = lambda x: mod_sorted[x]
except:
    mod_sort = lambda x: x
    # or a csv
    try:
        a_ij_sparse = None
        a_ij = loadtxt(args.input_path)
        num_n = a_ij.shape[0]
    except:
        print("Unable to load toplogy from {args.input_path}")
        a_ij = None
        num_n = args.num_n
        assert num_n > 0


# ------------------------------------------------------------------------------ #
# model
# ------------------------------------------------------------------------------ #

G = NeuronGroup(
    N=num_n,
    model="""
        dv/dt = ( k*(v-vr)*(v-vt) -u +I                     # [6] soma potential
                  +xi*(gs/tc)**0.5      )/tc   : volt       # white noise term
        dI/dt = -I/tA                          : volt       # [9, 10]
        du/dt = ( b*(v-vr) -u )/ta             : volt       # [7] inhibitory current
        dD/dt = ( 1-D)/tD                      : 1          # [11] recovery to one
    """,
    threshold="v > vp",
    reset="""
        v = vc           # [8]
        u = u + d        # [8]
        D = D * beta     # [11] delta-function term on spike
    """,
    method="euler",
    dt=defaultclock.dt,
)

S = Synapses(
    source=G,
    target=G,
    on_pre="""
        I_post += D_pre * gA    # [10]
    """,
)

# shot-noise:
# by targeting I with poisson, we should get pretty close to javiers version.
# rates = 0.01 / ms + (0.04 / ms)* rand(num_n) # we could have differen rates
# mini_g = PoissonGroup(num_n, rate)
# mini_s = Synapses(mini_g, G, on_pre="I_post+=gm", order=-1, name="Minis")
# mini_s.connect(j="i")

# treat minis as spikes, add directly to current
# for homogeneous rates, this is faster. here, N=1 is the input per neuron
mini_g = PoissonInput(target=G, target_var="I", N=1, rate=rate, weight=gm)

# connect synapses from loaded matrix or randomly
try:
    assert a_ij is not None or a_ij_sparse is not None
    if a_ij_sparse is None:
        print("Applying connectivity (non-sparse) ... ")
        pre, post = np.where(a_ij == 1)
        for idx, i in enumerate(pre):
            j = post[idx]
            S.connect(i=i, j=j)
    else:
        print("Applying connectivity (sparse) ... ")
        S.connect(i=a_ij_sparse[:, 0], j=a_ij_sparse[:, 1])
except Exception as e:
    print(e)
    print(f"Creating Synapses randomly.")
    S.connect(condition="i != j", p=0.01)

# initalize to a somewhat sensible state. we could have different neuron types
G.v = "vc + 5*mV*rand()"

# ------------------------------------------------------------------------------ #
# Stimulation if requested
# ------------------------------------------------------------------------------ #

if args.enable_stimulation:

    candidates = stimulation_pattern_random(n_per_mod=5, mod_targets=[0])
    candidates = np.array(candidates)
    stimulus_indices = []
    stimulus_times = []
    for step in range(
        1, int((args.duration + args.equil_duration) * second / stim_interval) - 1
    ):
        t = step * stim_interval
        n = stimulation_pattern_candidates(candidates=candidates).tolist()
        stimulus_indices += n
        stimulus_times += [t] * len(n)

    stim_g = SpikeGeneratorGroup(
        N=num_n,
        indices=stimulus_indices,
        times=stimulus_times,
        name="create_stimulation",
    )
    # because we project via artificial synapses, we get a delay of
    # approx (!) one timestep between the stimulation and the spike
    stim_s = Synapses(stim_g, G, on_pre="v_post = 2*vp", name="apply_stimulation",)
    stim_s.connect(condition="i == j")

# ------------------------------------------------------------------------------ #
# Running
# ------------------------------------------------------------------------------ #

# equilibrate
run(args.equil_duration * second, report="stdout", report_period=1 * 60 * second)

# disable state monitors that are not needed for production
# stat_m = StateMonitor(G, ["v", "I", "u", "D"], record=True)
spks_m = SpikeMonitor(G)
rate_m = PopulationRateMonitor(G)
# mini_m = SpikeMonitor(mini_g)

if args.enable_stimulation:
    stim_m = SpikeMonitor(stim_g)

run(args.duration * second, report="stdout", report_period=1 * 60 * second)


# ------------------------------------------------------------------------------ #
# Writing
# ------------------------------------------------------------------------------ #

print(f'#{"":#^75}#\n#{"saving to disk":^75}#\n#{"":#^75}#')

try:
    # make sure directory exists
    outdir = os.path.abspath(os.path.expanduser(args.output_path + "/../"))
    os.makedirs(outdir, exist_ok=True)
    copy2(args.input_path, args.output_path)
except Exception as e:
    print("Could not copy input file\n", e)

try:
    if args.output_path is None:
        raise ValueError
    f = h5py.File(args.output_path, "a")

    def convert_brian_spikes_to_pauls(spks_m):
        trains = spks_m.spike_trains()
        tmax = 0
        for tdx in trains.keys():
            if len(trains[tdx]) > tmax:
                tmax = len(trains[tdx])
        spiketimes = np.zeros(shape=(num_n, tmax))
        spiketimes_as_list = np.zeros(shape=(2, spks_m.num_spikes))
        last_idx = 0
        for n in range(0, num_n):
            t = trains[n]
            spiketimes[n, 0 : len(t)] = t / second - args.equil_duration
            spiketimes_as_list[0, last_idx : last_idx + len(t)] = [n] * len(t)
            spiketimes_as_list[1, last_idx : last_idx + len(t)] = (
                t / second - args.equil_duration
            )
            last_idx += len(t)
        return spiketimes, spiketimes_as_list.T

    # normal spikes, no stim in two different formats
    spks, spks_as_list = convert_brian_spikes_to_pauls(spks_m)

    dset = f.create_dataset("/data/spiketimes", data=spks)
    dset.attrs[
        "description"
    ] = "2d array of spiketimes, neuron x spiketime in seconds, zero-padded"

    dset = f.create_dataset("/data/spiketimes_as_list", data=spks_as_list)
    dset.attrs[
        "description"
    ] = "two-column list of spiketimes. first col is neuron id, second col the spiketime. effectively same data as in '/data/spiketimes'. neuron id will need casting to int for indexing."

    if args.enable_stimulation:
        # normal spikes, no stim in two different formats
        stim, stim_as_list = convert_brian_spikes_to_pauls(stim_m)

        dset = f.create_dataset("/data/stimulation_times_as_list", data=stim_as_list)
        dset.attrs[
            "description"
        ] = "two-column list of stimulation times. first col is target-neuron id, second col the stimulation time. Beware: we have approximateley one timestep delay between stimulation and spike."

    # rates
    # at the default timestep, the data files get huge.
    freq = int(50 * ms / defaultclock.dt)
    # dset = f.create_dataset(
    #     "/data/population_rate",
    #     data=np.array(
    #         [rate_m.t / second - args.equil_duration , rate_m.rate / Hz]
    #     ).T,
    # )
    # dset.attrs[
    #     "description"
    # ] = "2d array of the population rate in Hz, first dim is time in seconds"

    dset = f.create_dataset(
        "/data/population_rate_smoothed",
        data=np.array(
            [
                (rate_m.t / second - args.equil_duration)[::freq],
                (rate_m.smooth_rate(window="gaussian", width=50 * ms) / Hz)[::freq],
            ]
        ).T,
    )
    dset.attrs[
        "description"
    ] = "population rate in Hz, smoothed with gaussian kernel of 50ms width, first dim is time in seconds"


    # meta data of this simulation
    dset = f.create_dataset("/meta/dynamics_gA", data=gA / mV)
    dset.attrs["description"] = "AMPA current strength, in mV"

    dset = f.create_dataset("/meta/dynamics_gm", data=gm / mV)
    dset.attrs["description"] = "shot noise (minis) strength, in mV"

    dset = f.create_dataset("/meta/dynamics_tD", data=tD / second)
    dset.attrs["description"] = "characteristic decay time, in seconds"

    dset = f.create_dataset("/meta/dynamics_rate", data=rate / Hz)
    dset.attrs[
        "description"
    ] = "rate for the (global) poisson input (shot-noise), in Hz"

    dset = f.create_dataset("/meta/dynamics_simulation_duration", data=args.duration)
    dset.attrs["description"] = "in seconds"

    dset = f.create_dataset(
        "/meta/dynamics_equilibration_duration", data=args.equil_duration
    )
    dset.attrs["description"] = "in seconds"

    f.close()

    print(f'#{"":#^75}#\n#{"All done!":^75}#\n#{"":#^75}#')

except Exception as e:
    print("Unable to save to disk\n", e)

# remove cython caches
try:
    shutil.rmtree(cache_dir, ignore_errors=True)
except Exception as e:
    print("Unable to remove cached files: {e}")
