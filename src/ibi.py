# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-02-20 09:35:48
# @Last Modified: 2020-10-12 11:05:59
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
import sys
import numpy as np
from brian2 import *

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

# we want to run this on a cluster, assign a custom cache directory to each thread
cache_dir = os.path.expanduser(f"~/.cython/brian-pid-{os.getpid()}")
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
rate = 0.03 / ms   # rate for the poisson input (shot-noise), between 0.01 - 0.05 1/ms
gm =  25 * mV      # shot noise (minis) strength, between 10 - 50 mV
                   # (sum of minis arriving at target neuron)
gs = 300 * mV * mV * ms * ms  # white noise strength, via xi = dt**.5 * randn()
# fmt:on

# integration step size
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
parser.add_argument("-r", dest="r", default=rate * ms, help="in 1/ms", type=float)
parser.add_argument("-tD", dest="tD", default=tD / second, help="in seconds", type=float)
args = parser.parse_args()

numpy.random.seed(args.seed)
gA = args.gA * mV
gm = args.gm * mV
tD = args.tD * second
rate = args.r / ms

print(f'#{"":#^75}#\n#{"running dynamics in brian":^75}#\n#{"":#^75}#')
print(f"input topology: ", args.input_path)
print(f"seed: ", args.seed)
print(f"gA: ", gA)
print(f"gm: ", gm)
print(f"tD: ", tD)
print(f"rate: ", rate)

try:
    # load from hdf5
    a_ij = h5_load(args.input_path, "/data/connectivity_matrix")
    num_n = int(h5_load(args.input_path, "/meta/topology_num_neur"))
    # get the neurons sorted according to their modules
    mod_ids = h5_load(args.input_path, "/data/neuron_module_id")
    mod_sorted = np.zeros(num_n, dtype=int)
    temp = np.argsort(mod_ids)
    for i in range(0, num_n):
        mod_sorted[i] = np.argwhere(temp == i)

    mod_sort = lambda x: mod_sorted[x]
except:
    mod_sort = lambda x: x
    # or a csv
    try:
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
    assert a_ij is not None
    pre, post = np.where(a_ij == 1)
    for idx, i in enumerate(pre):
        j = post[idx]
        # i = np.argwhere(mod_sorted == i)[0, 0]
        # j = np.argwhere(mod_sorted == j)[0, 0]
        S.connect(i=i, j=j)
except:
    print(f"Creating Synapses randomly.")
    S.connect(condition="i != j", p=0.01)

# initalize to a somewhat sensible state. we could have different neuron types
G.v = "vc + 5*mV*rand()"

# ------------------------------------------------------------------------------ #
# Running and Writing
# ------------------------------------------------------------------------------ #

# assert False
# equilibrate
run(args.equil_duration * second, report="stdout", report_period=1 * 60 * second)

# disable state monitors that are not needed for production
# stat_m = StateMonitor(G, ["v", "I", "u", "D"], record=True)
spks_m = SpikeMonitor(G)
# rate_m = PopulationRateMonitor(G)
# mini_m = SpikeMonitor(mini_g)

run(args.duration * second, report="stdout", report_period=1 * 60 * second)

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

    # normal spikes, no stim in two different formats
    trains = spks_m.spike_trains()
    tmax = 0
    for tdx in trains.keys():
        if len(trains[tdx]) > tmax:
            tmax = len(trains[tdx])
    spiketimes = np.zeros(shape=(num_n, tmax))
    spiketimes_as_list = np.zeros(shape=(2,spks_m.num_spikes))
    last_idx = 0
    for n in range(0, num_n):
        t = trains[n]
        spiketimes[n, 0 : len(t)] = t / second - args.equil_duration
        spiketimes_as_list[0, last_idx:last_idx+len(t)] = [n]*len(t)
        spiketimes_as_list[1, last_idx:last_idx+len(t)] = t / second - args.equil_duration
        last_idx += len(t)

    try:
        bursts, time_series, summed_series, window = burst_times(spks_m, debug=True)
    except Exception as e:
        bursts = np.array([])
        summed_series = np.array([])


    ibi = args.duration * second / len(bursts)
    print(f"{len(bursts)} bursts occured in {args.duration} seconds. ibi: {ibi}")

    dset = f.create_dataset("/data/spiketimes", data=spiketimes)
    dset.attrs["description"] = "2d array of spiketimes, neuron x spiketime in seconds, zero-padded"

    dset = f.create_dataset("/data/spiketimes_as_list", data=spiketimes_as_list)
    dset.attrs["description"] = "two-column list of spiketimes. first col is neuron id, second col the spiketime. effectively same data as in '/data/spiketimes'. neuron id will need casting to int for indexing."



    dset = f.create_dataset("/data/bursttimes", data=bursts / second)
    dset.attrs["description"] = "time of detected busts in seconds"
    dset = f.create_dataset("/data/summed_series", data=bursts / second)
    dset.attrs[
        "description"
    ] = "timeseries of the number of neurons that spike in a time bin. based on 50 ms time bins. in each bin, a neuron spikes -possibly multiple times- (1) or not (0). In the summed series, in every bin we have an entry between 0 and number of neurons"
    dset = f.create_dataset("/data/ibi", data=ibi / second)
    dset.attrs["description"] = "inter burst interval in seconds"


    # meta data of this simulation
    dset = f.create_dataset("/meta/dynamics_gA", data=gA / mV)
    dset.attrs["description"] = "AMPA current strength, in mV"

    dset = f.create_dataset("/meta/dynamics_gm", data=gm / mV)
    dset.attrs["description"] = "shot noise (minis) strength, in mV"

    dset = f.create_dataset("/meta/dynamics_tD", data=tD / second)
    dset.attrs["description"] = "characteristic decay time, in seconds"

    dset = f.create_dataset("/meta/dynamics_rate", data=rate * ms)
    dset.attrs[
        "description"
    ] = "rate for the (global) poisson input (shot-noise), in 1/ms"

    dset = f.create_dataset("/meta/dynamics_simulation_duration", data=args.duration)
    dset.attrs["description"] = "in seconds"

    dset = f.create_dataset("/meta/dynamics_equilibration_duration", data=args.equil_duration)
    dset.attrs["description"] = "in seconds"

    f.close()


except Exception as e:
    print("Unable to save to disk\n", e)


print(f'#{"":#^75}#\n#{"All done!":^75}#\n#{"":#^75}#')


# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #

def plot_overview():
    ion()  # interactive plotting
    fig, ax = subplots(5, 1, sharex=True)

    n1 = randint(0, num_n)  # some neuron ton highlight
    sel = where(spks_m.i == n1)[0]

    # ax[1].plot(mini_m.t / second, mini_m.i, ".y")
    ax[1].plot(spks_m.t / second, mod_sort(spks_m.i), ".k")
    ax[1].plot(spks_m.t[sel] / second, mod_sort(spks_m.i[sel]), ".")
    # ax[1].plot(stim_m.t / second, mod_sort(stim_m.i), ".", color="orange")
    ax[1].set_ylabel("Raster")


    # bursts = burst_times(spks_m)
    bursts, time_series, summed_series, window = burst_times(spks_m, debug=True)
    ax[0].scatter(bursts / second, np.ones(len(bursts)))

    bin_size = 50 * ms
    ax[4].plot(np.arange(len(summed_series)) * bin_size / second, summed_series)

    ibi = args.duration * second / len(bursts)
    print("ibi: ", ibi)

    show()
