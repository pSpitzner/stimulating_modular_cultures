# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-02-20 09:35:48
# @Last Modified: 2020-02-27 13:39:38
# ------------------------------------------------------------------------------ #
# Dynamics described in Orlandi et al. 2013, DOI: 10.1038/nphys2686
# Loads topology from hdf5 or csv and runs the simulations in brian.
# ------------------------------------------------------------------------------ #

import h5py
import argparse

import numpy as np
from brian2 import *

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
gA = 170 * mV      # AMPA current strength, between 10 - 50 mV
                   # 170.612 value in javiers neurondyn
                   # this needs to scale with tc/tA

# noise
beta = 0.8         # D = beta*D after spike, to reduce efficacy, beta < 1
rate = 0.02 / ms   # rate for the poisson input (shot-noise), between 0.01 - 0.05 1/ms
gm =  30 * mV      # shot noise (minis) strength, between 10 - 50 mV
gs = 300 * mV * mV * ms * ms  # white noise strength, via xi = dt**.5 * randn()
# fmt:on

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


def stimulation_pattern(mod_ids, mod_sorted, mod_targets=[0, 1]):
    # we assume that the neuron ids are sorted according to modules
    first_idx = []
    num_in_mod = []
    mod_ids = sort(mod_ids)
    for i in mod_targets:
        assert i in mod_ids
        first_idx.append(where(mod_ids == i)[0])
        num_in_mod.append(sum(mod_ids == i))


# ------------------------------------------------------------------------------ #
# command line arguments
# ------------------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Brian")
parser.add_argument("-i", dest="input_path", help="input path", metavar="FILE")
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
parser.add_argument("-s", dest="seed", default=117, help="rng", type=int)
parser.add_argument("-N", dest="num_n", default=-1, type=int)
parser.add_argument("-d", dest="duration", default=30, help="in seconds", type=float)
parser.add_argument("--equilibrate", dest="equil_duration", default=10, help="in seconds", type=float)
args = parser.parse_args()

print(f"seed: {args.seed}")
numpy.random.seed(args.seed)

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
                +xi*(gs/tc)**0.5 )/tc          : volt       # white noise term
        dI/dt = -I/tA                          : volt       # [9, 10]
        du/dt = ( b*(v-vr) -u )/ta             : volt       # [7] inhibitory current
        dD/dt = ( 1-D)/tD                      : 1          # [11] recovery to one
        # vc                                   : volt       # different cell types
        # d                                    : volt       # different cell types
    """,
    threshold="v > vp",
    reset="""
        v = vc           # [8]
        u = u + d        # [8]
        D = D * beta     # [11] delta-function term on spike
    """,
    method="euler",
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
    S.connect(condition="i != j", p=0.1)

# initalize to a somewhat sensible state. we could have different neuron types
G.v = "vc + 5*mV*rand()"
# for n in range(0, num_n):
#     r = rand()
#     if r < .05:
#         G.vc[n] = -40 * mV
#         G.d[n]  =  55 * mV
#     elif r < .1:
#         G.vc[n] = -35 * mV
#         G.d[n]  =  60 * mV
#     elif r < .2:
#         G.vc[n] = -45 * mV
#         G.d[n]  =  50 * mV
#     else:
#         G.vc[n] = -50 * mV
#         G.d[n]  =  50 * mV

# ------------------------------------------------------------------------------ #
# Running and Writing
# ------------------------------------------------------------------------------ #

# equilibrate
run(args.equil_duration * second, report="stdout")

# disable state monitors that are not needed for production
stat_m = StateMonitor(G, ["v", "I", "u", "D"], record=True)
spks_m = SpikeMonitor(G)
rate_m = PopulationRateMonitor(G)
# mini_m = SpikeMonitor(mini_g)

run(args.duration * second, report="stdout")


try:
    copy2(args.input_path, args.output_path)
except Exception as e:
    print("Could not copy input file\n", e)

try:
    if args.output_path is None:
        raise ValueError
    f = h5py.File(args.output_path, "a")

    trains = spks_m.spike_trains()
    tmax = 0
    for tdx in trains.keys():
        if len(trains[tdx]) > tmax:
            tmax = len(trains[tdx])

    dset = np.zeros(shape=(num_n, tmax))
    for n in range(0, num_n):
        t = trains[n]
        dset[n, 0 : len(t)] = t / second - args.equil_duration

    f.create_dataset("/data/spiketimes", data=dset)
    f.close()


except Exception as e:
    print("Unable to save to disk\n", e)

# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #


ion()  # interactive plotting
fig, ax = subplots(5, 1, sharex=True)

ax[0].plot(rate_m.t / second, rate_m.smooth_rate(width=50 * ms) / Hz)
ax[0].set_ylabel("Pop. Rate (Hz)")
ax[0].set_title(f"{args.input_path}")

n1 = randint(0, num_n)  # some neuron ton highlight
sel = where(spks_m.i == n1)[0]


# ax[1].plot(mini_m.t / second, mini_m.i, ".y")
ax[1].plot(spks_m.t / second, mod_sort(spks_m.i), ".k")
ax[1].plot(spks_m.t[sel] / second, mod_sort(spks_m.i[sel]), ".")
ax[1].set_ylabel("Raster")

ax[2].plot(stat_m.t / second, stat_m.v[n1], label=f"Neuron {n1}")
ax[2].set_ylabel("v")
ax[2].legend()

# plot(stat_m.t / second, stat_m.I[n1], label="Neuron n1")
ax[3].plot(stat_m.t / second, stat_m.u[n1], label=f"Neuron {n1}")
ax[3].set_ylabel("u")

ax[4].plot(stat_m.t / second, stat_m.D[n1], label=f"Neuron {n1}")
ax[4].set_ylabel("D")
ax[4].set_xlabel("Time (s)")
ax[4].legend()

show()
