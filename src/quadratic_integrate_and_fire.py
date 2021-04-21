# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-02-20 09:35:48
# @Last Modified: 2021-04-21 14:03:45
# ------------------------------------------------------------------------------ #
# Dynamics described in Orlandi et al. 2013, DOI: 10.1038/nphys2686
# Loads topology from hdf5 and runs the simulations in brian.
# ------------------------------------------------------------------------------ #

import h5py
import argparse
import os
import tempfile
import sys
import shutil
import numpy as np
import logging
from brian2 import *

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

import stimulation as stim
import topology as topo

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
ta = 50 * ms  # time scale of recovery variable u

k = 0.5 / mV  # resistance over capacity(?), rescaled
b = 0.5       # sensitivity to sub-threshold fluctuations
d =  50 * mV  # after-spike reset of inhibitory current u

# synapse
tD =   2 * second  # characteristic recovery time, between 0.5 and 20 seconds
tA =  10 * ms      # decay time of post-synaptic current (AMPA current decay time)
gA =  35 * mV      # AMPA current strength, between 10 - 50 mV
                   # 170.612 value in javiers neurondyn
                   # this needs to scale with tc/tA

# noise
beta = 0.8         # D = beta*D after spike, to reduce efficacy, beta < 1
rate = 37 * Hz     # rate for the poisson input (shot-noise), between 10 - 50 Hz
gm =  25 * mV      # shot noise (minis) strength, between 10 - 50 mV
                   # (sum of minis arriving at target neuron)
gs = 300 * mV * mV * ms * ms  # white noise strength, via xi = dt**.5 * randn()


# ------------------------------------------------------------------------------ #
# simulation parameters
# ------------------------------------------------------------------------------ #

# integration step size
# this turns out to be quite crucial for synchonization:
# when too large (brian defaul 0.1ms) this forces sth like an integer cast at
# some point and may promote synchronized firing. (spike times are not precise enough)
# heuristically: do not go above 0.05 ms, better 0.01ms
defaultclock.dt = 0.05 * ms

# whether to record state variables
record_state = False
# which variables
record_state_vars = ["v", "I", "u", "D"]
# for which neurons
record_state_idxs = [0,1,2,3]

# whether to record population rates
record_rates = False
record_rates_freq = 50 * ms   # with which time resolution should rates be written to h5


# ------------------------------------------------------------------------------ #
# command line arguments
# ------------------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Brian")

parser.add_argument("-i",  dest="input_path",  help="input path",  metavar="FILE",  required=True)
parser.add_argument("-o",  dest="output_path", help="output path", metavar="FILE")
parser.add_argument("-gA", dest="gA",          help="in mV",       default=gA / mV,     type=float)
parser.add_argument("-gm", dest="gm",          help="in mV",       default=gm / mV,     type=float)
parser.add_argument("-r",  dest="r",           help="in Hz",       default=rate / Hz,   type=float)
parser.add_argument("-tD", dest="tD",          help="in seconds",  default=tD / second, type=float)
parser.add_argument("-s",  dest="seed",        help="rng",         default=117,         type=int)
parser.add_argument("-d",
    dest="sim_duration",   help="in seconds",  default=20 * 60, type=float)

parser.add_argument("-equil", "--equilibrate",
    dest="equil_duration", help="in seconds",  default= 2 * 60, type=float)

parser.add_argument("-stim",
    dest="stimulation_type", default="off", type=str,
    help="if/how to stimulate: 'off', 'poisson', 'hideaki'",)

parser.add_argument("-mod",
    dest="stimulation_module", default='0', type=str,
    help="modules to stimulate, e.g. `0`, or `02` for multiple",)

# we may want to give neurons that bridge two modules a smaller synaptic weight [0, 1]
parser.add_argument("--bridge_weight",
    dest="bridge_weight",  default= 1.0, type=float,
    help="synaptic weight of bridge neurons [0, 1]")

# fmt:on
args = parser.parse_args()

# RNG
numpy.random.seed(args.seed)

# correct units
gA = args.gA * mV
gm = args.gm * mV
tD = args.tD * second
rate = args.r * Hz
args.equil_duration *= second
args.sim_duration *= second
args.stimulation_module = [int(i) for i in args.stimulation_module]

print(f'#{"":#^75}#\n#{"running dynamics in brian":^75}#\n#{"":#^75}#')
log.info("input topology:   %s", args.input_path)
log.info("output path:      %s", args.output_path)
log.info("seed:             %s", args.seed)
log.info("gA:               %s", gA)
log.info("gm:               %s", gm)
log.info("tD:               %s", tD)
log.info("noise rate:       %s", rate)
log.info("duration:         %s", args.sim_duration)
log.info("equilibration:    %s", args.equil_duration)
log.info("stimulation:      %s", args.stimulation_type)
log.info("recording states: %s", record_state)
log.info("recording rates:  %s", record_rates)
if args.stimulation_type != "off":
    log.info("stim.   module: %s", args.stimulation_module)
log.info("bridge weight:    %s", args.bridge_weight)


# ------------------------------------------------------------------------------ #
# topology
# ------------------------------------------------------------------------------ #

num_n, a_ij_sparse, mod_ids = topo.load_topology(args.input_path)

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
        bw     : 1  (constant)                 # neuron specific synaptic weight
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
        I_post += D_pre * gA * bw_pre    # [10]
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
# maybe exclude the bridging neurons, when looking at `1x1_projected` topology
mini_g = PoissonInput(target=G, target_var="I", N=1, rate=rate, weight=gm)

# connect synapses
log.info("Applying connectivity from sparse matrix")
S.connect(i=a_ij_sparse[:, 0], j=a_ij_sparse[:, 1])

# initalize to a somewhat sensible state. we could have different neuron types
G.v = "vc + 5*mV*rand()"

# optionally, give bridging neurons different weights
bridge_ids = topo.load_bridging_neurons(args.input_path)
G.bw = 1.0
for n_id in bridge_ids:
    G.bw[n_id] = args.bridge_weight

# ------------------------------------------------------------------------------ #
# Stimulation if requested
# ------------------------------------------------------------------------------ #

if args.stimulation_type == "hideaki":

    stimulus_indices, stimulus_times = stim.stimulation_pattern(
        interval=400 * ms,
        duration=args.equil_duration + args.sim_duration,
        target_modules=args.stimulation_module,
        mod_ids=mod_ids,
        min_dt=defaultclock.dt * 1.001,
    )

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

elif args.stimulation_type == "poisson":
    # to target birde neurons only, assume bridge_ids are consecutive
    # stim_g = G[bridge_ids]
    # PoissonInput(target=stim_g, target_var="v", N=1, rate=rate, weight=vp)

    poisson_rate = np.zeros(num_n)
    for mod in args.stimulation_module:
        stim_idx = np.where(mod_ids == mod)[0]
        poisson_rate[stim_idx] = 0.4 * (5 / 25) * 1 / (400 * ms)

    print(poisson_rate)

    stim_g = PoissonGroup(num_n, poisson_rate, name="create_stimulation")
    stim_s = Synapses(stim_g, G, on_pre="v_post = 2*vp", name="apply_stimulation")
    stim_s.connect("i == j")


# ------------------------------------------------------------------------------ #
# Running
# ------------------------------------------------------------------------------ #

log.info("Equilibrating")
run(args.equil_duration, report="stdout", report_period=60 * 60 * second)

# add monitors after equilibration
spks_m = SpikeMonitor(G)

if record_state:
    stat_m = StateMonitor(G, record_state_vars, record=record_state_idxs)

if record_rates:
    rate_m = PopulationRateMonitor(G)

if args.stimulation_type != "off":
    stim_m = SpikeMonitor(stim_g)

log.info("Recording data")
run(args.sim_duration, report="stdout", report_period=60 * 60 * second)


# ------------------------------------------------------------------------------ #
# Writing
# ------------------------------------------------------------------------------ #

if args.output_path is None:
    log.error("No output path provided. try `-o`")
else:
    print(f'#{"":#^75}#\n#{"saving to disk":^75}#\n#{"":#^75}#')

    try:
        # make sure directory exists
        outdir = os.path.abspath(os.path.expanduser(args.output_path + "/../"))
        os.makedirs(outdir, exist_ok=True)
        shutil.copy2(args.input_path, args.output_path)
    except Exception as e:
        log.exception("Could not copy input file")

    try:
        f = h5py.File(args.output_path, "a")

        def convert_brian_spikes_to_pauls(spks_m):
            trains = spks_m.spike_trains()
            num_n = len(trains)  # monitor may be defined on a subgroup
            tmax = 0
            for tdx in trains.keys():
                if len(trains[tdx]) > tmax:
                    tmax = len(trains[tdx])
            spiketimes = np.zeros(shape=(num_n, tmax))
            spiketimes_as_list = np.zeros(shape=(2, spks_m.num_spikes))
            last_idx = 0
            for n in range(0, num_n):
                t = trains[n]
                spiketimes[n, 0 : len(t)] = (t - args.equil_duration) / second
                spiketimes_as_list[0, last_idx : last_idx + len(t)] = [n] * len(t)
                spiketimes_as_list[1, last_idx : last_idx + len(t)] = (
                    t - args.equil_duration
                ) / second
                last_idx += len(t)
            return spiketimes, spiketimes_as_list.T

        # normal spikes, no stim in two different formats
        spks, spks_as_list = convert_brian_spikes_to_pauls(spks_m)

        dset = f.create_dataset("/data/spiketimes", compression="gzip", data=spks)
        dset.attrs[
            "description"
        ] = "2d array of spiketimes, neuron x spiketime in seconds, zero-padded"

        dset = f.create_dataset(
            "/data/spiketimes_as_list", compression="gzip", data=spks_as_list
        )
        dset.attrs[
            "description"
        ] = "two-column list of spiketimes. first col is neuron id, second col the spiketime. effectively same data as in '/data/spiketimes'. neuron id will need casting to int for indexing."

        if args.stimulation_type != "off":
            # stimultation timestamps in two different formats
            stim, stim_as_list = convert_brian_spikes_to_pauls(stim_m)

            dset = f.create_dataset(
                "/data/stimulation_times_as_list", compression="gzip", data=stim_as_list
            )
            dset.attrs[
                "description"
            ] = "two-column list of stimulation times. first col is target-neuron id, second col the stimulation time. Beware: we have approximateley one timestep delay between stimulation and spike."

        if record_state:
            # write the time axis once for all variables and neurons (should be shared!)
            t_axis = (stat_m.t - args.equil_duration) / second
            dset = f.create_dataset(
                "/data/state_vars_time", compression="gzip", data=t_axis
            )
            dset.attrs["description"] = "time axis of all state variables, in seconds"

            for idx, var in enumerate(record_state_vars):
                data = stat_m.variables[var].get_value()
                dset = f.create_dataset(
                    f"/data/state_vars_{var}", compression="gzip", data=data.T
                )
                dset.attrs[
                    "description"
                ] = f"state variable {var}, dim 1 neurons, dim 2 value for time, recorded neurons: {record_state_idxs}"

        #     for state_var in record_state_vars:

        if record_rates:
            # we could write rates, but
            # at the default timestep, the data files (and RAM requirements) get huge.
            # write with lower frequency, and smooth to not miss sudden changes
            freq = int(record_rates_freq / defaultclock.dt)
            width = record_rates_freq

            def write_rate(mon, dsetname, description):
                tmp = [
                    (mon.t / second - args.equil_duration / second)[::freq],
                    (mon.smooth_rate(window="gaussian", width=width) / Hz)[::freq],
                ]
                dset = f.create_dataset(
                    dsetname, compression="gzip", data=np.array(tmp).T
                )
                dset.attrs["description"] = description

            # main rate monitor
            write_rate(
                rate_m,
                "/data/population_rate_smoothed",
                "population rate in Hz, smoothed with gaussian kernel (of 50ms? width), first dim is time in seconds",
            )

            # and one for every module ...
            # creating brians monitors in a for loop turned out problematic
            # for mdx, mon in enumerate(mod_rate_m):
            #     write_rate(
            #         mon,
            #         "/data/module_rate_smoothed_modid={mods[mdx]:d}",
            #         "same as population rate, just on a per module level",
            #     )

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

        dset = f.create_dataset(
            "/meta/dynamics_simulation_duration", data=args.sim_duration / second
        )
        dset.attrs["description"] = "in seconds"

        dset = f.create_dataset(
            "/meta/dynamics_equilibration_duration", data=args.equil_duration / second
        )
        dset.attrs["description"] = "in seconds"

        dset = f.create_dataset("/meta/dynamics_bridge_weight", data=args.bridge_weight)
        dset.attrs[
            "description"
        ] = "synaptic weight of bridging neurons. get applied as a factor to outgoing synaptic currents."

        f.close()

        print(f'#{"":#^75}#\n#{"All done!":^75}#\n#{"":#^75}#')

    except Exception as e:
        log.exception("Unable to save to disk")

# remove cython caches
try:
    shutil.rmtree(cache_dir, ignore_errors=True)
except Exception as e:
    log.exception("Unable to remove cached files")
