# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-02-20 09:35:48
# @Last Modified: 2022-11-28 16:14:28
# ------------------------------------------------------------------------------ #
# Dynamics described in Orlandi et al. 2013, DOI: 10.1038/nphys2686
# Creates a connectivity matrix matching the modular cultures (see `topology.py`)
# and runs the simulations in brian.
# ------------------------------------------------------------------------------ #

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

import topology as topo
from bitsandbobs import hi5 as h5

# we want to run this on a cluster, assign a custom cache directory to each thread
# putting this into a user-directory that gets backed-up turns out to be a bad idea.
# but shared dirs across users may cause trouble due to access-rights, defaults: 755
# cache_dir = os.path.expanduser(f"~/.cython/brian-pid-{os.getpid()}")
cache_dir = f"{tempfile.gettempdir()}/cython/brian-pid-{os.getpid()}"
prefs.codegen.runtime.cython.cache_dir = cache_dir
prefs.codegen.runtime.cython.multiprocess_safe = False

# Log level needs to be set in ~/.brian/user_preferences to work for all steps
prefs.logging.console_log_level = "INFO"
# prefs.logging.std_redirection = False
# import distutils.log
# distutils.log.set_verbosity(2)

# we want enforce simulation with c
prefs.codegen.target = "cython"
if prefs.codegen.target != "cython":
    log.warning("You are not using cython, are you sure?")

# ------------------------------------------------------------------------------ #
# model parameters
# ------------------------------------------------------------------------------ #

# fmt: off
# membrane potentials
vRef   = -60 * mV  # resting potential, neuron relaxes towards this without stimulation
vThr   = -45 * mV  # threshold potential
vPeak  =  35 * mV  # peak potential, after vThr is passed, rapid growth towards this
vReset = -50 * mV  # reset potential

# soma
tV = 50 * ms  # time scale of membrane potential
tU = 50 * ms  # time scale of recovery variable u

k = 0.5 / mV       # resistance over capacity(?), rescaled
b = 0.5            # sensitivity to sub-threshold fluctuations
uIncr =  50 * mV   # after-spike increment of recovery variable u

# synapse
tD =  20 * second  # characteristic recovery time, between 0.5 and 20 seconds
                   # Note: in the manuscript we renamed synaptic resources from D to R
tA =  10 * ms      # decay time of post-synaptic current (AMPA current decay time)
jA =  45 * mV      # AMPA current strength, between 10 - 50 mV
                   # 170.612 value in javiers neurondyn
                   # this needs to scale with tV/tA
tG = 20 * ms       # decay time of post-syanptic GABA current
jG = 50 * mV       # GABA current strength

# noise
beta = 0.8         # D = beta*D after spike, to reduce efficacy, beta < 1
rate = 80 * Hz     # rate for the poisson input (shot-noise), between 10 - 50 Hz
jM   = 15 * mV     # shot noise (minis) strength, between 10 - 50 mV
                   # (sum of minis arriving at target neuron)
# sigV = sqrt(0.06) * mV
sigV = 0.0 * mV
# amplitude of white noise membrane fluctuations
# Note: for 0.06^0.5 mV, membrane fluctuations are virtually zero.
# I made a mistake when converting the notation from the original paper,
# using a much lower noise amplitude.
# Since we focused on the noise through minis, its fine that we here use a lower sigV
# than the reference (where minis were not varied as a control parameter).
#
# Converting from javiers: sigma = sqrt(j / (2*tau**2)) = sqrt(300 / 2 / 50 / 50)
# Set sigV ~= 1.73 to get noise levels comparable to javiers paper.

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
# before enabeling this, do the math on ram requirements!
# 160 Neurons, at 1ms recording time step and one variable ~ 2GB RAM for 3600sec sim
record_state = True
# which variables
record_state_vars = ["D"]
# for which neurons, True for everything, or list of indices
record_state_idxs = True # [0, 1, 2, 3]
record_state_idxs = np.arange(0, 160)

# save state variables in steps of 25ms (the default) to save disk space
# use 0.5 for the highres simulations needed for nice rate-resource cycles
record_state_dt = 25 * ms

# whether to record population rates
record_rates = False
record_rates_freq = 50 * ms   # with which time resolution should rates be written to h5

# fmt:on

# ------------------------------------------------------------------------------ #
# command line arguments
# ------------------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Simulate modular cultures in Brian")

parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
parser.add_argument(
    "-jA",
    dest="jA",
    help="AMPA current strength, in mV",
    default=jA / mV,
    metavar=jA / mV,
    type=float,
)
parser.add_argument(
    "-jG",
    dest="jG",
    help="GABA current strength, in mV",
    default=jG / mV,
    metavar=jG / mV,
    type=float,
)
parser.add_argument(
    "-jM",
    dest="jM",
    help="Minis (noise amplitude), in mV",
    default=jM / mV,
    metavar=jM / mV,
    type=float,
)
parser.add_argument(
    "-r",
    dest="r",
    help="Poisson rate (minis), all neurons, in Hz",
    default=rate / Hz,
    metavar=rate / Hz,
    type=float,
)
parser.add_argument(
    "-tD",
    dest="tD",
    help="Characteristic recovery time, in seconds",
    default=tD / second,
    metavar=tD / second,
    type=float,
)
parser.add_argument("-s", dest="seed", help="RNG seed", default=42, metavar=42, type=int)
parser.add_argument(
    "-k", dest="k_inter", help="Number of bridging axons", default=5, metavar=5, type=int
)
parser.add_argument(
    "-kin",
    dest="k_in",
    help="Number of incoming connections per Neuron",
    default=30,
    metavar=30,
    type=int,
)
parser.add_argument(
    "-d",
    dest="sim_duration",
    help="Recording duration, in seconds",
    default=20 * 60,
    metavar=20 * 60,
    type=float,
)

parser.add_argument(
    "-equil",
    "--equilibrate",
    dest="equil_duration",
    help="Equilibration duration, in seconds",
    default=2 * 60,
    metavar=2 * 60,
    type=float,
)

parser.add_argument(
    "-stim",
    dest="stimulation_type",
    default="off",
    metavar="off",
    type=str,
    help="if/how to stimulate: 'off', 'poisson'",
)

parser.add_argument(
    "-stim_rate",
    dest="stimulation_rate",
    help="additional rate in stimulated mods, in Hz",
    default=20,
    metavar=20,
    type=float,
)

parser.add_argument(
    "-mod",
    dest="stimulation_module",
    default="0",
    metavar="0",
    type=str,
    help="modules to stimulate, e.g. `0`, or `02` for multiple",
)

# we may want to give neurons that bridge two modules a smaller synaptic weight [0, 1]
parser.add_argument(
    "--bridge_weight",
    dest="bridge_weight",
    default=1.0,
    metavar=1.0,
    type=float,
    help="synaptic weight of bridge neurons [0, 1]",
)

parser.add_argument(
    "--inhibition_fraction",
    dest="inhibition_fraction",
    default=0.2,
    metavar=0.2,
    type=float,
    help="fraction of neurons that should be inhibitory",
)

parser.add_argument(
    "--record-state-dt",
    dest="record_state_dt",
    default=record_state_dt / ms,
    metavar=record_state_dt / ms,
    type=float,
    help=(
        "time step for recording state variables, in ms. use 0.5ms to get smooth"
        " resource-rate cycles"
    ),
)

args = parser.parse_args()

# RNG
numpy.random.seed(args.seed)
topo.set_seed(args.seed)

# correct units
jA = args.jA * mV
jM = args.jM * mV
jG = args.jG * mV
tD = args.tD * second
rate = args.r * Hz
record_state_dt = args.record_state_dt * ms
args.equil_duration *= second
args.sim_duration *= second
args.stimulation_module = [int(i) for i in args.stimulation_module]
args.stimulation_rate *= Hz

print(f'#{"":#^75}#\n#{"running dynamics in brian":^75}#\n#{"":#^75}#')
log.info("output path:      %s", args.output_path)
log.info("seed:             %s", args.seed)
log.info("k_inter:          %s", args.k_inter)
log.info("k_in:             %s", args.k_in)
log.info("jA:               %s", jA)
log.info("jM:               %s", jM)
log.info("jG:               %s", jG)
log.info("tD:               %s", tD)
log.info("noise rate:       %s", rate)
log.info("duration:         %s", args.sim_duration)
log.info("equilibration:    %s", args.equil_duration)
log.info("recording states: %s", record_state)
log.info("  with time step: %s", record_state_dt)
log.info("recording rates:  %s", record_rates)
log.info("bridge weight:    %s", args.bridge_weight)
log.info("inhibition:       %s (fraction of all neurons)", args.inhibition_fraction)
log.info("stimulation:      %s", args.stimulation_type)
if args.stimulation_type != "off":
    log.info("stim. module:     %s", args.stimulation_module)
    log.info("stimulation rate: %s", args.stimulation_rate)


# ------------------------------------------------------------------------------ #
# topology
# ------------------------------------------------------------------------------ #


if args.k_in != -1:
    alpha = topo.find_alpha_for_target_in_degree(
        k_inter=args.k_inter, k_in_target=args.k_in, rerun=False
    )
    log.info(f"for k={args.k_inter}, using {alpha=} to get an in-degree of ~{args.k_in}.")
else:
    alpha = 0.0125
    log.warning(f"Not setting cusotm alpha, using default {alpha}")

if args.k_inter == -1:
    tp = topo.MergedTopology(assign_submodules=True, par_alpha=alpha)
    bridge_ids = np.array([], dtype="int")
else:
    tp = topo.ModularTopology(par_k_inter=args.k_inter, par_alpha=alpha)
    bridge_ids = tp.neuron_bridge_ids
num_n = tp.par_N
a_ij_sparse = tp.aij_sparse
mod_ids = tp.neuron_module_ids


# ------------------------------------------------------------------------------ #
# model, neurons
# ------------------------------------------------------------------------------ #

G = NeuronGroup(
    N=num_n,
    model="""
        dv/dt = ( k*(v-vRef)*(v-vThr) -u +IA -IG            # [1] soma potential
                  +xi*sigV*(2.0*tV)**0.5)/tV   : volt       # white noise term
        du/dt = ( b*(v-vRef) -u )/tU           : volt       # [2] recovery variable
        dIA/dt = -IA/tA                        : volt       # [4]
        dIG/dt = -IG/tG                        : volt       # [5]
        dD/dt = ( 1-D)/tD                      : 1          # [6] recovery to one
        j     : volt (constant)                             # neuron specific synaptic weight
    """,
    threshold="v > vPeak",
    reset="""
        v = vReset       # [3]
        u = u + uIncr    # [3]
        D = D * beta     # [7]
    """,
    method="euler",
    dt=defaultclock.dt,
)

# shot-noise:
# by targeting I with poisson, we should get pretty close to javiers version.
# rates = 0.01 / ms + (0.04 / ms)* rand(num_n) # we could have differen rates
# mini_g = PoissonGroup(num_n, rate)
# mini_s = Synapses(mini_g, G, on_pre="I_post+=jM", order=-1, name="Minis")
# mini_s.connect(j="i")

# treat minis as spikes, add directly to current
# for homogeneous rates, this is faster. here, N=1 is the input per neuron
mini_g = PoissonInput(target=G, target_var="IA", N=1, rate=rate, weight=jM)

# optionally, make a fraction of neurons inhibitiory.
num_inhib = int(num_n * args.inhibition_fraction)
num_excit = int(num_n - num_inhib)

# in brian, we can only create a subgroup from consecutive indices.
# thus, to separate GABA and AMPA (and bridge neuron) groups, we need to reorder
# indices. have two sets indices -> the ones in brian and the ones in the topology
t2b, b2t, inhib_ids, excit_ids, bridge_ids = topo.index_alignment(
    num_n, num_inhib, bridge_ids
)


G_inh = G[t2b[inhib_ids]]
G_exc = G[t2b[excit_ids]]
if len(bridge_ids) > 0:
    G_bridge = G[t2b[bridge_ids]]

# initalize according to neuron type
G.v = "vRef + 5*mV*(rand()-0.5)"
G.j = 0  # the lines below should overwrite this, sanity check
G_inh.j = jG
G_exc.j = jA
if len(bridge_ids) > 0:
    G_bridge.j *= args.bridge_weight

if not np.all(G.j != 0):
    log.warning(
        "Some synapse strenght `j` were zero after initialization. This should only"
        " happen if you manually set some `j` to zero!"
    )


# ------------------------------------------------------------------------------ #
# model, synapses
# ------------------------------------------------------------------------------ #

# We treat inhibitory and excitatory currents separately, so we need separete synapses.
S_exc = Synapses(
    source=G_exc,
    target=G,
    on_pre="""
        IA_post += D_pre * j_pre    # [8]
    """,
)
S_inh = Synapses(
    source=G_inh,
    target=G,
    on_pre="""
        IG_post += D_pre * j_pre    # [8]
    """,
)

# connect synapses
log.info("Applying connectivity from sparse matrix")
# we would like to do the simple thing, but need to work around inhibition
# and the different index conventions
# S.connect(i=a_ij_sparse[:, 0], j=a_ij_sparse[:, 1])

for n_id in inhib_ids:
    # i goes from 0 to num_inhib, and is already in brian index convention
    i = np.where(inhib_ids == n_id)[0][0]
    idx = np.where(a_ij_sparse[:, 0] == n_id)[0]
    if len(idx) == 0:
        continue
    ii = i * np.ones(len(idx), dtype="int64")
    # jj goes from 0 to num_n, and we still need to convert
    jj = t2b[a_ij_sparse[idx, 1]]
    S_inh.connect(i=ii, j=jj)

for n_id in excit_ids:
    i = np.where(excit_ids == n_id)[0][0]
    idx = np.where(a_ij_sparse[:, 0] == n_id)[0]
    if len(idx) == 0:
        continue
    ii = i * np.ones(len(idx), dtype="int64")
    jj = t2b[a_ij_sparse[idx, 1]]
    S_exc.connect(i=ii, j=jj)

# ------------------------------------------------------------------------------ #
# Stimulation if requested
# ------------------------------------------------------------------------------ #

if args.stimulation_type == "poisson":
    stim_ids = []
    for mod in args.stimulation_module:
        stim_ids.extend(np.where(mod_ids == mod)[0])
    stim_ids = t2b[np.array(stim_ids)]

    stim_g = PoissonGroup(len(stim_ids), args.stimulation_rate)
    stim_s = Synapses(stim_g, G, on_pre="IA_post += jM")
    stim_s.connect(i=np.arange(0, len(stim_ids)), j=stim_ids)


# ------------------------------------------------------------------------------ #
# Running
# ------------------------------------------------------------------------------ #

log.info("Equilibrating")
run(args.equil_duration, report="stdout", report_period=60 * 60 * second)

# add monitors after equilibration
spks_m = SpikeMonitor(G)

if record_state:
    if isinstance(record_state_idxs, bool):
        rec = record_state_idxs
    else:
        # list
        rec = t2b[record_state_idxs]
    stat_m = StateMonitor(G, record_state_vars, record=rec, dt=record_state_dt)

if record_rates:
    rate_m = PopulationRateMonitor(G)

log.info("Recording data")
run(args.sim_duration, report="stdout", report_period=60 * second)


# ------------------------------------------------------------------------------ #
# Output
# ------------------------------------------------------------------------------ #


if args.output_path is None:
    log.error("No output path provided. try `-o`")
else:
    print(f'#{"":#^75}#\n#{"Preparing to save to disk":^75}#\n#{"":#^75}#')
    try:
        # make sure directory exists
        outdir = os.path.abspath(os.path.expanduser(args.output_path + "/../"))
        os.makedirs(outdir, exist_ok=True)
        # shutil.copy2(args.input_path, args.output_path)
    except Exception as e:
        log.exception("Could not copy input file")

# these are python-benedicts (nested dictionaries) and we have a helper
# to dump the whole strucutre to hdf5 at the end
h5_data, h5_desc = tp.get_everything_as_nested_dict(return_descriptions=True)

# ------------------------------------------------------------------------------ #
# meta data of this simulation
# ------------------------------------------------------------------------------ #
# fmt: off
h5_data["meta.seed"] = args.seed

h5_data["meta.dynamics_jA"] = jA / mV
h5_desc["meta.dynamics_jA"] = "AMPA current strength, in mV"

h5_data["meta.dynamics_jG"] = jG / mV
h5_desc["meta.dynamics_jG"] = "GABA current strength, in mV"

h5_data["meta.dynamics_jM"] = jM / mV
h5_desc["meta.dynamics_jM"] = "shot noise (minis) strength, in mV"

h5_data["meta.dynamics_sigV"] = sigV / mV
h5_desc["meta.dynamics_sigV"] = "amplitude of white noise membrane fluctuations, in mV"

h5_data["meta.dynamics_tD"] = tD / second
h5_desc["meta.dynamics_tD"] = "characteristic decay time, in seconds"

h5_data["meta.dynamics_rate"] = rate / Hz
h5_desc["meta.dynamics_rate"] = "rate for the (global) poisson input (shot-noise), in Hz"

h5_data["meta.dynamics_stimulation_rate"] = args.stimulation_rate / Hz
h5_desc["meta.dynamics_stimulation_rate"] = "rate for the poisson input, added to stimulated modules, in Hz"

h5_data["meta.dynamics_simulation_duration"] = args.sim_duration / second
h5_desc["meta.dynamics_simulation_duration"] = "in seconds"

h5_data["meta.dynamics_equilibration_duration"] = args.equil_duration / second
h5_desc["meta.dynamics_equilibration_duration"] = "in seconds"

h5_data["meta.dynamics_bridge_weight"] = args.bridge_weight
h5_desc["meta.dynamics_bridge_weight"] = "synaptic weight of bridging neurons. get applied as a factor to outgoing synaptic currents."

h5_data["meta.topology_k_in"] = args.k_in
h5_desc["meta.topology_k_in"] = "desired in-degree for which alpha was optimized"

h5_data["data.neuron_g"] = G.j[t2b]
h5_desc["data.neuron_g"] = "synaptic weight that was ultimately used for each neuron in the dynamic simulation"

h5_data["data.neuron_inhibitory_ids"] = inhib_ids
h5_desc["data.neuron_inhibitory_ids"] = "List of neuron ids that were set to be inhibitory"

h5_data["data.neuron_excitatory_ids"] = excit_ids
h5_desc["data.neuron_excitatory_ids"] = "List of neuron ids that were set to be excitatory"

# ------------------------------------------------------------------------------ #
# simulation results
# ------------------------------------------------------------------------------ #

# I save the spike times as a nan-padded 2d array with shape (num_neurons, max_num_spikes)
# but also in what people might expect, a 2-col list (neuron_id, spike_time)
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
        # t = trains[n]
        t = trains[t2b[n]]  # convert back from brian to topology indices
        spiketimes[n, 0 : len(t)] = (t - args.equil_duration) / second
        spiketimes_as_list[0, last_idx : last_idx + len(t)] = [n] * len(t)
        spiketimes_as_list[1, last_idx : last_idx + len(t)] = (
            t - args.equil_duration
        ) / second
        last_idx += len(t)
    return spiketimes, spiketimes_as_list.T

try:
    # normal spikes, no stim, in two different formats
    spks, spks_as_list = convert_brian_spikes_to_pauls(spks_m)

    h5_data["data.spiketimes"] = spks
    h5_desc["data.spiketimes"] = "2d array of spiketimes, neuron x spiketime in seconds, zero-padded"

    h5_data["data.spiketimes_as_list"] = spks_as_list
    h5_desc["data.spiketimes_as_list"] = "two-column list of spiketimes. first col is neuron id, second col the spiketime. effectively same data as in 'data.spiketimes'. neuron id will need casting to int for indexing."

    if record_state:
        # write the time axis once for all variables and neurons (should be shared)
        t_axis = (stat_m.t - args.equil_duration) / second
        h5_data["data.state_vars_time"] = t_axis
        h5_desc["data.state_vars_time"] = "time axis of all state variables, in seconds"
        h5_data["data.state_vars_dt"] = record_state_dt / second
        h5_desc["data.state_vars_dt"] = "time step for state variable saving (seconds)"

        for idx, var in enumerate(record_state_vars):
            # careful to back-convert indices here
            if isinstance(record_state_idxs, bool):
                data = stat_m.variables[var].get_value()[:, t2b]
            else:
                # list
                # we already called t2b for the selection, no need again
                data = stat_m.variables[var].get_value()[:, :]

            h5_data[f"data.state_vars_{var}"] = data.T
            h5_desc[f"data.state_vars_{var}"] = f"state variable {var}, dim 1 neurons, dim 2 value for time, recorded neurons: {record_state_idxs}"


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
            h5_data[dsetname] = np.array(tmp).T
            h5_desc[dsetname] = description

        # main rate monitor
        write_rate(
            rate_m,
            "data.population_rate_smoothed",
            "population rate in Hz, smoothed with gaussian kernel (of 50ms? width), first dim is time in seconds",
        )

    if args.output_path is not None:
        print(f'#{"":#^75}#\n#{"Saving...":^75}#\n#{"":#^75}#')
        h5.recursive_write(args.output_path, h5_data, h5_desc)
        print(f'#{"":#^75}#\n#{"All done!":^75}#\n#{"":#^75}#')
    else:
        print("specify -o to save results to disk")


# fmt: on
except Exception as e:
    log.exception("Unable to save to disk")

# remove cython caches
try:
    shutil.rmtree(cache_dir, ignore_errors=True)
except Exception as e:
    log.exception("Unable to remove cached files")
