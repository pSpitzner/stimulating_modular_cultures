# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-05-06 09:35:48
# @Last Modified: 2021-08-04 20:41:35
# ------------------------------------------------------------------------------ #
# Here we try to find out what the impact of input from single bridges is.
# Todo:
# * [ ] empty neuron groups are a problem for brian. file an issue. bridges and inhib
# * [ ] loop it for probs / repeptition on same topo
# * [ ] create topo from exe in python
# * [ ] output?
# * [ ] can we check rate monitor life with a known threshold? we shoould.
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
from addict import Dict

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
# prefs.logging.std_redirection = False
# import distutils.log
# distutils.log.set_verbosity(2)

# we want enforce simulation with c
prefs.codegen.target = "numpy"

# ------------------------------------------------------------------------------ #
# model parameters
# ------------------------------------------------------------------------------ #

# fmt: off
# avoid cluttering global space, group parameters into a dictionary that is passed
# as the `namespace` argument to brians `run`
pars = Dict(

# membrane potentials
vRef   = -60 * mV,  # resting potential, neuron relaxes towards this without stimulation
vThr   = -45 * mV,  # threshold potential
vPeak  =  35 * mV,  # peak potential, after vThr is passed, rapid growth towards this
vReset = -50 * mV,  # reset potential

# soma
tV = 50 * ms,  # time scale of membrane potential
tU = 50 * ms,  # time scale of recovery variable u

k = 0.5 / mV,       # resistance over capacity(?), rescaled
b = 0.5,            # sensitivity to sub-threshold fluctuations
uIncr =  50 * mV,   # after-spike increment of recovery variable u

# synapse
tD =   2 * second,  # characteristic recovery time, between 0.5 and 20 seconds
tA =  10 * ms,      # decay time of post-synaptic current (AMPA current decay time)
jA =  45 * mV,      # AMPA current strength, between 10 - 50 mV
                    # 170.612 value in javiers neurondyn
                    # this needs to scale with tV/tA
tG = 20 * ms,       # decay time of post-syanptic GABA current
jG = 67.5 * mV,     # GABA current strength

# noise
beta = 0.8,         # D = beta*D after spike, to reduce efficacy, beta < 1
rate = 38 * Hz,     # rate for the poisson input (shot-noise), between 10 - 50 Hz
jM =  25 * mV,      # shot noise (minis) strength, between 10 - 50 mV
                    # (sum of minis arriving at target neuron)
jS = 300 * mV * mV * ms * ms,  # white noise strength, via xi = dt**.5 * randn()

)

# ------------------------------------------------------------------------------ #
# simulation parameters
# ------------------------------------------------------------------------------ #

# integration step size
# this turns out to be quite crucial for synchonization:
# when too large (brian defaul 0.1ms) this forces sth like an integer cast at
# some point and may promote synchronized firing. (spike times are not precise enough)
# heuristically: do not go above 0.05 ms, better 0.01ms
defaultclock.dt = 0.05 * ms


# ------------------------------------------------------------------------------ #
# command line arguments
# ------------------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Brian")

parser.add_argument("-o",  dest="output_path", help="output path", metavar="FILE")
parser.add_argument("-jA", dest="jA",          help="in mV",       default=pars.jA / mV,     type=float)
parser.add_argument("-jG", dest="jG",          help="in mV",       default=pars.jG / mV,     type=float)
parser.add_argument("-jM", dest="jM",          help="in mV",       default=pars.jM / mV,     type=float)
parser.add_argument("-r",  dest="r",           help="in Hz",       default=pars.rate / Hz,   type=float)
parser.add_argument("-tD", dest="tD",          help="in seconds",  default=pars.tD / second, type=float)
parser.add_argument("-s",  dest="seed",        help="rng",         default=117,         type=int)


parser.add_argument("-equil", "--equilibrate",
    dest="equil_duration", help="in seconds",  default=5, type=float)

# waiting time before we assume "no burst happened"
parser.add_argument("-d",
    dest="wait_duration",   help="in seconds",  default=120, type=float)

# match the ~8-10 ms we find as isi within bursts ~> 125 Hz
parser.add_argument("-stim", dest="r_stim", help="spike rate of the bridge neurons, in Hz",
    default = 125, type=float)

# we may want to give neurons that bridge two modules a smaller synaptic weight [0, 1]
parser.add_argument("--bridge_weight",
    dest="bridge_weight",  default= 1.0, type=float,
    help="synaptic weight of bridge neurons [0, 1]")

parser.add_argument("--inhibition_fraction",
    dest="inhibition_fraction",  default= 0.0, type=float,
    help="how many neurons should be inhibitory")

# fmt:on
args = parser.parse_args()

# RNG
numpy.random.seed(args.seed)

# correct units
pars.jA = args.jA * mV
pars.jM = args.jM * mV
pars.jG = args.jG * mV
pars.tD = args.tD * second
pars.rate = args.r * Hz
args.equil_duration *= second
args.wait_duration *= second
args.r_stim *= Hz

print(f'#{"":#^75}#\n#{"running dynamics in brian":^75}#\n#{"":#^75}#')
log.info("output path:      %s", args.output_path)
log.info("seed:             %s", args.seed)
log.info("jA:               %s", pars.jA)
log.info("jM:               %s", pars.jM)
log.info("jG:               %s", pars.jG)
log.info("tD:               %s", pars.tD)
log.info("noise rate:       %s", pars.rate)
log.info("equilibration:    %s", args.equil_duration)
log.info("wait duration:    %s", args.wait_duration)
log.info("stimulation rate: %s", args.r_stim)
# log.info("recording states: %s", record_state)
# log.info("recording rates:  %s", record_rates)
log.info("bridge weight:    %s", args.bridge_weight)
log.info("inhibition:       %s (fraction of all neurons)", args.inhibition_fraction)


# ------------------------------------------------------------------------------ #
# topology
# ------------------------------------------------------------------------------ #

topo_path = cache_dir + "/topology_file.hdf5"
topo_exe = f"/Users/paul/mpi/simulation/modular_cultures/_latest/exe/orlandi_standalone -o {topo_path} -N 25 -f 1x1_project -k 1 -a 0.0125 -a_weighted 1"
num_n = 26

# ------------------------------------------------------------------------------ #
# setup things that are kept across runs -> number of neurons, ODEs, monitors
# ------------------------------------------------------------------------------ #

G = NeuronGroup(
    N=num_n,
    model="""
        dv/dt = ( k*(v-vRef)*(v-vThr) -u +IA -IG          # [6] soma potential
                  +xi*(jS/tV)**0.5      )/tV   : volt       # white noise term
        dIA/dt = -IA/tA                        : volt       # [9, 10]
        dIG/dt = -IG/tG                        : volt       # [9, 10]
        du/dt = ( b*(v-vRef) -u )/tU         : volt       # [7] inhibitory current
        dD/dt = ( 1-D)/tD                      : 1          # [11] recovery to one
        j     : volt  (constant)                 # neuron specific synaptic weight
    """,
    threshold="v > vPeak",
    reset="""
        v = vReset           # [8]
        u = u + uIncr       # [8]
        D = D * beta        # [11] delta-function term on spike
    """,
    method="euler",
    dt=defaultclock.dt,
)

# dont hardcode
num_normal = 25
rate_m = PopulationRateMonitor(G[0:num_normal])
spks_m = SpikeMonitor(G[0:num_normal])
spks_m_debug = SpikeMonitor(G[num_normal:])
mini_g = PoissonInput(target=G, target_var="IA", N=1, rate=pars.rate, weight=pars.jM)

net = Network()
net.add(G)
net.add(rate_m)
net.add(spks_m)
net.add(spks_m_debug)
net.add(mini_g)
net.store()  # restore to time 0 later


def init():
    # global stimulus_indices, stimulus_times, t2b, b2t
    global net, S_exc, S_inh, inhib_ids, stim_g, stim_s

    # create a new topology, add synapses to network
    os.system(f"{topo_exe} -s {np.random.randint(1e6)} > /dev/null")
    num_n, a_ij_sparse, mod_ids = topo.load_topology(topo_path)
    assert num_n == 26

    # for 1x1_project, we have the convention that bridge have ids higher than normal ones
    bridge_ids = np.sort(topo.load_bridging_neurons(topo_path))

    # optionally, make a fraction of neurons inhibitiory. For now, only change `j`.
    num_normal = num_n - len(bridge_ids)
    num_inhib = int(num_normal * args.inhibition_fraction)
    num_excit = int(num_normal - num_inhib)

    # in brian, we can only create a subgroup from consecutive indices.
    # thus, to separate GABA and AMPA (and bridge neuron) groups, we need to reorder
    # indices. have two sets indices -> the ones in brian and the ones in the topology
    t2b, b2t, inhib_ids, excit_ids, _ = topo.index_alignment(num_normal, num_inhib, [])

    G_normal = G[0:num_normal]
    G_bridge = G[num_normal:]
    G_exc = G[t2b[excit_ids]]
    if len(inhib_ids) > 0:
        G_inh = G[t2b[inhib_ids]] # brian has a problem if lists of indices are empty.

    # initalize according to neuron type
    G.v = "vReset + 5*mV*rand()"
    G.j = 0  # the lines below should overwrite this, sanity check
    if len(inhib_ids) > 0:
        G_inh.j = jG
    G_exc.j = jA

    G_bridge.j = jA
    G_bridge.j *= args.bridge_weight
    assert np.all(G.j != 0)

    # ------------------------------------------------------------------------------ #
    # synapses
    # ------------------------------------------------------------------------------ #

    S_exc = Synapses(
        source=G_exc,
        target=G,
        on_pre="""
        IA_post += D_pre * j_pre    # [10]
        """,
    )

    if len(inhib_ids) > 0:
        S_inh = Synapses(
            source=G_inh,
            target=G,
            on_pre="""
                IG_post += D_pre * j_pre    # [10]
            """,
        )

    # apply connectivity
    for n_id in excit_ids:
        i = np.where(excit_ids == n_id)[0][0]
        idx = np.where(a_ij_sparse[:, 0] == n_id)[0]
        if len(idx) == 0:
            continue
        ii = i * np.ones(len(idx), dtype="int64")
        jj = t2b[a_ij_sparse[idx, 1]]
        S_exc.connect(i=ii, j=jj)

    for n_id in inhib_ids:
        i = np.where(inhib_ids == n_id)[0][0]
        idx = np.where(a_ij_sparse[:, 0] == n_id)[0]
        if len(idx) == 0:
            continue
        ii = i * np.ones(len(idx), dtype="int64")
        jj = t2b[a_ij_sparse[idx, 1]]
        S_inh.connect(i=ii, j=jj)

    net.add(S_exc)
    if len(inhib_ids) > 0:
        net.add(S_inh)

    # ------------------------------------------------------------------------------ #
    # stimulation, i.e. spiking of projecting neurons
    # ------------------------------------------------------------------------------ #


    stimulus_indices = []
    stimulus_times = []
    for ndx, n in enumerate(bridge_ids):
        # use rate and cv that we typically see in bursts inputs
        temp = stim.stimulation_at_rate_with_cv(
            rate=args.r_stim,
            cv=0.4,  # cv~=0.4 measured for the excitation-only system, for isis within bursts
            t_end=args.wait_duration + args.equil_duration,
            t_start=args.equil_duration,
            min_dt=defaultclock.dt * 1.001,
        )
        # this looses units, returned in seconds
        stimulus_times.extend(temp)
        stimulus_indices.extend([n] * len(temp))

    stimulus_times = np.array(stimulus_times) * second
    stimulus_indices = np.array(stimulus_indices, dtype="int64")

    # sort by time
    idx = np.argsort(stimulus_times)
    stimulus_times = stimulus_times[idx]
    stimulus_indices = stimulus_indices[idx]

    stim_g = SpikeGeneratorGroup(
        N=1,
        indices=[0]*len(stimulus_times),
        times=stimulus_times,
        name="create_stimulation",
    )
    # because we project via artificial synapses, we get a delay of
    # approx (!) one timestep between the stimulation and the spike
    stim_s = Synapses(
        stim_g, G[bridge_ids], on_pre="v_post = 2*vPeak", name="apply_stimulation",
    )
    stim_s.connect(j='i')
    net.add(stim_g)
    net.add(stim_s)


def cleanup():
    global net
    # remove old synapse stuff and reset network
    net.remove(S_exc)
    if len(inhib_ids) > 0:
        net.remove(S_inh)
    net.remove(stim_s)
    net.remove(stim_g)

    net.restore(restore_random_state=False)

from brian2tools import *
def iterate():
    rate_m.active = False
    spks_m.active = False
    net.run(args.equil_duration, report="stdout")

    rate_m.active = True
    spks_m.active = True
    # check every 5 seconds if we had a burst
    while net.t < args.wait_duration:
        net.run(5*second, report="stdout")
        # in the post-processing of exc-only, we used to use ~ 20ms smoothing
        # threshold at 7.5 Hz
        rate = rate_m.smooth_rate(window='gaussian', width=5*ms)
        if np.any(rate > 30 * Hz):
            subplots()
            brian_plot(spks_m)
            return rate_m.t[np.argmax(rate > 30 * Hz)] - args.equil_duration

    subplots()
    brian_plot(spks_m)
    return np.nan * second


res = []
for i in range(3):
    init()
    res.append(iterate())
    cleanup()


# ------------------------------------------------------------------------------ #
# Running
# ------------------------------------------------------------------------------ #

assert False

run(args.equil_duration, report="stdout", report_period=60 * 60 * second)

# add monitors after equilibration
spks_m = SpikeMonitor(G)

if record_state:
    stat_m = StateMonitor(G, record_state_vars, record=t2b[record_state_idxs])

if record_rates:
    rate_m = PopulationRateMonitor(G)


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
                # t = trains[n]
                t = trains[t2b[n]]  # convert back from brian to topology indices
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

            dset = f.create_dataset("/data/neuron_stimulation_ids", data=stim_ids)
            dset.attrs[
                "description"
            ] = "List of neuron ids that were stimulation targets"

            dset = f.create_dataset(
                "/meta/dynamics_stimulated_modules",
                data=np.array(args.stimulation_module),
            )
            dset.attrs[
                "description"
            ] = "List of module ids that were stimulation targets"

        if record_state:
            # write the time axis once for all variables and neurons (should be shared!)
            t_axis = (stat_m.t - args.equil_duration) / second
            dset = f.create_dataset(
                "/data/state_vars_time", compression="gzip", data=t_axis
            )
            dset.attrs["description"] = "time axis of all state variables, in seconds"

            for idx, var in enumerate(record_state_vars):
                # no need to back-convert indices here, we specified which neurons
                # to record
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
        dset = f.create_dataset("/meta/dynamics_jA", data=jA / mV)
        dset.attrs["description"] = "AMPA current strength, in mV"

        dset = f.create_dataset("/meta/dynamics_jG", data=jG / mV)
        dset.attrs["description"] = "GABA current strength, in mV"

        dset = f.create_dataset("/meta/dynamics_jM", data=jM / mV)
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

        dset = f.create_dataset("/data/neuron_g", data=G.j[t2b])
        dset.attrs[
            "description"
        ] = "synaptic weight that was ultimately used for each neuron in the dynamic simulation"

        dset = f.create_dataset("/data/neuron_inhibitory_ids", data=inhib_ids)
        dset.attrs["description"] = "List of neuron ids that were set to be inhibitory"

        dset = f.create_dataset("/data/neuron_excitatory_ids", data=excit_ids)
        dset.attrs["description"] = "List of neuron ids that were set to be excitatory"

        f.close()

        print(f'#{"":#^75}#\n#{"All done!":^75}#\n#{"":#^75}#')

    except Exception as e:
        log.exception("Unable to save to disk")

# remove cython caches
try:
    shutil.rmtree(cache_dir, ignore_errors=True)
except Exception as e:
    log.exception("Unable to remove cached files")


# import plot_helper as ph

# h5f = ph.ah.prepare_file(args.output_path)
# ph.overview_dynamic(h5f)
