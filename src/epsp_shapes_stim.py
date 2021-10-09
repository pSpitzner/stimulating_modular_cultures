# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-18 13:06:40
# @Last Modified: 2021-08-04 20:41:22
# ------------------------------------------------------------------------------ #
# Small script to investigate dynamic parameters and their impact on the
# single neuron level
# ------------------------------------------------------------------------------ #


import argparse
import os
import tempfile
import sys
import shutil
import numpy as np
import logging
from brian2 import *
import matplotlib
from tqdm import tqdm
import h5py

sys.path.append(os.path.abspath("/Users/paul/code/pyhelpers/"))


matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['figure.dpi'] = 150

# matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams['axes.labelcolor'] = "black"
matplotlib.rcParams['axes.edgecolor'] = "black"
matplotlib.rcParams['xtick.color'] = "black"
matplotlib.rcParams['ytick.color'] = "black"
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.bottom'] = False

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

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
tD =   2 * second  # characteristic recovery time, between 0.5 and 20 seconds
tA =  10 * ms      # decay time of post-synaptic current (AMPA current decay time)
jA =  45 * mV      # AMPA current strength, between 10 - 50 mV
                   # 170.612 value in javiers neurondyn
                   # this needs to scale with tV/tA
tG = 20 * ms       # decay time of post-syanptic GABA current
jG = 70 * mV       # GABA current strength

# noise
beta = 0.8         # D = beta*D after spike, to reduce efficacy, beta < 1
rate = 80 * Hz     # rate for the poisson input (shot-noise), between 10 - 50 Hz
jM   = 15 * mV     # shot noise (minis) strength, between 10 - 50 mV
                   # (sum of minis arriving at target neuron)
jS = 300 * mV * mV * ms * ms  # white noise strength, via xi = dt**.5 * randn()


# ------------------------------------------------------------------------------ #
# simulation parameters
# ------------------------------------------------------------------------------ #

# integration step size
# this turns out to be quite crucial for synchonization:
# when too large (brian defaul 0.1ms) this forces sth like an integer cast at
# some point and may promote synchronized firing. (spike times are not precise enough)
# heuristically: do not go above 0.05 ms, better 0.01ms
defaultclock.dt = 0.01 * ms

# whether to record state variables
record_state = True
# which variables
record_state_vars = ["Istim", "v", "u", "D", "IA"]
# for which neurons
record_state_idxs = [0]

# whether to record population rates
record_rates = False
record_rates_freq = 50 * ms   # with which time resolution should rates be written to h5

numpy.random.seed(6626)

# ------------------------------------------------------------------------------ #
# model
# ------------------------------------------------------------------------------ #

def init():

    G = NeuronGroup(
        N=1,
        model="""
            dv/dt = ( k*(v-vRef)*(v-vThr) -u +IA -IG + Istim  # [6] soma potential
                      +xi*(jS/tV)**0.5      )/tV   : volt       # white noise term
            dIA/dt = -IA/tA                        : volt       # [9, 10]
            dIG/dt = -IG/tG                        : volt       # [9, 10]
            du/dt = ( b*(v-vRef) -u )/tU         : volt       # [7] recovery variable
            dD/dt = ( 1-D)/tD                      : 1          # [11] recovery to one
            j     : volt  (constant)                 # neuron specific synaptic weight
            Istim : volt (constant)
            strength : volt (constant)
        """,
        threshold="v > vPeak",
        reset="""
            v = vReset        # [8]
            u = u + uIncr    # [8]
            D = D * beta     # [11] delta-function term on spike
        """,
        method="euler",
        dt=defaultclock.dt,
    )

    # 400 ms stimulus after 200 ms equilibration
    G.run_regularly("""
            stim_on = int(t >= 200 * ms and t <= 600 * ms)
            Istim = stim_on * strength
        """,
        dt = 200*ms)

    mini_g = PoissonInput(target=G, target_var="IA", N=1, rate=rate, weight=jM)

    # initalize to a somewhat sensible state. we could have different neuron types
    G.v = "vRef -2.5*mV + 5*mV*rand()"
    G.j = jA
    G.D = "1"
    G.IA = 0

    # add monitors after equilibration
    spks_m = SpikeMonitor(G)
    stat_m = StateMonitor(G, record_state_vars, record=record_state_idxs)


    net = Network()
    net_objs = dict(
        G=G, spks_m = spks_m, stat_m = stat_m, mini_g = mini_g,
    )
    for key in net_objs.keys():
        net.add(net_objs[key])
    net.store()
    net_objs["net"] = net
    return net_objs

def run(net_objs, stim_strength):
    net_objs["net"].restore()
    numpy.random.seed(None)
    net_objs["G"].strength = stim_strength * mV
    net_objs["net"].run(0.8*second, report=None, report_period=10 * second)
    spikes = net_objs["spks_m"].spike_trains()[0]
    spikes = spikes[spikes >= 200 * ms]
    # allow a 50 ms grace period more after stimulus ended
    spikes = spikes[spikes <= 650 * ms]
    num_spikes = len(spikes)
    time_to_first = np.nan*ms if len(spikes) == 0 else spikes[0] - 200*ms

    return num_spikes, time_to_first



def plot_panels(monitor, ax=None, palette='blues', n=0):
    desc = dict()
    desc["v"] = f"Membrane\npotential $\\bf v$"
    desc["u"] = f"Recovery\nvariable $\\bf u$"
    desc["IA"] = f"AMPA\ncurrent $\\bf I$"
    desc["Istim"] = f"External\ncurrent $\\bf I$"
    desc["D"] = f"Synaptic\ndepression $\\bf D$"

    if ax is None:
        fig, ax = plt.subplots(len(record_state_vars), 1, sharex=True,
            figsize=(3.1, 1.5*len(record_state_vars)))
    else:
        fig = ax[0].get_figure()

    import colors as cc
    colors = cc.cmap_cycle(palette, edge=False, N=4)
    colors.reverse()
    colors = [colors[n]]*len(record_state_vars)

    for vdx, var in enumerate(record_state_vars):
        x = monitor.t
        y = monitor.variables[var].get_value()[:,0]
        yf = 1
        if var in ["v", "u", "IA", "Istim"]:
            yf = 1000
        ax[vdx].plot(x, y*yf, c=colors[vdx], zorder=1,
            clip_on=True if vdx == 1 else False)
        ax[vdx].set_ylabel(desc[var])
        ax[vdx].axes.xaxis.set_visible(False)
        ax[vdx].spines["left"].set_position(("outward", 5))

    ax[1].set_ylim(-65, 40)
    ax[1].axhline(vRef/mV, ls=':', color='gray', zorder=0)
    ax[1].axhline(vThr/mV, ls=':', color='gray', zorder=0)
    ax[1].axhline(vPeak/mV, ls=':', color='gray', zorder=0)
    ax[1].axhline(vReset/mV, ls=':', color='gray', zorder=0)

    ax[0].set_ylim(0,60.0)
    ax[2].set_ylim(0,200.0)
    ax[3].set_ylim(0,1.0)


    ax[-1].set_xlim(0,None)
    ax[-1].spines["bottom"].set_visible(True)
    ax[-1].axes.xaxis.set_visible(True)
    ax[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()

    return ax

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge Multidm")
    parser.add_argument(
        "-o", dest="output_path", help="output path", metavar="FILE", required=True
    )
    parser.add_argument(
        "-s", dest="stim_strength", help="in mV", type=float,
    )
    return parser.parse_args()

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
    # import dask_helper as dh
    # dh.init_dask()
    args = parse_arguments()

    global times_to_first, num_spikes
    net_objs = init()
    num_reps = 1
    num_spikes = []
    times_to_first = []

    # f = functools.partial(run, net_objs=net_objs, stim_strength=stim_strength)
    # futures = dh.client.map(f, range(num_reps))
    # for future in tqdm(dh.as_completed(futures), total=len(futures)):
        # ns, tt = future.result()
    for rep in tqdm(range(num_reps), desc="repetition", leave=False):
        ns, tt = run(net_objs, args.stim_strength)
        num_spikes.append(ns)
        times_to_first.append(tt / ms)

        if rep == 0:
            plot_panels(net_objs["stat_m"])

    num_spikes = np.array(num_spikes)
    times_to_first = np.array(times_to_first)

    # f = h5py.File(args.output_path, "a")
    # dset = f.create_dataset("num_spikes", compression="gzip", data=num_spikes)
    # dset = f.create_dataset("times_to_first", compression="gzip", data=times_to_first)
    # f.close()
