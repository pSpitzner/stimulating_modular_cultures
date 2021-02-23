# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-18 13:06:40
# @Last Modified: 2021-02-22 20:03:32
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


matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

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

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut
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
vc = -35 * mV  # reset potential

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
defaultclock.dt = 0.01 * ms

# whether to record state variables
record_state = True
# which variables
record_state_vars = ["I", "v", "u", "D"]
# for which neurons
record_state_idxs = [0]

# whether to record population rates
record_rates = False
record_rates_freq = 50 * ms   # with which time resolution should rates be written to h5

numpy.random.seed(6626)

# ------------------------------------------------------------------------------ #
# model
# ------------------------------------------------------------------------------ #

G = NeuronGroup(
    N=1,
    model="""
        dv/dt = ( k*(v-vr)*(v-vt) -u +I                     # [6] soma potential
                  +xi*(gs/tc)**0.5      )/tc   : volt       # white noise term
        I : volt
        # dI/dt = -I/tA                          : volt       # [9, 10]
        du/dt = ( b*(v-vr) -u )/ta             : volt       # [7] membrane recovery
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

G2 = NeuronGroup(
    N=1,
    model="""
        dv/dt = ( k*(v-vr)*(v-vt) -u +I                     # [6] soma potential
                  +xi*(gs/tc)**0.5      )/tc   : volt       # white noise term
        dI/dt = -I/tA                          : volt       # [9, 10]
        du/dt = ( b*(v-vr) -u )/ta             : volt       # [7] membrane recovery
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
    target=G2,
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
# mini_g = PoissonInput(target=G2, target_var="I", N=1, rate=rate, weight=gm)
# mini_g2 = PoissonInput(target=G2, target_var="I", N=1, rate=75*Hz, weight=gm)

# connect synapses
S.connect('i==j')

# initalize to a somewhat sensible state. we could have different neuron types
G.v = "vr"
G.D = "1"
G.I = 0
G2.v = "vr"
G2.D = "1"
G2.I = 0


# ------------------------------------------------------------------------------ #
# Running
# ------------------------------------------------------------------------------ #

# add monitors after equilibration
spks_m = SpikeMonitor(G)

if record_state:
    stat_m = StateMonitor(G, record_state_vars, record=record_state_idxs)
    stat_m2 = StateMonitor(G2, record_state_vars, record=record_state_idxs)

if record_rates:
    rate_m = PopulationRateMonitor(G)


# add a plateau of more noise
# drive_times = np.random.uniform(low=30, high=40, size=int(10 * second * 74 * Hz)) * second

# stim_g = SpikeGeneratorGroup(
#     N=1,
#     indices=np.zeros(len(drive_times)),
#     times=drive_times,
#     name="extra_drive",
# )
# stim_s = Synapses(stim_g, G, on_pre="I_post += gm", name="apply_stimulation",)
# stim_s.connect(condition="i == j")


desc = dict()
desc["v"] = f"Membrane\npotential $\\bf v$"
desc["u"] = f"Recovery\nvariable $\\bf u$"
desc["I"] = f"Synaptic\ncurrent $\\bf I$"
desc["D"] = f"Synaptic\ndepression $\\bf D$"

def plot_panels(monitor, ax=None, palette='blues'):
    if ax is None:
        fig, ax = plt.subplots(len(record_state_vars), 1, sharex=True,
            figsize=(3.1, 1.5*len(record_state_vars)))
    else:
        fig = ax[0].get_figure()

    import colors as cc
    colors = cc.cmap_cycle(palette, edge=False, N=4)
    colors.reverse()

    for vdx, var in enumerate(record_state_vars):
        x = monitor.t
        y = monitor.variables[var].get_value()[:,0]
        yf = 1
        if var in ["v", "u", "I"]:
            yf = 1000
        ax[vdx].plot(x, y*yf, c=colors[vdx], zorder=1,
            clip_on=True if vdx == 1 else False)
        ax[vdx].set_ylabel(desc[var])
        ax[vdx].axes.xaxis.set_visible(False)
        ax[vdx].spines["left"].set_position(("outward", 5))

    ax[1].set_ylim(-65, 40)
    ax[1].axhline(vr/mV, ls=':', color='gray', zorder=0)
    ax[1].axhline(vt/mV, ls=':', color='gray', zorder=0)
    ax[1].axhline(vp/mV, ls=':', color='gray', zorder=0)
    ax[1].axhline(vc/mV, ls=':', color='gray', zorder=0)

    ax[0].set_ylim(0,60.0)
    ax[2].set_ylim(0,200.0)
    ax[3].set_ylim(0,1.0)


    ax[-1].set_xlim(0,None)
    ax[-1].spines["bottom"].set_visible(True)
    ax[-1].axes.xaxis.set_visible(True)
    ax[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()

    return fig, ax


# mini_g.active=False
# mini_g2.active=False
# run and record
G.I[0] = 0
run(.25*second, report="stdout", report_period=10 * second)

G.I[0] = 50 * mV
run(.5*second, report="stdout", report_period=10 * second)

G.I[0] = 0
run(.25*second, report="stdout", report_period=10 * second)

print(f"num spikes: {spks_m.spike_trains()[0].shape}")
print(np.diff(spks_m.spike_trains()[0]))

fig, ax = plot_panels(stat_m, palette='blues')
fig2, ax2 = plot_panels(stat_m2, palette='reds')

# remove cython caches
# try:
#     shutil.rmtree(cache_dir, ignore_errors=True)
# except Exception as e:
#     log.exception("Unable to remove cached files")
