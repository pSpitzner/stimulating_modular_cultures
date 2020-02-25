# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-02-20 09:35:48
# @Last Modified: 2020-02-25 10:40:49
# ------------------------------------------------------------------------------ #
# So we want to load my modular topology from hdf5 and run the simulations in
# brian. should make it easy for other people to reproduce ?!
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse

import numpy as np
from brian2 import *


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
        plot([0, 1], [i, j], "-k")
    xticks([0, 1], ["Source", "Target"])
    ylabel("Neuron index")
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, "ok")
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel("Source neuron index")
    ylabel("Target neuron index")


# ------------------------------------------------------------------------------ #
# model parameters
# ------------------------------------------------------------------------------ #

# fmt: off
# membrane potentials
vr = -60 * mV  # resting potential, neuron relaxes towards this without stimulation
vt = -45 * mV  # threshold potential
vp =  35 * mV  # peak potential, after vt is passed, rapid growth towards this
# vc = -50 * mV  # reset potential

# soma
tc = 50 * ms  # time scale of membrane potential
ta = 50 * ms  # time scale of inhibitory current u

k = 0.5 / mV  # resistance over capacity(?), rescaled
b = 0.5       # sensitivity to sub-threshold fluctuations
# d =  50 * mV  # after-spike reset of inhibitory current u

# synapse
tD = 1 *  second   # characteristic recovery time, between 0.5 and 20 seconds
tA = 10 * ms  # decay time of post-synaptic current (AMPA current decay time)
gA = 50 * mV  # AMPA current strength, between 10 - 50 mV

# noise
rate = 0.01 / ms  # rate for the poisson input (shot-noise), between 0.01 - 0.05 1/ms
gm =  30 * mV        # shot noise (minis) strength, between 10 - 50 mV
gs = 300 * mV * mV * ms * ms  # white noise strength, via xi = dt**.5 * randn()

beta = 0.8    # D = beta*D after spike, to reduce efficacy, beta < 1
# fmt:on

# defaultclock.dt = 0.01*ms

# ------------------------------------------------------------------------------ #
# command line arguments
# ------------------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Brian")
parser.add_argument("-i", dest="input_path", help="input path", metavar="FILE")
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
parser.add_argument("-gA", dest="gA", type=float)
parser.add_argument("-gm", dest="gm", type=float)
parser.add_argument("-rate", dest="rate", type=float)
parser.add_argument("-d", dest="d", type=float)

args = parser.parse_args()
if args.d is not None: duration = args.d * second
if args.gA is not None: gA = args.gA * mV
if args.gm is not None: gm = args.gm * mV
if args.rate is not None: rate = args.rate / ms


print("ga: ", gA)
print("rate: ", rate)
print("gm: ", gm)

seed(117)

num_n = int(h5_load(args.input_path, "/meta/topology_num_neur"))
a_ij = h5_load(args.input_path, "/data/connectivity_matrix")
mod_ids = h5_load(args.input_path, "/data/neuron_module_id")
mod_sorted = np.argsort(mod_ids)

# ------------------------------------------------------------------------------ #
# model
# ------------------------------------------------------------------------------ #

G = NeuronGroup(
    N=num_n,
    model="""
        dv/dt = (k*(v-vr)*(v-vt) -u +I +xi*(gs/tc)**0.5 )/tc : volt  # [6] soma potential
        dI/dt = -I/tA : volt
        du/dt = (b*(v-vr) -u )/ta : volt                 # [7] inhibitory current
        dD/dt = (1-D)/tD : 1                             # [11] recovery to one
        vc : volt
        d  : volt
    """,
    threshold="v > vp",
    reset="""
        v = vc        # [8]
        u = u + d     # [8]
    """,
    method="euler",
)

S = Synapses(
    source=G,
    target=G,
    on_pre="""
        I_post += gA*D_pre
        D_pre   = beta*D_pre             # [11] delta-function term on spike
    """,
)

# initalize to a sensible state
G.v = "vc + 5*mV*rand()"
for n in range(0, num_n):
    r = rand()
    if r < .05:
        G.vc[n] = -40 * mV
        G.d[n]  =  55 * mV
    elif r < .1:
        G.vc[n] = -35 * mV
        G.d[n]  =  60 * mV
    elif r < .2:
        G.vc[n] = -45 * mV
        G.d[n]  =  50 * mV
    else:
        G.vc[n] = -50 * mV
        G.d[n]  =  50 * mV

# shot-noise:
# by targeting I with poisson, we should get pretty close to javiers version.
# need to add dependence on the number of synapes/incoming connections
# P = PoissonInput(target=G, target_var="I", N=num_n, rate=rate, weight=gm)

ratess = 0.01 / ms + (0.04 / ms)* rand(num_n)
P = PoissonGroup(num_n, ratess)
Sp = Synapses(P, G, on_pre='I_post+=gm')
Sp.connect(j='i')

pre, post = np.where(a_ij == 1)
for idx, i in enumerate(pre):
    j = post[idx]
    # group modules close to each other
    i = np.argwhere(mod_sorted == i)[0, 0]
    j = np.argwhere(mod_sorted == j)[0, 0]
    S.connect(i=i, j=j)

run(10 * second, report='stdout') # equilibrate
# visualise_connectivity(S)


M = StateMonitor(G, ["v", "I", "u", "D"], record=True)
spikemon = SpikeMonitor(G)
spikemon_p = SpikeMonitor(P)
run(20 * second, report='stdout')


ion()  # interactive plotting
plot(spikemon_p.t / second, spikemon_p.i, ".y")
plot(spikemon.t / second, spikemon.i, ".k")
xlabel("Time (second)")
ylabel("Neuron index")

# figure()
# plot(G.v0 / mV, spikemon.count / duration)
# xlabel("v0 (mV)")
# ylabel("Firing rate (sp/s)")
# show()


figure()
plot(M.t / second, M.v[25], label="Neuron 25")
plot(M.t / second, M.v[130], label="Neuron 130")
xlabel("Time (s)")
ylabel("v")
legend()

figure()
# plot(M.t / second, M.I[130], label="130 I")
# plot(M.t / second, M.u[130], label="130 u")
plot(M.t / second, M.D[25], label="25 D")
xlabel("Time (s)")
ylabel("I")
legend()

# figure()
# plot(M.t / second, M.D[25], label="Neuron 25")
# plot(M.t / second, M.D[130], label="Neuron 130")
# xlabel("Time (s)")
# ylabel("D")
# legend()

show()


