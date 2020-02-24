# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-02-20 09:35:48
# @Last Modified: 2020-02-21 12:10:06
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


parser = argparse.ArgumentParser(description="Brian")
parser.add_argument("-i", dest="input_path", help="input path", metavar="FILE")
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
args = parser.parse_args()
# if args.input_path == None or args.output_path == None:
# print("use correct arguments: -i, -o need help? -h")
# exit()

num_n = int(h5_load(args.input_path, "/meta/topology_num_neur"))
a_ij = h5_load(args.input_path, "/data/connectivity_matrix")
mod_ids = h5_load(args.input_path, "/data/neuron_module_id")
mod_sorted = np.argsort(mod_ids)


# ------------------------------------------------------------------------------ #
# brian
# ------------------------------------------------------------------------------ #


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


duration = 500 * ms

# fmt: off
# membrane potentials
vr = -60 * mV  # resting potential, neuron relaxes towards this without stimulation
vt = -45 * mV  # threshold potential
vp =  35 * mV  # peak potential, after vt is passed, rapid growth towards this
vc = -50 * mV  # reset potential

# soma
tc = 50 * ms
ta = 50 * ms  # time scale of inhibitory current u

k = 0.5 / mV  # resistance over capacity(?), rescaled
b = 0.5       # sensitivity to sub-threshold fluctuations
d =  50 * mV  # after-spike reset of inhibitory current u

# synapse
tD = 500 * ms  # characteristic recovery time, between 0.5 and 20 seconds
tA = 10 * ms  # decay time of post-synaptic current (AMPA current decay time)
gA = 10 * mV  # AMPA current strength

beta = 0.8    # D = beta*D after spike, to reduce efficacy, beta < 1
# fmt:on

G = NeuronGroup(
    N=num_n,
    model="""
        dv/dt = (k*(v-vr)*(v-vt) -u +I)/tc : volt      # [6] soma potential
        dI/dt = -I/tA : volt
        du/dt = (b*(v-vr) -u )/ta : volt               # [7] inhibitory current
        dD/dt = (1-D)/tD : 1                           # [11] recovery to one
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
        D_pre  -= (1-beta)*D_pre             # [11] delta-function term on spike
    """,
)

G.v = 'vc + 5*mV*rand()'

pre, post = np.where(a_ij == 1)
for idx, i in enumerate(pre):
    j = post[idx]
    # group modules close to each other
    i = np.argwhere(mod_sorted == i)[0, 0]
    j = np.argwhere(mod_sorted == j)[0, 0]
    S.connect(i=i, j=j)

# visualise_connectivity(S)

M = StateMonitor(G, "v", record=True)
spikemon = SpikeMonitor(G)
run(duration)

plot(spikemon.t / ms, spikemon.i, ".k")
xlabel("Time (ms)")
ylabel("Neuron index")

# figure()
# plot(G.v0 / mV, spikemon.count / duration)
# xlabel("v0 (mV)")
# ylabel("Firing rate (sp/s)")
# show()


figure()
plot(M.t / ms, M.v[25], label="Neuron 25")
plot(M.t / ms, M.v[130], label="Neuron 130")
xlabel("Time (ms)")
ylabel("v")
legend()
