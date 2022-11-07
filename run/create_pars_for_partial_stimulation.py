# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2022-06-22 10:12:19
# @Last Modified: 2022-11-07 16:56:08
# ------------------------------------------------------------------------------ #
# This creates a `parameter.tsv` where each line contains one parameter set,
# which can be directly called from the command line (i.e. on a cluster).
# * Here: parameters for stimulation of only two modules,
# * output file names have the form
#   `stim=02_k=5_jA=45.0_jG=0.0_jM=15.0_tD=20.0_rate=80.0_stimrate=0.0_rep=001.hdf5`
#   where `stim=02` means that modules at positon 0 and 2 receive an additonal
#   `stimrate` in addition the `rate` all modules receive.
# ------------------------------------------------------------------------------ #

import os
import numpy as np
from itertools import product

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))
out_path = os.path.abspath(os.path.curdir + f"./dat/simulations/lif/raw")
print(f"simulation results will go to {out_path}")

# seed for rank 0, will increase per thread
seed = 7_000

# parameters to scan, noise rate, ampa strength, and a few repetitons for statistics
l_k_inter = np.array([-1, 1, 5, 10])
l_mod = np.array(["02"])
l_rep = np.arange(0, 50)
l_jA = [45.0]
# as a control, check without inhibition
# l_jG = [0, 50.0]
l_jG = [50.0]
l_jM = [15.0]
l_tD = [20.0]
rate = 80
# for 20 Hz noise extra, results seemed consistent with experiments when inhib. enabled
l_stim_rate = [0, 20]

print("l_jA  ", l_jA)
print("l_jG  ", l_jG)
print("l_jM  ", l_jM)
print("l_tD  ", l_tD)
print("l_stim_rate", l_stim_rate)

bridge_weight = 1.0
inh_frac = 0.20

arg_list = product(l_k_inter, l_jA, l_jG, l_jM, l_tD, l_rep)

count_dynamic = 0
count_topo = 0
with open("./parameters.tsv", "w") as f_dyn:
    # set the cli arguments
    f_dyn.write("# commands to run, one line per realization\n")

    for args in arg_list:
        k_inter = args[0]
        jA = args[1]
        jG = args[2]
        jM = args[3]
        tD = args[4]
        rep = args[-1]
        mod = l_mod[0]

        seed += 1

        # same seeds for all rates so that topo matches
        for stim_rate in l_stim_rate:
            f_base = f"k={k_inter:d}_jA={jA:.1f}_jG={jG:.1f}_jM={jM:.1f}_tD={tD:.1f}_rate={rate:.1f}_stimrate={stim_rate:.1f}_rep={rep:03d}.hdf5"

            dyn_path = f"{out_path}/stim={mod}_{f_base}"

            if mod == "off":
                stim_arg = ""
            else:
                stim_arg = f"-stim poisson -mod {mod} -stim_rate {stim_rate:.1f} "

            f_dyn.write(
                # dynamic command
                f"python ./src/quadratic_integrate_and_fire.py "
                + f'-o "{dyn_path}" '
                + f"-k {k_inter} "
                + f"-d 1800 -equil 300 -s {seed:d} "
                + f"--bridge_weight {bridge_weight} "
                + f"--inhibition {inh_frac} "
                + f"-jA {jA} -jG {jG} -jM {jM} -r {rate} -tD {tD} "
                + f"{stim_arg}\n"
            )

            count_dynamic += 1

print(f"number of argument combinations for dynamics: {count_dynamic}")
