import os
import numpy as np
from itertools import product

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# seed for rank 0, will increase per thread
seed = 27_000

# parameters to scan, noise rate, ampa strength, and a few repetitons for statistics
l_topo = ["2x2_fixed"]
l_k_inter = np.array([5])
l_mod = np.array(["off"])
l_rep = np.arange(0, 100)
l_jA = np.arange(40, 46, 2.5).tolist()
l_jA.reverse()
# l_jG = np.arange(40, 101, 2.5)
l_jG = [67.5]
l_jM = [25.0]
l_rate = np.arange(24.0, 67.0, 2.0)

print("l_jA  ", l_jA)
print("l_jG  ", l_jG)
print("l_jM  ", l_jM)
print("l_rate", l_rate)

bridge_weight = 1.0
inh_frac = 0.20

arg_list = product(l_topo, l_k_inter, l_jA, l_jG, l_jM, l_rate, l_rep)

count_dynamic = 0
count_topo = 0
with open("./parameters_topo.tsv", "w") as f_topo:
    with open("./parameters_dyn.tsv", "w") as f_dyn:

        # set the cli arguments
        f_topo.write("# commands to create topology\n")
        f_dyn.write("# commands to run dynamics on existing topology\n")

        for args in arg_list:
            topo = args[0]
            k_inter = args[1]
            jA = args[2]
            jG = args[3]
            jM = args[4]
            rate = args[5]
            rep = args[6]

            f_base = f"k={k_inter:d}_jA={jA:.1f}_jG={jG:.1f}_jM={jM:.1f}_rate={rate:.1f}_rep={rep:03d}.hdf5"

            topo_path = f"./dat/inhibition_sweep_rate_1/topo/{f_base}"
            f_topo.write(
                # topology command
                f"/data.nst/share/projects/paul_brian_modular_cultures/topology_orlandi_standalone/exe/orlandi_standalone "
                + f"-N 100 -s {seed:d} -o {topo_path} "
                + f"-f {topo} -a 0.0125 -a_weighted 1 -k {k_inter}\n"
            )
            count_topo += 1

            for mod in l_mod:
                dyn_path = f"./dat/inhibition_sweep_rate_1/dyn/stim={mod}_{f_base}"

                if mod == "off":
                    stim_arg = ""
                else:
                    stim_arg = f"-stim hideaki -mod {mod}"

                f_dyn.write(
                    # dynamic command
                    f"python ./src/quadratic_integrate_and_fire.py -i {topo_path} "
                    + f"-o {dyn_path} "
                    + f"-d 3600 -equil 300 -s {seed:d} "
                    + f"--bridge_weight {bridge_weight} "
                    + f"--inhibition {inh_frac} "
                    + f"-jA {jA} -jG {jG} -jM {jM} -r {rate} "
                    + f"{stim_arg}\n"
                )

                count_dynamic += 1
                seed += 1


print(f"number of argument combinations for topology: {count_topo}")
print(f"number of argument combinations for dynamics: {count_dynamic}")
