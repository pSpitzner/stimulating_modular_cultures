import os
import numpy as np
from itertools import product

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# seed for rank 0, will increase per thread
seed = 3_000

# parameters to scan, noise rate, ampa strength, and a few repetitons for statistics
l_topo = ["2x2_fixed"]
l_k_inter = np.array([3])
l_mod = np.array(["off", "0", "02", "012", "0123"])
l_rep = range(0, 1)
l_gA = np.array([30, 35, 37, 40, 45])
l_gG = np.array([40, 60, 70, 80, 100])

bridge_weight = 1.0
inh_frac = 0.20

arg_list = product(l_gA, l_gG)

count_dynamic = 0
count_topo = 0
with open("./parameters_topo.tsv", "w") as f_topo:
    with open("./parameters_dyn.tsv", "w") as f_dyn:

        # set the cli arguments
        f_topo.write("# commands to create topology\n")
        f_dyn.write("# commands to run dynamics on existing topology\n")

        for i in arg_list:
            topo = "2x2_fixed"
            k_inter = 3
            rep = 0
            gA = i[0]
            gG = i[1]

            topo_path = f"./dat/inhibition_test/topo/k={k_inter:d}_bw={bridge_weight:03.2f}_inh={inh_frac:03.2f}_gA={gA:.1f}_gG={gG:.1f}_rep={rep:02d}.hdf5"
            f_topo.write(
                # topology command
                f"/data.nst/share/projects/paul_brian_modular_cultures/topology_orlandi_standalone/exe/orlandi_standalone "
                + f"-N 100 -s {seed:d} -o {topo_path} "
                + f"-f {topo} -a 0.0125 -a_weighted 1 -k {k_inter}\n"
            )
            count_topo += 1

            for mod in l_mod:
                dyn_path = f"./dat/inhibition_test/dyn/k={k_inter:d}_stim={mod}_bw={bridge_weight:03.2f}_inh={inh_frac:03.2f}_gA={gA:.1f}_gG={gG:.1f}_rep={rep:02d}.hdf5"

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
                    + f"--inhibition {inh_frac} -gA {gA} -gG {gG} "
                    + f"{stim_arg}\n"
                )

                count_dynamic += 1
                seed += 1


print(f"number of argument combinations for topology: {count_topo}")
print(f"number of argument combinations for dynamics: {count_dynamic}")
