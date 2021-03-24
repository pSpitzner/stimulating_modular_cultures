import os
import numpy as np
from itertools import product

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# seed for rank 0, will increase per thread
seed = 1_000

# parameters to scan, noise rate, ampa strength, and a few repetitons for statistics
l_topo = ["2x2_fixed"]
l_k_inter = np.array([1, 2, 3, 5])
l_mod = np.array(["off", "0", "02", "012", "0123"])
l_rep = range(0, 25)

bridge_weight = 0.5

arg_list = product(l_topo, l_k_inter, l_rep)

count_dynamic = 0
count_topo = 0
with open("./parameters_topo.tsv", "w") as f_topo:
    with open("./parameters_dyn.tsv", "w") as f_dyn:

        # set the cli arguments
        f_topo.write("# commands to create topology\n")
        f_dyn.write("# commands to run dynamics on existing topology\n")

        for i in arg_list:
            topo = i[0]
            k_inter = i[1]
            rep = i[2]

            topo_path = f"./dat/bridge_weights/topo/k={k_inter:d}_bw={bridge_weight:03.2f}_rep={rep:02d}.hdf5"
            f_topo.write(
                # topology command
                f"/data.nst/share/projects/paul_brian_modular_cultures/topology_orlandi_standalone/exe/orlandi_standalone "
                + f"-N 100 -s {seed:d} -o {topo_path} "
                + f"-f {topo} -a 0.0125 -a_weighted 1 -k {k_inter}\n"
            )
            count_topo += 1

            for mod in l_mod:
                dyn_path = f"./dat/bridge_weights/dyn/k={k_inter:d}_stim={mod}_bw={bridge_weight:03.2f}_rep={rep:02d}.hdf5"

                if mod == "off":
                    stim_arg = ""
                else:
                    stim_arg = f"-stim hideaki -mod {mod}"

                f_dyn.write(
                    # dynamic command
                    f"python ./src/quadratic_integrate_and_fire.py -i {topo_path} "
                    + f"-o {dyn_path} "
                    + f"-d 10800 -equil 300 -s {seed:d} "
                    + f"{stim_arg}\n"
                )

                count_dynamic += 1
                seed += 1


print(f"number of argument combinations for topology: {count_topo}")
print(f"number of argument combinations for dynamics: {count_dynamic}")
