import os
import numpy as np
from itertools import product

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# seed for rank 0, will increase per thread
seed = 5_000

# parameters to scan, noise rate, ampa strength, and a few repetitons for statistics
l_topo = ["2x2_fixed"]
l_k_inter = np.array([5])
l_mod = np.array(["off", "02"])
l_rep = np.arange(0, 25)
l_jA = [35.0, 30.0, 40.0]
l_jG = [50.0]
l_jM = [15.0]
l_jE = [10, 20, 30]
rate = 80

print("l_jA  ", l_jA)
print("l_jG  ", l_jG)
print("l_jM  ", l_jM)
print("l_jE", l_jE)
print("l_mod", l_mod)

bridge_weight = 1.0
inh_frac = 0.20

arg_list = product(l_topo, l_k_inter, l_jA, l_jG, l_jM, l_rep)

count_dynamic = 0
count_topo = 0
with open("./parameters_topo.tsv", "w") as f_topo:
    with open("./parameters_dyn.tsv", "w") as f_dyn:

        # set the cli arguments
        f_topo.write("# commands to create topology\n")
        f_dyn.write("# commands to run dynamics on existing topology\n")

        for args in arg_list:
            k_inter = args[1]
            topo = args[0]
            jA = args[2]
            jG = args[3]
            jM = args[4]
            rep = args[-1]

            seed += 1


            # same seeds for all stimulation strength so that topo matches
            f_base = f"k={k_inter:d}_jA={jA:.1f}_jG={jG:.1f}_jM={jM:.1f}_rep={rep:03d}.hdf5"
            topo_path = f"/scratch03.local/pspitzner/inhib02/dat/inhibition_sweep_jE_160/topo/{f_base}"
            f_topo.write(
                # topology command
                f"/data.nst/share/projects/paul_brian_modular_cultures/topology_orlandi_standalone/exe/orlandi_standalone "
                + f"-N 160 -s {seed:d} -o {topo_path} "
                + f"-f {topo} -a 0.0125 -a_weighted 1 -k {k_inter}\n"
            )
            count_topo += 1

            for mod in l_mod:
                for jE in l_jE:

                    if mod == "off":
                        if jE != l_jE[0]:
                            continue
                        jE = 0
                        stim_arg = "-jE 0 "
                    else:
                        stim_arg = f"-stim hideaki -mod {mod} -jE {jE} "
                    dyn_path = f"/scratch03.local/pspitzner/inhib02/dat/inhibition_sweep_jE_160/dyn/stim={mod}_jE={jE:.1f}_{f_base}"

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


print(f"number of argument combinations for topology: {count_topo}")
print(f"number of argument combinations for dynamics: {count_dynamic}")
