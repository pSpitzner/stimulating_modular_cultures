import os
import numpy as np
from itertools import product

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# seed for rank 0, will increase per thread
seed_0 = 3_000

# parameters to scan, noise rate, ampa strength, and a few repetitons for statistics
l_topo = ['2x2_fixed']
# l_rate = np.arange(25,41)
# l_gampa = np.arange(20,51)
# extend region
l_rate = np.array([37])
l_gampa = np.array([35])
l_recovery = np.array([2.0])
l_alpha = np.array([0.0125])
l_k_inter = np.array([1,5])
l_mod = ['02','012','0123']
l_rep = range(0, 50)

arg_list = list(product(l_topo, l_rate, l_gampa, l_recovery, l_alpha, l_k_inter, l_rep))

# we need to create the topology first for every seed!

# we want the same seeds for all modules.
for mod in l_mod:
    with open(f"./parameters_stim_{mod}.tsv", "w") as f_stim:
        f_stim.write("# commands with stimulation enabled\n")

        seed = seed_0
        count = 0
        for i in arg_list:
            topo = i[0]
            rate = i[1]
            gampa = i[2]
            recovery = i[3]
            alpha  = i[4]
            k_inter  = i[5]
            rep  = i[6]


            topo_path = f"./dat/topo/{topo}/gampa={gampa:04.2f}_rate={rate:04.2f}_recovery={recovery:04.2f}_alpha={alpha:.04f}_k={k_inter:d}_rep={rep:02d}.hdf5"
            dyn_path = f"./dat/dyn/{topo}/gampa={gampa:04.2f}_rate={rate:04.2f}_recovery={recovery:04.2f}_alpha={alpha:.04f}_k={k_inter:d}_rep={rep:02d}.hdf5"
            stim_path = f"./dat/jitter_{mod}/gampa={gampa:04.2f}_rate={rate:04.2f}_recovery={recovery:04.2f}_alpha={alpha:.04f}_k={k_inter:d}_rep={rep:02d}.hdf5"

            f_stim.write(
                # dynamic command
                f"python ./src/quadratic_integrate_and_fire.py -i {topo_path} " +
                f"-o {stim_path} " +
                f"-d 10800 -equil 300 -s {seed:d} " +
                f"-stim -mod {mod} " +
                f"-gA {gampa:04.2f} -tD {recovery:04.2f} -r {rate:04.2f}\n"
            )

            count += 1
            seed += 1


print(f"number of argument combinations: {count}")
