import os
import numpy as np
from itertools import product

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# seed for rank 0, will increase per thread
seed = 10_000

# parameters to scan, noise rate, gaba strength, and a few repetitons for statistics
l_topo = ['2x2merged', '2x2_fixed']
l_rate = np.arange(0.01, 0.051, 0.005)
l_gaba = np.arange(10, 51, 5)
l_rep = range(0, 5)

arg_list = product(l_topo, l_rate, l_gaba, l_rep)

# we need to create the topology first for every seed!

count = 0
with open("./parameters.tsv", "w") as f:

    # set the cli arguments
    f.write("# _topologycmd_; _dynamiccmd_\n")

    for i in arg_list:
        topo = i[0]
        rate = i[1]
        gaba = i[2]
        rep  = i[3]
        path = f"./dat/{topo}/gaba={gaba:04.2f}_rate={rate:.4f}_rep={rep:d}.hdf5"

        f.write(
            # topology command
            f"/Users/paul/mpi/simulation/modular_cultures/_latest/exe/orlandi_standalone -N 100 -s {seed:d} -o {path} -f {topo}; " +
            # dynamic command
            f"python ./src/ibi.py -i {path} " +
            f"-o {path} " +
            f"-d 3600 -equil 300 -s {seed:d} " +
            f"-gA {gaba:04.2f} -r {rate:.4f}\n"
            )

        seed += 1
        count += 1


print(f"number of argument combinations: {count}")
