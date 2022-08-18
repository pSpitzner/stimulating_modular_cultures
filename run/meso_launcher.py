# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

import numpy as np

import os
import sys

sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/../src/")
import mesoscopic_model as mm
from tqdm import tqdm


rng_start_seed = 55436434
n_trajectories = 15 # number of repetitons per parameter combination

gating_mechanism = True
simulation_time = 1_000
output_folder = "./dat/simulations/meso/raw"

# ------------------------------------------------------------------------------ #
# Note on file naming convention:
# - Some file paths in the the ana/paper_plots.py are hardcoded
#   If you want the reproduce the paper plots 1:1
# - first, add `_no_gates` ot output_folder when disabling the gating mechanism
# - second, add `_long_ts` to output_folder to use a longer simulation time, e.g.
# ------------------------------------------------------------------------------ #
# gating_mechanism = False
# output_folder = "./dat/simulations/meso/raw_no_gates_long_ts"
# simulation_time = 10_000
# ------------------------------------------------------------------------------ #


# Combinations of coupling and external inputs.
# We create a folder per coupling and a file per input step.
# coupling_span = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 1.0, 5.0])
coupling_span = np.array([0.0, 0.025, 0.04, 0.1, 5.0])
external_inputs = np.arange(0, 0.31, 0.025)


def main():
    """
    Used to execute the launcher and start the simulations.
    """
    print(f"Gating mechanism is {'on' if gating_mechanism else 'off'}")
    print(f"Saving to {output_folder}")
    # iterate over repetitions last so we can explore all phase space while waiting
    for rep in tqdm(range(n_trajectories), desc="repetitions", leave=True):
        for c in tqdm(coupling_span, desc="coupling", leave=False):
            # one folder for every coupling
            coupling_folder = f"{output_folder}/coup{c:0.3f}-{rep:d}"
            # yes, the .2f precision turned out to be insufficient for the cpl values
            # but we load the value from metadata, not file name.
            os.makedirs(coupling_folder, exist_ok=True)

            # Loop over different external inputs,
            for j, h in tqdm(
                enumerate(external_inputs),
                total=len(external_inputs),
                desc="noise",
                leave=False,
            ):
                mm.simulate_and_save(
                    # output_filename=f"{coupling_folder}/noise{j}",
                    output_filename=f"{coupling_folder}/noise_{h:0.3f}",
                    simulation_time=simulation_time,
                    ext_str=h,
                    w0=c,
                    rseed=rng_start_seed + rep * 41533,
                    gating_mechanism=gating_mechanism,
                    meta_data=dict(
                        coupling=c,
                        noise=h,
                        rep=rep,
                        gating_mechanism=gating_mechanism,
                        seed = rng_start_seed + rep * 41533,
                    ),
                )


if __name__ == "__main__":
    main()
