# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

import numpy as np

import os
import sys

sys.path.append("./src/")
import mesoscopic_model as mm
from tqdm import tqdm

# Set RNG seed and configure output folders
rng_start_seed = 55436434
n_trajectories = 15
output_folder = "./dat/meso_in_new_p_yes_gates/"

gating_mechanism = True

# Parameters we will use for the simulations. For each coupling span, all external inputs will be evaluated.
# Each simulation can have a different length if needed.
# coupling_span = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 1.0, 5.0])
coupling_span = np.array([0.0, 0.025, 0.04, 0.1, 5.0])
external_inputs = np.arange(0, 0.31, 0.025)
simulation_time = 1000


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
            coupling_folder = f"{output_folder}coup{c:.2f}-{rep:d}"
            os.makedirs(coupling_folder, exist_ok=True)

            # Loop over different external inputs,
            for j, h in tqdm(
                enumerate(external_inputs),
                total=len(external_inputs),
                desc="noise",
                leave=False,
            ):
                mm.simulate_and_save(
                    output_filename=f"{coupling_folder}/noise{j}",
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
                    ),
                )


if __name__ == "__main__":
    main()
