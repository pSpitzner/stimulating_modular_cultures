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
n_trajectories = 50
output_folder = "./dat/meso_in/"

# Parameters we will use for the simulations. For each coupling span, all external inputs will be evaluated.
# Each simulation can have a different length if needed.
coupling_span = np.array([0.05, 0.1, 0.4])
external_inputs = np.linspace(0.0, 0.3, 16)
simulation_time = 1000


def main():
    """
    Used to execute the launcher and start the simulations.
    """
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
                    meta_data=dict(
                        coupling=c,
                        noise=h,
                        rep=rep,
                    ),
                )


if __name__ == "__main__":
    main()
