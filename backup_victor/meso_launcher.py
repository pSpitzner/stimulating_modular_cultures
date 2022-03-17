# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

import numpy as np
import mesoscopic_model as mm
import pathlib
import os
from tqdm import tqdm

print("Start...")

use_cpp = False
random_seed = 55436434

n_trajectories = 20


if use_cpp:

    # Then simulate it for several values of noise
    def simulate_model_noise(
        hvalues, tf, folder="var_noise", coupling=0.3, rseed=random_seed
    ):
        # Simulate the model many times with different values of noise
        for j, (h, t) in enumerate(zip(hvalues, tf)):
            os.system(
                f"./a.out {h} {coupling} {t} 0.0 2 0 {rseed} modeldata/{folder}/noise{j}"
            )
            print("...h=" + str(h) + " - done")

    timescale_span = np.linspace(0.0, 0.8, 30)
    coupling_span = np.array([0.1, 0.3, 0.6])
    tf = np.array([3000 for j in range(30)])

    # Different executable program for each random seed (yep, kinda overkill, but...)
    os.system(f"g++ -O3 mesoscopic_model.cpp")

    # For each coupling, generate several trajectories
    print("Simulating with several values external inputs...")
    for c in coupling_span:
        folder = f"coup{c:.2f}"
        for j in range(n_trajectories):
            rseed = (
                random_seed + 41533 * j * j + 12323 * j
            )  # Use a different random seed for each trajectory
            pathlib.Path(f"modeldata/{folder}-{j}").mkdir(exist_ok=True)
            simulate_model_noise(
                timescale_span, tf, folder=folder, coupling=c, rseed=rseed
            )

else:
    # Then simulate it for several values of noise
    def simulate_model_noise(hvalues, tf, folder="var_noise", coupling=0.3):
        # Simulate the model many times with different values of noise
        for j, (h, t) in tqdm(
            enumerate(zip(hvalues, tf)), desc="noise", leave=False, total=len(tf)
        ):
            mm.simulate_model(
                t,
                0.0,
                t,
                output_filename=f"{folder}/noise{j}",
                ext_str_=h,
                w0_=coupling,
                sigma_=0.15,
            )
            # print("...h=" + str(h) + " - done")

    timescale_span = np.linspace(0.0, 0.8, 30)
    coupling_span = np.array([0.1, 0.3, 0.6])
    tf = np.array([3000 for j in range(30)])

    print("Simulating with several values external inputs...")
    for j in tqdm(range(n_trajectories), desc="reps", leave=True):
        for c in tqdm(coupling_span, desc="coupling", leave=False):
            folder = f"coup{c:.2f}"
            simulate_model_noise(
                timescale_span, tf, folder=f"./modeldata/{folder}-{j}", coupling=c
            )
