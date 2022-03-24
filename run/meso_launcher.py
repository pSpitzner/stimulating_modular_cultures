# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

import numpy as np

import pathlib
import os
import sys
from tqdm import tqdm

use_cpp = False

if not use_cpp:
    sys.path.append("./src/")
    import mesoscopic_model as mm

rng_start_seed = 55436434
n_trajectories =  20
output_folder = "./dat/meso_in/"
cpp_exe_folder = "./exe/"

timescale_span = np.linspace(0.0, 0.8, 30)
coupling_span = np.array([0.01, 0.05, 0.5])
tf = np.array([1000 for j in range(30)])


def main():

    if use_cpp:
        func = simulate_model_noise_cpp
    else:
        func = simulate_model_noise_python

    # iterate over repetitions last so we can explore all phase space while waiting
    for rep in tqdm(range(n_trajectories), desc="repetitions", leave=True):
        for c in tqdm(coupling_span, desc="coupling", leave=False):
            # one folder for every coupling
            coupling_folder = f"{output_folder}/coup{c:.2f}-{rep:d}"
            os.makedirs(coupling_folder, exist_ok=True)

            # and one file for every noise value
            # the iteration over noise values happens inside the `simulate_model_noise()`
            # all noise values share the same RNG seed
            func(
                hvalues=timescale_span,
                tf=tf,
                coupling_folder=coupling_folder,
                coupling=c,
                rseed=rng_start_seed + rep * 41533,
            )


def simulate_model_noise_python(hvalues, tf, coupling_folder, coupling=0.3, rseed=815):

    for j, (h, t) in tqdm(
        enumerate(zip(hvalues, tf)), desc="noise", leave=False, total=len(tf)
    ):
        np.random.seed(rseed)
        mm.simulate_model(
            t,
            0.0,
            t,
            output_filename=f"{coupling_folder}/noise{j}",
            ext_str_=h,
            w0_=coupling,
            sigma_=0.15,
        )


def simulate_model_noise_cpp(hvalues, tf, coupling_folder, coupling=0.3, rseed=815):

    cpp_exe = f"{cpp_exe_folder}/mesoscopic_model"
    if not os.path.isfile(cpp_exe):
        print(f"Compiling cpp binary {cpp_exe}")
        os.makedirs(cpp_exe_folder, exist_ok=True)
        os.system(f"g++ -O3 ./src/mesoscopic_model.cpp -o {cpp_exe}")

    for j, (h, t) in tqdm(
        enumerate(zip(hvalues, tf)), desc="noise", leave=False, total=len(tf)
    ):
        os.system(
            f"{cpp_exe} {h} {coupling} {t} 0.0 2 0 {rseed} {coupling_folder}/noise{j}"
        )


if __name__ == "__main__":
    main()
