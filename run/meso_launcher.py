# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

import numpy as np

import pathlib
import os
import sys
from tqdm import tqdm

# Select if we are using Python implementation or the C++ one.
# Python is preferred
use_cpp = False
if not use_cpp:
    sys.path.append("./src/")
    import mesoscopic_model as mm

# Set RNG seed and configure output folders
rng_start_seed = 55436434
n_trajectories = 1
output_folder = "./dat/meso_in/"
cpp_exe_folder = "./exe/"

# Parameters we will use for the simulations. For each coupling span, all external inputs will be evaluated.
# Each simulation can have a different length if needed.
coupling_span = np.array([0.1, 0.3, 0.6])
external_inputs = np.linspace(0.0, 0.5, 30)
tf = np.array([1000 for j in range(30)])


def main():
    """
    Used to execute the launcher and start the simulations.
    """

    # Select C++ or Python implementations, both are similar
    if use_cpp:
        func = simulate_model_noise_cpp
    else:
        func = simulate_model_noise_python

    # iterate over repetitions last so we can explore all phase space while waiting
    for rep in tqdm(range(n_trajectories), desc="repetitions", leave=True):
        for c in tqdm(coupling_span, desc="coupling", leave=False):
            # one folder for every coupling
            coupling_folder = f"{output_folder}coup{c:.2f}-{rep:d}"
            os.makedirs(coupling_folder, exist_ok=True)

            # and one file for every noise value
            # the iteration over noise values happens inside the `simulate_model_noise()`
            # all noise values share the same RNG seed
            func(
                ext_inputs=external_inputs,
                tf=tf,
                coupling_folder=coupling_folder,
                coupling=c,
                rseed=rng_start_seed + rep * 41533,
            )


def simulate_model_noise_python(
    ext_inputs, tf, coupling_folder, coupling, rseed=56454154
):
    """
    Perform a simulation of the model using the Python implementation.

    #Parameters
    ext_inputs : numpy array
        Array with different externals inputs where we want to perform simulations.
    tf : numpy array
        Array with the time duration of each simulation (low estimulation might require larger times for statistics)
    coupling_folder : str
        Folder where all these results will be stored as HDF files
    coupling : float
        Coupling between modules to use in this simulation, as a parameter
    rseed : float, optional
        A random seed to ensure reproducibility of the simulations
    """

    # Loop over different external inputs,
    for j, (h, t) in tqdm(
        enumerate(zip(ext_inputs, tf)), desc="noise", leave=False, total=len(tf)
    ):
        mm.simulate_model(
            t,
            output_filename=f"{coupling_folder}/noise{j}",
            ext_str=h,
            w0=coupling,
            sigma=0.1,
            tc=40.0,
            td=5.0,
            b=1.0,
            gatethr=1.4,
            rseed=rseed,
        )


def simulate_model_noise_cpp(ext_inputs, tf, coupling_folder, coupling=0.3, rseed=815):
    """
    Perform a simulation of the model using the C++ implementation.

    #Parameters
    ext_inputs : numpy array
        Array with different externals inputs where we want to perform simulations.
    tf : numpy array
        Array with the time duration of each simulation (low estimulation might require larger times for statistics)
    coupling_folder : str
        Folder where all these results will be stored as HDF files
    coupling : float
        Coupling between modules to use in this simulation, as a parameter
    rseed : float, optional
        A random seed to ensure reproducibility of the simulations
    """

    cpp_exe = f"{cpp_exe_folder}/mesoscopic_model"
    if not os.path.isfile(cpp_exe):
        print(f"Compiling cpp binary {cpp_exe}")
        os.makedirs(cpp_exe_folder, exist_ok=True)
        os.system(f"g++ -O3 ./src/mesoscopic_model.cpp -o {cpp_exe}")

    for j, (h, t) in tqdm(
        enumerate(zip(ext_inputs, tf)), desc="noise", leave=False, total=len(tf)
    ):
        os.system(
            f"{cpp_exe} {h} {coupling} {t} 0.0 2 0 {rseed} {coupling_folder}/noise{j}"
        )


if __name__ == "__main__":
    main()
