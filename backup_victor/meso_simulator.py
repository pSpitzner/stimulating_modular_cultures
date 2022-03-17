# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga 
# @Email:         vbuendiar@onsager.ugr.es 
# ------------------------------------------------------------------------------ #

import numpy as np 
import mesoscopic_model as mm
import pathlib
import os

print("Start...")

use_cpp = True
random_seed = 51456434

if use_cpp:

    w0 = 0.3
    h = 0.0
    tf = 1800.0

    #Make a simulation of the model
    os.system(f"g++ -O3 mesoscopic_model.cpp")
    os.system(f"./a.out {h} {w0} {tf} 0.0 2 0 {random_seed} results_model")

    print("Simulation - done")

    #Do the same thing but without gates
    os.system(f"./a.out {h} {w0} {tf} 0.0 2 1 {random_seed} results_model_nogates")
    print("Simulation no gates - done")

    #Then simulate it for several values of noise
    def simulate_model_noise(hvalues, tf, folder="var_noise", coupling=0.3):
        #Simulate the model many times with different values of noise    
        for j,(h,t) in enumerate(zip(hvalues,tf)):
            os.system(f"./a.out {h} {coupling} {t} 0.0 2 0 {random_seed} ./modeldata/{folder}/noise{j}")
            print("...h="+str(h)+ " - done")

    timescale_span = np.linspace(0.0, 0.8, 30)
    coupling_span = np.array([0.1, 0.3, 0.6]) 
    tf = np.array([3000 for j in range(30)]) 


    simulate_model_noise(timescale_span, tf, folder="var_noise", coupling=0.3)
    exit()
    print("Simulating with several values external inputs...")
    for c in coupling_span:
        folder = f"coup{c:.2f}"
        print(folder)
        pathlib.Path(f"modeldata/{folder}").mkdir(exist_ok=True)
        simulate_model_noise(timescale_span, tf, folder=folder, coupling=c)

else:

    #Make a simulation of the model
    mm.simulate_model(1800.0, 600.0, 1200.0, "results_model")
    print("Simulation - done")

    #Do the same thing but without gates
    mm.simulate_model(1800.0, 600.0, 1200.0, "results_model_nogates", no_gates=True)
    print("Simulation no gates - done")

    #Then simulate it for several values of noise
    def simulate_model_noise(hvalues, tf, folder="var_noise", coupling=0.3):
        #Simulate the model many times with different values of noise    
        for j,(h,t) in enumerate(zip(hvalues,tf)):
            mm.simulate_model(t, 0.0, t, output_filename=f"modeldata/{folder}/noise{j}", ext_str_=h, w0_=coupling, sigma_=0.15)
            print("...h="+str(h)+ " - done")

    timescale_span = np.linspace(0.0, 0.8, 30)
    coupling_span = np.array([0.1, 0.3, 0.6]) 
    tf = np.array([3000 for j in range(30)]) 


    print("Simulating with several values external inputs...")
    for c in coupling_span:
        folder = f"coup{c:.2f}"
        print(folder)
        pathlib.Path(f"modeldata/{folder}").mkdir(exist_ok=True)
        simulate_model_noise(timescale_span, tf, folder=folder, coupling=c)

