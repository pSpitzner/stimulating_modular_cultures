# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

from genericpath import exists
import numpy as np
import pandas as pd
import os

def simulate_model(tf, t_sti_start, t_sti_end, output_filename, no_gates=False, m_=2.0, tc_=5.0, td_=0.55, b_=1.55, sigma_=0.1, basefiring_=0.01, w0_=0.3, lamb_=0.6, g_=0.1, ext_str_=1.5, ksig_=1.6, basethr_=0.4, gain_=10.0, dt=0.01):
    """
    Simulate the simulated from t=0 to t=tf, stimulating from t_sti_start to t_sti_end. For model parameters please check function implementation.
    """
    #Stochastic dt
    sqdt = np.sqrt(dt)

    #Binnings associated to such a time
    nt = int(tf / dt)
    nt_start, nt_end = int(t_sti_start/dt), int(t_sti_end/dt)

    #Initialize variables
    x = np.empty((4,nt))                #activity
    mvar = np.empty((4,nt))             #current resources
    gate = np.zeros((4,4), dtype=int)   #state of each gate at this time (directed)

    #Model parameters!
    m0= m_      #Max amount of resources (spont, sti)
    tc, td = tc_, td_   #Charge and discharge timescale of resources
    b = b_            #Spontaneous decay of activity
    sigma = sigma_         #Noise intensity
    basefiring = np.ones(4) * basefiring_ #Minimum (small) firing rate

    #Coupling matrix
    w = np.zeros((4,4), dtype=int)  #Adjacency matrix
    w0 = w0_                        #Coupling between clusters
    w[0,1] = 1
    w[1,0] = 1
    w[0,2] = 1
    w[2,0] = 1
    w[1,3] = 1
    w[3,1] = 1
    w[2,3] = 1
    w[3,2] = 1

    lamb = lamb_      #Gate open timescale
    g = g_            #Gate recovery

    #External input
    ext_str = ext_str_
    ext_input = np.zeros(4)

    #Sigmoid definition (Wilson-Cowan type, such that f(x<=0)=0 and f(+infty)=gain)
    ksig = ksig_      #Slope
    basethr = basethr_   #Activation threshold
    gain = gain_      #Final gain
    thrsig = np.exp(ksig * basethr) #Shortcut to make computation faster

    def f(inpt):
        expinpt = np.exp(-ksig*(inpt-basethr))
        return gain*(1.0 - expinpt) / (thrsig*expinpt + 1.0) if inpt >= basethr else 0.0
    def f2(inpt, thrs, gamma, lmbda):
        return lmbda / (1.0 + np.exp(-gamma * (inpt - thrs)))

    #Initialize parameters to spontaneous, get initial value of resources
    m = np.ones(4) * m0

    mvar[:,0] = m.copy()
    x[:,0] = 0.01

    #Define some auxiliary constants for gates
    GATE_OPEN = 0
    GATE_CLOSED = 1

    # -------------
    # Simulation
    # -------------

    #Main computation loop: Milstein algorithm, assuming Ito interpretation
    t = 0.0
    for j in range(nt-1):

        #Change parameters during stimulation
        if (j < nt_start):
            ext_input[:2] = 0.0
        elif j < nt_end:
            ext_input[:2] = ext_str, ext_str
        else:
            ext_input[:2] = 0.0

        #Update each cluster
        old_gate = gate.copy()
        for c in range(4):

            suma = 0.0
            #Interaction with other connected clusters (usually one keeps a list of neighbours, etc. but we only have 4 clusters...)
            for neigh in range(4):
                if w[neigh, c] > 0.0:

                    suma += ((old_gate[neigh,c]==GATE_OPEN) or no_gates) * w0 * x[neigh,j]

                    gatethr = 0.5
                    #Close door depending on activity of source
                    if old_gate[c, neigh] == GATE_OPEN:
                        #prob = 1.0 - np.exp(-dt*lamb*x[c,j])
                        prob = 1.0 - np.exp(-dt * (1.0-f2(mvar[c,j], gatethr, 40.0, 1.0)))
                        cosa = np.random.rand()
                        if cosa < prob:
                            gate[c, neigh] = GATE_CLOSED
                    #Open door with a characteristic time
                    else:
                        prob = 1.0 - np.exp(-dt*g)
                        #prob = 1.0 - np.exp(-dt * f2(mvar[c,j], gatethr, 40.0, 1.0))
                        if np.random.rand() < prob:
                            gate[c, neigh] = GATE_OPEN

            #Multiplicative noise (+ extra additive)
            noise = np.random.standard_normal() * sigma
            if (x[c,j] > basefiring[c]):
                noise += 2*np.sqrt(x[c,j]) * np.random.standard_normal() * sigma

            #Terms for the deterministic system
            t1 = b*(x[c,j]-basefiring[c])                                           #Spontaneous decay to small firing
            t2 = f(mvar[c,j] * (x[c,j] + suma + ext_input[c]))     #Input to the cluster

            #Update our variables
            x[c,j+1] = x[c,j] + dt*(-t1 + t2) + sqdt*noise
            mvar[c,j+1] = mvar[c,j] + dt*((m[c] - mvar[c,j])/tc - mvar[c,j]*x[c,j]/td)

        t += dt #Update the time





    #------------
    # Output
    # -----------

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    #Create a DataFrame easy to read in our workflow
    df = pd.DataFrame(columns=["time"] + [f"mod_{m_cd}" for m_cd in range(1,5)])
    df["time"] = np.arange(0, tf, dt)
    for m_cd in range(4):
        df[f"mod_{m_cd+1}"] = x[m_cd,:]
        df[f"mod_{m_cd+1}_res"] = mvar[m_cd,:]
    # df.to_csv(f"{output_filename}.csv")
    df.to_hdf(f"{output_filename}.hdf5", f"/dataframe", complevel=9)
