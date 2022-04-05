# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

from genericpath import exists
import numpy as np
import pandas as pd
import os

import logging
import warnings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

try:
    from numba import jit, prange

    # raise ImportError
    log.info("Using numba for parallelizable functions")

    try:
        from numba.typed import List
    except:
        # older numba versions dont have this
        def List(*args):
            return list(*args)

    # silence deprications
    try:
        from numba.core.errors import (
            NumbaDeprecationWarning,
            NumbaPendingDeprecationWarning,
        )

        warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
        warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
    except:
        pass

except ImportError:
    log.info("Numba not available, skipping compilation")
    # replace numba functions if numba not available:
    # we only use jit and prange
    # helper needed for decorators with kwargs
    def parametrized(dec):
        def layer(*args, **kwargs):
            def repl(f):
                return dec(f, *args, **kwargs)

            return repl

        return layer

    @parametrized
    def jit(func, **kwargs):
        return func

    def prange(*args):
        return range(*args)

    def List(*args):
        return list(*args)


# fmt:off
@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def simulate_model(
    simulation_time,
    no_gates    = False,
    max_rsrc    = 2.0,
    tc          = 40.0,
    td          = 5.0,
    decay_r     = 1.0,
    sigma       = 0.1,
    basefiring  = 0.01,
    w0          = 0.3,
    gate_cls    = 0.7,
    gate_rec    = 0.03,
    ext_str     = 1.5,
    k_sigm      = 1.6,
    thres_sigm  = 0.4,
    gain        = 10.0,
    thres_gate  = 1.0,
    k_gate      = 10.0,
    dt          = 0.01,
    rseed       = None
):
# fmt:on
    """
    Simulate the mesoscopic model up to time tf, with the given parameters.

    #Parameters:
    simulation_time: float
        When does the simulation end.
    output_filename : str
        Name of the file that will be stored in disk, as output_filename.csv.

    no_gates : bool, optional
        Control whether gates are active. If False (default), the gates are used.

    max_rsrc : float, optional
        Maximum amount of synaptic resources.

    tc : float, optional
        Timescale of synaptical resource charging
    td : float, optional
        Timescale of synaptical resource discharge

    decay_r : float, optional
        Timescale of activity exponential decay

    sigma : float, optional
        Strenght of background noise fluctuations

    basefiring : float, optional
        Firing rate in absence of stimulation.

    w0 : float, optional
        Coupling strenght between different nodes

    gate_cls : float, optional
        Rate of gate becoming inactive, thus not letting activity go through
    gate_rec : float, optional
        Rate of gate recovery

    ext_str : float, optional
        Stimulation strength

    thres_sigm: float, optional
        Threshold for the non-linear sigmoidal function. Any activity which falls below this value will not affect the module
    gain : float, optional
        Gain that multiplies the result of the sigmoidal function, thus increasing the effect of the input

    thres_gate : float, optional
        Threshold of activity needed in order to be able to affect the gate. Levels of activity below cannot inactivate a gate.
    k_gate : float, optional
        Slope of the gate's response sigmoid

    dt_ : float, optional
        Timestep of the Euler integrator (default=0.01)

    rseed : float, optional
        Use a custom random seed to ensure reproducitibility. If None (default), will use whatever Numpy selects

    # Returns

    activity : 2d array, timeseries of module rate. Shape: (n_module, n_timepoints)
    resources : 2d array, timeseries of module resources. Shape: (n_module, n_timepoints)

    """

    #Set random seed
    if rseed != None:
        np.random.seed(rseed)

    # Stochastic dt
    sqdt = np.sqrt(dt)

    # Binnings associated to such a time
    nt = int(simulation_time / dt)

    # Initialize activity, resources, and gates 
    x = np.ones(shape=(4, nt), dtype="float")*np.nan     
    rsrc = np.ones(shape=(4, nt), dtype="float")*np.nan  
    gate = np.ones(shape=(4, 4), dtype="int")  # state of each gate at this time (directed) gate[from, to]

    # Coupling matrix
    w = np.zeros(shape=(4, 4), dtype="int")  # Adjacency matrix
    w[0, 1] = 1
    w[1, 0] = 1
    w[0, 2] = 1
    w[2, 0] = 1
    w[1, 3] = 1
    w[3, 1] = 1
    w[2, 3] = 1
    w[3, 2] = 1

    #Gate timescales
    #gate_rec = gate_rec / tc

    # External input
    ext_input = np.ones(4) * ext_str
    
    # Auxiliary shortcut to pre-compute this constant 
    # which is used in sigmoid transfer function
    aux_thrsig = np.exp(k_sigm * thres_sigm)  

    # Initialize parameters to spontaneous, get initial value of resources
    m = np.ones(4) * max_rsrc

    rsrc[:, 0] = np.random.rand(4) * max_rsrc
    x[:, 0] = np.random.rand(4)

    # Define some auxiliary constants for gates
    GATE_OPEN = 0
    GATE_CLOSED = 1

    # -------------
    # Simulation
    # -------------

    # Main computation loop: Milstein algorithm, assuming Ito interpretation
    t = 0.0
    for j in range(nt - 1):

        # Update each cluster
        old_gate = gate.copy()
        for c in range(4):

            module_input = 0.0
            # Interaction with other connected clusters (usually one keeps a list of neighbours, etc. but we only have 4 clusters...)
            for neigh in range(4):
                if w[neigh, c] > 0.0:

                    #Sum input to cluster c, only through open gates
                    if (no_gates or (old_gate[neigh, c] == GATE_OPEN)):
                        module_input += w0 * x[neigh, j]

                    # Close door depending on activity of source
                    if old_gate[c, neigh] == GATE_OPEN:
                        prob = probability_to_close(rsrc[c, j], dt, thres_gate, k_gate, gate_cls)
                        ranumb = np.random.rand()
                        if ranumb < prob:
                            gate[c, neigh] = GATE_CLOSED

                    # Open door with a characteristic time
                    else:
                        prob = 1.0 - np.exp(-dt * gate_rec)
                        if np.random.rand() < prob:
                            gate[c, neigh] = GATE_OPEN
            module_input *= 0.5

            # Multiplicative noise (+ extra additive)
            noise = np.random.standard_normal() * sigma
            if x[c, j] > basefiring:
                noise += np.sqrt(x[c, j]) * np.random.standard_normal() * sigma

            # Terms for the deterministic system
            t1 = decay_r * (x[c, j] - basefiring)  # Spontaneous decay to small firing
            t2 = transfer_function(rsrc[c, j] * (x[c, j] + module_input + ext_input[c]), gain, k_sigm, thres_sigm, aux_thrsig)  # Input to the cluster

            # Update our variables
            x[c, j + 1] = x[c, j] + dt * (-t1 + t2) + sqdt * noise
            rsrc[c, j + 1] = rsrc[c, j] + dt * (
                (m[c] - rsrc[c, j]) / tc - rsrc[c, j] * x[c, j] / td
            )

        t += dt  # Update the time

    time_axis = np.arange(0, simulation_time, dt)
    return time_axis, x, rsrc



# this should be the same?
def gate_deactivation_function(src_resources):
    """
    Auxiliar implementation of the response function of the gate to the number of resources. For plotting purposes only.

    #Parameters:
    src_resources: ndarray
        Returns the response function of the given array
    """

    def f2(inpt, thrs, gamma, lmbda):
        return lmbda / (1.0 + np.exp(-gamma * (inpt - thrs)))

    dt = 0.01
    prob = 1.0 - np.exp(-dt * (1.0 - f2(src_resources, 0.5, 40.0, 1.0)))

    return prob

@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def probability_to_close(resources, dt, thres_gate, k_gate, gate_cls):
    """
    Returns the probability of the gate to be closed depending on sigmoid response and currently available resources 

    #Parameters:
    resources : float
        Level of resources
    dt : float
        Integration timestep
    thrs_gate : float
        Threshold for sigmoidal. Below this value output is low, but not cut to zero
    k_gate : float
        Knee of the sigmoidal
    gate_cls : float
        Typical rate at which the gate closes when all resources are available
    
    # Returns
    prob_close: float
        Probability of gate closing for the currently available number of resources.
    
    """
    return 1.0 - np.exp(-dt * (gate_cls - gate_sigm(resources, thres_gate, k_gate, gate_cls)))

#Sigmoid for gate response
@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def gate_sigm(inpt, thrs, k_gate, gate_cls):
    """
    Sigmoid that gives the response of the gate to the current level of resources

    #Parameters:
    inpt : float
        Level of resources
    thrs : float
        Threshold. Below this value output is low, but not cut to zero
    k_gate : float
        Knee of the sigmoidal
    gate_cls : float
        Typical rate at which the gate closes when all resources are available
    
    # Returns
    gate_cls_effective : float
        Rate of gate closing for the currently available number of resources.
    
    """
    return gate_cls / (1.0 + np.exp(-k_gate * (inpt - thrs)))

#
@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def transfer_function(inpt, gain, k_sigm, thres_sigm, aux_thrsig):
    """
    Gets the input to a module, given its input

    #Parameters
    inpt : float
        Neuronal activity input to the transfer function
    gain : float
        Maximum value returned by sigmoid for large inputs
    k_sigm : float
        Knee of the sigmoid
    thres_sigm : float
        Threshold. Below this value, function returns 0
    aux_thrsig : float
        Auxiliary variable defined as exp(k_sigm * thres_sigm), precomputed for speed 

    #Returns
    feedback : float
        The result of applying the transfer function
    """
    expinpt = np.exp(-k_sigm * (inpt - thres_sigm))
    return (
        gain * (1.0 - expinpt) / (aux_thrsig * expinpt + 1.0) if inpt >= thres_sigm else 0.0
    )

def simulate_and_save(output_filename, **kwargs):
    """
    Perform a simulation of the system and save it to the indicated path

    #Parameters
    output_filename : str
        Path to the output file. Extension (.hdf5) will be added automatically.
    **kwargs : dict
        Any parameters that can be given to mesoscopic_model.simulate_model
    """

    #Perform model simulation
    time, activity, resources = simulate_model(**kwargs)

    #Create the path if needed
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Create a DataFrame easy to read in our workflow and export as HDF
    df = pd.DataFrame(columns=["time"] + [f"mod_{m_cd}" for m_cd in range(1, 5)])
    df["time"] = time
    for m_cd in range(4):
        df[f"mod_{m_cd+1}"] = activity[m_cd, :]
        df[f"mod_{m_cd+1}_res"] = resources[m_cd, :]


    df.to_hdf(f"{output_filename}.hdf5", f"/dataframe", complevel=9)
