# ------------------------------------------------------------------------------ #
# @Author:        Victor Buendia Ruiz-Azuaga
# @Email:         vbuendiar@onsager.ugr.es
# ------------------------------------------------------------------------------ #

from genericpath import exists
import numpy as np
import pandas as pd
import os
import h5py

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


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def simulate_model(
    # fmt:off
    simulation_time,
    gating_mechanism  = True,
    max_rsrc          = 1.0,
    tc                = 40.0,
    td                = 5.0,
    decay_r           = 1.0,
    sigma             = 0.1,
    basefiring        = 0.0,
    w0                = 0.01,
    gate_cls          = 1.0,
    gate_rec          = 0.025, # 1/50, here still specified as rate
    ext_str           = 0.0,
    k_sigm            = 1.6,
    thres_sigm        = 0.2,
    gain              = 10.0,
    thres_gate        = 0.5,
    k_gate            = 10.0,
    dt                = 0.01,
    rseed             = None,
    # fmt:on
):
    """
    Simulate the mesoscopic model up to time tf, with the given parameters.

    #Parameters:
    simulation_time: float
        When does the simulation end.
    output_filename : str
        Name of the file that will be stored in disk, as output_filename.csv.

    gating_mechanism  : bool, optional
        Control whether the gating mechanism is used (default: yes).
        If False, gates are not updated and activity can pass at all times.

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

    time_axis : 1d array, time stemps for all other timeseries
    activity : 2d array, timeseries of module rate. Shape: (n_module, n_timepoints)
    resources : 2d array, timeseries of module resources. Shape: (n_module, n_timepoints)

    """

    # Set random seed
    if rseed != None:
        np.random.seed(rseed)

    thermalization_time = simulation_time * 0.1
    recording_time = simulation_time
    simulation_time = simulation_time + thermalization_time

    # Binnings associated to such a time
    nt = int(simulation_time / dt)

    GATE_CONNECTED = 1  # allow transmission
    GATE_DISCONNECTED = 0  # nothing goes through

    # time series of variables
    # activity (firing rate), init to random
    rate = np.ones(shape=(4, nt), dtype="float") * np.nan
    rate[:, 0] = np.random.rand(4)
    # reources
    rsrc = np.ones(shape=(4, nt), dtype="float") * np.nan
    rsrc[:, 0] = np.random.rand(4) * max_rsrc

    # state of each gate at this time (directed) gate[from, to]
    gate = np.ones(shape=(4, 4), dtype="int") * GATE_CONNECTED

    # keep track of geates. lets keep the shape simple and set all non-existing gates to zero.
    gate_history = np.zeros(shape=(4, 4, nt), dtype="int")


    # Coupling matrix
    Aij = np.zeros(shape=(4, 4), dtype="int")  # Adjacency matrix
    Aij[0, 1] = 1
    Aij[1, 0] = 1
    Aij[0, 2] = 1
    Aij[2, 0] = 1
    Aij[1, 3] = 1
    Aij[3, 1] = 1
    Aij[2, 3] = 1
    Aij[3, 2] = 1

    # External input
    ext_input = np.ones(4) * ext_str

    # Auxiliary shortcut to pre-compute this constant
    # which is used in sigmoid transfer function
    aux_thrsig = np.exp(k_sigm * thres_sigm)

    # -------------
    # Simulation
    # -------------

    # Main computation loop: Milstein algorithm, assuming Ito interpretation
    for t in range(nt - 1):

        # Update each module, `src` -> source module, `tar` -> target module
        for tar in range(4):
            # Collect the part of input that arrives from other modules
            module_input = 0.0
            for src in range(4):
                # connection matrix. [from, to]. ajj is 0.
                if Aij[src, tar] == 1:
                    # Sum input to module tar, only through open gates
                    if gate[src, tar] == GATE_CONNECTED:
                        # module_input += w0 * rate[src, t]
                        module_input += w0 * rate[src, t] * rsrc[src, t]

            # this should not happen.
            # module_input *= 0.5


            # Collect pieces to update our firing rate, Milstein algorithm
            # Spontaneous decay to small firing
            term1 = dt*( - decay_r * (rate[tar, t] - basefiring))

            # Input from all sources
            # total_input = rsrc[tar, t] * (rate[tar, t] + module_input + ext_input[tar])
            total_input = rsrc[tar, t] * rate[tar, t] + module_input + ext_input[tar]

            term2 = dt*transfer_function(
                total_input,
                gain,
                k_sigm,
                thres_sigm,
                aux_thrsig,
            )

            # Noise (multiplicative under conditions! and additive)
            noise = 0
            noise += np.random.standard_normal() * sigma
            if rate[tar, t] > basefiring:
                noise += np.sqrt(rate[tar, t]) * np.random.standard_normal() * sigma
            # noise only sqrt dt?
            term3 = np.sqrt(dt) * noise

            rate[tar, t + 1] = rate[tar, t] + term1 + term2 + term3

            # resources are easier
            rsrc[tar, t + 1] = rsrc[tar, t] + dt * (
                (max_rsrc - rsrc[tar, t]) / tc - rate[tar, t] * rsrc[tar, t]  / td
            )

        # update gates for next time step
        old_gate = gate.copy()
        for src in range(4):
            for tar in range(4):
                # Store gate history for export, before updating [from, to, time]
                gate_history[src, tar, t] = gate[src, tar]
                gate_history[src, tar, t] = gate[src, tar]

                # update outgoing(!) gates, but only if the mechanism is enabled
                if not gating_mechanism:
                    continue

                # dont touch non-existing gate
                # we could skip before saving the gate history, but this helps debugging
                if Aij[src, tar] == 0:
                    continue

                # Close gate depending on activity of source
                if old_gate[src, tar] == GATE_CONNECTED:
                    prob = probability_to_disconnect(
                        rsrc[src, t], dt, thres_gate, k_gate, gate_cls
                    )
                    if np.random.rand() < prob:
                        gate[src, tar] = GATE_DISCONNECTED

                # Open gate with a characteristic time
                else:
                    prob = 1.0 - np.exp(-dt * gate_rec)
                    if np.random.rand() < prob:
                        gate[src, tar] = GATE_CONNECTED

    # this is a bit hacky...
    rec_start = int(thermalization_time / dt)
    rate = rate[:, rec_start:]
    rsrc = rsrc[:, rec_start:]
    gate_history = gate_history[:, :, rec_start:]

    time_axis = np.arange(0, nt - rec_start) * dt
    return time_axis, rate, rsrc, gate_history


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def probability_to_disconnect(
    resources, dt=0.01, thres_gate=1.0, k_gate=10.0, gate_cls=1.0
):
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
    return 1.0 - np.exp(
        -dt * (gate_cls - gate_sigm(resources, thres_gate, k_gate, gate_cls))
    )


# Sigmoid for gate response
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
        gain * (1.0 - expinpt) / (aux_thrsig * expinpt + 1.0)
        if inpt >= thres_sigm
        else 0.0
    )


def simulate_and_save(output_filename, meta_data=None, **kwargs):
    """
    Perform a simulation of the system and save it to the indicated path

    #Parameters
    output_filename : str
        Path to the output file. Extension (.hdf5) will be added automatically.
    meta_data : dict, optional
        key value pairs to save into the hdf5 file in the `/meta/` group.
        (use for parameters so they can be read back in)

    **kwargs : dict
        Any parameters that can be given to mesoscopic_model.simulate_model
    """

    # Perform model simulation
    time, activity, resources, gate_history = simulate_model(**kwargs)

    # Create the path if needed
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Create a DataFrame easy to read in our workflow and export as HDF
    df = pd.DataFrame(columns=["time"] + [f"mod_{m_cd}" for m_cd in range(1, 5)])
    df["time"] = time
    for m_cd in range(4):
        df[f"mod_{m_cd+1}"] = activity[m_cd, :]
        df[f"mod_{m_cd+1}_res"] = resources[m_cd, :]

    # For the first module, store also the dynamics of its gate
    # for gateind in range(2):
    # df[f"mod_gate_{gateind+1}"] = gate_history[gateind, :]

    # overwrite data if it already exists
    if ".hdf5" not in output_filename.lower():
        output_filename += ".hdf5"

    if os.path.exists(f"{output_filename}"):
        os.remove(f"{output_filename}")
    df.to_hdf(f"{output_filename}", f"/dataframe", complevel=9)

    # This is quite inconsistent, most data is saved with pandas, only this is
    # native hdf5. fixing requires a rewrite of meso_helper
    file = h5py.File(f"{output_filename}", "r+")
    file.create_dataset(f"/data/gate_history", data=gate_history, compression="gzip")
    file.close()

    if meta_data is not None:
        file = h5py.File(f"{output_filename}", "r+")
        for key in meta_data.keys():
            try:
                file.create_dataset(f"/meta/{key}", data=meta_data[key])
            except Exception as e:
                log.exception(e)
        file.close()
