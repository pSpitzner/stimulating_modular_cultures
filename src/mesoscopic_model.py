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
log.setLevel("WARNING")
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
# see `simulate_model` parameter description
default_pars = dict(
    gating_mechanism  = True,
    max_rsrc          = 1.0,  # @victor used to be 2
    tau_charge        = 40.0,
    tau_discharge     = 5.0,
    tau_rate          = 1.0,
    sigma             = 0.1,
    w0                = 0.01,
    tau_disconnect    = 1.0,
    tau_connect       = 20.0, # @victor used to be 1/50
    ext_str           = 0.0,
    k_inpt            = 1.6,
    thrs_inpt         = 0.2, # @victor used to be 0.4
    gain_inpt         = 20.0, # @victor used to be 10
    thrs_gate         = 0.5,
    k_gate            = 10.0,
    dt                = 0.01,
    rseed             = None,
)
# fmt:on


def simulate_model(simulation_time, **kwargs):
    """
    Simulate the mesoscopic model with the given parameters.

    # Parameters:
    simulation_time: float
        Duration of the simulation in arbitrary units. use 1000 as a starting point
    gating_mechanism  : bool, optional
        Control whether the gating mechanism is used (default: True).
        If False, gates are not updated and activity can pass at all times.
    max_rsrc : float, optional
        Maximum amount of synaptic resources.
    tau_charge : float, optional
        Timescale of synaptical resource charging
    tau_discharge : float, optional
        Timescale of synaptical resource discharge
    tau_rate : float, optional
        Timescale of firing rate (activity) going to zero (exponential decay)
    sigma : float, optional
        Strength of background noise fluctuations
    w0 : float, optional
        Coupling strenght between different nodes
    tau_disconnect : float, optional
        Timescale of gate becoming inactive, thus not letting activity go through
    tau_connect : float, optional
        Timescale of gate recovery
    ext_str : float or np array of floats, optional
        Stimulation strength (for each module)
    k_inpt : float, optional
        Knee of the input sigmoid
    thrs_inpt: float, optional
        Threshold for the non-linear sigmoidal function mapping input to rate change
        Any activity which falls below this value will not affect the module
    gain_inpt : float, optional
        Gain that multiplies the result of the sigmoidal function,
        thus increasing the effect of the input
    thrs_gate : float, optional
        Threshold of activity needed in order to be able to affect the gate. Levels
        of activity below cannot inactivate a gate.
    k_gate : float, optional
        Knee of the gate's response sigmoid
    dt : float, optional
        Timestep of the Euler integrator (default=0.01)
    rseed : int, optional
        Use a custom random seed to ensure reproducitibility.
        If None (default), will use whatever Numpy selects

    # Returns
    time_axis : 1d array,
        time stamps for all other timeseries
    activity : 2d array,
        timeseries of module rate. Shape: (n_module, n_timepoints)
    resources : 2d array,
        timeseries of module resources. Shape: (n_module, n_timepoints)
    """
    pars = default_pars.copy()
    for key, value in kwargs.items():
        assert key in default_pars.keys(), f"unknown kwarg for mesoscopic model: '{key}'"
        pars[key] = value
    # make sure ext_str is upcast from float to vector of floats
    try:
        len(pars["ext_str"])
    except TypeError:
        log.debug("ext_str is a float, upcasting to array")
        pars["ext_str"] = np.array([pars["ext_str"]])
    if len(pars["ext_str"]) == 1:
        pars["ext_str"] = pars["ext_str"] * np.ones(4)
    elif len(pars["ext_str"]) == 4:
        pass
    else:
        raise ValueError("ext_str must be a float or a vector of length 4")
    log.debug(f"ext_str: {pars['ext_str']}")
    return _simulate_model(simulation_time, **pars)


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def _simulate_model(
    simulation_time,
    gating_mechanism,
    max_rsrc,
    tau_charge,
    tau_discharge,
    tau_rate,
    sigma,
    w0,
    tau_disconnect,
    tau_connect,
    # to do partial stimulation, we need this to be a vector.
    # wrap correctly on the python side.
    ext_str,
    k_inpt,
    thrs_inpt,
    gain_inpt,
    thrs_gate,
    k_gate,
    dt,
    rseed,
):
    """
    This guy is wrapped, so we can set default arguments via the dictionary.
    Numba does not like this.
    """

    # Set random seed
    if rseed != None:
        np.random.seed(rseed)

    thermalization_time = simulation_time * 0.1
    simulation_time = simulation_time + thermalization_time

    # Binnings associated to such a time
    nt = int(simulation_time / dt)

    GATE_CONNECTED = 1  # allow transmission
    GATE_DISCONNECTED = 0  # nothing goes through

    # time series of variables
    # activity (firing rate), init to random
    rate = np.ones(shape=(4, nt), dtype="float") * np.nan
    rate[:, 0] = np.random.rand(4)
    # resources
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

    # Auxiliary shortcut to pre-compute this constant
    # which is used in sigmoid transfer function
    aux_thrsig = np.exp(k_inpt * thrs_inpt)

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
                        module_input += w0 * rate[src, t] * rsrc[src, t]

            # this should not happen.
            # @victor, can you confirm, that we do not need this?
            # module_input *= 0.5

            # Collect pieces to update our firing rate, Milstein algorithm
            # Spontaneous decay, firing rate to zero
            term1 = dt * (-rate[tar, t] / tau_rate)

            # Input from all sources, recurrent, neighbours, external
            total_input = rsrc[tar, t] * rate[tar, t] + module_input + ext_str[tar]

            term2 = dt * transfer_function(
                total_input,
                gain_inpt,
                k_inpt,
                thrs_inpt,
                aux_thrsig,
            )

            # additive noise
            # @victor: noise only gets added with sqrt dt?
            term3 = np.sqrt(dt) * np.random.standard_normal() * sigma

            rate[tar, t + 1] = rate[tar, t] + term1 + term2 + term3

            # resources are easier
            rsrc[tar, t + 1] = rsrc[tar, t] + dt * (
                -(rate[tar, t] * rsrc[tar, t]) / tau_discharge
                + (max_rsrc - rsrc[tar, t]) / tau_charge
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

                # Disconnect gate depending on activity of source
                if old_gate[src, tar] == GATE_CONNECTED:
                    prob = _probability_to_disconnect(
                        rsrc[src, t], dt, thrs_gate, k_gate, tau_disconnect
                    )
                    if np.random.rand() < prob:
                        gate[src, tar] = GATE_DISCONNECTED

                # Connect gate with a characteristic time
                else:
                    prob = 1.0 - np.exp(-dt / tau_connect)
                    if np.random.rand() < prob:
                        gate[src, tar] = GATE_CONNECTED

    # this is a bit hacky...
    # to thermalize, simply chop off the indices that correspond to thermalization time
    rec_start = int(thermalization_time / dt)
    rate = rate[:, rec_start:]
    rsrc = rsrc[:, rec_start:]
    gate_history = gate_history[:, :, rec_start:]

    time_axis = np.arange(0, nt - rec_start) * dt
    return time_axis, rate, rsrc, gate_history


def probability_to_disconnect(resources, **kwargs):
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
    tau_disconnect : float
        Time scale at which the gate disconnects when no resources are available

    # Returns
    prob_close: float
        Probability of gate closing for the currently available number of resources.
    """
    pars = default_pars.copy()
    for key, value in kwargs.items():
        assert key in default_pars.keys(), f"unknown kwarg for mesoscopic model: '{key}'"
        pars[key] = value
    return _probability_to_disconnect(
        resources, pars["dt"], pars["thrs_gate"], pars["k_gate"], pars["tau_disconnect"]
    )


@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def _probability_to_disconnect(resources, dt, thrs_gate, k_gate, tau_disconnect):

    return 1.0 - np.exp(
        -dt
        * ((1 / tau_disconnect) - gate_sigm(resources, thrs_gate, k_gate, tau_disconnect))
    )


# Sigmoid for gate response
@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def gate_sigm(inpt, thrs_gate, k_gate, tau_disconnect):
    """
    Sigmoid that gives the response of the gate to the current level of resources

    #Parameters:
    inpt : float
        Level of resources
    thrs_gate : float
        Threshold. Below this value output is low, but not cut to zero
    k_gate : float
        Knee of the sigmoidal
    tau_disconnect : float
        Time scale at which the gate disconnects when no resources are available

    # Returns
    tau_disconnect_effective : float
        Rate of gate closing for the currently available number of resources.

    """
    return (1 / tau_disconnect) / (1.0 + np.exp(-k_gate * (inpt - thrs_gate)))


#
@jit(nopython=True, parallel=False, fastmath=False, cache=True)
def transfer_function(total_input, gain_inpt, k_inpt, thrs_inpt, aux_thrsig=None):
    """
    Gets the input to a module, given its input

    x/b = (1-exp( -k(rx+h-t) ) )/(1+exp(kt)*exp(-k(xr+h-t)))

    #Parameters
    inpt : float, or array of floats
        Neuronal activity input to the transfer function
    gain_inpt : float
        Maximum value returned by sigmoid for large inputs
    k_inpt : float
        Knee of the sigmoid
    thrs_inpt : float
        Threshold. Below this value, function returns 0
    aux_thrsig : float, optional
        Auxiliary variable defined as exp(k_inpt * thrs_inpt), can be precomputed

    #Returns
    feedback : float
        The result of applying the transfer function
    """

    if aux_thrsig is None:
        aux_thrsig = np.exp(k_inpt * thrs_inpt)

    expinpt = np.exp(-k_inpt * (total_input - thrs_inpt))

    if total_input >= thrs_inpt:
        return gain_inpt * (1.0 - expinpt) / (aux_thrsig * expinpt + 1.0)
    else:
        # we need to get consistent shapes.
        return total_input * 0.0


def single_module_odes(y, t, **pars):
    """
    Defines the coupled ODEs of the mesoscopic model.
    Use with numeric solver to explore nullclines and plot trajectories.

    Note that the `simulate_meso` function does not use this, and the ODEs are
    hard-coded in both places.

    # Parameters
    y : array-like
        (rate, resource)
    t : float
        time at which to evaluate the ODE, only needed for the solver, not used in def
    **pars : dict
        Parameters of the model, as a dict.
        at least needs to contain those keys:
            - "ext_str"
            - "thrs_inpt"
            - "gain_inpt"
            - "k_inpt"
            - "tau_discharge"
            - "tau_charge"
            - "tau_rate"
            - "max_rsrc"



    # Example
    ```
    pars = mm.default_pars.copy()
    ode_with_kwargs = functools.partial(single_module_odes, **pars)
    trajectory = scipy.integrate.odeint(
        func=ode_with_kwargs,
        y0=np.array([0.5, 1.0]),
        t=np.linspace(0, 1000, 5000),
    )
    ```
    """

    rate, rsrc = y
    # fmt:off
    rate_ode = \
        - rate / pars["tau_rate"] + 0.0 \
        + transfer_function(
            total_input = rate * rsrc + pars["ext_str"],
            gain_inpt   = pars["gain_inpt"],
            k_inpt      = pars["k_inpt"],
            thrs_inpt   = pars["thrs_inpt"],
        )
    rsrc_ode = \
        - rsrc * rate / pars["tau_discharge"] \
        + (pars["max_rsrc"] - rsrc) / pars["tau_charge"]
    # fmt:on

    return np.array([rate_ode, rsrc_ode])


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
