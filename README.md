# Notes for Victor

I did quite a bit of refactoring.
A simple example of what works now: launch python from base directory,
```
import sys
sys.path.append("./ana")
sys.path.append("./run")

import meso_launcher as launcher
import meso_helper as mh
import plot_helper as ph
import ana_helper as ah

# simulate,
# for the example you can ctrl-c after the first iteration of the noise is done
launcher.main()

h5f = mh.prepare_file("./dat/meso_in/coup0.10-0/noise0.hdf5")
mh.find_system_bursts_and_module_contributions(h5f)
fig = ph.overview_dynamic(h5f)
fig.axes[-1].set_ylim(0, 5)


```


# Stimulating Modular Cultures

The code is structured into folders as follows:

- `./src/` contains simulation files to generate topologies and run neuron dynamics:

    - `topology.py` for merged and modular topologies and the axons growth algorithm proposed by Orlandi et al.

    - `quadratic_integrate_and_fire.py` for brian2 code, neuron dynamics and generation of spiking data

- `./run/` contains helpers to run the simulations from `src` on the cluster, and to create parameter combinations / sweeps that can be computed in serial.

- `./ana/` contains all post processing (and many non-relevant files, which will be removed). The ones that are relevant are:

    - `paper_plots.py` is the high-level wrapper that creates our figures.
        * at the top, we find functions named `fig_x()` that call all lower-level plotting routines, and load analyzed data form disk. This is a good starting point to trace back what comes from where.
        * statistical tests are implemented here, and use the dataframes produced by `process_conditions`

    - `process_conditions.py` analyses the raw spiking data from either experiments or simulations
        * provides pandas dataframes that are stored to a hdf5 file.
        * Frontmost, these dataframes contain the analysed data used to plot violins, esp. Fig 2. Calls `ana_helper` for most tasks.

    - `ndim_merge.py` is used only for simulation data
        * runs more analysis than `process_conditions`, but (mostly) focusus on scalar observables rather than distributions.
        * Produces the analyzed files need for Fig. 4 (some observable as a function of the simulated noise level). Calls `ana_helper` for most tasks.

    - `ana_helper.py` contains all the low-level analysis functions that operate on spike times, importers of simulation and experimental data and much more.
        * All analysis work on a nested dictionary, mostly  called `h5f`. Each `h5f` corresponds to one trial, and we attach all analysis results to this dict.
        * The idea is that later functions (such as plotting) can use previous results. Note that this is done live, and that although we "add" to the `h5f`, nothing is written to disk.

    - `plot_helper.py` provides various functions that take such a `h5f` dict and plot some aspect of it with consistent styling. Examples include timeseries, rasters, and distributions.

# Further Notes:

- Most code relies [my helpers](https://github.com/pSpitzner/pyhelpers), which I use across projects.
- Paths are so far hard coded, and will be refactored before publication.

