# Notes for Victor

I did quite a bit of refactoring.
A simple example of what works now: launch python from base directory, then
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

# plot a resource cycle
pp.meso_resource_cycle(h5f)

# to analyse all data and save
dset = mh.process_data_from_folder("./dat/meso_in/")
mh.write_xr_dset_to_hdf5(dset, "./dat/meso_out/analysed.hdf5")

# and to load back in
import xarray as xr
dset = xr.load_dataset("./dat/meso_out/analysed.hdf5"

# plot event size or rij
ax = pp.meso_obs_for_all_couplings(dset, "event_size")
```

Locations where code was moved:

- the models (cpp and py) are located in `src/`
- the the launcher is in [`run/meso_launcher.py`](./run/meso_launcher.py)
- the analysis and input/output are in [`ana/meso_helper.py`](ana/meso_helper.py)
- the plot functions are in `ana/paper_plots.py` where there is a big "meso" heading [near line 2692](ana/paper_plots.py#L2692) :P

Major changes:

- I restructured the launcher and have not tested the cpp.
- The `src/mesoscopic_model.py` now writes hdf5 files for compression and all code that relies on it has been updated accordingly - it should handle the old `.csv` and the new `.hdf5` just fine.  (If you want to add this to cpp, too I can provide some resources. For now, that does not seem important.)
- My code has the convention to start counting modules with `mod_0`. As a workaround, the data loader (`ana/meso_helper.py/_load_if_path()`) now makes an index shift whenever it does not find `mod_0` as a column.
- For preparing the data, I updated `read_csv_and_format` (for consistency, now called `prepare_file` in the `meso_helper.py`).
- Reworked the `module_contribution` function, now called `find_system_bursts_and_module_contributions`. Together, the two functions generate a few more details of the `h5f`, which allows more of the microscopic plot functions to work.

Notes:
- Depending on how much we want to persue this, we should come up with a better conventions for storing the simulation files. (its quite easy to write / extract parameter values either to metat data of hdf5 or to filenames via regex)
- For combining across many realizations I use [xarrays](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataset). Super cool stuff highly commended.
    - xarray dataframes are just n-dimensional numpy arrays with labels
    - xarray datasets are essentially a dict of multiple such frames that share the same coordinate systems (here I use one data frame for each observable)

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

