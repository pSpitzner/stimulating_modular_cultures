# Stimulating Modular Cultures


## Dependencies

All dependencies should be setup automatically if you create a new conda environment from the repository directory.
```bash
conda env create --file environment.yml
conda activate modular_cultures
```

## Getting started
We provide [interactive notebooks](notebooks) which serve as a good starting point, they showcase the models a bit and link which parts of the code led to which plot.


## Code structure
The repo is structured into folders as follows:

- [`src/`](src) contains the models for all simulations. In particular:
    - Merged and modular [topologies](src/topology.py) using axon growth algorithm proposed by Orlandi et al.
    - Leaky integrate and fire [neuron dynamics](src/quadratic_integrate_and_fire.py) that run in [Brian2](https://brian2.readthedocs.io/en/stable/) to generate artificial spiking data.
    - The [mesoscopic model](src/mesoscopic_model.py) where modules are treated as the smallest spacial unit.

- [`run/`](run) contains helpers to run the simulations from `src` on a cluster, and to create parameter combinations / sweeps that can be computed in serial.

- [`ana/](ana) contains all post-processing. Frontmost:
    - [`paper_plots.py`](ana/paper_plots.py) is the high-level wrapper that creates our figures.
    - [`process_conditions.py`](ana/process_conditions.py) analyses the raw spiking data from either experiments or simulations to create before-after comparisons
    - [`ndim_merge.py`](ana/ndim_merge.py) analyses simulation data, sweeping different input strength.
    - [`ana_helper.py`](ana/ana_helper.py) contains all the low-level analysis functions that operate on spike times, importers of simulation and experimental data and much more.
        * All analysis work on a nested dictionary, mostly  called `h5f`. Each `h5f` corresponds to one trial, and we attach all analysis results to this dict.
        * The idea is that later functions (such as plotting) can use previous results. Note that this is done live, and that although we "add" to the `h5f`, nothing is written to disk.
    - [`plot_helper.py`](ana/plot_helper.py) provides various functions that take such a `h5f` dict and plot some aspect of it with consistent styling. Examples include timeseries, rasters, and distributions.
