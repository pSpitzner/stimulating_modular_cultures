Dynamics described in [Orlandi et al. 2013, DOI: 10.1038/nphys2686](https://dx.doi.org/10.1038/nphys2686).

Loads topologies from hdf5 and runs the simulations in brian2.

A simple example can be found in `src/dynamics.py`

We want to model optogenetic stimulation.
Stimulation Example:

```
python -it ./src/stimulate.py -i ./topologies/2x2_single.hdf5 -d 10 -o ./dat/test.hdf5 -stim
python ./ana/create_spike_movie_stim.py -i ./dat/test.hdf5 -o ./dat/test.mov -l 10 -tmin 0 -tmax 10
```


