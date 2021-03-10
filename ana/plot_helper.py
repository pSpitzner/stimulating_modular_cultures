# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-09 11:16:44
# @Last Modified: 2021-03-10 11:43:16
# ------------------------------------------------------------------------------ #
# What's a good level of abstraction?
# Basic routines that plot on thing or the other, directly from file.
# ------------------------------------------------------------------------------ #


# fmt: off
import os
import sys
import glob
import h5py
import argparse
import logging

import matplotlib
# matplotlib.rcParams['font.sans-serif'] = "Arial"
# matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
# matplotlib.rcParams["axes.spines.right"] = False
# matplotlib.rcParams["axes.spines.top"] = False
# matplotlib.rcParams["axes.spines.left"] = False
# matplotlib.rcParams["axes.spines.bottom"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
    "#233954", "#ea5e48", "#1e7d72", "#f49546", "#e8bf58", # dark
    "#5886be", "#f3a093", "#53d8c9", "#f2da9c", "#f9c192", # light
    ]) # qualitative, somewhat color-blind friendly, in mpl words 'tab5'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from brian2.units.allunits import *

log = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut
import logisi as logisi
# requires my python helpers https://github.com/pSpitzner/pyhelpers
import hi5 as h5
from hi5 import BetterDict
import colors as cc

# fmt: on


def plot_raster(h5f, ax=None, sort_by_module=True, apply_formatting=True):
    """
        Plot a raster plot

        # Parameters:
        h5f : BetterDict
            of a loaded hdf5 file with the conventional entries.
            call `prepare_file(h5f)` to set everything up!
        sort_by_module : bool
            set to False to avoid reordering by module
        apply_formatting : bool
            if false, no styling changes will be done to ax
    """

    assert h5f.ana is not None, "`prepare_file(h5f)` before plotting!"

    log.info("Plotting Raster")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.set_rasterization_zorder(-1)

    for n_id in h5f.ana.neuron_ids:

        if sort_by_module:
            n_id_sorted = h5f.ana.mod_sort(n_id)
        else:
            n_id_sorted = n_id

        m_id = h5f.data.neuron_module_id[n_id]
        spikes = h5f.ana.spikes_2d[n_id]

        ax.plot(
            spikes,
            n_id_sorted * np.ones(len(spikes)),
            "|",
            alpha=0.5,
            zorder=-2,
            color=h5f.ana.mod_colors[m_id],
        )

    if apply_formatting:
        ax.margins(x=0, y=0)
        ax.set_ylabel("Raster")
        ax.set_xlabel("Time [seconds]")

    fig.tight_layout()

    return ax


def plot_module_rates(h5f, ax=None, mark_bursts=True, apply_formatting=True):

    assert h5f.ana is not None, "`prepare_file(h5f)` before plotting!"

    log.info("Plotting Module Rates")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    dt = h5f.ana.rates.dt

    for m_id in h5f.ana.mods:
        pop_rate = h5f.ana.rates.module_level[m_id]
        mean_rate = np.nanmean(pop_rate)
        ax.plot(
            np.arange(0, len(pop_rate)) * dt,
            pop_rate,
            label=f"{m_id:d}: ({mean_rate:.2f} Hz)",
            color=h5f.ana.mod_colors[m_id],
        )
        if mark_bursts:
            beg_times = h5f.ana.bursts.module_level[m_id].beg_times
            end_times = h5f.ana.bursts.module_level[m_id].end_times
            ax.plot(
                beg_times,
                np.ones(len(beg_times)) * (20 + m_id),
                marker="4",
                color=h5f.ana.mod_colors[m_id],
                lw=0,
            )
            ax.plot(
                end_times,
                np.ones(len(end_times)) * (20 + m_id),
                marker="3",
                color=h5f.ana.mod_colors[m_id],
                lw=0,
            )

    if mark_bursts:
        beg_times = h5f.ana.bursts.system_level.beg_times
        end_times = h5f.ana.bursts.system_level.end_times

        ax.plot(
            beg_times, np.ones(len(beg_times)) * (25), marker="4", color="black", lw=0
        )
        ax.plot(
            end_times, np.ones(len(end_times)) * (25), marker="3", color="black", lw=0
        )
        try:
            ax.axhline(y=h5f.ana.bursts.module_level[0].threshold, ls=":", color="gray")
        except:
            pass

    leg = ax.legend(loc=1)

    if apply_formatting:
        leg.set_title("Module Rates")
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("#e4e5e6")
        leg.get_frame().set_alpha(0.95)

        ax.margins(x=0, y=0)
        ax.set_ylabel("Rates [Hz]")
        ax.set_xlabel("Time [seconds]")

        fig.tight_layout()

    return ax


def plot_distribution_burst_duration(h5f, ax=None, apply_formatting=True):

    assert h5f.ana is not None, "`prepare_file(h5f)` before plotting!"

    log.info("Plotting Burst Duration Distribution")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kwargs = {
        "ax": ax,
        "kde": False,
        "binwidth": 2.5 / 1000,  # ms
        # "binrange": (0.06, 0.12),
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "poly",
    }

    for m_id in h5f.ana.mods:

        beg_times = h5f.ana.bursts.module_level[m_id].beg_times
        end_times = h5f.ana.bursts.module_level[m_id].end_times
        sns.histplot(
            data=end_times - beg_times,
            color=h5f.ana.mod_colors[m_id],
            alpha=0.2,
            **kwargs,
            label = f"Module {m_id}"
        )

    beg_times = h5f.ana.bursts.system_level.beg_times
    end_times = h5f.ana.bursts.system_level.end_times
    sns.histplot(
        data=end_times - beg_times,
        color='black',
        alpha=0,
        **kwargs,
        label = "System-wide"
    )

    if apply_formatting:
        ax.set_xlabel(r"Burst duration $D$ (seconds)")
        ax.set_ylabel(r"Probability $P(D)$")
        ax.legend()
    fig.tight_layout()

    return ax



def plot_parameter_info(h5f, ax=None, apply_formatting=True):

    log.info("Plotting Parameter Info")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if apply_formatting:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.margins(x=0, y=0)
        ax.set_ylabel("Parameters")
        ax.set_xlabel("")

    dat = []
    dat.append(["Connections k", h5f.meta.topology_k_inter])
    dat.append(["", ""])
    dat.append(["Noise Rate [Hz]", h5f.meta.dynamics_rate])
    dat.append(["gA [mV]", h5f.meta.dynamics_gA])
    try:
        stim_neurons = np.unique(h5f.data.stimulation_times_as_list[:, 0]).astype(int)
        stim_mods = np.unique(h5f.data.neuron_module_id[stim_neurons])
        stim_str = f"On {str(tuple(stim_mods)).replace(',)', ')')}"
    except:
        stim_str = "Off"
    dat.append(["Stimulation", stim_str])

    for d in dat:
        if d[0] != "" and d[1] != "":
            log.info(f"{d[0]:>22}:    {d[1]}")

    tab = ax.table(dat, loc="center", edges="open")
    cells = tab.properties()["celld"]
    for i in range(0, int(len(cells) / 2)):
        cells[i, 1]._loc = "left"

    if apply_formatting:
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()

    return ax


def prepare_file(h5f, mod_colors="auto"):
    """
        modifies h5f in place! (not on disk, only in RAM)

        # adds the following attributes:
        h5f.ana.mod_sort   : function that maps from neuron_id to sorted id, by module
        h5f.ana.mods       : list of unique module ids
        h5f.ana.mod_colors : list of colors associated with each module
        h5f.ana.neuron_ids
    """

    log.info("Preparing File")

    if isinstance(h5f, str):
        h5f = h5.recursive_load(h5f, hot=False)

    h5f.ana = BetterDict()
    num_n = h5f.meta.topology_num_neur

    # ------------------------------------------------------------------------------ #
    # mod sorting
    # ------------------------------------------------------------------------------ #
    try:
        # get the neurons sorted according to their modules
        mod_sorted = np.zeros(num_n, dtype=int)
        mod_ids = h5f.data.neuron_module_id[:]
        mods = np.sort(np.unique(mod_ids))
        if len(mods) == 1:
            raise NotImplementedError  # avoid resorting.
        temp = np.argsort(mod_ids)
        for n_id in range(0, num_n):
            mod_sorted[n_id] = np.argwhere(temp == n_id)

        h5f.ana.mods = mods
        h5f.ana.mod_sort = lambda x: mod_sorted[x]
    except:
        h5f.ana.mods = [0]
        h5f.ana.mod_sort = lambda x: x

    # ------------------------------------------------------------------------------ #
    # assign colors to modules so we can use them in every plot consistently
    # ------------------------------------------------------------------------------ #
    if mod_colors is False:
        h5f.ana.mod_colors = ["black"] * len(h5f.ana.mods)
    elif mod_colors is "auto":
        h5f.ana.mod_colors = [f"C{x}" for x in range(0, len(h5f.ana.mods))]
    else:
        assert isinstance(mod_colors, list)
        assert len(mod_colors) == len(h5f.ana.mods)
        h5f.ana.mod_colors = mod_colors

    # ------------------------------------------------------------------------------ #
    # spikes
    # ------------------------------------------------------------------------------ #

    # maybe change this to exclude neurons that did not spike
    # neuron_ids = np.unique(spikes[:, 0]).astype(int, copy=False)
    neuron_ids = np.arange(0, h5f.meta.topology_num_neur, dtype=int)
    h5f.ana.neuron_ids = neuron_ids

    # make sure that the 2d_spikes representation is nan-padded, requires loading!
    spikes = h5f.data.spiketimes[:]
    spikes[spikes == 0] = np.nan
    h5f.data.spiketimes = spikes

    # # now we need to load things. [:] loads to ram and makes everything else faster
    # # convert spikes in the convenient nested (2d) format, first dim neuron,
    # # then ndarrays of time stamps in seconds
    # spikes = h5f.data.spiketimes_as_list[:]
    # spikes_2d = []
    # for n_id in neuron_ids:
    #     idx = np.where(spikes[:, 0] == n_id)[0]
    #     spikes_2d.append(spikes[idx, 1])
    # # the outer array is essentially a list but with fancy indexing.
    # # this is a bit counter-intuitive
    # h5f.ana.spikes_2d = np.array(spikes_2d, dtype=object)

    # ------------------------------------------------------------------------------ #
    # bursts and module rates
    # ------------------------------------------------------------------------------ #

    bs_large = 0.02
    bs_small = 0.002  # seconds, small bin size
    rate_threshold = 15  # Hz
    merge_threshold = 0.1  # seconds, merge bursts if separated by less than this

    bursts = BetterDict()
    bursts.module_level = BetterDict()
    rates = BetterDict()
    rates.dt = bs_small
    rates.module_level = BetterDict()

    beg_times = []  # lists of length num_modules
    end_times = []

    log.info("Finding Bursts from Rates")

    for m_id in h5f.ana.mods:
        selects = np.where(h5f.data.neuron_module_id[:] == m_id)[0]
        pop_rate = logisi.population_rate(spikes[selects], bin_size=bs_small)
        pop_rate = logisi.smooth_rate(pop_rate, clock_dt=bs_small, width=bs_large)
        pop_rate = pop_rate / bs_small

        beg_time, end_time = logisi.burst_detection_pop_rate(
            spikes[selects],
            bin_size=bs_large,
            rate_threshold=rate_threshold,
            highres_bin_size=bs_small,
        )

        beg_time, end_time = logisi.merge_if_below_separation_threshold(
            beg_time, end_time, threshold=merge_threshold
        )

        beg_times.append(beg_time)
        end_times.append(end_time)

        rates.module_level[m_id] = pop_rate
        bursts.module_level[m_id] = BetterDict()
        bursts.module_level[m_id].beg_times = beg_time
        bursts.module_level[m_id].end_times = end_time
        bursts.module_level[m_id].rate_threshold = rate_threshold


    pop_rate = logisi.population_rate(spikes[:], bin_size=bs_small)
    pop_rate = logisi.smooth_rate(pop_rate, clock_dt=bs_small, width=bs_large)
    pop_rate = pop_rate / bs_small
    rates.system_level = pop_rate

    all_begs, all_ends, all_seqs = logisi.system_burst_from_module_burst(
        beg_times, end_times, threshold=merge_threshold,
    )

    bursts.system_level = BetterDict()
    bursts.system_level.beg_times = all_begs
    bursts.system_level.end_times = all_ends
    bursts.system_level.module_sequences = all_seqs

    h5f.ana.bursts = bursts
    h5f.ana.rates = rates

    return h5f
