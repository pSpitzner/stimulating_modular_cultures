# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-09 11:16:44
# @Last Modified: 2021-03-12 13:09:16
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
# import logisi as logisi
# requires my python helpers https://github.com/pSpitzner/pyhelpers
import hi5 as h5
from hi5 import BetterDict
import colors as cc
import ana_helper as ah

# fmt: on


def plot_raster(h5f, ax=None, apply_formatting=True, sort_by_module=True):
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
        spikes = h5f.data.spiketimes[n_id]

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


def plot_module_rates(h5f, ax=None, apply_formatting=True, mark_bursts=True):

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

    if h5f.ana.bursts is None:
        log.info("Finding Bursts from Rates")
        h5f.ana.bursts, h5f.ana.rates = ah.find_bursts_from_rates(h5f)

    bursts = h5f.ana.bursts

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
        beg_times = np.array(bursts.module_level[m_id].beg_times)
        end_times = np.array(bursts.module_level[m_id].end_times)
        sns.histplot(
            data=end_times - beg_times,
            color=h5f.ana.mod_colors[m_id],
            alpha=0.2,
            **kwargs,
            label=f"Module {m_id}",
        )

    beg_times = np.array(bursts.system_level.beg_times)
    end_times = np.array(bursts.system_level.end_times)
    sns.histplot(
        data=end_times - beg_times,
        color="black",
        alpha=0,
        **kwargs,
        label="System-wide",
    )

    if apply_formatting:
        ax.set_xlabel(r"Burst duration $D$ (seconds)")
        ax.set_ylabel(r"Probability $P(D)$")
        ax.legend()
    fig.tight_layout()

    return ax


def plot_distribution_isi(
    h5f, ax=None, apply_formatting=True, log_binning=True, which="all"
):
    """
        Plot the inter spike intervals in h5f.

        # Parameters
        h5f : BetterDict
            if `h5f.ana.isi` is not set, it will be calculated (and added to the file)
        which : str
            `all`, `in_bursts` or `out_bursts`. which isi to focus on, default `all`.

        log_binning : bool, True
            set to false for linear bins
    """
    if h5f.ana.isi is None:
        log.info("Finding ISIS")
        assert which == "all", "`find_bursts` before plotting isis other than `all`"
        h5f.ana.isi = ah.find_isis(h5f)

    isi = h5f.ana.isi

    log.info("Plotting ISI")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # we want to use the same bins across modules
    max_isi = 0
    for m_id in h5f.ana.mods:
        assert len(h5f.ana.isi[m_id][which]) > 0, f"No isis in h5f for '{which}'"
        isi = h5f.ana.isi[m_id][which]
        this_max_isi = np.max(isi)
        if this_max_isi > max_isi:
            max_isi = this_max_isi

    log.info(f"Largest ISI: {max_isi} (seconds)")

    if log_binning:
        max_isi = np.ceil(np.log10(max_isi))
        # start with one ms
        br = np.logspace(np.log10(1 / 1000), max_isi, num=100)
    else:
        br = np.linspace(0, max_isi / 10, num=100)

    kwargs = {
        "ax": ax,
        "kde": False,
        # "binwidth": 2.5 / 1000,  # ms
        # "binrange": (0.06, 0.12),
        "bins": br,
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "poly",
    }

    for m_id in h5f.ana.mods:
        sns.histplot(
            data=h5f.ana.isi[m_id][which],
            color=h5f.ana.mod_colors[m_id],
            alpha=0.2,
            **kwargs,
            label=f"Module {m_id} ({which})",
        )
    if log_binning:
        ax.set_xscale("log")

    if apply_formatting:
        ax.set_xlabel(r"Inter-spike Interval (seconds)")
        ax.set_ylabel(r"Probability")
        ax.legend()
    fig.tight_layout()


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
