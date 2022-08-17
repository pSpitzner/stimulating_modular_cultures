# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-09 11:16:44
# @Last Modified: 2022-08-17 16:16:20
# ------------------------------------------------------------------------------ #
# All the plotting is in here.
#
# What's a good level of abstraction?
# * Basic routines that plot one thing or the other, directly from file.
# * target an mpl ax element with normal functions and
# * provide higher level ones that combine those to `overview` panels
# ------------------------------------------------------------------------------ #

# fmt: off
import os
import sys
import logging

import matplotlib
# matplotlib.rcParams['font.sans-serif'] = "Arial"
# matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"]=8
matplotlib.rcParams["ytick.labelsize"]=8
matplotlib.rcParams["axes.titlesize"]= 8
matplotlib.rcParams["axes.labelsize"]= 8
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
# matplotlib.rcParams["axes.spines.right"] = False
# matplotlib.rcParams["axes.spines.top"] = False
# matplotlib.rcParams["axes.spines.left"] = False
# matplotlib.rcParams["axes.spines.bottom"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
    "#233954", "#ea5e48", "#1e7d72", "#f49546", "#e8bf58", # dark
    "#5886be", "#f3a093", "#53d8c9", "#f9c192", "#f2da9c" # light
    ]) # qualitative, somewhat color-blind friendly, in mpl words 'tab5'


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from tqdm import tqdm
from brian2.units.allunits import *
from benedict import benedict

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))

# our custom modules
from bitsandbobs import hi5 as h5
from bitsandbobs.plt import alpha_to_solid_on_bg
import ana_helper as ah

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("WARNING")
# fmt: on

# ------------------------------------------------------------------------------ #
# overview panels
# ------------------------------------------------------------------------------ #


def overview_topology(h5f, filenames=None, skip_graph=False):
    """
    Plots an overview figure of the topology of `h5f`.

    If `filenames` are provided, they are passed to the underlying plot
    functions and, where possible, ensemble statistics across filenames are
    used.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[12, 8])

    if not "ana" in h5f.keypaths():
        ah.prepare_file(h5f)

    plot_distribution_dendritic_tree_size(h5f, axes[0, 0], filenames=filenames)
    plot_distribution_axon_length(h5f, axes[1, 0], filenames=filenames)
    plot_distribution_degree_k(h5f, axes[1, 1], filenames=filenames)
    plot_parameter_info(h5f, axes[0, 1])

    if not skip_graph:
        plot_connectivity_layout(h5f, axes[0, 2])
        plot_axon_layout(h5f, axes[1, 2])
        axes[0, 2].set_title("Connectivity")
        axes[1, 2].set_title("Axons")
        # share zoom between connectivity and axons
        axes[0, 2].get_shared_x_axes().join(axes[0, 2], axes[1, 2])
        axes[0, 2].get_shared_y_axes().join(axes[0, 2], axes[1, 2])
        axes[1, 2].set_aspect(1)
        axes[1, 2].autoscale()

    fig.tight_layout()

    return fig


def overview_dynamic(h5f, filenames=None, threshold=None, states=True, skip=[]):

    # if we have recorded depletion, let's plot that, too
    depletion = False
    if "data.state_vars_D" in h5f.keypaths() and states:
        depletion = True

    fig = plt.figure(figsize=(4, 7.5))
    axes = []
    if depletion:
        gs = fig.add_gridspec(5, 2)  # [row, column]
    else:
        gs = fig.add_gridspec(4, 2)  # [row, column]
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[1, :]))
    axes.append(fig.add_subplot(gs[2, :], sharex=axes[1]))
    axes.append(fig.add_subplot(gs[3, :], sharex=axes[1]))
    axes.append(fig.add_subplot(gs[0, 0]))
    if depletion:
        axes.append(fig.add_subplot(gs[4, :], sharex=axes[1]))

    if not "ana" in h5f.keypaths():
        ah.prepare_file(h5f)

    if not "ana.rates" in h5f.keypaths():
        ah.find_rates(h5f, bs_large=20 / 1000)

    if threshold is not None or "ana.bursts" not in h5f.keypaths():
        # (re) do the detection so that we do not have default values
        # ah.find_bursts_from_rates(h5f, rate_threshold=threshold)
        if threshold == "max" or "ana.bursts" not in h5f.keypaths():
            rate_threshold = 0.025 * np.nanmax(h5f["ana.rates.system_level"])
        else:
            rate_threshold = threshold

        ah.find_system_bursts_from_global_rate(
            h5f, rate_threshold=rate_threshold, merge_threshold=0.1
        )
        ah.remove_bursts_with_sequence_length_null(h5f)
        ah.find_ibis(h5f)

    plot_parameter_info(h5f, axes[0])
    if not "raster" in skip:
        try:
            # this may happen for the mesoscopic model
            plot_raster(h5f, axes[1])
        except Exception as e:
            log.info("Failed to plot raster")
            log.debug(e)
    if not "rates" in skip:
        plot_module_rates(h5f, axes[2])
        plot_system_rate(h5f, axes[2])
    if not "bursts" in skip:
        ax = plot_bursts_into_timeseries(h5f, axes[3], style="markers")
        _style_legend(ax.legend(loc=1))
        ax = plot_bursts_into_timeseries(
            h5f, axes[1], apply_formatting=False, style="fill_between"
        )
        try:
            plot_gate_history(h5f, axes[3])
        except:
            log.debug("Gate states are only defined for the mesoscopic model")
    if not "init" in skip:
        try:
            ax = plot_initiation_site(h5f, axes[4])
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.xaxis.set_minor_locator(plt.NullLocator())
        except Exception as e:
            log.info("Failed to plot initiating module")
            log.debug(e)

    axes[1].set_xlabel("")
    axes[2].set_xlabel("")

    if depletion and not "depletion" in skip:
        axes[3].set_xlabel("")
        ax = plot_state_variable(h5f, axes[5], variable="D")

    fig.tight_layout()

    # plot_distribution_participating_fraction(h5f)

    return fig


def overview_burst_duration_and_isi(h5f, filenames=None, which="all"):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(4, 6),
        gridspec_kw=dict(height_ratios=[1, 3, 3]),
    )
    plot_distribution_burst_duration(h5f, ax=axes[1], filenames=filenames)
    plot_distribution_isi(h5f, ax=axes[2], filenames=filenames, which=which)
    comments = [
        ["hist bin size", "1ms"],
        ["rate bin size", f'{h5f[f"ana.rates.dt"]*1000}ms'],
    ]
    plot_parameter_info(h5f, ax=axes[0], add=comments)

    axes[1].set_xlim(0, 0.4)
    axes[2].set_xlim(-3, 3)

    for i in range(4):
        fig.tight_layout()

    return fig


# ------------------------------------------------------------------------------ #
# main plot level functions
# ------------------------------------------------------------------------------ #

# decorator for lower level plot functions to continue if subplot fails
def warntry(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log.exception(f"{func.__name__}: {e}")

    return wrapper


def plot_raster(
    h5f,
    ax=None,
    apply_formatting=True,
    sort_by_module=True,
    neuron_id_as_y=True,
    neurons=None,
    base_color=None,
    exclude_nids=[],
    **kwargs,
):
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
    kwargs = kwargs.copy()
    assert "ana" in h5f.keypaths(), "`prepare_file(h5f)` before plotting!"

    log.info("Plotting Raster")
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 1.3))
    else:
        fig = ax.get_figure()

    ax.set_rasterization_zorder(-1)

    if len(h5f["ana.neuron_ids"]) > 500:
        marker = "."
        kwargs.setdefault("alpha", 1)
        kwargs.setdefault("markersize", 1.5)
        kwargs.setdefault("markeredgewidth", 0)
    else:
        marker = "."
        kwargs.setdefault("alpha", 0.75)
        kwargs.setdefault("markersize", 2.0)
        kwargs.setdefault("markeredgewidth", 0)

    # sort by module, access as [:] to load, in case h5 dataset
    if neurons is None:
        neurons = h5f["ana.neuron_ids"][:].copy()
    else:
        neurons = np.asarray(neurons)
    if sort_by_module:
        idx = _argsort_neuron_ids_by_module(h5f, neurons)
        neurons = neurons[idx]

    mods = h5f["data.neuron_module_id"][:][neurons]
    last_mod = np.nan
    num_mods = len(h5f["ana.mod_ids"])
    # offset = len(neurons)
    offset = 0
    offset_add = len(neurons) / (len(neurons) + num_mods)

    for ndx, n_id in enumerate(neurons):
        # if n_id > 1000:
        # continue

        m_id = mods[ndx]
        # m_id = h5f["data.neuron_module_id"][n_id]

        spikes = h5f["data.spiketimes"][n_id]
        spikes = spikes[np.where(np.isfinite(spikes))]

        if neuron_id_as_y:
            offset = ndx
        else:
            offset += offset_add
            if last_mod != m_id:
                offset += offset_add
            last_mod = m_id

        log.debug(f"neuron {n_id} module {m_id} at {offset}")

        plot_kws = kwargs.copy()
        if base_color is None:
            plot_kws.setdefault("color", h5f["ana.mod_colors"][m_id])
        else:
            plot_kws.setdefault(
                "color",
                alpha_to_solid_on_bg(base_color, (num_mods - m_id) / num_mods),
            )

        ax.plot(
            spikes,
            offset * np.ones(len(spikes)),
            marker,
            **plot_kws,
        )

    if apply_formatting:
        ax.margins(x=0, y=0)
        ax.set_xlim(0, None)
        ax.set_ylabel("Raster")
        ax.set_xlabel("Time [seconds]")
        ax.set_ylim(0, len(h5f["ana.neuron_ids"]))
        try:
            if len(h5f["ana.mods"]) == 4 and len(h5f["ana.neuron_ids"]) == 160:
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
                ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.set_ylim(0, 160)
        except:
            pass
        fig.tight_layout()

    return ax


def plot_module_rates(
    h5f, ax=None, apply_formatting=True, mark_burst_threshold=False, **kwargs
):

    assert "ana" in h5f.keypaths(), "`prepare_file(h5f)` before plotting!"
    kwargs = kwargs.copy()
    log.info("Plotting Module Rates")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if not "ana.rates" in h5f.keypaths():
        ah.find_rates(h5f)

    dt = h5f["ana.rates.dt"]

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        pop_rate = h5f[f"ana.rates.module_level"][m_dc]
        # log.info(f"Threshold from SNR: {ah.get_threshold_via_signal_to_noise_ratio(pop_rate)}")
        log.info(f'CV {m_dc}: {h5f[f"ana.rates.cv.module_level"][m_dc]:.3f}')
        mean_rate = np.nanmean(pop_rate)
        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault("color", h5f[f"ana.mod_colors"][m_id])
        plot_kwargs.setdefault("alpha", 0.5)
        plot_kwargs.setdefault("label", f"{m_id:d}: {mean_rate:.2f} Hz")
        ax.plot(np.arange(0, len(pop_rate)) * dt, pop_rate, **plot_kwargs)

        if mark_burst_threshold:
            try:
                ax.axhline(
                    y=h5f["ana.bursts.module_level.mod_0.rate_threshold"],
                    ls=":",
                    color=h5f[f"ana.mod_colors"][m_id],
                )
            except Exception as e:
                log.debug(e)

    leg = ax.legend(loc=1)

    if apply_formatting:
        leg.set_title("Module Rates")
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("#e4e5e6")
        leg.get_frame().set_alpha(0.95)

        ax.margins(x=0, y=0)
        ax.set_xlim(0, len(pop_rate) * dt)
        ax.set_ylabel("Rates [Hz]")
        ax.set_xlabel("Time [seconds]")

        fig.tight_layout()

    return ax


def plot_system_rate(
    h5f, ax=None, apply_formatting=True, mark_burst_threshold=True, **kwargs
):
    assert "ana" in h5f.keypaths(), "`prepare_file(h5f)` before plotting!"

    kwargs = kwargs.copy()

    log.info("Plotting System Rate")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if not "ana.rates" in h5f.keypaths():
        ah.find_rates(h5f)

    dt = h5f["ana.rates.dt"]

    pop_rate = h5f["ana.rates.system_level"]
    mean_rate = np.nanmean(pop_rate)

    kwargs.setdefault("color", "black")
    kwargs.setdefault("label", f"system: {mean_rate:.2f} Hz")

    ax.plot(np.arange(0, len(pop_rate)) * dt, pop_rate, **kwargs)
    log.info(f'CV system rate: {h5f["ana.rates.cv.system_level"]:.3f}')

    if mark_burst_threshold:
        try:
            ax.axhline(
                y=h5f["ana.bursts.system_level.rate_threshold"],
                ls=":",
                color="black",
            )
        except Exception as e:
            log.debug(e)

    if apply_formatting:
        _style_legend(ax.legend(loc=1))
        ax.margins(x=0, y=0)
        ax.set_ylabel("Rates [Hz]")
        ax.set_xlabel("Time [seconds]")

        fig.tight_layout()

    return ax

def plot_gate_history(
    h5f, ax=None, apply_formatting=True, mark_burst_threshold=False, **kwargs
):
    """
    Plots times when gates allow transmission as continuous lines.
    spreads all gates into the y interval between 0 and 1,
    and colors by module id.
    """

    kwargs = kwargs.copy()
    log.info("Plotting Gate states")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()



    num_time_steps = h5f["data.gate_history"].shape[-1]
    dt = h5f["ana.rates.dt"]
    time_axis = np.arange(0, num_time_steps) * dt

    # Coupling matrix, those modules are connected by a gate
    w = np.zeros(shape=(4, 4), dtype="int")
    w[0, 1] = 1
    w[1, 0] = 1
    w[0, 2] = 1
    w[2, 0] = 1
    w[1, 3] = 1
    w[3, 1] = 1
    w[2, 3] = 1
    w[3, 2] = 1

    # lets reshape the gate history so we have only the two actually existing gates:
    # old: [from, to, time] 4,4,t
    # new: [from, (target_one, target_two), time] 4,2,t
    gate_history = np.ones(shape=(4,2,num_time_steps), dtype="int")*-1
    for m_id in h5f["ana.mod_ids"]:
        t1, t2 = np.where(w[m_id, :] == 1)[0]
        gate_history[m_id,0,:] = h5f["data.gate_history"][m_id, t1, :]
        gate_history[m_id,1,:] = h5f["data.gate_history"][m_id, t2, :]

    # lets hard code the y coordinates.
    total = 1.0
    padding = 0.1
    content = total - padding
    dy = content / 8  # 4 modules, two gates each
    y_pos = padding

    for m_id in h5f["ana.mod_ids"]:
        m_dc = h5f["ana.mods"][m_id]
        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault("color", h5f[f"ana.mod_colors"][m_id])
        plot_kwargs.setdefault("solid_capstyle", "butt")

        for g_id in range(2):
            array = gate_history[m_id, g_id, :].astype("float")
            # we want a continuous line, where the gate is active that
            # disappears when the gate closes.
            array[array == 0] = np.nan
            ax.plot(time_axis, array*y_pos, **plot_kwargs)
            y_pos += dy

    if apply_formatting:
        ax.set_ylabel("Gate State")
        ax.set_xlabel("Time [seconds]")

        fig.tight_layout()

    return ax



def plot_state_variable(h5f, ax=None, apply_formatting=True, variable="D", **kwargs):
    """
    We may either have an average value, then the h5f dataset has shape(1, t)
    or an entry for every neuron, then shape(N, t)

    could add a strides argument for high t-resolutions
    """

    log.info(f"Plotting state variable '{variable}'")
    assert f"data.state_vars_{variable}" in h5f.keypaths()
    stat_vals = h5f[f"data.state_vars_{variable}"]  # avoid loading this with [:]!

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # check if this seems to be a quantity between
    # 0 and one
    geqo = False
    leqz = False
    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]

        selects = np.where(h5f["data.neuron_module_id"][:] == m_id)[0]

        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault("zorder", 1)
        plot_kwargs.setdefault("color", h5f["ana.mod_colors"][m_id])

        # mean across neurons
        x = h5f[f"data.state_vars_time"][:]
        y = np.nanmean(stat_vals[selects, :], axis=0)
        ax.plot(x,y,**kwargs)
        if np.any(y > 1):
            geqo = True
        if np.any(y < 0):
            leqz = True

        # show some faint lines for individual neurons from each module
        num_examples = 0
        if len(selects) >= num_examples:
            selects = selects[0:num_examples]
        for s in selects:
            ax.plot(
                h5f[f"data.state_vars_time"][:],
                stat_vals[s],
                lw=0.5,
                color=h5f["ana.mod_colors"][m_id],
                zorder=0,
                alpha=0.5,
            )

    if apply_formatting:
        ax.margins(x=0, y=0)
        if variable == "D":
            ax.set_ylabel("Resources")
        else:
            ax.set_ylabel(f"state {variable}")

        if (not geqo) and (not leqz):
            ax.set_ylim(0, 1)
        elif geqo and not (leqz):
            ax.set_ylim(0, None)

        ax.set_xlabel("Time [seconds]")
        fig.tight_layout()

    return ax


def plot_fluorescence_trace(
    h5f,
    ax=None,
    neurons=None,
    base_color=None,
    **kwargs,
):
    """
    Only makes sense for experimental files, where we have the entries set.
    requires `h5f["data.neuron_fluorescence_trace"]`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if neurons is None:
        neurons = h5f["ana.neuron_ids"].copy()
    else:
        neurons = np.asarray(neurons)

    idx = _argsort_neuron_ids_by_module(h5f, neurons)
    neurons = neurons[idx]

    mods = h5f["data.neuron_module_id"][neurons]
    last_mod = np.nan
    num_mods = len(h5f["ana.mod_ids"])
    offset = 0

    for ndx, n_id in enumerate(neurons):
        plot_kws = kwargs.copy()
        m_id = mods[ndx]

        offset += 0.15
        if last_mod != m_id:
            offset += 0.1
        last_mod = m_id

        log.debug(f"neuron {n_id} module {m_id} at {offset}")

        if base_color is None:
            try:
                color = h5f["ana.mod_colors"][m_id]
            except:
                color = "black"
        else:
            color = alpha_to_solid_on_bg(base_color, (num_mods - m_id) / num_mods)

        plot_kws.setdefault("color", color)
        plot_kws.setdefault("lw", 0.5)
        plot_kws.setdefault("zorder", -m_id)

        dt = h5f["data.neuron_fluorescence_timestep"]
        y = h5f["data.neuron_fluorescence_trace"][n_id].copy()
        x = np.arange(0, len(y) * dt, dt)

        y_smooth = ah.smooth_rate(y, clock_dt=dt, width=40 * dt)

        bg = np.nanmin(y_smooth)

        # zscoring
        # y = (y - np.nanmean(y)) / np.nanstd(y)

        # df/f
        y = (y - bg) / bg

        ax.plot(
            x,
            y + offset,
            **plot_kws,
        )

    return ax


def plot_bursts_into_timeseries(h5f, ax=None, apply_formatting=True, **kwargs):

    kwargs = kwargs.copy()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # calculate ibi to make a meaningful legend
    if "ana.ibi" not in h5f.keypaths():
        try:
            ah.find_ibis(h5f)
        except Exception as e:
            log.debug(e)

    total_num_b = 0
    pad = 3
    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        try:
            beg_times = h5f[f"ana.bursts.module_level.{m_dc}.beg_times"]
            end_times = h5f[f"ana.bursts.module_level.{m_dc}.end_times"]
        except KeyError:
            log.debug("No module-level burst in h5f. Cannot plot those burst times")
            continue

        num_b = len(beg_times)
        total_num_b += num_b
        try:
            ibi = np.nanmean(h5f[f"ana.ibi.module_level.{m_dc}"])
        except:
            ibi = np.nan

        plot_kws = kwargs.copy()
        plot_kws.setdefault("color", h5f["ana.mod_colors"][m_id])
        plot_kws.setdefault("label", f"{num_b} bursts, ~{ibi:.1f} s")

        _plot_bursts_into_timeseries(
            ax, beg_times, end_times, y_offset=pad + 1 + m_id, **plot_kws
        )

    log.info(f"Found {total_num_b} bursts across modules")

    # only interested in bursts that extended across all modules
    beg_times = np.array(h5f["ana.bursts.system_level.beg_times"])
    end_times = np.array(h5f["ana.bursts.system_level.end_times"])
    l = [len(seq) for seq in h5f["ana.bursts.system_level.module_sequences"]]
    # idx = np.where(np.array(l) >= len(h5f["ana.mods"])) # swap these lines to show all
    idx = np.where(np.array(l) >= 0)  # or only system wide bursts in black
    beg_times = beg_times[idx]
    end_times = end_times[idx]

    num_b = len(beg_times)
    log.info(f"Found {num_b} bursts of any length")
    try:
        ibi = np.nanmedian(h5f[f"ana.ibi.system_level.all_modules"])
    except:
        ibi = np.nan
    log.info(f"System-wide IBI (median): {ibi:.2f} seconds")
    try:
        ibi = np.nanmedian(h5f[f"ana.ibi.system_level.any_module"])
    except:
        ibi = np.nan
    log.info(f"Any-length IBI (median): {ibi:.2f} seconds")

    plot_kws = kwargs.copy()
    plot_kws.setdefault("color", "black")
    plot_kws.setdefault("label", f"{num_b} bursts, ~{ibi:.1f} s")

    _plot_bursts_into_timeseries(ax, beg_times, end_times, y_offset=pad, **plot_kws)

    if apply_formatting:
        ax.set_ylim(0, len(h5f["ana.mods"]) + 2 * pad)
        ax.margins(x=0, y=0)
        ax.set_ylabel("Bursts")
        ax.set_xlabel("Time [seconds]")
        ax.set_yticks([])
        fig.tight_layout()

    return ax


def _plot_bursts_into_timeseries(
    ax, beg_times, end_times, y_offset=3, style="markers", **kwargs
):
    """
    lower level helper to plot beginning and end times of bursts.

    # Parameters
    style : str
        "markers" to show in and outs via small triangles
        "fill_between" to highlight the the background with a `fill_between`
    y_offset: float
        y position if style is "markers"
    """

    kwargs = kwargs.copy()

    if style == "markers":
        kwargs.setdefault("color", "black")
        ax.plot(
            beg_times,
            np.ones(len(beg_times)) * y_offset,
            marker="4",
            lw=0,
            **kwargs,
        )

        try:
            kwargs.pop("label")
        except KeyError:
            pass

        ax.plot(
            end_times,
            np.ones(len(end_times)) * y_offset,
            marker="3",
            lw=0,
            **kwargs,
        )

    elif style == "fill_between":
        kwargs.setdefault("color", "black")
        kwargs.setdefault("alpha", 0.1)
        kwargs.setdefault("zorder", 0)

        try:
            kwargs.pop("label")
        except KeyError:
            pass

        # simply doing a fill between for every burst causes glitches.
        # construct the `where` array, of booleans matching x to fill
        x = []
        where = []
        for idx in range(0, len(beg_times)):
            beg = beg_times[idx]
            end = end_times[idx]
            x.extend([beg, end, beg + (end - beg) / 2])
            where.extend([True, True, False])

        trans = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )

        ax.fill_between(
            x=x,
            y1=0,
            y2=1,
            where=where,
            interpolate=False,
            lw=0,
            transform=trans,
            **kwargs,
        )


def plot_parameter_info(h5f, ax=None, apply_formatting=True, add=[]):

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
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title("Parameters")

    dat = []
    # dat.append(["Connections k", h5f["meta.topology_k_inter"]])
    # dat.append(["Stimulation", h5f["ana.stimulation_description"]])

    try:
        dat.append(["", ""])
        dat.append(["AMPA [mV]", h5f["meta.dynamics_jA"]])
        dat.append(["GABA [mV]", h5f["meta.dynamics_jG"]])
        dat.append(["Noise [mV]", h5f["meta.dynamics_jM"]])
        dat.append(["Noise Rate [Hz]", h5f["meta.dynamics_rate"]])
    except:
        pass

    try:
        dat.append(["Stim [mV]", h5f["meta.dynamics_jE"]])
        dat.append(["Adapt. [s]", h5f["meta.dynamics_tD"]])
    except:
        pass

    try:
        dat.append(
            [
                "E/I",
                len(h5f["data.neuron_excitatiory_ids"])
                / len(h5f["data.neuron_inhibitory_ids"]),
            ]
        )
    except:
        pass

    try:
        dat.append(["Connections", f'{h5f["meta.dynamics_k_frac"]*100:.1f}%'])
    except:
        pass

    if len(add) > 0:
        dat.append(["", ""])
        for el in add:
            assert len(el) == 2
            dat.append(el)

    left = ""
    right = ""
    for d in dat:
        left += f"{d[0]}\n"
        right += f"{d[1]}\n"
        if d[0] != "" and d[1] != "":
            log.info(f"{d[0]:>22}:    {d[1]}")

    # tables stretch rows of axis, so manually place some text instead
    # tab = ax.table(dat, loc="center", edges="open")
    # cells = tab.properties()["celld"]
    # for i in range(0, int(len(cells) / 2)):
    #     cells[i, 1]._loc = "left"

    # ax.plot([0,1], [0,1], ls=None)

    ax.text(
        0.46,
        0.9,
        left,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.54,
        0.9,
        right,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    if apply_formatting:
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()

    return ax


def plot_initiation_site(h5f, ax=None, apply_formatting=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_ylabel("Histogram")
    ax.set_xlabel("Initiation module")

    assert "ana.bursts" in h5f.keypaths()

    sequences = h5f["ana.bursts.system_level.module_sequences"]
    if len(sequences) == 0:
        return ax

    # first_mod = np.ones(len(sequences), dtype=int) - 1
    first_mod = []
    for idx, seq in enumerate(sequences):
        if len(seq) > 0:
            first_mod.append(seq[0])
    first_mod = np.array(first_mod, dtype=int)

    # unique = np.sort(np.unique(first_mod))
    unique = np.array([0, 1, 2, 3])
    bins = (unique - 0.5).tolist()
    bins = bins + [bins[-1] + 1]

    N, bins, patches = ax.hist(first_mod, bins=bins)
    # assume they are ordered correctly
    try:
        for idx, patch in enumerate(patches):
            patches[idx].set_facecolor(h5f["ana.mod_colors"][idx])
    except Exception as e:
        log.debug(e)

    fig.tight_layout()

    return ax


def plot_resources_vs_activity(
    h5f, ax=None, apply_formatting=True, mod_ids=None, max_traces_per_mod=100, **kwargs
):
    """
    Helper to illustrate the charge-and-release cycle of synaptic resources.

    # Parameters
    mod_ids : list of modules to plot e.g. `[0, 1, 2]`, default all modules in h5f
    max_traces_per_mod : only show this many traces for each module
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if apply_formatting:
        ax.set_xlabel("Synaptic resources")
        ax.set_ylabel("Population rate (Hz)")

    if mod_ids is None:
        mod_ids = h5f["ana.mod_ids"]

    if "ana.adaptation" not in h5f.keypaths():
        ah.find_module_level_adaptation(h5f)

    for mdx, mod_id in enumerate(mod_ids):
        mod = f"mod_{mod_id}"
        assert f"ana.rates.module_level.{mod}" in h5f.keypaths(), f"rate for {mod} needed"
        mod_adapt = h5f[f"ana.adaptation.module_level.{mod}"]
        mod_rate = h5f[f"ana.rates.module_level.{mod}"]

        # we need matching time steps. unfortunately, i have not always saved
        # the dt of the state vars, so this is a bit fishy.
        stride = int(
            np.round(
                h5f["ana.adaptation.dt"]
                / h5f["ana.rates.dt"]
            )
        )
        log.debug(f"strides for rate vs adaptation: {stride}")
        assert (
            len(mod_rate) == len(mod_adapt) * stride
        ), "len of rate v resources did not match."
        assert stride >= 1, "NotImplemented"
        # in our default parameters, stride is 50:
        # rate dt is 0.5 ms
        # adaptation dt is 25 ms
        # subsample or smooth over window ?
        mod_rate = mod_rate[0 : len(mod_rate) : stride]

        # we could do this via dt
        adap_times = h5f["data.state_vars_time"][:]

        try:
            clr = h5f["ana.mod_colors"][mod_id]
        except:
            clr = "black"
        plot_kwargs = {"color": clr, "alpha": 0.3}
        plot_kwargs.update(kwargs)

        split_by_burst = True
        if split_by_burst and stride == 1:
            beg_times = h5f["ana.bursts.system_level.beg_times"]
            end_times = h5f["ana.bursts.system_level.end_times"]
            last_end = 0.0
            num_traces = 0
            for bdx in range(0, len(end_times)):
                if num_traces == max_traces_per_mod:
                    break
                num_traces += 1
                next_end = end_times[bdx]
                next_beg = beg_times[bdx]
                # do we want the recharging part of the trace?
                # idx = np.where((adap_times >= next_beg) & (adap_times <= next_end))
                idx = np.where((adap_times > last_end) & (adap_times <= next_end))
                last_end = next_end
                ax.plot(mod_adapt[idx], mod_rate[idx], **plot_kwargs)
            log.info(f"plotted {num_traces} cycle-traces for {mod}")

        else:
            ax.plot(mod_adapt, mod_rate, **plot_kwargs)
            # ax.scatter(mod_rate, mod_adapt, alpha=0.6, s=0.5)

        if apply_formatting:
            ax.set_xlim(0, 1)
            ax.set_ylim(-5, 100)
            ax.get_figure().tight_layout()

    return ax




# ------------------------------------------------------------------------------ #
# plots of pandas dataframes
# ------------------------------------------------------------------------------ #


def pd_sequence_length(df):

    df_res = None
    # prequery
    for query in tqdm(
        [
            "`Bridge weight` == 1 & `Connections` == 1 & `Number of inhibitory"
            " neurons` == 0",
            "`Bridge weight` == 1 & `Connections` == 2 & `Number of inhibitory"
            " neurons` == 0",
            "`Bridge weight` == 1 & `Connections` == 3 & `Number of inhibitory"
            " neurons` == 0",
            "`Bridge weight` == 1 & `Connections` == 5 & `Number of inhibitory"
            " neurons` == 0",
        ],
        desc="Queries",
        leave=False,
    ):
        # as hue value, we want to compare stimulation scenarios, need concat
        dfc = None
        for sq in [
            "`Stimulation` == 'Off'",
            "`Stimulation` == 'On (0)'",
            "`Stimulation` == 'On (0, 2)'",
            "`Stimulation` == 'On (0, 1, 2)'",
            "`Stimulation` == 'On (0, 1, 2, 3)'",
        ]:
            dfq = df.query(query + " & " + sq)
            if dfq.shape[0] == 0:
                log.debug(f"Skipping {sq}, no rows.")
                continue
            # calculate the probability of sequence length for every repetition
            dfq = ah.sequence_length_histogram_from_pd_df(
                dfq,
                keepcols=[
                    "Stimulation",
                    "Bridge weight",
                    "Connections",
                    "Number of inhibitory neurons",
                ],
            )
            dfc = pd.concat([dfc, dfq], ignore_index=True)
        try:
            plot_pd_boxplot(dfc, x="Sequence length", y="Probability")
        except Exception as e:
            log.error(e)

        # for returning everything
        df_res = pd.concat([df_res, dfc], ignore_index=True)

    return df_res


def pd_burst_duration(df):

    x = "Stimulation"
    y = "Duration"
    df = df.query("Connections > 0")
    # df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 2)'")
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 1"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 2"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 3"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 1 & `Connections` == 5"), x, y)

    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 1"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 2"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 3"), x, y)
    # plot_pd_violin(df.query("`Bridge weight` == 0.5 & `Connections` == 5"), x, y)

    x = "Connections"
    # df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 2)'")
    df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 1, 2, 3)'")
    df2 = df2.query("`Number of inhibitory neurons` > 0")
    ax = plot_pd_violin(df2.query("`Bridge weight` == 1 & `Sequence length` == 1"), x, y)
    ax.set_title("L=1", loc="left")
    ax = plot_pd_violin(df2.query("`Bridge weight` == 1 & `Sequence length` == 2"), x, y)
    ax.set_title("L=2", loc="left")
    ax = plot_pd_violin(df2.query("`Bridge weight` == 1 & `Sequence length` > 1"), x, y)
    ax.set_title("L>1", loc="left")
    # plot_pd_violin(df2.query("`Bridge weight` == 0.75"), x, y)
    # plot_pd_violin(df2.query("`Bridge weight` == 0.5"), x, y)

    # df2 = df.query("Stimulation == 'Off' | Stimulation == 'On (0, 1, 2, 3)'")
    # plot_pd_violin(df2.query("`Bridge weight` == 1"), x, y)
    # plot_pd_violin(df2.query("`Bridge weight` == 0.75"), x, y)
    # plot_pd_violin(df2.query("`Bridge weight` == 0.5"), x, y)

    return df


def pd_burst_duration_change_under_stim(df):
    """
    df from `ah.batch_pd_bursts`
    """

    fig, ax = plt.subplots()
    ax.set_title("L=all, Inh, k=5, bw=1")

    # query so what remains is repetitions and the condition to compare
    df1 = df.query(
        "`Sequence length` >= 1 & `Number of inhibitory neurons` > 0 &"
        " `Connections` == 5 & `Bridge weight` == 1.0 & (Stimulation == 'Off' |"
        " Stimulation == 'On (0, 2)')"
    )
    # use the mean duration as the observable, to which contributions are exactly
    # one data point from one realization, and
    # compare between different `stimulation` conditions
    df2 = df1.groupby(["Stimulation", "Repetition"])["Duration"].mean().reset_index()

    # ax = plot_pd_violin(df2, x="Stimulation", y="Duration", split=False)
    # ax = plot_pd_boxplot(df2, x="Stimulation", y="Duration")
    palette = _stimulation_color_palette()
    order = [s for s in palette.keys() if s in df2["Stimulation"].unique()]
    sns.pointplot(
        data=df2,
        x="Stimulation",
        y="Duration",
        order=order,
        ax=ax,
        hue="Repetition",
        scale=0.5,
        color=".4",
    )
    sns.violinplot(
        data=df2,
        x="Stimulation",
        y="Duration",
        hue="Stimulation",
        split=True,
        order=order,
        ax=ax,
        palette=palette,
    )
    # sns.boxplot(data=df2, x="Stimulation", y="Duration", order=order, ax=ax, fliersize=0, palette=palette)
    # sns.swarmplot(data=df2, x="Stimulation", y="Duration", order=order, ax=ax, color=".4", size=4)
    ax.get_legend().set_visible(False)

    ax.set_ylim(0.0, 0.3)
    fig.tight_layout()

    return df2


def plot_pd_boxplot(
    df, x="Sequence length", y="Probability", ax=None, apply_formatting=True
):
    """
    Boxplot across stimulation conditions.
    `df` needs to be filtered already:
    ```
    data = df.loc[((df["Connections"] == k) & (df["Bridge weight"] == bw))]
    ```
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    palette = _stimulation_color_palette()

    order = None
    hue_order = None
    if x == "Stimulation":
        order = palette.keys()
    if x == "Connections":
        hue_order = []
        stim_of_df = df["Stimulation"].unique()
        for stim in palette.keys():
            if stim in stim_of_df:
                hue_order.append(stim)

    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue="Stimulation",
        order=order,
        hue_order=hue_order,
        fliersize=0.5,
        linewidth=0.5,
        ax=ax,
        palette=palette,
    )

    if apply_formatting:

        # if those columns are equal across rows, print it as title
        try:
            bw = df["Bridge weight"].to_numpy()
            if np.all(bw[0] == bw):
                ax.set_title(f"bw = {bw[0]}", loc="right")
        except:
            pass
        try:
            k = df["Connections"].to_numpy()
            if np.all(k[0] == k):
                ax.set_title(f"k = {k[0]}", loc="left")
        except:
            pass

        ax.legend()
        ax.get_legend().set_visible(False)
        ax.axhline(0, ls=":", color="black", lw=0.75, zorder=-1)
        ax.axhline(1, ls=":", color="black", lw=0.75, zorder=-1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.tick_params(axis="x", which="major", length=0)
        fig.tight_layout()

    ax.set_ylim(-0.01, 1.01)
    # ax.set_ylim(-.01,.2)

    return ax


def plot_pd_violin(df, x, y, ax=None, apply_formatting=True, split=True):
    """
    Violinplot across stimulation conditions.
    `df` needs to be filtered already:
    ```
    data = df.loc[((df["Connections"] == k) & (df["Bridge weight"] == bw))]
    ```
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # make a nice color map
    c0 = "#1f77b4"  # matplotlib.cm.get_cmap("tab10").colors[0] # blue
    c1 = "#ff7f0e"  # matplotlib.cm.get_cmap("tab10").colors[1] # orange

    palette = {
        "Off": c0,
        "On (0)": alpha_to_solid_on_bg(base=c1, alpha=0.25, bg="white"),
        "On (0, 2)": alpha_to_solid_on_bg(base=c1, alpha=0.50, bg="white"),
        "On (0, 1, 2)": alpha_to_solid_on_bg(base=c1, alpha=0.75, bg="white"),
        "On (0, 1, 2, 3)": alpha_to_solid_on_bg(base=c1, alpha=1.00, bg="white"),
    }

    order = None
    hue_order = None
    if x == "Stimulation":
        order = palette.keys()
    if x == "Connections":
        hue_order = []
        stim_of_df = df["Stimulation"].unique()
        for stim in palette.keys():
            if stim in stim_of_df:
                hue_order.append(stim)

    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue="Stimulation",
        hue_order=hue_order,
        order=order,
        size=0.3,
        edgecolor=(0, 0, 0, 0),
        linewidth=0.5,
        dodge=True,
        ax=ax,
        palette=palette,
    )

    sns.violinplot(
        data=df,
        x=x,
        y=y,
        hue="Stimulation",
        order=order,
        hue_order=hue_order,
        scale_hue=False,
        scale="area",
        split=split,
        inner="quartile",
        linewidth=0.5,
        ax=ax,
        palette=palette,
    )

    if apply_formatting:

        # if those columns are equal across rows, print it as title
        try:
            bw = df["Bridge weight"].to_numpy()
            if np.all(bw[0] == bw):
                ax.set_title(f"bw = {bw[0]}", loc="right")
        except:
            pass
        try:
            k = df["Connections"].to_numpy()
            if np.all(k[0] == k):
                ax.set_title(f"k = {k[0]}", loc="left")
        except:
            pass

        ax.legend()
        ax.get_legend().set_visible(False)
        ax.axhline(0, ls=":", color="black", lw=0.75, zorder=-1)
        ax.axhline(1, ls=":", color="black", lw=0.75, zorder=-1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.tick_params(axis="x", which="major", length=0)
        fig.tight_layout()

    ax.set_ylim(-0.01, 0.5)
    # ax.set_ylim(-.01,.2)

    return ax


def plot_pd_seqlen_vs_burst_duration(df, ax=None, apply_formatting=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sns.scatterplot(
        x="Sequence length",
        y="Duration",
        hue="Sequence length",
        data=df,
        ax=ax,
    )

    if apply_formatting:
        ax.legend()
        ax.get_legend().set_visible(False)
        fig.tight_layout()

    return ax


def _stimulation_color_palette():

    c0 = "#1f77b4"  # matplotlib.cm.get_cmap("tab10").colors[0] # blue
    c1 = "#ff7f0e"  # matplotlib.cm.get_cmap("tab10").colors[1] # orange

    palette = {
        "Off": c0,
        "On (0)": alpha_to_solid_on_bg(base=c1, alpha=0.25, bg="white"),
        "On (0, 2)": alpha_to_solid_on_bg(base=c1, alpha=0.50, bg="white"),
        "On (0, 1, 2)": alpha_to_solid_on_bg(base=c1, alpha=0.75, bg="white"),
        "On (0, 1, 2, 3)": alpha_to_solid_on_bg(base=c1, alpha=1.00, bg="white"),
    }

    return palette


def plot_comparison_rij_between_conditions_serial(
    h5f_1=None,
    h5f_2=None,
    fname_1=None,
    fname_2=None,
    label_1="f1",
    label_2="f2",
    which="modules",
    plot_matrix=False,
):
    """
    Plot a rij for every neuron pair between conditions.
    Assumes that fname_1 and fname_2 lead to h5files with the same neuron positions
    under different dynamic conditions.

    #Parameters
    which : str, "modules" or "stim" to compare within/across modules or groups
        of stimulated vs non-stimulated modules. stimulated mods are hardcoded to 02
    """

    assert which in ["modules", "stim"]

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(f"rij_{which}_{label_1}_{label_2}")

    if h5f_1 is None:
        assert fname_1 is not None
        h5f_1 = ah.prepare_file(fname_1)
    if h5f_2 is None:
        assert fname_2 is not None
        h5f_2 = ah.prepare_file(fname_2)

    rij_1 = ah.find_rij(h5f_1, which="neurons", time_bin_size=40 / 1000)
    rij_2 = ah.find_rij(h5f_2, which="neurons", time_bin_size=40 / 1000)

    log.debug(fname_1)
    log.debug(fname_2)

    if plot_matrix:
        temp_ax = _plot_matrix(rij_1, vmin=0, vmax=1)
        temp_ax.set_title(label_1)
        temp_ax.get_figure().tight_layout()
        temp_ax.get_figure().canvas.manager.set_window_title(f"rij_{label_1}")

        temp_ax = _plot_matrix(rij_2, vmin=0, vmax=1)
        temp_ax.set_title(label_2)
        temp_ax.get_figure().tight_layout()
        temp_ax.get_figure().canvas.manager.set_window_title(f"rij_{label_2}")

    if which == "modules":
        pairings = ["within_modules", "across_modules"]
    elif which == "stim":
        # pairings = ["within_modules", "across_groups_02_13"]
        pairings = [
            "within_group_0",
            # "within_group_1",
            # "within_group_2",
            # "within_group_3",
        ]
        # pairings = [ "across_groups_0_1", "across_groups_2_3", "across_groups_1_3", "across_groups_0_2"]

    for idx, p in enumerate(pairings):
        ax.plot(
            ah.find_rij_pairs(h5f_1, rij=rij_1, pairing=p),
            ah.find_rij_pairs(h5f_2, rij=rij_2, pairing=p),
            ".",
            color=f"C{idx}",
            markersize=1.5,
            markeredgewidth=0,
            label=p,
            alpha=1,
            zorder=10 - idx,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], zorder=-2, color="gray")

    ax.set_xlabel(label_1)
    ax.set_ylabel(label_2)
    ax.legend(
        labelcolor="linecolor",
        markerscale=5,
        handlelength=0,
    )

    fig.tight_layout()

    h5.close_hot(h5f_1)
    h5.close_hot(h5f_2)

    # return rij_within_1, rij_within_2
    # return rij_1, rij_2
    return ax


def plot_comparison_rij_between_conditions_batch(
    h5f,
    fname_1,
    fname_2,
    label_1="f1",
    label_2="f2",
    which="modules",
    plot_matrix=False,
):
    """
    not sure if this guy works
    """

    assert which in ["modules", "stim"]

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(f"rij_{which}_{label_1}_{label_2}")

    def ana_func(filename):
        h5f = ah.prepare_file(filename)
        rij = ah.find_rij(h5f, which="neurons", time_bin_size=40 / 1000)
        check = h5f["data.neuron_module_id"][:]
        h5.close_hot(h5f["h5.filename"])
        return rij, check, filename

    global rij_1
    rij_1, check = ah.batch_across_filenames(fname_1, ana_function=ana_func)
    rij_2, check = ah.batch_across_filenames(fname_2, ana_function=ana_func)

    if plot_matrix:
        temp_ax = _plot_matrix(rij_1[0], vmin=0, vmax=1)
        temp_ax.set_title(label_1)
        temp_ax.get_figure().tight_layout()
        temp_ax.get_figure().canvas.manager.set_window_title(f"rij_{label_1}")

        temp_ax = _plot_matrix(rij_2[0], vmin=0, vmax=1)
        temp_ax.set_title(label_2)
        temp_ax.get_figure().tight_layout()
        temp_ax.get_figure().canvas.manager.set_window_title(f"rij_{label_2}")

    if which == "modules":
        pairings = ["within_modules", "across_modules"]
    elif which == "stim":
        # pairings = ["within_modules", "across_groups_02_13"]
        pairings = [
            "within_group_0",
            "within_group_1",
            "within_group_2",
            "within_group_3",
        ]
        # pairings = [ "across_groups_0_1", "across_groups_2_3", "across_groups_1_3", "across_groups_0_2"]

    for idx, p in enumerate(tqdm(pairings, desc="Pairings")):
        rij_1_pairs = []
        for rij in rij_1:
            rij_1_pairs.extend(ah.find_rij_pairs(h5f, rij=rij, pairing=p))

        rij_2_pairs = []
        for rij in rij_2:
            rij_2_pairs.extend(ah.find_rij_pairs(h5f, rij=rij, pairing=p))

        ax.plot(
            rij_1_pairs,
            rij_2_pairs,
            ".",
            color=f"C{idx}",
            markersize=1.5,
            markeredgewidth=0,
            label=p,
            alpha=1,
            zorder=10 - idx,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], zorder=-2, color="gray")

    ax.set_xlabel(label_1)
    ax.set_ylabel(label_2)
    ax.legend(
        labelcolor="linecolor",
        markerscale=5,
        handlelength=0,
    )

    fig.tight_layout()

    h5.close_hot(h5f_1)
    h5.close_hot(h5f_2)

    # return rij_within_1, rij_within_2
    # return rij_1, rij_2
    return ax


# ------------------------------------------------------------------------------ #
# distributions
# ------------------------------------------------------------------------------ #


def _plot_matrix(c, **kwargs):
    fig, ax = plt.subplots()
    im = ax.matshow(c, **kwargs)
    plt.colorbar(im)

    return ax


def plot_distribution_correlation_coefficients(
    h5f,
    ax=None,
    apply_formatting=True,
    num_bins=20,
    which="neurons",
):
    """
    Uses (naively) binned spike counts per neuron to calculate correlation coefficients
    with `numpy.corrcoef`.

    # Parameters
    num_bins : number of bins to discretize correlation coefficients.
    """

    assert which in ["neurons", "modules"]

    C, rij = ah.find_functional_complexity(
        h5f, which=which, num_bins=num_bins, return_res=True, write_to_h5f=False
    )

    log.info(f"Functional Complexity: {C:.3f}")

    _plot_matrix(rij, vmin=0, vmax=1)

    bw = 1.0 / num_bins
    bins = np.arange(0, 1 + 0.1 * bw, bw)

    try:
        color = (
            h5f["ana.mod_colors"][0] if len(h5f["ana.mod_colors"]) == 1 else "black",
        )
    except:
        color = "black"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kwargs = {
        "ax": ax,
        "kde": False,
        "bins": bins,
        "stat": "density",
        # 'multiple' : 'stack',
        "element": "step",
    }

    # im pretty sure sns ignores nans (rij diagonal)
    sns.histplot(
        data=rij.flatten(),
        color=color,
        alpha=0.2,
        **kwargs,
    )

    if apply_formatting:
        ax.set_title(f"{which} C = {C:.3f}")
        ax.set_xlabel(r"Correlation coefficients $r_{ij}$")
        ax.set_ylabel(r"Prob. density $P(r_{ij})$")
        # ax.legend()
        fig.tight_layout()

    return ax


def plot_distribution_burst_duration(h5f, ax=None, apply_formatting=True, filenames=None):

    if not "ana" in h5f.keypaths():
        ah.prepare_file(h5f)

    if not "ana.bursts" in h5f.keypaths():
        log.info("Finding Bursts from Rates")
        ah.find_bursts_from_rates(h5f)

    if filenames is None:
        log.info("Plotting Bursts")
        bursts = h5f["ana.bursts"]
    else:
        log.info("Gathering Bursts across provided filenames")
        # merge across files, and save the result
        if not "ana.ensemble" in h5f.keypaths():
            h5f["ana.ensemble"] = benedict()
            h5f["ana.ensemble.filenames"] = filenames
            ens = ah.batch_candidates_burst_times_and_isi(filenames, hot=True)
            h5f["ana.ensemble.bursts"] = ens["ana.bursts"]
            h5f["ana.ensemble.isi"] = ens["ana.isi"]
            del ens

        assert (
            h5f["ana.ensemble.filenames"] == filenames
        ), "`filenames` dont match. Using multiple ensembles is not implemented yet."
        bursts = h5f["ana.ensemble.bursts"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kwargs = {
        "ax": ax,
        "kde": False,
        # "binwidth": 1 / 1000,  # ms, beware the timestep of burst analysis (usually 2ms)
        # "binrange": (0.06, 0.12),
        "stat": "density",  # must use density when comparing across different binsizes!
        # 'multiple' : 'stack',
        "element": "poly",
    }

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        beg_times = np.array(bursts[f"module_level.{m_dc}.beg_times"])
        end_times = np.array(bursts[f"module_level.{m_dc}.end_times"])
        sns.histplot(
            data=end_times - beg_times,
            binwidth=1 / 1000,
            color=h5f["ana.mod_colors"][m_id],
            alpha=0.2,
            **kwargs,
            label=m_dc,
        )

    beg_times = np.array(bursts["system_level.beg_times"])
    end_times = np.array(bursts["system_level.end_times"])
    sns.histplot(
        data=end_times - beg_times,
        binwidth=1 / 1000,
        color="black",
        alpha=0,
        **kwargs,
        label="System-wide",
    )

    if apply_formatting:
        ax.set_xlabel(r"Burst duration $D$ (seconds)")
        ax.set_ylabel(r"Prob. density $P(D)$")
        ax.legend()
        fig.tight_layout()

    return ax


def plot_distribution_isi(
    h5f,
    ax=None,
    apply_formatting=True,
    log_binning=True,
    which="all",
    filenames=None,
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

    filenames : str
        if `filenames` are provided, bursts an
    """
    assert which in ["all", "in_bursts", "out_bursts"]

    if "ana.isi" not in h5f.keypaths():
        log.info("Finding ISIS")
        assert which == "all", "`find_bursts` before plotting isis other than `all`"
        assert "ana.bursts" in h5f.keypaths()
        ah.find_isis(h5f)

    if filenames is None:
        log.info("Plotting ISI")
        isi = h5f["ana.isi"]
    else:
        log.info("Gathering ISI across provided filenames")
        # merge across files, and save the result
        if not "ana.ensemble" in h5f.keypaths():
            h5f["ana.ensemble"] = benedict()
            h5f["ana.ensemble.filenames"] = filenames
            ens = ah.batch_candidates_burst_times_and_isi(filenames, hot=True)
            h5f["ana.ensemble.bursts"] = ens["ana.bursts"]
            h5f["ana.ensemble.isi"] = ens["ana.isi"]
            del ens

        assert (
            h5f["ana.ensemble.filenames"] == filenames
        ), "`filenames` dont match. Using multiple ensembles is not implemented yet."
        isi = h5f["ana.ensemble.isi"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # we want to use the same bins across modules
    max_isi = 0
    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        assert len(isi[m_dc][which]) > 0, f"No isis found for '{which}'"
        mod_isi = isi[m_dc][which]
        this_max_isi = np.max(mod_isi)
        if this_max_isi > max_isi:
            max_isi = this_max_isi
        try:
            log.info(
                f"Mean ISI, in bursts, {m_dc}:"
                f' {np.mean(isi[f"{m_id}.in_bursts"])*1000:.1f} (ms)'
            )
        except:
            pass

    log.info(f"Largest ISI: {max_isi:.3f} (seconds)")

    # do log binning manuall -> first transform to log scale, then apply the histogram
    # and (if desired) manually relabel.
    # Beware, probability is p(log10 isi) but if relabeling xaxis, we dont show
    # log isi on a linear scale (for readability). this might be confusing.
    kwargs = {
        "ax": ax,
        "kde": False,
        # "binwidth": 2.5 / 1000,  # ms
        # "binrange": (0.06, 0.12),
        "bins": np.linspace(-3, 3, num=200),
        # "stat": "density",
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "poly",
    }

    for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
        m_dc = h5f["ana.mods"][mdx]
        sns.histplot(
            data=np.log10(isi[m_dc][which]),
            color=h5f["ana.mod_colors"][m_id],
            alpha=0.2,
            **kwargs,
            label=f"{m_dc} ({which})",
        )

    if apply_formatting:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(_ticklabels_lin_to_log10_power)
        )
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(_ticklocator_lin_to_log_minor())
        # maybe skip the relabeling and just set xlabel to "log10 ISI (log10 seconds)"
        ax.set_xlabel(r"Inter-spike Interval (seconds)")
        ax.set_ylabel(r"Prob. Density (log ISI)")
        ax.legend()
        fig.tight_layout()

    return ax


def plot_distribution_participating_fraction(
    h5f,
    ax=None,
    apply_formatting=True,
    num_bins=20,
):
    """
    Plot the fraction of neurons participating in a burst
    """

    log.info("Plotting participating fraction")
    assert "ana.bursts" in h5f.keypaths()
    if "ana.bursts.system_level.participating_fraction" not in h5f.keypaths():
        ah.find_participating_fraction_in_bursts(h5f)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    fractions = h5f["ana.bursts.system_level.participating_fraction"]
    log.info(f"Mean fraction: {np.nanmean(fractions)*100:.1f}%")

    bw = 1.0 / num_bins
    bins = np.arange(0, 1 + 0.1 * bw, bw)

    kwargs = {
        "ax": ax,
        "kde": False,
        # "binwidth": 2.5 / 1000,  # ms
        # "binrange": (0.06, 0.12),
        "bins": bins,
        # "stat": "density",
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "step",
    }

    sns.histplot(
        data=fractions,
        # binwidth=1 / 1000,
        color=h5f["ana.mod_colors"][0] if len(h5f["ana.mod_colors"]) == 1 else "black",
        alpha=0.2,
        **kwargs,
        label="system-wide",
    )

    # now this is hacky:
    try:
        C = ah._functional_complexity(np.array(fractions), num_bins)
    except Exception as e:
        log.debug(e)
        C = np.nan

    if apply_formatting:
        ax.set_xlabel(r"Fraction of neurons in bursts")
        ax.set_ylabel(r"Prob. Density")
        ax.set_title(f"C={C:.3f}")
        # ax.legend()
        fig.tight_layout()

    return ax


@warntry
def plot_distribution_sequence_length(h5f, ax=None, apply_formatting=True):
    """
    Plot the distribution of sequence lengths' during bursts
    """

    assert "ana.bursts" in h5f.keypaths()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    seq_lens = h5f["ana.bursts.system_level.module_sequences"]
    seq_lens = np.array([len(s) for s in seq_lens])
    log.info(seq_lens)

    num_bins = len(h5f["ana.mods"])
    # bins = np.arange(-0.5, num_bins + 1 + 0.6, 1)

    # log.info(bins)
    # counts, _ = np.histogram(seq_lens, bins=bins)
    vals, counts = np.unique(seq_lens, return_counts=True)
    log.info(f"seq lenghts: {vals} {counts}")

    kwargs = {
        "ax": ax,
        "kde": False,
        "binwidth": 1,
        # "binrange": (0.06, 0.12),
        # "bins": bins,
        "stat": "density",
        # 'multiple' : 'stack',
        "element": "step",
    }

    sns.histplot(
        data=seq_lens,
        # binwidth=1 / 1000,
        color=h5f["ana.mod_colors"][0] if len(h5f["ana.mod_colors"]) == 1 else "black",
        alpha=0.2,
        **kwargs,
        label="system-wide",
    )

    if apply_formatting:
        ax.set_xlabel(r"Sequence Length")
        ax.set_ylabel(r"Prob. Density (log ISI)")
        # ax.legend()
        fig.tight_layout()

    return ax


@warntry
def plot_distribution_degree_k(h5f, ax=None, apply_formatting=True, filenames=None):
    """
    Plot the distribution of the in-and out degree for `h5f`.

    if `filenames` is provided, distribution data is accumulated from all files,
    and only styling/meta information is used from `h5f`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if filenames is None:
        k_in = h5f["data.neuron_k_in"][:]
        k_out = h5f["data.neuron_k_out"][:]
    else:
        log.info("Loading multiple files to plot degree distributions")
        k_in = h5.load(filenames, "/data/neuron_k_in")
        k_out = h5.load(filenames, "/data/neuron_k_out")
        k_in = np.concatenate(k_in).ravel()
        k_out = np.concatenate(k_out).ravel()

    maxbin = np.nanmax([k_in, k_out])
    br = np.arange(0, maxbin, 1)

    kwargs = {
        "ax": ax,
        "kde": True,
        "bins": br,
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "step",
        "alpha": 0.25,
    }

    sns.histplot(
        k_in,
        label=r"$k_{in}$",
        color=matplotlib.cm.get_cmap("tab10").colors[0],
        **kwargs,
    )
    sns.histplot(
        k_out,
        label=r"$k_{out}$",
        color=matplotlib.cm.get_cmap("tab10").colors[1],
        **kwargs,
    )

    if apply_formatting:
        ax.set_xlabel(f"Degree $k$ (per neuron)")
        ax.set_ylabel(f"Probability $p(k)$")
        ax.set_title("Degree distribution")
        ax.legend(loc="upper right")

        ax.text(
            0.95,
            0.7,
            f"median:\n" + r"$k_{in} \sim$" + f"{np.nanmedian(k_in):g}\n"
            r"$k_{out} \sim$"
            + f"{np.nanmedian(k_out):g}\n\n"
            + f"mean:\n"
            + r"$k_{in} \sim$"
            + f"{np.nanmean(k_in):g}\n"
            r"$k_{out} \sim$" + f"{np.nanmean(k_out):g}\n",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    return ax


@warntry
def plot_distribution_degree_for_neuron_distance(
    h5f, ax=None, apply_formatting=True, filenames=None
):
    """
    Plot the distribution of the network degree for `h5f`, as a function
    of the distance between the neurons.

    uses `data.connectivity_matrix_sparse` and `data.neuron_pos_x` (and y)

    if `filenames` is provided, distribution data is accumulated from all files,
    and only styling/meta information is used from `h5f`.
    """

    from itertools import combinations

    def local_prep(h5f):
        try:
            # dont load this guys via [:], lets keep them "hot"
            a_ij_sparse = h5f["data.connectivity_matrix_sparse"]
            pos_x = h5f["data.neuron_pos_x"]
            pos_y = h5f["data.neuron_pos_y"]
            k_dists = np.ones(a_ij_sparse.shape[0]) * np.nan
            # assume len of positions match number of neurons in a_ij
            for row in tqdm(range(0, a_ij_sparse.shape[0])):
                src, tar = a_ij_sparse[row, :]
                dist_sq = (pos_x[src] - pos_x[tar]) ** 2 + (pos_y[src] - pos_y[tar]) ** 2
                k_dists[row] = np.sqrt(dist_sq)

            if len(pos_x) > 5000:
                log.info("Large population, using random sample")
                # for large populations, we cant fit N^2 distances to ram
                selects = np.sort(np.random.choice(len(pos_x), size=5000, replace=False))
            else:
                selects = slice(None)

            # arrays of combinations shape (N, 2)
            xc = np.array(list(combinations(pos_x[selects], 2)))
            yc = np.array(list(combinations(pos_y[selects], 2)))
            n_dists = np.sqrt(
                np.power(xc[:, 0] - xc[:, 1], 2) + np.power(yc[:, 0] - yc[:, 1], 2)
            )

            return k_dists, n_dists

        except Exception as e:
            log.error(e)
            return np.array([])

    if filenames is None:
        k_dists, n_dists = local_prep(h5f)
    else:
        raise NotImplementedError

    n_h, bins = np.histogram(n_dists, bins=50)
    n_k, _ = np.histogram(k_dists, bins=bins)
    bin_centers = (bins[0:-1] + bins[1:]) / 2

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kwargs = {
        "ax": ax,
        "kde": True,
        # "bins": br,
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "step",
        "alpha": 0.25,
    }

    sns.histplot(
        x=bin_centers,
        weights=n_k / n_h,
        label=r"$k/h$",
        color=matplotlib.cm.get_cmap("tab10").colors[0],
        **kwargs,
    )

    sns.histplot(
        x=bin_centers,
        weights=n_k,
        label=r"$k$",
        color=matplotlib.cm.get_cmap("tab10").colors[1],
        **kwargs,
    )

    sns.histplot(
        x=bin_centers,
        weights=n_h,
        label=r"$h$",
        color=matplotlib.cm.get_cmap("tab10").colors[2],
        **kwargs,
    )

    if apply_formatting:
        ax.set_xlabel(f"Distance between neurons")
        ax.set_ylabel(f"Probability of a conneciton")
        ax.set_title("Degree vs. distance")
        ax.legend(loc="upper right")

    return ax


@warntry
def plot_distribution_axon_length(h5f, ax=None, apply_formatting=True, filenames=None):
    """
    Plot the distribution of the axon length for `h5f`.

    if `filenames` is provided, distribution data is accumulated from all files,
    and only styling/meta information is used from `h5f`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if filenames is None:
        alen = h5f["data.neuron_axon_length"][:]
        aete = h5f["data.neuron_axon_end_to_end_distance"][:]
    else:
        log.info("Loading multiple files for axon length distribution")
        alen = h5.load(filenames, "/data/neuron_axon_length")
        aete = h5.load(filenames, "/data/neuron_axon_end_to_end_distance")
        alen = np.concatenate(alen).ravel()
        aete = np.concatenate(aete).ravel()

    kwargs = {
        "ax": ax,
        "kde": True,
        "stat": "probability",
        "element": "step",
        "alpha": 0.25,
    }

    sns.histplot(
        alen,
        label="length",
        color=matplotlib.cm.get_cmap("tab10").colors[0],
        **kwargs,
    )
    sns.histplot(
        aete,
        label="end to end",
        color=matplotlib.cm.get_cmap("tab10").colors[1],
        **kwargs,
    )

    if apply_formatting:
        ax.set_xlabel(f"Length $l\,[\mu m]$")
        ax.set_ylabel(f"Probability $p(l)$")
        ax.set_title("Axon length distribution")
        ax.legend()
        fig.tight_layout()

    return ax


@warntry
def plot_distribution_dendritic_tree_size(
    h5f, ax=None, apply_formatting=True, filenames=None
):
    """
    Plot the distribution of the size of the dendritic tree for `h5f`.

    if `filenames` is provided, distribution data is accumulated from all files,
    and only styling/meta information is used from `h5f`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if filenames is None:
        n_R_d = h5f["data.neuron_radius_dendritic_tree"][:]
    else:
        log.info("Loading multiple files for distribution of dendritic tree size")
        n_R_d = h5.load(filenames, "/data/neuron_radius_dendritic_tree")
        n_R_d = np.concatenate(n_R_d).ravel()

    kwargs = {
        "ax": ax,
        "kde": True,
        "stat": "probability",
        "element": "step",
        "alpha": 0.25,
    }

    sns.histplot(n_R_d, label="dendritic radius", color="gray", **kwargs)

    if apply_formatting:
        ax.set_xlabel(f"Radius $r\,[\mu m]$")
        ax.set_ylabel(f"Probability $p(r)$")
        ax.set_title("Dendritic tree size")
        fig.tight_layout()

    return ax


def _plot_distribution_from_series(series, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    default_kwargs = {
        "ax": ax,
        # "stat": "density",
        "binwidth": 1,
        "stat": "probability",
        # 'multiple' : 'stack',
        "element": "poly",
    }

    plot_kwargs = dict(default_kwargs)
    plot_kwargs.update(kwargs)
    if "bins" in kwargs.keys():
        plot_kwargs.pop("binwidth")
    # if "bins" not in plot_kwargs.keys():
    #     min_val = np.floor(np.nanmin(series))
    #     max_val = np.ceil(np.nanmax(series))
    #     plot_kwargs["bins"] = np.arange(min_val-0.5, max_val+0.51, 1)

    # print(plot_kwargs)

    sns.histplot(series, **plot_kwargs)

    return ax


# ------------------------------------------------------------------------------ #
# topology
# ------------------------------------------------------------------------------ #


@warntry
def plot_axon_layout(
    h5f, ax=None, apply_formatting=True, axon_kwargs=None, soma_kwargs=None
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if soma_kwargs is None:
        soma_kwargs = dict()
    _plot_soma(h5f, ax, **soma_kwargs)

    if axon_kwargs is None:
        axon_kwargs = dict()
    _plot_axons(h5f, ax, **axon_kwargs)
    # _plot_dendrites(h5f, ax)

    if apply_formatting:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_aspect(1)
        ax.set_xlabel(f"Position $l\,[\mu m]$")
        fig.tight_layout()

    return ax


@warntry
def plot_connectivity_layout(h5f, ax=None, apply_formatting=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if "ana.networkx" in h5f.keypaths():
        G = h5f["ana.networkx.G"]
        pos = h5f["ana.networkx.pos"]
    else:
        # add nodes, generate positions in usable format
        G = nx.DiGraph()
        G.add_nodes_from(h5f["ana.neuron_ids"])
        pos = dict()
        for idx, n in enumerate(h5f["ana.neuron_ids"]):
            pos[n] = (
                h5f["data.neuron_pos_x"][idx],
                h5f["data.neuron_pos_y"][idx],
            )

        # add edges
        try:
            # for large data, we might not have loaded the matrix.
            G.add_edges_from(h5f["data.connectivity_matrix_sparse"][:])
        except Exception as e:
            log.info(e)

        # add to h5f
        h5f["ana.networkx"] = benedict()
        h5f["ana.networkx.G"] = G
        h5f["ana.networkx.pos"] = pos

    log.debug("Drawing graph nodes")
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_size=0,  # we draw them with fixed size using _circles
        node_color="black",
        # with_labels=False,
    )
    log.debug("Drawing graph edges")
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="black", arrows=False, width=0.1)
    log.debug("Drawing soma")
    _plot_soma(h5f, ax)

    if apply_formatting:
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect(1)
        fig.tight_layout()


def _plot_soma(h5f, ax, n_R_s=7.5, **soma_kwargs):
    n_x = h5f["data.neuron_pos_x"][:]
    n_y = h5f["data.neuron_pos_y"][:]
    kwargs = soma_kwargs.copy()
    kwargs.setdefault("fc", "white")
    kwargs.setdefault("ec", "black")
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("lw", 0.25)
    kwargs.setdefault("zorder", 4)
    _circles(n_x, n_y, n_R_s, ax=ax, **kwargs)
    # _circles(n_x, n_y, n_R_s, ax=ax, fc="white", ec="none", alpha=1, lw=0.5, zorder=4)
    # _circles(n_x, n_y, n_R_s, ax=ax, fc="none", ec="black", alpha=0.3, lw=0.5, zorder=5)


def _plot_axons(h5f, ax, **axon_kwargs):
    # axon segments
    # zero-or-nan-padded 2d arrays
    seg_x = h5f["data.neuron_axon_segments_x"][:]
    seg_y = h5f["data.neuron_axon_segments_y"][:]
    seg_x = np.where(seg_x == 0, np.nan, seg_x)
    seg_y = np.where(seg_y == 0, np.nan, seg_y)

    # iterate over neurons to plot axons
    for n in range(len(seg_x)):
        try:
            m_id = h5f["data.neuron_module_id"][n]
            clr = h5f["ana.mod_colors"][m_id]
        except:
            clr = "black"

        kwargs = axon_kwargs.copy()
        kwargs.setdefault("color", clr)
        kwargs.setdefault("lw", 0.35)
        kwargs.setdefault("zorder", 0)
        kwargs.setdefault("alpha", 0.5)

        ax.plot(seg_x[n], seg_y[n], **kwargs)


def _plot_dendrites(h5f, ax):
    n_x = h5f["data.neuron_pos_x"][:]
    n_y = h5f["data.neuron_pos_y"][:]
    n_R = h5f["data.neuron_radius_dendritic_tree"][:]
    _circles(
        n_x,
        n_y,
        n_R,
        ax=ax,
        ls="--",
        fc="none",
        ec="gray",
        alpha=0.5,
        lw=0.9,
        zorder=0,
    )


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def _argsort_neuron_ids_by_module(h5f, neuron_ids):
    """
    return indices that would sort the provided list of neuron ids
    according to module, and, secondly, neuron id
    """
    module_ids = h5f["data.neuron_module_id"][neuron_ids]
    sort_idx = np.lexsort((neuron_ids, module_ids))
    return sort_idx


def _style_legend(leg):
    try:
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("#e4e5e6")
        leg.get_frame().set_alpha(0.9)
    except:
        log.warning("Failed to style legend")

    return leg


def _ticklabels_lin_to_log10(x, pos):
    """
    converts ticks of manually logged data (lin ticks) to log ticks, as follows
     1 -> 10
     0 -> 1
    -1 -> 0.1

    # Example
    ```
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(_ticklabels_lin_to_log10_power)
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(_ticklocator_lin_to_log_minor())
    ```
    """
    prec = int(np.ceil(-np.minimum(x, 0)))
    return "{{:.{:1d}f}}".format(prec).format(np.power(10.0, x))


def _ticklabels_lin_to_log10_power(x, pos, nicer=True, nice_range=[-1, 0, 1]):
    """
    converts ticks of manually logged data (lin ticks) to log ticks, as follows
     1 -> 10^1
     0 -> 10^0
    -1 -> 10^-1
    """
    if x.is_integer():
        # use easy to read formatter if exponents are close to zero
        if nicer and x in nice_range:
            return _ticklabels_lin_to_log10(x, pos)
        else:
            return r"$10^{{{:d}}}$".format(int(x))
    else:
        # return r"$10^{{{:f}}}$".format(x)
        return ""


def _ticklocator_lin_to_log_minor(vmin=-10, vmax=10, nbins=10):
    """
    get minor ticks on when manually converting lin to log
    """
    locs = []
    orders = int(np.ceil(vmax - vmin))
    for o in range(int(np.floor(vmin)), int(np.floor(vmax + 1)), 1):
        locs.extend([o + np.log10(x) for x in range(2, 10)])
    return matplotlib.ticker.FixedLocator(locs, nbins=nbins * orders)


# circles in data scale
# https://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker
def _circles(x, y, s, c="b", vmin=None, vmax=None, ax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None
    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection
