# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-01-24 13:43:39
# @Last Modified: 2022-04-12 16:05:53
# ------------------------------------------------------------------------------- #
# Create a movie of the network for a given time range and visualize
# firing neurons. Saves to mp4.
# ------------------------------------------------------------------------------- #

import os
from re import L
import sys
import glob
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib
import hi5 as h5
import plot_helper as ph
import ana_helper as ah
import seaborn as sns
import colors as cc

# fmt:off
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter

plt.style.use("dark_background")
plt.ioff()
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
    "#5886be", "#f3a093", "#53d8c9", "#f9c192", "#f2da9c", # light
])
matplotlib.rcParams["figure.dpi"] = 300
# fmt:on


def main(args):

    global topo, axes, x, y, dt, flr, h5f

    metadata = dict(title=f"{args.title}", artist="Matplotlib", comment="Yikes! Spikes!")
    writer = FFMpegWriter(fps=args.fps, metadata=metadata)
    num_frames = int(args.fps * args.movie_duration) + 1

    # keep a list of renderer objects that will be updated in the main loop.
    # everyone needs to have a function `set_time` that takes the experimental
    # timestamp as input.
    renderers = []

    # 1800x900 px at 300 dpi, where fonts are decent
    # fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    fig = plt.figure(figsize=(6, 3))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[0.33, 0.66],
        # wspace=0.05,
        hspace=0.1,
        left=0.1,
        right=0.99,
        top=0.99,
        bottom=0.15,
    )


    # ------------------------------------------------------------------------------ #
    # Topology plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[0, 0])
    topo = TopologyRenderer(input_path=args.input_path, ax=ax)

    # its helpful to set the decay time propto movie playback speed
    time_ratio = args.movie_duration / (args.tend - args.tbeg)
    topo.decay_time = 0.5 / time_ratio
    renderers.append(topo)

    # ------------------------------------------------------------------------------ #
    # population activity plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[1, 1])
    h5f = ah.prepare_file(args.input_path)
    ah.find_rates(h5f)

    ph.plot_system_rate(h5f=h5f, ax=ax, color="white")
    ax.get_legend().set_visible(False)
    sns.despine(ax=ax, right=True, top=True, offset=5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Rate (Hz)")
    sns.despine(ax=ax, bottom=True)
    # cc.detick(axis=ax.xaxis, keep_labels=True)

    ar = MovingWindowRenderer(
        ax=ax,
        data_beg=args.tbeg,
        data_end=args.tend,
        window_from=-20.0,
        window_to=100,
    )

    renderers.append(ar)

    # ------------------------------------------------------------------------------ #
    # raster plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[0, 1])

    ph.plot_raster(h5f=h5f, ax=ax)
    sns.despine(ax=ax, right=True, top=True, left=True, bottom=True)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_xlabel("")

    sr = MovingWindowRenderer(
        ax=ax,
        data_beg=args.tbeg,
        data_end=args.tend,
        window_from=-20.0,
        window_to=100,
    )

    renderers.append(sr)

    # ------------------------------------------------------------------------------ #
    # Resource cycles
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[1, 0])

    ah.find_module_level_adaptation(h5f)
    x_sets = []
    y_sets = []
    colors = []
    for mod_id in [0, 1, 2, 3]:
        x = h5f[f"ana.adaptation.module_level.mod_{mod_id}"]
        y = h5f[f"ana.rates.module_level.mod_{mod_id}"]
        dt = h5f[f"ana.rates.dt"]
        assert (
            h5f[f"ana.adaptation.dt"] == dt
        ), "adaptation and rates need to share the same time step"
        # limit data range to whats needed
        x = x[0 : int(args.tend / dt) + 2]
        y = y[0 : int(args.tend / dt) + 2]
        x_sets.append(x)
        y_sets.append(y)
        colors.append(f"C{mod_id}")

    flr = FadingLineRenderer(
        x=x_sets, y=y_sets, colors=colors, dt=dt, ax=ax, tbeg=args.tbeg
    )
    flr.decay_time = 0.5 / time_ratio

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))
    ax.set_ylim(-15, 145)
    sns.despine(ax=ax, right=True, top=True, trim=True, offset=1)
    ax.set_xlabel("Resources")
    ax.set_ylabel("Rates (Hz)")

    renderers.append(flr)

    fig.tight_layout()

    # ------------------------------------------------------------------------------ #
    # movie loop
    # ------------------------------------------------------------------------------ #

    with writer.saving(fig=topo.fig, outfile=args.output_path, dpi=300):
        print(f"Rendering {args.movie_duration:.0f} seconds with {num_frames} frames ...")

        for fdx in tqdm(range(num_frames)):
            exp_time = frame_index_to_experimental_time(
                this_frame=fdx,
                total_frames=num_frames,
                exp_start=args.tbeg,
                exp_end=args.tend,
            )

            for r in renderers:
                r.set_time(exp_time)

            writer.grab_frame(facecolor=topo.canvas_clr)


# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Movie from Spiketrains")
    parser.add_argument(
        "-i", dest="input_path", help="input path", metavar="FILE", required=True
    )
    parser.add_argument(
        "-o", dest="output_path", help="output path", metavar="FILE", required=True
    )
    parser.add_argument("--title", dest="title", help="movie title")
    parser.add_argument(
        "--fps", dest="fps", help="frames per second for the movie", type=int, default=30
    )
    parser.add_argument(
        "--length",
        dest="movie_duration",
        help="desired length of the movie, in seconds",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--tbeg",
        dest="tbeg",
        help="movie starts at this time point, experimental time (seconds)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--tend",
        dest="tend",
        help="movie ends at this time point, experimental time (seconds)",
        type=float,
        default=60,
    )

    args = parser.parse_args()
    if args.title == None:
        args.title = args.input_path

    return args


def rgba_to_rgb(c, bg="white"):
    bg = mcolors.to_rgb(bg)
    alpha = c[-1]

    res = (
        (1 - alpha) * bg[0] + alpha * c[0],
        (1 - alpha) * bg[1] + alpha * c[1],
        (1 - alpha) * bg[2] + alpha * c[2],
    )
    return res


def frame_index_to_experimental_time(this_frame, total_frames, exp_start, exp_end):
    # in experimental time
    exp_duration = exp_end - exp_start

    # every frame corresponds to experimental time
    exp_timer_per_frame = exp_duration / total_frames

    return exp_start + this_frame * exp_timer_per_frame


# ------------------------------------------------------------------------------ #
# Topology
# ------------------------------------------------------------------------------ #


class TopologyRenderer(object):
    """
    This guy provides the helpers to plot soma and axons and lets them light
    up on spikes. the glow fades with a customizable duration (in time units of)
    the experiment
    """

    def __init__(self, input_path, ax=None):
        self.input_path = input_path

        # some styling options
        self.canvas_clr = "black"
        self.title_clr = "white"
        self.neuron_clr = "white"

        # try to color neurons differently, according to modules
        mod_ids = h5.load(self.input_path, "/data/neuron_module_id")
        self.n_id_clr = []

        # neuron radius, um
        self.rad_n = 7.5

        # fade time of neurons after spiking mimicing calcium indicator decay
        # time in seconds of the experimental time
        self.decay_time = 5.0

        # experimental time
        self.time_unit = "s"

        # number of neurons
        self.num_n = int(h5.load(self.input_path, "/meta/topology_num_neur"))

        # keep a list of spiketimes for every neuron.
        self.spiketimes = []
        # load event times
        # two-column list of spiketimes. first col is neuron id, second col the spiketime.
        spikes = h5.load(self.input_path, "/data/spiketimes_as_list")
        # convert to something we can work with, list of arrays: [neuron][spiketimes]
        for n_id in range(self.num_n):
            idx = np.where(spikes[:, 0] == n_id)[0]
            self.spiketimes.append(spikes[idx, 1])
            self.n_id_clr.append(f"C{mod_ids[n_id]}")

        # create a bunch of artists that we can update later
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=[6.4, 6.4])
        else:
            self.ax = ax
            self.fig = ax.get_figure()
        self.init_background()

    def init_background(
        self,
        axon_edge=(1.0, 1.0, 1.0, 0.1),
        soma_edge=(1.0, 1.0, 1.0, 0.1),
        soma_face=(1.0, 1.0, 1.0, 0.1),
    ):
        """
        This changes the style of the figure element, associated with ax.
        """
        # default colors, spikes will drawn over this. its nice to have them solid, cast alpha
        axon_edge = rgba_to_rgb(axon_edge, self.canvas_clr)
        soma_edge = rgba_to_rgb(soma_edge, self.canvas_clr)
        soma_face = rgba_to_rgb(soma_face, self.canvas_clr)

        ax = self.ax
        fig = self.fig

        fig.patch.set_facecolor(self.canvas_clr)
        # ax.set_title(f"{args.title}", fontsize=16, color=title_clr)
        ax.set_facecolor(self.canvas_clr)
        ax.set_axis_off()
        ax.set_aspect(1)

        # load data
        pos_x = h5.load(self.input_path, "/data/neuron_pos_x")  # soma centers
        pos_y = h5.load(self.input_path, "/data/neuron_pos_y")
        seg_x = h5.load(self.input_path, "/data/neuron_axon_segments_x")  # 2d array
        seg_y = h5.load(self.input_path, "/data/neuron_axon_segments_y")
        seg_x = np.where(seg_x == 0, np.nan, seg_x)  # overwrite padding 0 at the end
        seg_y = np.where(seg_y == 0, np.nan, seg_y)

        # keep handles for drawn elements so we can update them later
        self.art_time = fig.text(
            0.02, 0.95, "current time", fontsize=8, color=self.title_clr
        )

        art_axons = []
        for i in range(len(seg_x)):
            # background
            tmp = ax.plot(
                seg_x[i],
                seg_y[i],
                color=axon_edge,
                lw=0.5,
                zorder=0,
            )
            # foreground overlay, when spiking
            tmp = ax.plot(
                seg_x[i],
                seg_y[i],
                color=(0, 0, 0, 0),
                lw=0.7,
                zorder=3,
            )
            art_axons.append(tmp[0])
        self.art_axons = art_axons

        art_soma = []
        for i in range(len(pos_x)):
            # background
            circle = plt.Circle((pos_x[i], pos_y[i]), radius=self.rad_n, lw=0.5, zorder=1)
            circle.set_facecolor(soma_face)
            circle.set_edgecolor(soma_edge)
            circle.set_linewidth(0.25)
            ax.add_artist(circle)
            # foreground overlay, when spiking
            circle = plt.Circle((pos_x[i], pos_y[i]), radius=self.rad_n, lw=0.7, zorder=4)
            circle.set_facecolor((0, 0, 0, 0))
            circle.set_edgecolor((0, 0, 0, 0))
            circle.set_linewidth(0.25)
            ax.add_artist(circle)
            art_soma.append(circle)
        self.art_soma = art_soma

    def set_time(self, time):
        """
        Update the whole figure to a given experimental time stamp
        """
        self.art_time.set_text(f"t = {time :.2f} {self.time_unit}")

        for n_id in range(self.num_n):
            self.style_neuron_to_time(n_id, time)

    def style_neuron_to_time(self, n_id, time):
        """
        set the appearence of a single neuron to the current time stamp
        """
        spikes = self.spiketimes[n_id]
        try:
            times_to_show = np.where(
                (spikes <= time) & (spikes >= time - 10 * self.decay_time)
            )[0]
            times_to_show = spikes[times_to_show]
        except:
            # no spike occured yet
            times_to_show = []

        # calculate total alpha (brightness) by adding up past spikes
        total_alpha = 0
        for last_time in times_to_show:
            dt = time - last_time
            alpha = np.exp(-dt / self.decay_time)
            # alpha = 1.0 - dt / self.decay_time
            # at least x consecutive spikes needed to reach full brightness
            alpha /= 10
            total_alpha += alpha
        total_alpha = np.clip(total_alpha, 0.0, 1.0)

        try:
            clr = self.n_id_clr[n_id]
        except:
            clr = self.neuron_clr

        # rgba as 4-tuple, between 0 and 1
        # make soma a bit brighter than axons
        ax_edge = (*mcolors.to_rgb(clr), total_alpha * 0.6)
        sm_edge = (*mcolors.to_rgb(clr), total_alpha * 0.8)
        sm_face = (*mcolors.to_rgb(clr), total_alpha * 1.0)

        # update the actual foreground layer
        self.art_axons[n_id].set_color(ax_edge)
        self.art_soma[n_id].set_edgecolor(sm_edge)
        self.art_soma[n_id].set_facecolor(sm_face)


# ------------------------------------------------------------------------------ #
# Fading Lines
# ------------------------------------------------------------------------------ #


class FadingLineRenderer(object):
    """
    Think of an oldschool osci, where the electron beam draws the line and fades,
    draw line transparency as exponential

    Create plot lines as line segments so that styles can be updated later
    c.f. https://github.com/dpsanders/matplotlib-examples/blob/master/colorline.py
    """

    def __init__(
        self, x, y, ax=None, dt=1, tbeg=0, colors=None, background="transparent"
    ):
        """
        # Parameters:
        x, y : 1d arrays of the data to plot, or list of arrays, if multiple lines
            should go into the same axis
        dt : float, how to map from time to data index, every point in x is assumed
            to have length dt (and do start at t=0)
        tbeg : float, if x does not start at time 0, when does it start?
        ax : matplotlib axis element to draw into
        kwargs : passed to LineCollection
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=[6.4, 6.4])
        else:
            self.ax = ax
            self.fig = ax.get_figure()

        # cast to list if only one array given
        if not isinstance(x, list):
            x = [x]
            y = [y]

        if not isinstance(colors, list):
            colors = [colors] * len(x)

        # default colors
        for idx, c in enumerate(colors):
            if c is None:
                if len(colors) == 1:
                    colors[idx] = "white"
                else:
                    colors[idx] = f"C{idx}"

        self.num_timesteps = len(x[0])

        self.x_sets = x
        self.y_sets = y
        self.colors = colors
        self.lcs = []
        self.cmaps = []

        for clr in self.colors:
            if background == "transparent":
                background = cc.to_rgb(clr)
                background += (0,)
            cmap = cc.create_cmap(
                start=background,
                end=clr,
            )
            self.cmaps.append(cmap)

        # casting experimental time to the data array
        self.dt = dt
        self.tbeg = tbeg

        # fade transparency,
        self.decay_time = 15.0  # exponential decay time in experimental time units

        # keep handles for drawn elements so we can update them later
        self.art_time = ax.text(0.02, 0.95, "current time", fontsize=8, color="white")

        # print("FadineLineRenderer:")
        # print(f"{self.num_timesteps} timesteps")
        # print(f"{self.dt} dt")
        # print(f"{self.tbeg + self.num_timesteps*self.dt} data duration")
        # print(f"{self.decay_bins} decay bins")
        # print(f"{self.decay_time} decay time")

    @property
    def decay_time(self):
        return self._decay_time

    @decay_time.setter
    def decay_time(self, tau):
        self._decay_time = tau
        # now update mask and bins
        # after around 3 decay times we are sufficiently close to zero
        self.decay_bins = int(3 * tau / self.dt)

        t = np.arange(0, self.decay_bins) * -self.dt
        mask = np.exp(-t / tau)
        # make sure last entry is zero
        m_min = np.nanmin(mask)
        m_max = np.nanmax(mask)
        m_diff = m_max - m_min
        self.decay_mask = (mask - m_min) / m_diff

    def set_time(self, time):

        self.art_time.set_text(f"t = {time :.2f}")

        # get time range we need to show
        time_index = int((time - self.tbeg) / self.dt)
        first_visible = time_index - self.decay_bins
        if time_index < 0:
            time_index = 0
        if time_index >= self.num_timesteps:
            time_index = self.num_timesteps
        if first_visible < 0:
            first_visible = 0
        num_bins = time_index - first_visible

        # remove old lines
        for lc in self.lcs:
            lc.remove()
            del lc
        self.lcs = []

        # draw new lines
        for idx in range(0, len(self.x_sets)):
            x = self.x_sets[idx]
            y = self.y_sets[idx]

            x = x[time_index - num_bins : time_index]
            y = y[time_index - num_bins : time_index]
            z = self.decay_mask[-num_bins:]

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, array=z, cmap=self.cmaps[idx], capstyle="round")
            self.ax.add_collection(lc)
            self.lcs.append(lc)


class MovingWindowRenderer(object):
    """
    Plot into an axis and dynamically change the shown range, and maybe
    an indicator for the current time point.

    Used for e.g. raster plots.
    """

    def __init__(
        self,
        window_from,
        window_to,
        time_indicator = "default",
        data_beg = None,
        data_end = None,
        ax=None,
    ):
        """
        # Parameters:
        window_from : float, relative to time, where does the window start
            e.g. at -20 seconds
        window_to : float, relative to time, where does the window end
        time_indicator : "default" to use a dashed line to show current time, or
            provide a callback function that takes the timestamp
        data_beg : where does the data start / end. dont change window to exceed.
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=[6.4, 6.4])
        else:
            self.ax = ax
            self.fig = ax.get_figure()

        self.window_from = window_from
        self.window_to = window_to
        self.window_size = window_to - window_from

        if time_indicator is None:
            self.set_time_indicator = lambda x : None
        if time_indicator == "default":
            indicator = ax.axvline(window_from, ls=":")
            self.set_time_indicator = indicator.set_xdata
        else:
            self.set_time_indicator = time_indicator

        if data_beg is not None:
            self.data_beg = data_beg
        else:
            # try to infer
            self.data_beg = ax.get_xlim()[0]

        if data_end is not None:
            self.data_end = data_end
        else:
            self.data_end = ax.get_xlim()[1]

        assert self.window_size <= self.data_end - self.data_beg

        self.ax.set_xlim(self.data_beg, self.data_beg + self.window_size)

    def set_time(self, time):

        old_beg, old_end = self.ax.get_xlim()
        beg = time + self.window_from
        end = time + self.window_to
        if beg >= self.data_beg and end <= self.data_end:
            # all good
            pass
        elif beg <= self.data_beg:
            # starting, let the time indicator run from zero to where it stays
            beg = self.data_beg
            end = self.data_beg + self.window_size
        elif end >= self.data_end:
            end = self.data_end
            beg = self.data_end - self.window_size

        if beg != old_beg and end != old_end:
            self.ax.set_xlim(beg, end)

        self.set_time_indicator(time)


if __name__ == "__main__":
    args = parse_args()
    main(args)
