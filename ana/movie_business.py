# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-01-24 13:43:39
# @Last Modified: 2022-09-01 11:46:47
# ------------------------------------------------------------------------------- #
# Classed needed to create a movie of the network.
# ------------------------------------------------------------------------------- #

import os
import numpy as np
from tqdm import tqdm
import matplotlib
import hi5 as h5
import colors as cc
import logging

# fmt:off
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter

from bitsandbobs.plt import alpha_to_solid_on_bg

# theme_bg = "black"
theme_bg = "white"
if theme_bg == "black":
    plt.style.use("dark_background")
    # use custom colors to match the paper
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
        "#5886be", "#f3a093", "#53d8c9", "#f9c192", "#f2da9c",
    ])
else:
    plt.style.use("default")
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
        "#233954", "#ea5e48", "#1e7d72", "#f49546", "#e8bf58",
    ])
plt.ioff()
matplotlib.rcParams["figure.dpi"] = 300
# fmt:on


class MovieWriter(object):
    """
    Thin wrapper around matplotlib FFMpegWriter.

    Takes a list of renderers that get updated at every frame, by calling their
    `set_time` function.
    The renderers should share the same data time, where the conversion from
    current frame to data time is done by the MovieWriter using the `tbeg` and
    `tend`. And they should be plotting into the same matplotlib figure object.

    # Parameters:
    movie_duration : float
        movie duration in seconds
    fps : int
        frame rate passed to ffmpeg
    tbeg : float
        time when the visualized data start, in units of the data (experiment)
    tend : float
        time when the data ends, units of the experiment
    renderers : list of arbitrary objects
        at every frame, the `set_time` function will be called
    kwargs : dict
        passed to FFMpegWriter.
    """

    def __init__(
        self, output_path, tbeg, tend, renderers=None, fps=30, movie_duration=30, **kwargs
    ):

        kwargs = kwargs.copy()
        kwargs.setdefault(
            "metadata",
            dict(
                title="Noise-driven control of synchrony",
                artist="Matplotlib",
                comment="Yikes! Spikes!",
            ),
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.writer = FFMpegWriter(fps=fps, **kwargs)
        self.fps = fps
        self.movie_duration = movie_duration
        self.num_frames = int(fps * movie_duration) + 1

        if renderers is None:
            renderers = []
        self.renderers = renderers

        self.tbeg = tbeg
        self.tend = tend
        self.output_path = output_path

        log.info(
            f"Created MovieWriter for {output_path} and {movie_duration} seconds movie"
            " duration"
        )

    def render(self):
        assert len(self.renderers) > 0, "Dont forget to add renderers"
        fig = self.renderers[0].ax.get_figure()
        fig.patch.set_facecolor(theme_bg)
        with self.writer.saving(fig=fig, outfile=self.output_path, dpi=300):
            log.info(f"Rendering {self.movie_duration:.0f} seconds at {self.fps} fps")

            for fdx in tqdm(range(self.num_frames), desc="Frames", leave=False):
                exp_time = self.frame_index_to_experimental_time(frame=fdx)

                for r in self.renderers:
                    r.set_time(exp_time)

                self.writer.grab_frame(facecolor=fig.patch.get_facecolor())

    def frame_index_to_experimental_time(self, frame):
        # in experimental time
        exp_duration = self.tend - self.tbeg

        # every frame corresponds to experimental time
        exp_timer_per_frame = exp_duration / self.num_frames

        return self.tbeg + frame * exp_timer_per_frame


# ------------------------------------------------------------------------------ #
# Renderers, standard guys that seem usable in other contexts
# ------------------------------------------------------------------------------ #


class FadingLineRenderer(object):
    """
    Think of an oldschool osci, where the electron beam draws the line and fades,
    draw line transparency as exponential.

    Create plot lines as line segments so that styles can be updated later
    c.f. https://github.com/dpsanders/matplotlib-examples/blob/master/colorline.py

    For my use case, it turned out to be faster to recreate the linecollection rather
    then changing the colors of an existing (super large) one.

    # Parameters

    x, y : 1d arrays of the data to plot, or list of arrays, if multiple lines
        should go into the same axis
    dt : float, how to map from time to data index, every point in x is assumed
        to have length dt (and do start at t=0)
    tbeg : float, if x does not start at time 0, when does it start?
    ax : matplotlib axis element to draw into
    kwargs : passed to LineCollection
    """

    def __init__(
        self,
        x,
        y,
        ax=None,
        dt=1,
        tbeg=0,
        colors=None,
        background="transparent",
        decay_time=10.0,
        show_time=False,
        **kwargs,
    ):
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
        # exponential decay time in experimental time units
        self.decay_time = decay_time

        # keep handles for drawn elements so we can update them later
        self.show_time = show_time
        self.art_time = ax.text(0.02, 0.95, "", fontsize=8, color="white")

        assert "cmap" not in kwargs.keys(), "cmaps are set via `colors` argument"
        self.lc_kwargs = kwargs.copy()
        self.lc_kwargs.setdefault("capstyle", "round")

        log.info(f"Created FadingLineRender for {len(self.x_sets)} lines")
        log.info(f"{self.num_timesteps} timesteps at {self.dt}")
        log.info(f"data time from {self.tbeg} to {self.num_timesteps*self.dt}")
        log.info(
            f"decay time is {self.decay_time}, spanning {self.decay_bins} time steps"
        )

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

        if self.show_time:
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
            lc = LineCollection(segments, array=z, cmap=self.cmaps[idx], **self.lc_kwargs)
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
        time_indicator="default",
        tbeg=None,
        tend=None,
        ax=None,
    ):
        """
        # Parameters:
        window_from : float, relative to time, where does the window start
            e.g. at -20 seconds
        window_to : float, relative to time, where does the window end
        time_indicator : "default" to use a dashed line to show current time, or
            provide a callback function that takes the timestamp
        tbeg, tend : where does the data start / end. window should not be larger
            than the data range.
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
            self.set_time_indicator = lambda x: None
        if time_indicator == "default":
            indicator = ax.axvline(window_from, ls=":")
            self.set_time_indicator = indicator.set_xdata
        else:
            self.set_time_indicator = time_indicator

        if tbeg is not None:
            self.tbeg = tbeg
        else:
            # try to infer
            self.tbeg = ax.get_xlim()[0]

        if tend is not None:
            self.tend = tend
        else:
            self.tend = ax.get_xlim()[1]

        assert self.window_size <= self.tend - self.tbeg

        self.ax.set_xlim(self.tbeg, self.tbeg + self.window_size)

        log.info(f"Created MovingWindowRenderer with window size {self.window_size}")
        log.info(f"data time from {self.tbeg} to {self.tend}")

    def set_time(self, time):

        old_beg, old_end = self.ax.get_xlim()
        beg = time + self.window_from
        end = time + self.window_to
        if beg >= self.tbeg and end <= self.tend:
            # all good
            pass
        elif beg <= self.tbeg:
            # starting, let the time indicator run from zero to where it stays
            beg = self.tbeg
            end = self.tbeg + self.window_size
        elif end >= self.tend:
            end = self.tend
            beg = self.tend - self.window_size

        if beg != old_beg and end != old_end:
            self.ax.set_xlim(beg, end)

        self.set_time_indicator(time)


class TextRenderer(object):
    def __init__(self, text_object):
        self.text_object = text_object

    def set_time(self, time):
        self.text_object.set_text(f"t = {time :.2f}")


# ------------------------------------------------------------------------------ #
# Topology, this one is only useful for neurons in a h5f following my data format
# ------------------------------------------------------------------------------ #


class TopologyRenderer(object):
    """
    This guy provides the helpers to plot soma and axons and lets them light
    up on spikes. the glow fades with a customizable duration (in time units of)
    the experiment

    # Parameters
    color : str
        used for foreground elements
    background : str
        canvas.
    """

    def __init__(
        self,
        input_path,
        ax=None,
        decay_time=10.0,
        title_color="white",
        neuron_color=None,
        background="black",
        show_time=False,
    ):

        self.input_path = input_path

        # some styling options
        self.background = background
        self.title_clr = title_color
        if neuron_color is None:
            self.neuron_color = alpha_to_solid_on_bg("#bbb", 0.1, bg=background)
            log.info(f"Using default neuron color {self.neuron_color}")
        else:
            self.neuron_clr = neuron_color

        # try to color neurons differently, according to modules
        mod_ids = h5.load(self.input_path, "/data/neuron_module_id")
        self.n_id_clr = []

        # neuron radius, um
        self.rad_n = 7.5

        # fade time of neurons after spiking mimicing calcium indicator decay
        # time in seconds of the experimental time
        self.decay_time = decay_time

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

        self.show_time = show_time

        # create a bunch of artists that we can update later
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=[6.4, 6.4])
        else:
            self.ax = ax
            self.fig = ax.get_figure()
        self.init_background(
            # axon_edge = (1.0, 1.0, 1.0, 0.1),
            # soma_edge = (1.0, 1.0, 1.0, 0.1),
            # soma_face = (1.0, 1.0, 1.0, 0.1),
            axon_edge = self.neuron_color,
            soma_edge = self.neuron_color,
            soma_face = self.neuron_color,
        )

        log.info(f"Created TopologyRenderer for {input_path}")
        try:
            log.info(f"Spiketimes between {spikes[0, 1]} and {spikes[-1, 1]}")
        except:
            log.warning("No spikes found in the data")

    def init_background(self, axon_edge, soma_edge, soma_face):
        """
        This changes the style of the figure element, associated with ax.
        """
        # default colors, spikes will drawn over this. its nice to have them solid, cast alpha
        axon_edge = _rgba_to_rgb(axon_edge, self.background)
        soma_edge = _rgba_to_rgb(soma_edge, self.background)
        soma_face = _rgba_to_rgb(soma_face, self.background)

        ax = self.ax
        fig = self.fig

        # ax.set_title(f"{args.title}", fontsize=16, color=title_clr)
        ax.set_facecolor(self.background)
        ax.set_axis_off()
        ax.set_aspect(1)

        # keep handles for drawn elements so we can update them later
        self.art_time = fig.text(0.02, 0.95, "", fontsize=8, color=self.title_clr)

        # ------------------------------------------------------------------------------ #
        # axons
        # ------------------------------------------------------------------------------ #

        try:
            seg_x = h5.load(
                self.input_path, "/data/neuron_axon_segments_x", raise_ex=True
            )
            seg_y = h5.load(
                self.input_path, "/data/neuron_axon_segments_y", raise_ex=True
            )
            # overwrite padding 0 at the end
            seg_x = np.where(seg_x == 0, np.nan, seg_x)
            seg_y = np.where(seg_y == 0, np.nan, seg_y)
        except:
            log.warning("No axons found in the data")
            # this happens for experimental data
            seg_x = []
            seg_y = []

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

        # ------------------------------------------------------------------------------ #
        # soma
        # ------------------------------------------------------------------------------ #

        pos_x = h5.load(self.input_path, "/data/neuron_pos_x")
        pos_y = h5.load(self.input_path, "/data/neuron_pos_y")

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
        if self.show_time:
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
        self.art_soma[n_id].set_edgecolor(sm_edge)
        self.art_soma[n_id].set_facecolor(sm_face)
        if len(self.art_axons) != 0:
            self.art_axons[n_id].set_color(ax_edge)


# ------------------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------------------ #


def _rgba_to_rgb(c, bg="white"):
    bg = mcolors.to_rgb(bg)
    c = mcolors.to_rgba(c)
    alpha = c[-1]

    res = (
        (1 - alpha) * bg[0] + alpha * c[0],
        (1 - alpha) * bg[1] + alpha * c[1],
        (1 - alpha) * bg[2] + alpha * c[2],
    )
    return res
