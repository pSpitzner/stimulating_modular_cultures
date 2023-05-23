# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-01-24 13:43:39
# @Last Modified: 2023-05-11 10:55:48
# ------------------------------------------------------------------------------- #
# Classes needed to create a movie of the network.
# ------------------------------------------------------------------------------- #

import os
import numpy as np
from tqdm import tqdm
import matplotlib
import bitsandbobs as bnb
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
plt.ioff()
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
        # "#5886be", "#f3a093", "#53d8c9", "#f9c192", "#f2da9c",
        "#F7A233", "#2F8DC8", "#DBAD70", "#77A8C6",
    ])
else:
    plt.style.use("default")
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color", [
        # "#233954", "#ea5e48", "#1e7d72", "#f49546", "#e8bf58",
        "#BD6B00", "#135985", "#ca8933", "#427A9D",
    ])
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
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
                background = bnb.plt.to_rgb(clr)
                background += (0,)
            cmap = bnb.plt.create_cmap(
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
        # check file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File {input_path} does not exist")
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
        mod_ids = bnb.hi5.load(self.input_path, "/data/neuron_module_id")
        self.n_id_clr = []

        # neuron radius, um
        self.rad_n = 7.5

        # fade time of neurons after spiking mimicing calcium indicator decay
        # time in seconds of the experimental time
        self.decay_time = decay_time

        # experimental time
        self.time_unit = "s"

        # number of neurons
        self.num_n = int(bnb.hi5.load(self.input_path, "/meta/topology_num_neur"))

        # keep a list of spiketimes for every neuron.
        self.spiketimes = []
        # load event times
        # two-column list of spiketimes. first col is neuron id, second col the spiketime.
        spikes = bnb.hi5.load(self.input_path, "/data/spiketimes_as_list")
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
            axon_edge=self.neuron_color,
            soma_edge=self.neuron_color,
            soma_face=self.neuron_color,
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
            seg_x = bnb.hi5.load(
                self.input_path, "/data/neuron_axon_segments_x", raise_ex=True
            )
            seg_y = bnb.hi5.load(
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

        pos_x = bnb.hi5.load(self.input_path, "/data/neuron_pos_x")
        pos_y = bnb.hi5.load(self.input_path, "/data/neuron_pos_y")

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


class CultureGrowthRenderer(object):
    def __init__(
        self,
        input_path,
        ax=None,
    ):
        if isinstance(input_path, str):
            self.h5f = bnb.hi5.recursive_load(input_path)
        else:
            self.h5f = input_path

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            self.ax = ax
        else:
            self.ax = ax

        ax.set_aspect("equal")
        # if we overwrite this to a number, the animations below will be tweaked to
        # focus the specified neuron id
        self.focus_id = None

        # disable all the axis and ticks
        # ax.axis("off")

        self.num_neurons = len(self.h5f["data"]["neuron_pos_x"])
        self.neurons = []
        for n_id in range(0, self.num_neurons):
            self.neurons.append(
                SingleNeuronGrowth(
                    self.h5f,
                    n_id,
                    base_color=mcolors.hsv_to_rgb((np.random.uniform(0, 1), 0.8, 0.8)),
                    ax=self.ax,
                )
            )

        # lets init a bezier curve that we can use for animation interpolation.
        # its 1000 steps long (x) and goes from 0 to 1 (y)
        _, self._bezier = _bezier(0, 1, 0, 1, 1001)

    def smooth(self, x, beg=0, end=1):
        """
        smooth a value between beg and end, returning beg for x=0.0 and end for x=1.0
        """
        # get the index of the bezier curve
        idx = int(np.clip(x, 0, 1) * 1000)
        # get the value from the bezier curve
        return beg + (end - beg) * self._bezier[idx]

    def set_axon_alpha(self, alpha, n_id=None):
        alpha = np.clip(alpha, 0, 1)
        for neuron in self.neurons:
            if n_id is None or neuron.n_id == n_id:
                neuron.artists["axon"].set_alpha(alpha)

    def set_connections_alpha(self, alpha, n_id=None):
        alpha = np.clip(alpha, 0, 1)
        for neuron in self.neurons:
            if n_id is None or neuron.n_id == n_id:
                for line in neuron.artists["connections"]:
                    line.set_alpha(alpha)

    def set_dendritic_tree_alpha(self, alpha, n_id=None):
        alpha = np.clip(alpha, 0, 1)
        for neuron in self.neurons:
            if n_id is None or neuron.n_id == n_id:
                circle = neuron.artists["dendritic_tree"]
                base = neuron.base_color
                inner = (*mcolors.to_rgb(base), neuron.dendrite_alpha_inner * alpha)
                outer = (*mcolors.to_rgb(base), neuron.dendrite_alpha_outer * alpha)
                circle.set_facecolor(inner)
                circle.set_edgecolor(outer)

    def set_soma_alpha(self, alpha, n_id=None):
        alpha = np.clip(alpha, 0, 1)
        for neuron in self.neurons:
            if n_id is None or neuron.n_id == n_id:
                neuron.artists["soma"].set_alpha(alpha)

    def set_num_visible_segments(self, num_segs, n_id=None):
        for neuron in self.neurons:
            if n_id is None or neuron.n_id == n_id:
                neuron.set_num_visible_segments(num_segs)

    # ------------------------------------------------------------------------------ #
    # here we implement our animation scripting
    # ------------------------------------------------------------------------------ #
    def set_time(self, time):
        """
        experimental time, does not have meaning here. arbitrary units,
        lets call them frames.
        make sure that the renderer passes integer-like time values
        e.g. when movie_duration is 25, fps is 30 then set `tbeg=0` and `tend=751`
        """
        time = int(time)

        # hide everything
        if time == 0:
            self.set_axon_alpha(0)
            self.set_num_visible_segments(0)
            self.set_connections_alpha(0)
            self.set_dendritic_tree_alpha(0)
            self.set_soma_alpha(1)

        # get a smoothed out interpolation between 0 and 1 with ease-in ease-out effect.
        sm = self.smooth

        # make soma appear over 100 frames
        if time <= 100:
            # f is a "relative frame count" for this animation part
            f = time - 0
            self.set_soma_alpha(sm(f / 100))

        # make dendritic tree appear over 100 frames
        if time >= 100 and time <= 150:
            f = time - 100
            self.set_dendritic_tree_alpha(sm(f / 50))

        # fade dendritic tree out a bit again
        if time >= 150 and time <= 250:
            f = time - 150
            self.set_dendritic_tree_alpha(1 - 0.5 * sm(f / 100))

        # if we have a focus neuron, we fade the remaining soma a bit
        if self.focus_id is not None and time >= 150 and time <= 250:
            f = time - 150
            self.set_soma_alpha(1 - 0.3 * sm(f / 100))

        # let axons grow over 550 frames
        if time == 250:
            self.set_num_visible_segments(0)
            # by passing a n_id to the set_... functions, we only alter the given id.
            # Otherwise, everything if focus_single is None
            self.set_axon_alpha(1, n_id=self.focus_id)

        if time >= 250 and time <= 600:
            f = time - 250
            self.set_num_visible_segments(int(f), n_id=self.focus_id)

        # fade axons out
        if time >= 600 and time <= 750:
            f = time - 600
            self.set_axon_alpha(1 - sm(f / 150), n_id=self.focus_id)

        # also fade out whats left of the dendrites
        if time >= 700 and time <= 775:
            f = time - 700
            self.set_dendritic_tree_alpha(0.5 - 0.5 * sm(f / 75))

        # fade in connections
        if time >= 625 and time <= 750:
            f = time - 625
            self.set_connections_alpha(sm(f / 125), n_id=self.focus_id)

        # go down to 0.05 alpha for connections, there the locality is visible
        if time >= 750 and time <= 800:
            f = time - 750
            if self.focus_id is None:
                self.set_connections_alpha(1 - 0.9 * sm(f / 50))


class SingleNeuronGrowth(object):
    """
    Illustrate the growth of a single neuron, one axon segment at a time
    not a normal renderer, just a helper for the culture one

    # Parameters
    h5f : dict like
        loaded from h5py.File
    neuron_id : int
        id of the neuron to show
    base_color : str or tuple
        color of the neuron, RGB only! Alpha are hardcoded in here
    """

    def __init__(self, h5f, neuron_id, base_color="white", ax=None):
        """ """
        self.h5f = h5f
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            self.ax = ax
        else:
            self.ax = ax

        self.base_color = base_color
        self.n_id = neuron_id
        self.pos_x = h5f["data"]["neuron_pos_x"][self.n_id]
        self.pos_y = h5f["data"]["neuron_pos_y"][self.n_id]
        self.seg_x = np.hstack(
            [[self.pos_x], h5f["data"]["neuron_axon_segments_x"][self.n_id]]
        )
        self.seg_y = np.hstack(
            [[self.pos_y], h5f["data"]["neuron_axon_segments_y"][self.n_id]]
        )
        self.segs_shown = 0

        # neuron radius, um
        # self.rad_n = 7.5
        self.rad_n = 20  # made them larger for this animation to be more visible
        self.rad_dendrite = h5f["data"]["neuron_radius_dendritic_tree"][self.n_id]

        # which connections will get created? sparse matrix: from -> to
        aij_sparse = h5f["data"]["connectivity_matrix_sparse"]
        # find the outgoing connections
        idx = np.where(aij_sparse[:, 0] == self.n_id)[0]
        self.target_ids = aij_sparse[idx, 1]
        try:
            # get the id of the axon segment that creates the connection to target neuron
            # this was added later tot he file convention and might be missing
            # because we insert an extra seg at the soma, we need to shift by 1
            self.tar_seg_ids = h5f["data"]["connectivity_segments"][idx] + 1
        except:
            self.tar_seg_ids = np.ones_like(self.target_ids, dtype="int") * (
                len(self.seg_x) - 1
            )

        # store the plot artists (lines etc) so we can modify them
        self.artists = dict()

        # ------------------------------------------------------------------------------ #
        # soma
        # ------------------------------------------------------------------------------ #

        edge_color = "white" if theme_bg == "dark" else "black"

        circle = plt.Circle((self.pos_x, self.pos_y), radius=self.rad_n, lw=0.5, zorder=4)
        circle.set_facecolor(base_color)
        circle.set_edgecolor(None)
        circle.set_linewidth(0.0)
        self.ax.add_artist(circle)
        self.artists["soma"] = circle

        # ------------------------------------------------------------------------------ #
        # dendritic tree
        # ------------------------------------------------------------------------------ #

        circle = plt.Circle(
            (self.pos_x, self.pos_y),
            radius=self.rad_dendrite,
            lw=2,
            zorder=1,
            linestyle="-",
        )
        self.dendrite_alpha_inner = 0.2
        self.dendrite_alpha_outer = 0
        circle.set_facecolor((*mcolors.to_rgb(base_color), self.dendrite_alpha_inner))
        circle.set_edgecolor((*mcolors.to_rgb(base_color), self.dendrite_alpha_outer))
        circle.set_linewidth(0.25)
        self.ax.add_artist(circle)
        self.artists["dendritic_tree"] = circle

        # ------------------------------------------------------------------------------ #
        # axon
        # ------------------------------------------------------------------------------ #

        self.artists["axon"] = self.ax.plot([], [], lw=0.5, zorder=3, color=base_color)[0]
        # to update:
        # ax.figure.canvas.draw()
        # self.artists["axon"].set_data(xdata, ydata)

        # ------------------------------------------------------------------------------ #
        # connections, soma to soma
        # ------------------------------------------------------------------------------ #
        self.artists["connections"] = []
        for i, (tar_id, tar_seg_id) in enumerate(zip(self.target_ids, self.tar_seg_ids)):
            # get the position of the target neuron
            tar_pos_x = h5f["data"]["neuron_pos_x"][tar_id]
            tar_pos_y = h5f["data"]["neuron_pos_y"][tar_id]

            # get the position of the axon segment that creates the connection
            # we might want to show a flash or something here
            seg_pos_x = self.seg_x[tar_seg_id]
            seg_pos_y = self.seg_y[tar_seg_id]

            # plot the connection
            line = self.ax.plot(
                [self.pos_x, tar_pos_x],
                [self.pos_y, tar_pos_y],
                lw=0.7,
                zorder=2,
                color=base_color,
            )[0]
            self.artists["connections"].append(line)

    def set_num_visible_segments(self, num_segs):
        """
        set the number of axon segments to show
        """
        try:
            self.seg_x[num_segs]
        except:
            num_segs = len(self.seg_x)
        self.artists["axon"].set_data(self.seg_x[0:num_segs], self.seg_y[0:num_segs])


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


def _bezier(y_beg, y_end, x_beg, x_end, n_steps=None):
    """
    poormans bezier for slow-fast-slow animations.
    create a smooth interpolation curve beteen two points.
    linspace x and cos y, then interpolate the y values
    """
    if n_steps is None:
        n_steps = x_end - x_beg + 1
    x = np.linspace(x_beg, x_end, n_steps)
    y = -0.5 * np.cos(np.linspace(0, np.pi, n_steps)) + 0.5
    y = y_beg + (y_end - y_beg) * y
    return x, y
