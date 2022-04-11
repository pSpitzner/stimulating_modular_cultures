# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-01-24 13:43:39
# @Last Modified: 2022-04-11 15:31:46
# ------------------------------------------------------------------------------- #
# Create a movie of the network for a given time range and visualize
# firing neurons. Save to mp4.
# This version includes the visualization of stimuli.
#
# conda install h5py matplotlib ffmpeg tqdm
# ------------------------------------------------------------------------------- #

import os
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter

plt.ioff()
plt.style.use("dark_background")


def main(args):

    metadata = dict(title=f"{args.title}", artist="Matplotlib", comment="Yikes! Spikes!")
    writer = FFMpegWriter(fps=args.fps, metadata=metadata)
    num_frames = (args.fps * args.length) + 1

    # 1800x900 px at 300 dpi, where fonts are decent
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

    global topo
    topo = TopologyRenderer(input_path=args.input_path, ax=axes[0])

    # its helpful to set the decay time propto movie playback speed
    time_ratio = args.length / (args.tend - args.tbeg)
    topo.decay_time = 0.5 / time_ratio

    # ------------------------------------------------------------------------------ #
    # population activity plot
    # ------------------------------------------------------------------------------ #

    ax = axes[1]
    h5f = ah.prepare_file(args.input_path)
    ah.find_rates(h5f)

    ph.plot_system_rate(h5f=h5f, ax=ax, color="white")

    ax.get_legend().set_visible(False)
    # window = 5
    # ax.set_xlim(-window/2, window/2)
    ax.set_xlim(args.tbeg, args.tend)
    reference = ax.axvline(0, ls=":")
    sns.despine(ax=ax, right=True, top=True, offset=5)

    # ------------------------------------------------------------------------------ #
    # movie loop
    # ------------------------------------------------------------------------------ #

    with writer.saving(fig=topo.fig, outfile=args.output_path, dpi=300):
        print(f"Rendering {args.length:.0f} seconds with {num_frames} frames ...")

        for fdx in tqdm(range(num_frames)):
            exp_time = frame_index_to_experimental_time(
                this_frame=fdx,
                total_frames=num_frames,
                exp_start=args.tbeg,
                exp_end=args.tend,
            )
            topo.update_to_time(exp_time)
            # ax.set_xlim(exp_time - window/2, exp_time + window/2)
            reference.set_xdata(exp_time)

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
        dest="length",
        help="desired length of the movie, in seconds",
        type=int,
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

    def update_to_time(self, time):
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

        # rgba as 4-tuple, between 0 and 1
        # make soma a bit brighter than axons
        ax_edge = (*mcolors.to_rgb(self.neuron_clr), total_alpha * 0.6)
        sm_edge = (*mcolors.to_rgb(self.neuron_clr), total_alpha * 0.8)
        sm_face = (*mcolors.to_rgb(self.neuron_clr), total_alpha * 1.0)

        # update the actual foreground layer
        self.art_axons[n_id].set_color(ax_edge)
        self.art_soma[n_id].set_edgecolor(sm_edge)
        self.art_soma[n_id].set_facecolor(sm_face)


class FadingLineRenderer(object):
    """
    Think of an oldschool osci, where the electron beam draws the line and fades,
    draw line transparency as exponential

    Create plot lines as line segments so that styles can be updated later
    c.f. https://github.com/dpsanders/matplotlib-examples/blob/master/colorline.py
    """

    def __init__(self, x, y, ax=None, dt=1, tbeg=0, **kwargs):
        """
        # Parameters:
        x, y : 1d arrays of the data to plot,
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

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        kwargs = kwargs.copy()
        # this cmap is white at 1 and transparent at 0
        kwargs.setdefault("cmap", cc.cmap["alpha_white"])
        if isinstance(kwargs["cmap"], str):
            self.cmap = plt.get_cmap(kwargs["cmap"])
        else:
            self.cmap = kwargs["cmap"]

        self.num_timesteps = len(x)

        self.lc = LineCollection(segments, array=np.ones_like(x, dtype="float"), **kwargs)
        # now, the lc array is a helper array of same length as data, to set the color
        # along the line, e.g. with floats between 0 and 1, that is mapped using colormap
        self.ax.add_collection(self.lc)

        # casting experimental time to the data array
        self.dt = dt
        self.tbeg = tbeg

        # fade transparency,
        self.decay_time = 15.0  # exponential decay time in experimental time units
        # after around 3 decay times we are sufficiently close to zero
        self.decay_bins = int(3 * self.decay_time / self.dt)

        t = np.arange(0, self.decay_bins) * -self.dt
        mask = np.exp(-t / self.decay_time)
        # make sure last entry is zero
        m_min = np.nanmin(mask)
        m_max = np.nanmax(mask)
        m_diff = m_max - m_min
        self.decay_mask = (mask - m_min) / m_diff

    def set_array(self, z):
        self.lc.set_array(z)

        # unfortunately, setting the lc array does not directly
        # refresh the colors of the line collection.
        self.lc.set_colors([self.cmap(f) for f in z])

    def set_time(self, time):
        time_index = int((time - self.tbeg) / self.dt)
        first_visible = time_index - self.decay_bins
        if time_index < 0:
            time_index = 0
        if first_visible < 0:
            first_visible = 0

        num_bins = time_index - first_visible
        new_z = np.zeros(self.num_timesteps)
        if num_bins > 0:
            new_z[time_index-num_bins:time_index] = self.decay_mask[-num_bins:]

        self.set_array(new_z)


if __name__ == "__main__":
    args = parse_args()
    main(args)
