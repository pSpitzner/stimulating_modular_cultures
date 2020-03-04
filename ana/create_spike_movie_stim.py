# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-01-24 13:43:39
# @Last Modified: 2020-03-04 14:03:06
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter

plt.ioff()

# ------------------------------------------------------------------------------- #
# arguments. note that argparse does not work with ipython
# ------------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(description="Create a Movie from Spiketrains")
parser.add_argument("-i", dest="input_path", help="input path", metavar="FILE")
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
parser.add_argument("-t", "--title", dest="title", help="movie title")
parser.add_argument(
    "-fps", dest="fps", help="frames per second for the movie", type=int, default=30
)
parser.add_argument(
    "-l",
    "--length",
    dest="length",
    help="desired length of the movie, in seconds",
    type=int,
    default=10,
)
parser.add_argument(
    "-tunit",
    "--tunit",
    dest="tunit",
    help="time unit of the ('--rescale'd) bins in the eventlist. only used for displaying the time index in the movie. default: 'sec'",
    type=str,
    default="sec",
)
parser.add_argument(
    "-r",
    "--rescale",
    dest="rescale",
    help="multiply eventlist by a factor e.g. to convert from s to ms, or to create a timelapse",
    type=float,
    default=1.0,
)
parser.add_argument(
    "-tmin",
    "--tmin",
    dest="tmin",
    help="rendering starts at this time point in the eventlist. in units of the ('--rescale'd) eventlist",
    type=float,
    default=0,
)
parser.add_argument(
    "-tmax",
    "--tmax",
    dest="tmax",
    help="rendering ends at this time point in the eventlist. in units of the ('--rescale'd) eventlist",
    type=float,
    default=1000,
)

args = parser.parse_args()
if args.input_path == None or args.output_path == None:
    print("use correct arguments: -i, -o, -t need help? -h")
    exit()
if args.title == None:
    args.title = args.input_path


fps = args.fps  # frames per second shown in the movie
num_frames = int(np.floor(args.length * fps))

rescale = args.rescale  # factor by which to rescale the event list

# how many time bins of the eventlist to include in each rendered frame
bpf = (args.tmax - args.tmin) / num_frames

frame_offset = int(args.tmin / bpf)
decay_s = 0.75  # decay of spike display in seconds
decay_b = bpf * fps * decay_s  # in time bins

stim_t_threshold = decay_b*1.5  # do not show normal spike this short after stimulus

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #


def h5_load(filename, dsetname, raise_ex=True):
    try:
        file = h5py.File(filename, "r")
        res = file[dsetname][:]
        file.close()
        return res
    except Exception as e:
        print(f"failed to load {dsetname} from {filename}")
        if raise_ex:
            raise e


def rgba_to_rgb(c, bg="white"):
    bg = mcolors.to_rgb(bg)
    alpha = c[-1]

    res = (
        (1 - alpha) * bg[0] + alpha * c[0],
        (1 - alpha) * bg[1] + alpha * c[1],
        (1 - alpha) * bg[2] + alpha * c[2],
    )
    return res


# ------------------------------------------------------------------ #
# load data
# ------------------------------------------------------------------ #

rad_n = 7.5  # soma radius
num_n = int(h5_load(args.input_path, "/meta/topology_num_neur"))  # number of neurons
pos_x = h5_load(args.input_path, "/data/neuron_pos_x")  # soma centers
pos_y = h5_load(args.input_path, "/data/neuron_pos_y")
seg_x = h5_load(args.input_path, "/data/neuron_axon_segments_x")  # 2d array
seg_y = h5_load(args.input_path, "/data/neuron_axon_segments_y")
seg_x = np.where(seg_x == 0, np.nan, seg_x)  # overwrite padding 0 at the end
seg_y = np.where(seg_y == 0, np.nan, seg_y)

# assuming integer spiketimes to check if they match frames
event_list = h5_load(args.input_path, "/data/spiketimes")
event_list = event_list * rescale
event_list = np.where(event_list == 0, np.nan, event_list)

stim_list = h5_load(args.input_path, "/data/stimulationtimes")
stim_list = stim_list * rescale
stim_list = np.where(stim_list == 0, np.nan, stim_list)

# ------------------------------------------------------------------------------ #
# figure setup and color choices
# ------------------------------------------------------------------------------ #

canvas_clr = "black"  # the overall background
# canvas_clr='white'
title_clr = "white"

# needs to return a color that will be applied to the foreground of axons
def render_spike_axon(time_ago, decay_time, basecolor="white"):
    alpha = 1.0 - (time_ago) / decay_time  # linear
    # alpha = 1.0 / (time_ago) # 1/x, pretty steep
    alpha = np.clip(alpha, 0.0, 1.0)
    rgb = mcolors.to_rgb(basecolor)
    return (*rgb, alpha * 0.6)


# needs to return a two colors, for cell body and outline
def render_spike_soma(time_ago, decay_time, basecolor="white"):
    alpha = 1.0 - (time_ago) / decay_time  # linear
    alpha = np.clip(alpha, 0.0, 1.0)
    rgb = mcolors.to_rgb(basecolor)
    # edge = (1,1,1, alpha * 0.8)
    edge = (*rgb, alpha * 0.8)
    face = (*rgb, alpha)
    return edge, face


# default colors, spikes will drawn over this. its nice to have them solid, cast alpha
axon_edge = rgba_to_rgb((1.0, 1.0, 1.0, 0.1), canvas_clr)
soma_edge = rgba_to_rgb((1.0, 1.0, 1.0, 0.1), canvas_clr)
soma_face = rgba_to_rgb((1.0, 1.0, 1.0, 0.1), canvas_clr)

fig, ax = plt.subplots(figsize=[6.4, 6.4])
ax.set_title(f"{args.title}", fontsize=16, color=title_clr)
fig.patch.set_facecolor(canvas_clr)
ax.set_facecolor(canvas_clr)
plt.axis("off")
ax.set_aspect(1)
art_time = fig.text(0.02, 0.95, "current time", fontsize=14, color=title_clr)

art_axons = []
for i in range(len(seg_x)):
    # background
    tmp = ax.plot(seg_x[i], seg_y[i], color=axon_edge, lw=0.5, zorder=0,)
    # foreground overlay, when spiking
    tmp = ax.plot(seg_x[i], seg_y[i], color=(0, 0, 0, 0), lw=0.7, zorder=3,)
    art_axons.append(tmp[0])

art_soma = []
for i in range(len(pos_x)):
    # background
    circle = plt.Circle((pos_x[i], pos_y[i]), radius=rad_n, lw=0.5, zorder=1)
    circle.set_facecolor(soma_face)
    circle.set_edgecolor(soma_edge)
    circle.set_linewidth(0.25)
    ax.add_artist(circle)
    # foreground overlay, when spiking
    circle = plt.Circle((pos_x[i], pos_y[i]), radius=rad_n, lw=0.7, zorder=4)
    circle.set_facecolor((0, 0, 0, 0))
    circle.set_edgecolor((0, 0, 0, 0))
    circle.set_linewidth(0.25)
    ax.add_artist(circle)
    art_soma.append(circle)


# ------------------------------------------------------------------------------ #
# Movie making
# ------------------------------------------------------------------------------ #

metadata = dict(title=f"args.input_path", artist="Matplotlib", comment="Yikes! Spikes!")
writer = FFMpegWriter(fps=fps, metadata=metadata)

# keep track of each neurons last spike time to calculate current color
prev_spk = np.ones((num_n), dtype=np.int) * -10000
prev_spk_idx = np.ones((num_n), dtype=np.int) * 0  # the index in the event list
last_spk_idx = event_list.shape[1]  # so we do not run out of arrays

# same again for stimulus
prev_stm = np.ones((num_n), dtype=np.int) * -10000
prev_stm_idx = np.ones((num_n), dtype=np.int) * 0  # the index in the event list
last_stm_idx = stim_list.shape[1]  # so we do not run out of arrays

with writer.saving(fig=fig, outfile=args.output_path, dpi=100):
    print(f"Rendering {args.length:.0f} seconds with {num_frames} frames ...")
    for f in tqdm(range(frame_offset, num_frames + frame_offset)):
        time_stamp = f * bpf
        art_time.set_text(f"t = {time_stamp :.2f} {args.tunit}")

        for n in range(num_n):
            # normal spikes
            idx = prev_spk_idx[n]
            while idx + 1 < last_spk_idx and event_list[n, idx + 1] < time_stamp:
                idx = idx + 1
            spike_time = event_list[n, idx]
            prev_spk_idx[n] = idx

            # stimulation
            idx = prev_stm_idx[n]
            while idx + 1 < last_stm_idx and stim_list[n, idx + 1] < time_stamp:
                idx = idx + 1
            stim_time = stim_list[n, idx]
            prev_stm_idx[n] = idx

            spk_ago = time_stamp - spike_time
            stm_ago = time_stamp - stim_time

            # only draw the stimulation, not the spike that was also recorded
            if np.fabs(stm_ago - spk_ago) < stim_t_threshold:
                spk_ago = -1

            # paint something cool
            if spk_ago >= 0 and spk_ago < decay_b:
                ax_edge = render_spike_axon(spk_ago, decay_b)
                sm_edge, sm_face = render_spike_soma(spk_ago, decay_b)
            elif stm_ago >= 0 and stm_ago < decay_b:
                ax_edge = render_spike_axon(spk_ago, decay_b, 'orange')
                sm_edge, sm_face = render_spike_soma(spk_ago, decay_b, 'orange')
            else:
                ax_edge = (0, 0, 0, 0)
                sm_edge = (0, 0, 0, 0)
                sm_face = (0, 0, 0, 0)
            art_axons[n].set_color(ax_edge)
            art_soma[n].set_edgecolor(sm_edge)
            art_soma[n].set_facecolor(sm_face)

        writer.grab_frame(facecolor=canvas_clr)
