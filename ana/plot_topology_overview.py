# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-01-24 14:13:56
# @Last Modified: 2020-07-16 12:19:02
# ------------------------------------------------------------------ #
# script that takes a hdf5 as produced by my cpp simulation
# as first argument and visualizes the topological features
#
# conda install matplotlib h5py seaborn networkx
# ------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import matplotlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import networkx as nx

# interactive plotting
plt.ion()

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
        else:
            return np.nan


# https://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker
def circles(x, y, s, c="b", vmin=None, vmax=None, ax=None, **kwargs):
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


parser = argparse.ArgumentParser(description="Topology Overview")
parser.add_argument("-i", dest="input_path", help="input path", metavar="FILE")
parser.add_argument("-o", dest="output_path", help="output path", metavar="FILE")
args = parser.parse_args()
file = args.input_path

# ------------------------------------------------------------------------------ #
# setup merged figure
# ------------------------------------------------------------------------------ #

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[12, 8])

# ------------------------------------------------------------------ #
# distributions
# ------------------------------------------------------------------ #

print("plotting distributions")

# dendritic tree radius
ax = axes[0, 0]
try:
    n_R_d = h5_load(file, "/data/neuron_radius_dendritic_tree")
    # fig, ax = plt.subplots()
    sns.distplot(n_R_d, ax=ax, label="radius", hist=True, kde_kws={"alpha": 1})
    ax.set_xlabel(f"Distance $l\,[\mu m]$")
    ax.set_ylabel(f"Probability $p(l)$")
    ax.set_title("Dendritic tree size distribution")
    ax.legend()
    # axes.append(ax)
    # fig.savefig(f"./fig/degree_distribution.pdf", transparent=True, pad_inches=0.0)
except Exception as e:
    print("plotting dendritic tree size distribution failed")


# neuron positions
ax = axes[0, 1]
try:
    n_x = h5_load(file, "/data/neuron_pos_x")
    n_y = h5_load(file, "/data/neuron_pos_y")
    # fig, ax = plt.subplots()
    sns.distplot(n_x, ax=ax, label="x", hist=True, kde_kws={"alpha": 1})
    sns.distplot(n_y, ax=ax, label="y", hist=True, kde_kws={"alpha": 1})
    ax.set_xlabel(f"Position $l\,[\mu m]$")
    ax.set_ylabel(f"Probability $p(l)$")
    ax.set_title("Neuron position distribution")
    ax.legend()
    # axes.append(ax)
    # fig.savefig(f"./fig/degree_distribution.pdf", transparent=True, pad_inches=0.0)
except Exception as e:
    print("plotting neuron position distribution failed")


# in and out degree
ax = axes[1, 1]
try:
    k_in = h5_load(file, "/data/neuron_k_in")
    k_out = h5_load(file, "/data/neuron_k_out")
    # fig, ax = plt.subplots()
    maxbin = np.nanmax([k_in, k_out])
    bins = np.arange(0, maxbin, 1)
    sns.distplot(
        k_in,
        ax=ax,
        label=r"$k_{in}$",
        bins=bins,
        hist=True,
        norm_hist=True,
        kde_kws={"alpha": 1},
    )
    sns.distplot(
        k_out,
        ax=ax,
        label=r"$k_{out}$",
        bins=bins,
        hist=True,
        norm_hist=True,
        kde_kws={"alpha": 1},
    )
    ax.set_xlabel(f"Degree $k$")
    ax.set_ylabel(f"Probability $p(k)$")
    ax.set_title("Degree distribution")
    ax.legend()
    # axes.append(ax)
    # fig.savefig(f"./fig/degree_distribution.pdf", transparent=True, pad_inches=0.0)
except Exception as e:
    print("plotting in/out-degree distribution failed")

# axon length distribution
ax = axes[1,0]
try:
    k_len = h5_load(file, "/data/neuron_axon_length")
    k_ee = h5_load(file, "/data/neuron_axon_end_to_end_distance")
    # fig, ax = plt.subplots()
    sns.distplot(k_len, ax=ax, label="length", hist=True, kde_kws={"alpha": 1})
    sns.distplot(k_ee, ax=ax, label="end to end", hist=True, kde_kws={"alpha": 1})
    ax.set_xlabel(f"Distance $l\,[\mu m]$")
    ax.set_ylabel(f"Probability $p(l)$")
    ax.set_title("Axon length distribution")
    ax.legend()
    # axes.append(ax)
    # fig.savefig(f"./fig/degree_distribution.pdf", transparent=True, pad_inches=0.0)
except Exception as e:
    print("plotting axon length distribution failed")


# ------------------------------------------------------------------ #
# load network map data
# ------------------------------------------------------------------ #

print("creating network map")

# neuron positions
try:
    n_x
    n_y
except NameError:
    try:
        n_x = h5_load(file, "/data/neuron_pos_x")
        n_y = h5_load(file, "/data/neuron_pos_y")
    except:
        print("failed to load neuron positions for network map")

# dendritic tree
try:
    n_R_s = 7.5
    n_R_d
except NameError:
    try:
        n_R_d = h5_load(file, "/data/neuron_radius_dendritic_tree")
    except:
        print("failed to load dendritic radius")

# axons
try:
    axon_segments_x = h5_load(file, "/data/neuron_axon_segments_x")
    axon_segments_y = h5_load(file, "/data/neuron_axon_segments_y")
    # no nans in hdf5, 0 is the default padding
    axon_segments_x = np.where(axon_segments_x == 0, np.nan, axon_segments_x)
    axon_segments_y = np.where(axon_segments_y == 0, np.nan, axon_segments_y)
except:
    print("failed to load axon segments")

# connectivity matrix
try:
    a_ij = h5_load(file, "/data/connectivity_matrix")
except:
    print("failed to load connectivity matrix")

# fraction of inter-module connections
try:
    mod_ids = h5_load(file, "/data/neuron_module_id")
    intra = 0
    inter = 0
    for i in range(0, a_ij.shape[0]):
        outgoing = np.where(a_ij[i,:] == 1)[0]
        for j in outgoing:
            if mod_ids[j] == mod_ids[i]:
                intra += 1
            else:
                inter += 1

    assert intra + inter == np.sum(a_ij == 1)
    print(f"Connections between modules: {inter}")
    print(f"Connections within modules:  {intra}")
    print(f"Ratio:                       {inter/intra*100 :.2f}%")
except:
    pass

# using networkx for connectivity plotting
G = nx.DiGraph()
pos = {}

# ------------------------------------------------------------------ #
# network map helper
# ------------------------------------------------------------------ #


def plot_axons(ax, **kwargs):
    try:
        # for i in range(1000):
        for i in range(len(axon_segments_x)):
            ax.plot(
                axon_segments_x[i],
                axon_segments_y[i],
                color="black",
                lw=0.25,
                zorder=0,
                alpha=0.3,
            )
    except:
        print("failed to plot axons")


def plot_soma(ax):
    try:
        circles(
            n_x, n_y, n_R_s, ax=ax, fc="white", ec="none", alpha=1, lw=0.5, zorder=4
        )
        circles(
            n_x, n_y, n_R_s, ax=ax, fc="none", ec="black", alpha=0.3, lw=0.5, zorder=5
        )
        # circles(n_x, n_y, n_R_d, ax=ax, fc='gray',  ec='none', alpha=0.001,
        #     zorder=-5)
    except:
        print("failed to plot soma")


def plot_graph_nodes(ax, with_labels=False):

    try:
        G.add_nodes_from(np.arange(len(n_x)))
        for i in range(len(n_x)):
            pos[i] = (n_x[i], n_y[i])
            G.nodes[i]["pos"] = (n_x[i], n_y[i])
    except:
        print("failed to plot connectivity nodes")

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        node_size=0,
        node_color="black",
        edge_color="gray",
        with_labels=with_labels,
    )


def plot_graph_edges(ax, tar=np.arange(len(n_x))):
    try:
        for n in tar:
            connected = np.where(a_ij[n] == 1)
            for c in connected[0]:
                G.add_edge(n, c, weight=1)
    except:
        print("failed to plot connectivity edges")

    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color="black", arrows=False, width=0.1
    )


color = 0
n = -1


def highlight_single(ax=ax, stay=False):
    try:
        global color
        global n
        if not stay:
            color = (color + 1) % 10
            n = n + 1
        ax.plot(
            axon_segments_x[n], axon_segments_y[n], color=f"C{color}", lw=0.9, zorder=10
        )
        circles(
            n_x[n],
            n_y[n],
            n_R_s,
            ax=ax,
            ls="-",
            lw=0.9,
            alpha=1.0,
            zorder=15,
            fc="white",
            ec=f"C{color}",
        )
        circles(
            n_x[n],
            n_y[n],
            n_R_d[n],
            ax=ax,
            ls="--",
            lw=0.9,
            alpha=0.9,
            zorder=7,
            fc="none",
            ec=f"C{color}",
        )
        circles(
            n_x[n],
            n_y[n],
            n_R_d[n],
            ax=ax,
            ls="--",
            lw=0.0,
            alpha=0.1,
            zorder=6,
            fc=f"C{color}",
            ec="none",
        )
    except:
        print("failed to highlight single")


# ------------------------------------------------------------------ #
# plotting
# ------------------------------------------------------------------ #

# fig, ax = plt.subplots(figsize=[6.4, 6.4])
ax = axes[0,2]
ax.set_title("Connectivity")
# plt.axis("off")
ax.set_aspect(1)
plot_soma(ax)
plot_graph_nodes(ax)
plot_graph_edges(ax)
# axes.append(ax)

# share the zooming
ax.get_shared_x_axes().join(ax, axes[1,2])
ax.get_shared_y_axes().join(ax, axes[1,2])

ax = axes[1,2]
ax.set_title("Axons")
plot_soma(ax)
plot_axons(ax)
ax.set_aspect(1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_aspect(1)
ax.autoscale()
ax.set_xlabel(f"Position $l\,[\mu m]$")

fig.tight_layout()

plt.show()
