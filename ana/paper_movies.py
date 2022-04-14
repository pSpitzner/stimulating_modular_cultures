import numpy as np
import plot_helper as ph
import ana_helper as ah
import seaborn as sns

# import those last, so that the matpllotlb rc params are not overwritten
# they are tweaked for movies, dark background, agg backened, etc
from movie_business import matplotlib, plt
from movie_business import MovieWriter
from movie_business import FadingLineRenderer, MovingWindowRenderer, TextRenderer
from movie_business import TopologyRenderer


def main():
    input_path = "./dat/the_last_one/dyn/highres_stim=off_k=1_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_rep=001.hdf5"
    movie_from_simulation(input_path)


def movie_from_simulation(
    input_path,
    output_path="./mov/simulation_test.mp4",
    movie_duration=30,
    window_from=-20,
    window_to=100,
    tbeg=0,
    tend=360,
    only_test_layout=False,
):
    """
    Wrapper to create the layout used for movies of simulions.

    Here we do the data analysis in place, using default parameters.
    (For Experiments we use the analyzed files coming out of process_conditions.)

    # Parameters
    input_path: str
        Path to the HDF5 file containing the simulation data.
    output_path: str
        Path to the output movie file.
    movie_duration: float
        Duration of the movie in seconds.
    window_from, to: float
        Start, end of the time window showing the data, in data units.
    tbeg, tend: float
        Start, end time of the data, in data units.
    only_test_layout: bool
        Use this to test the layout without actually creating a movie.
    """

    # keep a list of renderers that will be updated in the writers render loop.
    # every renderer needs to have a function `set_time` that takes the experimental
    # timestamp as input.
    writer = MovieWriter(
        output_path=output_path,
        movie_duration=movie_duration,
        tbeg=tbeg,
        tend=tend,
    )

    # I like to set the decay time propto movie playback speed
    time_ratio = writer.movie_duration / (writer.tend - writer.tbeg)
    decay_time = 0.5 / time_ratio

    # ------------------------------------------------------------------------------ #
    # Figure layout
    # ------------------------------------------------------------------------------ #

    fig = plt.figure(figsize=(1920 / 300, 800 / 300), dpi=300)
    fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        # use one dummy column for the axis ticks
        width_ratios=[0.4, 0.6 * 0.333, 0.07, 0.6 * 0.666],
        height_ratios=[0.6, 0.4],
        wspace=0.02,
        hspace=0.2,
        left=0.01,
        right=0.99,
        top=0.95,
        bottom=0.15,
    )

    # ------------------------------------------------------------------------------ #
    # Topology plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[:, 0])
    tpr = TopologyRenderer(input_path=input_path, ax=ax)
    tpr.decay_time = decay_time

    ax.set_xlim(-40, 640)
    ax.set_ylim(-40, 640)
    if "k=-1" in input_path:
        # align merged guys differently
        ax.set_xlim(-140, 540)
        ax.set_ylim(-140, 540)
    ax.set_clip_on(False)
    writer.renderers.append(tpr)

    # ------------------------------------------------------------------------------ #
    # population activity plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[1, 3])
    h5f = ah.prepare_file(input_path)
    ah.find_rates(h5f)

    ph.plot_module_rates(h5f=h5f, ax=ax, alpha=0.5)
    ph.plot_system_rate(h5f=h5f, ax=ax, color="white")
    ax.get_legend().set_visible(False)
    ax.set_ylim(-15, 150)
    sns.despine(ax=ax, right=True, top=True, bottom=True, left=True, offset=1)
    # ax.yaxis.set_visible(False)

    ax.set_xlabel("Time (seconds)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(30.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel("")
    # ax.set_ylabel("Rates (Hz)")
    # ax.yaxis.labelpad = 10
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.tick_params(left=False, labelleft=False, bottom=False)

    writer.renderers.append(
        MovingWindowRenderer(
            ax=ax,
            tbeg=writer.tbeg,
            tend=writer.tend,
            # this plot only has 2/3 the width of the raster
            window_from=window_from,
            window_to=window_from + (window_to - window_from),
        )
    )

    # ------------------------------------------------------------------------------ #
    # Resource cycles
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[1, 1])

    if not only_test_layout:
        # speed things up by not doing the analysis
        ah.find_module_level_adaptation(h5f)
    x_sets = []
    y_sets = []
    colors = []
    for mod_id in [0, 1, 2, 3]:
        y = h5f[f"ana.rates.module_level.mod_{mod_id}"]
        dt = h5f[f"ana.rates.dt"]
        try:
            x = h5f[f"ana.adaptation.module_level.mod_{mod_id}"]
            assert (
                h5f[f"ana.adaptation.dt"] == dt
            ), "adaptation and rates need to share the same time step"
        except:
            x = np.ones_like(y) * np.nan
        # limit data range to whats needed
        x = x[0 : int(writer.tend / dt) + 2]
        y = y[0 : int(writer.tend / dt) + 2]
        x_sets.append(x)
        y_sets.append(y)
        colors.append(f"C{mod_id}")

    flr = FadingLineRenderer(
        x=x_sets,
        y=y_sets,
        colors=colors,
        dt=dt,
        ax=ax,
        tbeg=writer.tbeg,
        clip_on=False,
    )
    flr.decay_time = decay_time

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(150))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))
    ax.set_ylim(-15, 150)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position("both")
    sns.despine(ax=ax, left=True, right=False, top=True, trim=True, offset=1)
    # workaround for despine
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))

    ax.set_xlabel("Resources")
    ax.set_ylabel("")
    # ax.yaxis.set_visible(False)

    writer.renderers.append(flr)

    # ------------------------------------------------------------------------------ #
    # raster plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[0, 3])

    ph.plot_raster(h5f=h5f, ax=ax)
    sns.despine(ax=ax, right=True, top=True, left=True, bottom=True)
    # ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.set_ylabel("Raster")
    # ax.yaxis.labelpad = 10
    ax.tick_params(left=False, labelleft=False)

    writer.renderers.append(
        MovingWindowRenderer(
            ax=ax,
            tbeg=writer.tbeg,
            tend=writer.tend,
            window_from=window_from,
            window_to=window_to,
        )
    )

    # ------------------------------------------------------------------------------ #
    # Finalize and movie loop
    # ------------------------------------------------------------------------------ #

    fig.text(0.61, 0.7, "Raster", fontsize=8, ha="right")
    fig.text(0.61, 0.5, "Rates (Hz)", fontsize=8, ha="right")

    writer.renderers.append(TextRenderer(fig.text(0.45, 0.9, "Time", fontsize=8)))

    fig.tight_layout()

    if only_test_layout:
        for r in writer.renderers:
            r.set_time(tbeg + (tend - tbeg) / 2)
    else:
        writer.render()


def movie_from_experiment(
    input_path,
    output_path="./mov/simulation_test.mp4",
    movie_duration=30,
    window_from=-20,
    window_to=100,
    tbeg=0,
    tend=360,
    only_test_layout=False,
):
    """
    Pretty much a copy of `movie_from_simulation`, but we cannot draw
    the resource cycles, and the layout is adapted.
    """

    writer = MovieWriter(
        output_path=output_path,
        movie_duration=movie_duration,
        tbeg=tbeg,
        tend=tend,
    )

    time_ratio = writer.movie_duration / (writer.tend - writer.tbeg)
    decay_time = 0.5 / time_ratio

    # ------------------------------------------------------------------------------ #
    # Figure layout
    # ------------------------------------------------------------------------------ #

    fig = plt.figure(figsize=(1920 / 300, 800 / 300), dpi=300)
    fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        # use one dummy column for the axis ticks
        width_ratios=[0.4, 0.6 * 0.333, 0.07, 0.6 * 0.666],
        height_ratios=[0.6, 0.4],
        wspace=0.02,
        hspace=0.2,
        left=0.01,
        right=0.99,
        top=0.95,
        bottom=0.15,
    )

    # ------------------------------------------------------------------------------ #
    # Topology plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[:, 0])
    tpr = TopologyRenderer(input_path=input_path, ax=ax)
    tpr.decay_time = decay_time

    ax.set_ylim(0, 600)
    ax.set_xlim(0, 600)
    ax.set_clip_on(False)
    writer.renderers.append(tpr)

    # ------------------------------------------------------------------------------ #
    # population activity plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[1, 3])
    h5f = ah.prepare_file(input_path)
    ah.find_rates(h5f)

    ph.plot_module_rates(h5f=h5f, ax=ax, alpha=0.5)
    ph.plot_system_rate(h5f=h5f, ax=ax, color="white")
    ax.get_legend().set_visible(False)
    ax.set_ylim(-15, 150)
    sns.despine(ax=ax, right=True, top=True, bottom=True, left=True, offset=1)

    ax.set_xlabel("Time (seconds)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(30.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel("")
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.tick_params(left=False, labelleft=False, bottom=False)

    writer.renderers.append(
        MovingWindowRenderer(
            ax=ax,
            tbeg=writer.tbeg,
            tend=writer.tend,
            # this plot only has 2/3 the width of the raster
            window_from=window_from,
            window_to=window_from + (window_to - window_from),
        )
    )

    # ------------------------------------------------------------------------------ #
    # raster plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[0, 3])

    ph.plot_raster(h5f=h5f, ax=ax)
    sns.despine(ax=ax, right=True, top=True, left=True, bottom=True)
    ax.xaxis.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, labelleft=False)

    writer.renderers.append(
        MovingWindowRenderer(
            ax=ax,
            tbeg=writer.tbeg,
            tend=writer.tend,
            window_from=window_from,
            window_to=window_to,
        )
    )

    # ------------------------------------------------------------------------------ #
    # Finalize and movie loop
    # ------------------------------------------------------------------------------ #

    fig.text(0.61, 0.7, "Raster", fontsize=8, ha="right")
    fig.text(0.61, 0.5, "Rates (Hz)", fontsize=8, ha="right")

    writer.renderers.append(TextRenderer(fig.text(0.45, 0.9, "Time", fontsize=8)))

    fig.tight_layout()

    if only_test_layout:
        for r in writer.renderers:
            r.set_time(tbeg + (tend - tbeg) / 2)
    else:
        writer.render()


if __name__ == "__main__":
    main()
