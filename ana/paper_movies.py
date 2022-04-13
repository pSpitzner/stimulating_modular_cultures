# use matplotlib with args that help movies, dark background, agg backened, etc
from movie_business import matplotlib, plt
from movie_business import MovieWriter, FadingLineRenderer, MovingWindowRenderer
from movie_business import TopologyRenderer
import plot_helper as ph
import ana_helper as ah
import seaborn as sns


def main():

    input_path = "./dat/the_last_one/dyn/highres_stim=off_k=1_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_rep=001.hdf5"

    # keep a list of renderer objects that will be updated in the main loop.
    # everyone needs to have a function `set_time` that takes the experimental
    # timestamp as input.
    writer = MovieWriter(
        output_path="./mov/lorem.mp4",
        movie_duration=30,
        tbeg=0,
        tend=360,
    )

    # for some plots, it is helpful to set the decay time propto movie playback speed
    time_ratio = writer.movie_duration / (writer.tend - writer.tbeg)
    decay_time = 0.5 / time_ratio

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
    tpr = TopologyRenderer(input_path=input_path, ax=ax)
    tpr.decay_time = decay_time
    writer.renderers.append(tpr)

    # ------------------------------------------------------------------------------ #
    # population activity plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[1, 1])
    h5f = ah.prepare_file(input_path)
    ah.find_rates(h5f)

    ph.plot_system_rate(h5f=h5f, ax=ax, color="white")
    ax.get_legend().set_visible(False)
    sns.despine(ax=ax, right=True, top=True, offset=5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Rate (Hz)")
    sns.despine(ax=ax, bottom=True)
    # cc.detick(axis=ax.xaxis, keep_labels=True)

    writer.renderers.append(
        MovingWindowRenderer(
            ax=ax,
            data_beg=writer.tbeg,
            data_end=writer.tend,
            # use a 120 second sliding window
            window_from=-20.0,
            window_to=100,
        )
    )

    # ------------------------------------------------------------------------------ #
    # raster plot
    # ------------------------------------------------------------------------------ #

    ax = fig.add_subplot(gs[0, 1])

    ph.plot_raster(h5f=h5f, ax=ax)
    sns.despine(ax=ax, right=True, top=True, left=True, bottom=True)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_xlabel("")

    writer.renderers.append(
        MovingWindowRenderer(
            ax=ax,
            data_beg=writer.tbeg,
            data_end=writer.tend,
            window_from=-20.0,
            window_to=100,
        )
    )

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
        x = x[0 : int(writer.tend / dt) + 2]
        y = y[0 : int(writer.tend / dt) + 2]
        x_sets.append(x)
        y_sets.append(y)
        colors.append(f"C{mod_id}")

    flr = FadingLineRenderer(
        x=x_sets, y=y_sets, colors=colors, dt=dt, ax=ax, tbeg=writer.tbeg
    )
    flr.decay_time = decay_time

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))
    ax.set_ylim(-15, 145)
    sns.despine(ax=ax, right=True, top=True, trim=True, offset=1)
    ax.set_xlabel("Resources")
    ax.set_ylabel("Rates (Hz)")

    writer.renderers.append(flr)

    # ------------------------------------------------------------------------------ #
    # Finalize and movie loop
    # ------------------------------------------------------------------------------ #

    fig.tight_layout()

    writer.render()


if __name__ == "__main__":
    main()
