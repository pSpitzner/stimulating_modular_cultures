# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-18 19:38:37
# @Last Modified: 2021-02-22 17:55:20
# ------------------------------------------------------------------------------ #
# Helper functions for dealing with colors
# ------------------------------------------------------------------------------ #


from matplotlib.colors import LinearSegmentedColormap as _ls
from matplotlib.colors import to_hex, to_rgb, to_rgba, Normalize
from matplotlib.patches import Rectangle as _Rectangle
from matplotlib.colorbar import ColorbarBase as _ColorbarBase
import matplotlib.pyplot as plt
import numpy as np

palettes = dict()
# good with edge = False
palettes["cold"] = [
    (0, "0.92"),  # single float as a string for grayscale
    (0.25, "#DCEDC8"),
    (0.45, "#42B3D5"),
    (0.75, "#1A237E"),
    (1, "black"),
]
palettes["hot"] = [
    (0, "0.95"),
    (0.3, "#FEEB65"),
    (0.65, "#E4521B"),
    (0.85, "#4D342F"),
    (1, "black"),
]
palettes["pinks"] = [
    (0, "0.95"),
    (0.2, "#E0CB8F"),
    # (0.2, "#FFECB3"),
    (0.45, "#E85285"),
    (0.65, "#6A1B9A"),
    (1, "black"),
]

# good with edge = True
palettes["volcano"] = [
    (0, "#E8E89C"),
    (0.25, "#D29C65"),
    (0.65, "#922C40"),
    (1, "#06102E"),
]

palettes["pastel_1"] = [
    (0, "#E7E7B6"),
    (0.25, "#ffad7e"),
    (0.5, "#cd6772"),
    (0.75, "#195571"),
    (1, "#011A39"),
]

palettes["reds"] = [
    (0, "#d26761"),
    (0.25, "#933a3e"),
    (0.5, "#6b354c"),
    (0.75, "#411c2f"),
    (1, "#050412"),
]
palettes["blues"] = [
    (0, "#bbc2d2"),
    (0.25, "#2865a6"),
    (0.5, "#11395d"),
    (0.75, "#091d35"),
    (1, "#030200"),
]

# hm not super color blind friendly
palettes["bambus"] = [
    (0, "#D9DFD3"),
    (0.25, "#8FA96D"),
    (0.5, "#9C794F"),
    (1, "#3F2301"),
]


# enable the colormaps for matplotlib cmaps and getting discrete values, eg
# cmap["pinks"](0.5)
cmap = dict()
for key in palettes.keys():
    cmap[key] = _ls.from_list(key, palettes[key], N=512)


def cmap_cycle(palette="hot", N=5, edge=True):
    if palette not in palettes.keys():
        raise KeyError(f"Unrecognized palette '{palette}'")

    assert N >= 1

    res = []
    for idx in range(0, N):
        if N == 1:
            arg = 0.5
        else:
            if edge:
                arg = idx / (N - 1)
            else:
                arg = (idx + 1) / (N + 1)
        res.append(to_hex(cmap[palette](arg)).upper())

    return res


def demo_cmap(palette="hot", Nmax=7, edge=True):

    dpi = 72
    cell_width = 120
    cell_height = 28
    swatch_width = 117
    swatch_height = 25
    margin = 12
    topmargin = 40
    cbar_width = 50

    ncols = Nmax
    nrows = Nmax
    width = cell_width * ncols + 2 * margin + cbar_width
    height = cell_height * nrows + margin + topmargin

    fig, axes = plt.subplots(
        ncols=2,
        gridspec_kw={"width_ratios": [width - cbar_width, cbar_width]},
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
    )
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - topmargin) / height,
    )
    ax = axes[0]
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(
        f"palette: {palette}", fontsize=24, fontweight="bold", loc="center", pad=10
    )

    print(f"palette: {palette}")
    for N in range(1, Nmax + 1):
        # columns
        colors = cmap_cycle(palette, N, edge)
        print(f"N = {N}: {' '.join(colors)}")
        for n in range(1, N + 1):
            # rows

            swatch_clr = colors[n - 1]
            # get a text color that is gray but brightness invert to shown patch
            r, g, b = to_rgb(swatch_clr)
            gray = 0.2989 * (1 - r) + 0.5870 * (1 - g) + 0.1140 * (1 - b)
            if gray < 0.5:
                gray = "black"
            else:
                gray = "white"

            col = N - 1
            row = Nmax / 2 - n + N / 2

            swatch_start_x = col * cell_width
            swatch_start_y = row * cell_height
            text_pos_x = swatch_start_x + cell_width / 2
            text_pos_y = swatch_start_y + cell_height / 2

            if n == N:
                ax.text(
                    text_pos_x,
                    text_pos_y - cell_height,
                    f"N = {N}",
                    fontsize=14,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            ax.text(
                text_pos_x,
                text_pos_y,
                swatch_clr,
                fontsize=14,
                color=str(gray),
                horizontalalignment="center",
                verticalalignment="center",
            )

            ax.add_patch(
                _Rectangle(
                    xy=(swatch_start_x, swatch_start_y),
                    width=swatch_width,
                    height=swatch_height,
                    facecolor=swatch_clr,
                )
            )


    # add the full color bar to the right
    cbax = axes[1]
    cbar = _ColorbarBase(
        ax=cbax,
        cmap=cmap[palette],
        norm=Normalize(vmin=0, vmax=1),
        orientation="vertical",
    )
    cbax.axis("off")
    cbar.outline.set_visible(False)

    fig.tight_layout()

    return fig, axes
