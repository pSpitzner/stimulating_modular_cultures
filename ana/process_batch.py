import os
import sys
import glob
import h5py
import argparse
import logging
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut
import ana_helper as ah
import plot_helper as ph
import colors as cc
import hi5 as h5
from hi5 import BetterDict
import transfer_entropy as treant


def process_candidates_burst_times_and_isi(input_path, hot=False):
    """
        get the burst times based on rate for every module and merge it down, so that
        we have ensemble average statistics

        uses remaing info from h5f (spiketimes, modules etc) from first file in input_path
    """

    candidates = glob.glob(input_path)

    assert len(candidates) > 0, "Is the input_path correct?"

    res = None

    for cdx, candidate in enumerate(
        tqdm(candidates, desc="Burst times for files", position=2, leave=False)
    ):
        h5f = h5.recursive_load(candidate, hot=hot)
        ah.prepare_file(h5f)
        ah.find_bursts_from_rates(h5f)
        ah.find_isis(h5f)

        this_burst = h5f.ana.bursts
        this_isi = h5f.ana.isi

        if cdx == 0:
            res = h5f

        # copy over system level burst
        b = res.ana.bursts.system_level
        b.beg_times.extend(this_burst.system_level.beg_times)
        b.end_times.extend(this_burst.system_level.end_times)
        b.module_sequences.extend(this_burst.system_level.module_sequences)
        for m_id in h5f.ana.mods:
            # copy over module level bursts
            b = res.ana.bursts.module_level[m_id]
            b.beg_times.extend(this_burst.module_level[m_id].beg_times)
            b.end_times.extend(this_burst.module_level[m_id].end_times)

            # and isis
            i = res.ana.isi[m_id]
            for var in ["all", "in_bursts", "out_bursts"]:
                i[var].extend(this_isi[m_id][var])

        if hot:
            h5.close_hot()

    return res


def isi_across_conditions():

    figs = []
    conds = _conditions()
    for k in tqdm(conds.varnames, desc="k values", position=0, leave=False):
        for stim in tqdm(
            conds[k].varnames, desc="stimulation targets", position=1, leave=False
        ):
            h5f = process_candidates_burst_times_and_isi(conds[k][stim])
            logging.getLogger("plot_helper").setLevel("WARNING")
            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=(4, 6),
                gridspec_kw=dict(height_ratios=[1, 3, 3]),
            )
            ph.plot_parameter_info(h5f, ax=axes[0])
            ph.plot_distribution_burst_duration(h5f, ax=axes[1])
            ph.plot_distribution_isi(h5f, ax=axes[2])
            for i in range(4):
                fig.tight_layout()
            figs.append(fig)
            logging.getLogger("plot_helper").setLevel("INFO")
            del h5f

    # figures=[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

    for f in figs:
        axes = f.get_axes()
        axes[1].set_xlim(0, 0.3)
        axes[2].set_xlim(1e-3, 1e2)

    return figs



def _conditions():
    # fmt:off
    path_base = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/"
    stim = BetterDict()
    for k in [0,1,2,3,5]:
        stim[k] = BetterDict()
        stim[k].off = f"{path_base}/dyn/2x2_fixed/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k}_rep=*.hdf5"
        for s in ['0', '02', '012', '0123']:
            stim[k][s] = f"{path_base}/jitter_{s}/gampa=35.00_rate=37.00_recovery=2.00_alpha=0.0125_k={k}_rep=*.hdf5"

    return stim
