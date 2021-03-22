import os
import sys
import glob
import h5py
import argparse
import logging
import warnings
import pickle

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

import ana_helper as ah
import plot_helper as ph
import colors as cc
import hi5 as h5
from hi5 import BetterDict
import transfer_entropy as treant


def process_candidates_burst_times_and_isi(input_path, hot=True):
    """
        get the burst times based on rate for every module and merge it down, so that
        we have ensemble average statistics
    """

    candidates = glob.glob(input_path)

    assert len(candidates) > 0, "Is the input_path correct?"

    res = None
    mods = None

    for cdx, candidate in enumerate(
        tqdm(candidates, desc="Bursts and ISIs for files", leave=False)
    ):
        h5f = h5.recursive_load(candidate, hot=hot)
        ah.prepare_file(h5f)
        ah.find_bursts_from_rates(h5f)
        ah.find_isis(h5f)

        this_burst = h5f.ana.bursts
        this_isi = h5f.ana.isi

        if cdx == 0:
            res = h5f
            mods = h5f.ana.mods

        # todo: consistency checks
        # lets at least check that the modules are consistent across candidates.
        assert np.all(h5f.ana.mods == mods), "Modules differ between files"

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
            # only close the last file (which we opened), and let's hope no other file
            # was opened in the meantime
            h5.close_hot(which=-1)

    return res


def isi_across_conditions():

    conds = _conditions()
    for k in tqdm(conds.varnames, desc="k values", position=0, leave=False):
        for stim in tqdm(
            conds[k].varnames, desc="stimulation targets", position=1, leave=False
        ):
            h5f = process_candidates_burst_times_and_isi(conds[k][stim])
            # preprocess so that plot functions wont do it again.
            # todo: make api consistent
            h5f.ana.ensemble = BetterDict()
            h5f.ana.ensemble.filenames = conds[k][stim]
            h5f.ana.ensemble.bursts = h5f.ana.bursts
            h5f.ana.ensemble.isi = h5f.ana.isi

            logging.getLogger("plot_helper").setLevel("WARNING")
            fig = ph.plot_overview_burst_duration_and_isi(h5f, filenames=conds[k][stim])
            logging.getLogger("plot_helper").setLevel("INFO")

            fig.savefig(f"/Users/paul/mpi/simulation/brian_modular_cultures/_figures/isis/{k}_{stim}.pdf", dpi=300)
            with open(f"/Users/paul/mpi/simulation/brian_modular_cultures/_figures/isis/pkl/{k}_{stim}.pkl",'wb') as fid:
                pickle.dump(fig, fid)

            del fig
            del h5f
            h5.close_hot()


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
