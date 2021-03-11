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
import colors as cc
import hi5 as h5
from hi5 import BetterDict
import transfer_entropy as treant


def process_candidates_burst_times_and_isi(input_path):
    """
        get the burst times based on rate for every module and merge it down, so that
        we have ensemble average statistics
    """

    candidates = glob.glob(input_path)

    for cdx, candidate in enumerate(tqdm(candidates, desc="Burst times for files")):
        h5f = h5.recursive_load(candidate, hot=True)
        ah.prepare_file(h5f)

        this_burst, _ = ah.find_bursts_from_rates(h5f)
        this_isi = ah.find_isis_from_bursts(h5f, bursts = this_burst)

        if cdx == 0:
            all_bursts = this_burst
            all_isi = this_isi
        else:
            # copy over system level burst
            b = all_bursts.system_level
            b.beg_times.extend(this_burst.system_level.beg_times)
            b.end_times.extend(this_burst.system_level.end_times)
            b.module_sequences.extend(this_burst.system_level.module_sequences)
            for m_id in h5f.ana.mods:
                # copy over module level bursts
                b = all_bursts.module_level[m_id]
                b.beg_times.extend(this_burst.module_level[m_id].beg_times)
                b.end_times.extend(this_burst.module_level[m_id].end_times)

                i = all_isi[m_id]
                for var in ['all', 'in_bursts', 'out_bursts']:
                    i[var].extend(this_isi[m_id][var])

        h5.close_hot()

    return all_bursts, all_isi

