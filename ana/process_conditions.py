# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-25 17:28:21
# @Last Modified: 2021-11-12 13:53:26
# ------------------------------------------------------------------------------ #
# Hard coded script to analyse experimental data
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import h5py
import argparse
import logging
import warnings
import functools
import itertools
import tempfile
import psutil
import re
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
# import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from benedict import benedict

# from addict import Dict

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # suppress numpy warnings

import dask_helper as dh
import ana_helper as ah
import plot_helper as ph
import hi5 as h5
import colors as cc

# only affects simulations, as in the experiments we have only few neurons
# per module, thus the 20% of neurons in the module are just one neuron.
remove_null_sequences = False

# for correlation coefficients, size of time steps in which number of spikes are counted
time_bin_size_for_rij = 200 / 1000 # in seconds

def main():
    parser = argparse.ArgumentParser(description="Merge Multidm")
    parser.add_argument(
        "-t", dest="etype", required=True, help="'exp', 'exp_chemical', or 'sim'",
    )
    args = parser.parse_args()

    if "exp" in args.etype:
        input_base = "/Users/paul/mpi/simulation/brian_modular_cultures/data_for_jordi/2021-10-11/Experimental data - ML spike/ConsistentRasterWith12000frames"
        output_path = (
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/exp_out"
        )
    elif args.etype == "sim":
        input_base = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/inhibition_sweep_rate_160/dyn"
        output_path = (
            "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/sim_out"
        )
    else:
        assert False, "Choose type from 'sim' 'exp' 'exp_chemical'"

    if args.etype == "exp":
        layouts = ["1b", "3b", "merged"]
        conditions = ["1_pre", "2_stim", "3_post"]
    elif args.etype == "exp_chemical":
        layouts = ["KCl_1b"]
        conditions = ["1_KCl_0mM", "2_KCl_2mM"]
    elif args.etype == "sim":
        layouts = ["k=5"]  # number of axons between modules
        conditions = ["82", "90"]  # Hz

    # ------------------------------------------------------------------------------ #
    # iterate over all combination
    # ------------------------------------------------------------------------------ #

    for layout in layouts:
        dataframes = dict()
        for key in ["bursts", "isis", "rij", "rij_paired", "trials"]:
            dataframes[key] = []

        for condition in conditions:
            if "exp" in args.etype:
                input_paths = glob.glob(f"{input_base}/{layout}/*")
            elif args.etype == "sim":
                input_paths = glob.glob(
                    f"{input_base}/stim=off_{layout}_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate={condition}.0_rep=*.hdf5"
                )

            # trials / realizations
            for path in input_paths:
                trial = os.path.basename(path)
                if args.etype == "sim":
                    trial = trial.split("rep=")[-1].split(".")[0]

                print(f"\n{args.etype} {layout} {condition} {trial}\n")

                # for the dataframes, we need to tidy up some labels
                if "exp" in args.etype:
                    condition_string = condition[2:]
                    stimulation_string = "On" if condition[0:2] == "2_" else "Off"
                elif args.etype == "sim":
                    condition_string = f"{condition} Hz"
                    stimulation_string = "On" if float(condition) < 90  else "Off"

                # the path still contains the trial
                h5f = prepare_file(args.etype, condition, path)

                # ------------------------------------------------------------------------------ #
                # overview plot
                # ------------------------------------------------------------------------------ #

                # plot overview panels for experiments
                if "exp" in args.etype:
                    os.makedirs(f"{output_path}/{layout}/{trial}", exist_ok=True)
                    fig = ph.overview_dynamic(h5f)
                    fig.savefig(
                        f"{output_path}/{layout}/{trial}/{condition}_overview.pdf"
                    )

                    # get a nice zoom in on some bursts
                    try:
                        max_pos = np.nanargmax(h5f["ana.rates.system_level"])
                        max_pos *= h5f["ana.rates.dt"]
                        beg = max_pos
                    except:
                        beg = 0
                    beg = np.fmax(0, beg - 10)
                    fig.get_axes()[-2].set_xlim(beg, beg + 20)
                    fig.savefig(f"{output_path}/{layout}/{trial}/{condition}_zoom.pdf")
                    plt.close(fig)

                # ------------------------------------------------------------------------------ #
                # statistics of bursts
                # ------------------------------------------------------------------------------ #

                # we have already done a bunch of analysis in `prepare_file`
                fracs = np.array(h5f["ana.bursts.system_level.participating_fraction"])
                blen = np.array(h5f["ana.bursts.system_level.end_times"]) - np.array(
                    h5f["ana.bursts.system_level.beg_times"]
                )
                slen = np.array(
                    [len(x) for x in h5f["ana.bursts.system_level.module_sequences"]]
                )
                olen = ah.find_onset_durations(h5f, return_res=True)

                # we have num_bursts -1 inter-burst intervals, use time to next burst
                # and last burst gets a nan.
                ibis = h5f["ana.ibi.system_level.any_module"]
                ibis.extend([np.nan] * (len(blen) - len(ibis)))

                df = pd.DataFrame(
                    {
                        "Duration": blen,
                        "Sequence length": slen,
                        "Fraction": fracs,
                        "Onset duration": olen,
                        "Inter-burst-interval": ibis,
                        "Condition": condition_string,
                        "Trial": trial,
                        "Stimulation": stimulation_string,
                        "Type": args.etype,
                    }
                )
                dataframes["bursts"].append(df)

                # ------------------------------------------------------------------------------ #
                # Inter spike intervals
                # ------------------------------------------------------------------------------ #

                isis = []
                for mdx, m_id in enumerate(h5f["ana.mod_ids"]):
                    m_dc = h5f["ana.mods"][mdx]
                    isis.extend(h5f[f"ana.isi.{m_dc}.all"])

                df = pd.DataFrame(
                    {
                        "ISI": isis,
                        "Condition": condition_string,
                        "Trial": trial,
                        "Stimulation": stimulation_string,
                        "Type": args.etype,
                    }
                )
                dataframes["isis"].append(df)

                # ------------------------------------------------------------------------------ #
                # trials and correlation coefficients
                # ------------------------------------------------------------------------------ #

                # NxN matrix
                rij = ah.find_rij(h5f, time_bin_size=time_bin_size_for_rij)
                np.fill_diagonal(rij, np.nan)
                rij_flat = rij.flatten()
                df = pd.DataFrame(
                    {
                        "Correlation Coefficient": rij_flat,
                        "Condition": condition_string,
                        "Trial": trial,
                        "Stimulation": stimulation_string,
                        "Type": args.etype,
                    }
                )
                # just the bunch of all rijs
                dataframes["rij"].append(df)

                # we also want to compare the correlation coefficients for different
                # combinations ("parings") of neurons from certain modules
                for pairing in [
                    "within_group_02",
                    "within_group_13",
                    "across_groups_02_13",
                ]:
                    rij_paired = ah.find_rij_pairs(h5f, rij=rij, pairing=pairing)
                    df = pd.DataFrame(
                        {
                            "Correlation Coefficient": rij_paired,
                            "Condition": condition_string,
                            "Trial": trial,
                            "Pairing": pairing,
                            "Pair ID": np.arange(len(rij_paired)),
                            "Stimulation": stimulation_string,
                            "Type": args.etype,
                        }
                    )
                    dataframes["rij_paired"].append(df)

                # and some summary statistics on the trial level
                fc = ah._functional_complexity(rij)
                mean_rij = np.nanmean(rij)
                df = pd.DataFrame(
                    {
                        "Num Bursts": [len(blen)],
                        "Mean Correlation": [mean_rij],
                        "Mean IBI": [np.nanmean(ibis)],
                        "Median IBI": [np.nanmedian(ibis)],
                        "Mean Fraction": [np.nanmean(fracs)],
                        "Median Fraction": [np.nanmedian(fracs)],
                        "Functional Complexity": [fc],
                        "Condition": condition_string,
                        "Trial": trial,
                        "Stimulation": stimulation_string,
                        "Type": args.etype,
                    }
                )
                dataframes["trials"].append(df)

                h5.close_hot()
                del h5f

        # for every layout, join list of dataframes and save
        for key in dataframes.keys():
            dataframes[key] = pd.concat(dataframes[key], ignore_index=True)
            if key == "isis":
                dataframes[key]["logISI"] = dataframes[key].apply(
                    lambda row: np.log10(row["ISI"]), axis=1
                )

        dict_of_dfs_to_hdf5(dataframes, f"{output_path}/{layout}.hdf5")


# ------------------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------------------ #


def prepare_file(etype, condition, path_prefix):
    if etype == "exp" or etype == "exp_chemical":
        bs_large = 200 / 1000 # for pop. rate, width of gaussian placed on every spike
        threshold_factor = 10 / 100 # threshold for burst detection [% max peak height]
        h5f = ah.load_experimental_files(
            path_prefix=f"{path_prefix}/", condition=condition
        )
    elif etype == "sim":
        bs_large = 20 / 100
        threshold_factor = 2.5 / 100
        h5f = ah.prepare_file(path_prefix)

    ah.find_rates(h5f, bs_large=bs_large)
    ah.find_system_bursts_from_global_rate(
        h5f,
        rate_threshold=threshold_factor * np.nanmax(h5f["ana.rates.system_level"]),
        merge_threshold=0.1,
        skip_sequences=False,
    )

    # this is a global setting for now
    if remove_null_sequences:
        ah.remove_bursts_with_sequence_length_null(h5f)

    ah.find_ibis(h5f)
    ah.find_participating_fraction_in_bursts(h5f)
    ah.find_isis(h5f)

    return h5f


def dict_of_dfs_to_hdf5(df_dict, df_path):
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    for key in df_dict.keys():
        df = df_dict[key]
        df.to_hdf(df_path, f"/data/df_{key}", complevel=6)


if __name__ == "__main__":
    main()
