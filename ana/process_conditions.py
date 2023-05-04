# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-10-25 17:28:21
# @Last Modified: 2023-05-02 09:45:04
# ------------------------------------------------------------------------------ #
# Analysis script that preprocesses experiments and creates dataframes to compare
# across condtions. Plots and more detailed analysis are in `paper_plots.py`
# * input files are globbed from the provided input directory using a
# hardcoded wildcard, depending on the type, e.g.  `-t sim`, `-t exp`.
# * output file names are given automatically, `-o` specifies the output directory.
# ------------------------------------------------------------------------------ #

import os
import glob
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# import enlighten
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")
warnings.filterwarnings("ignore")  # suppress numpy warnings

# our custom modules
import bitsandbobs as bnb
import ana_helper as ah
import plot_helper as ph

# only affects simulations, as in the experiments we have only few neurons
# per module, thus the 20% of neurons in the module are just one neuron.
remove_null_sequences = False

# whether to store the analysis of each trial as hdf5 in the usual format
save_analysed_h5f = False

# for correlation coefficients, size of time steps in which number of spikes are counted
time_bin_size_for_rij = 500 / 1000  # in seconds

# threshold for burst detection [% of max peak height]
# we found that simulations needed different parameters due to the higher (sampled)
# number of neurons and time resolution
def threshold_factor(etype):
    if etype[0:3] == "exp":
        return 10 / 100
    elif etype[0:3] == "sim":
        return 2.5 / 100
    else:
        raise ValueError(f"etype {etype} not recognized")


# for pop. rate, width of gaussian placed on every spike, in seconds
def bs_large(etype):
    if etype[0:3] == "exp":
        return 200 / 1000
    elif etype[0:3] == "sim":
        return 20 / 1000
    else:
        raise ValueError(f"etype {etype} not recognized")


dataframes = None


def main():
    global dataframes
    global h5f
    parser = argparse.ArgumentParser(description="Process conditions")
    parser.add_argument(
        "-t",
        dest="etype",
        required=True,
        help=(
            "'exp', 'exp_chemical', 'exp_bic', 'sim', 'sim_partial',"
            " 'sim_partial_no_inhib'"
        ),
    )
    parser.add_argument(
        "-i",
        dest="input_base",
        required=True,
        help="Root directory for files, `./dat/exp_in/`",
    )
    parser.add_argument(
        "-o",
        dest="output_path",
        required=True,
        help="`./dat/exp_out/`",
    )
    args = parser.parse_args()

    output_path = args.output_path

    conditions = dict()
    if args.etype == "exp":
        for layout in ["1b", "3b", "merged"]:
            conditions[layout] = ["1_pre", "2_stim", "3_post"]
    elif args.etype == "exp_chemical":
        conditions["KCl_1b"] = ["1_KCl_0mM", "2_KCl_2mM"]
    elif args.etype == "exp_bic":
        conditions["Bicuculline_1b"] = ["1_spon_Bic_20uM", "2_stim_Bic_20uM"]
    elif args.etype == "exp_mod_comp":
        # comparison between different targeted regions
        conditions["partial_2u"] = ["1_pre", "2_stim2", "3_post", "4_stim1"]
        conditions["partial_5u"] = ["1_pre", "2_stim2", "3_post", "4_stim1"]
        conditions["global_2u"] = ["1_pre", "2_stim", "3_post"]
        conditions["global_5u"] = ["1_pre", "2_stim", "3_post"]
    elif args.etype == "sim":
        # number of axons between modules as layouts
        # first rate gets stimulated "off" value assigned, second becomes "on"
        # motiviation here was to get similar IEI for all k,
        # which occurs at different levels of noise, depending on k.
        conditions["k=5"] = ["80.0", "90.0"]  # Hz
        conditions["k=1"] = ["75.0", "85.0"]
        conditions["k=10"] = ["85.0", "92.5"]
    elif args.etype == "sim_partial":
        # for the case where we only stimulate 2 modules instead of uniform
        # noise to all, we need a bit more tweaking below
        conditions["k=0"] = ["0.0", "20.0"]
        conditions["k=1"] = ["0.0", "20.0"]
        conditions["k=3"] = ["0.0", "20.0"]
        conditions["k=5"] = ["0.0", "20.0"]
        conditions["k=10"] = ["0.0", "20.0"]
        conditions["k=-1"] = ["0.0", "20.0"]
    elif args.etype == "sim_partial_no_inhib":
        # this is the control for blocked inhibition, we only did that for k=5
        conditions["k=3"] = ["0.0", "20.0"]
    else:
        raise KeyError("type should be 'exp', 'exp_chemical' or 'sim_partial'")

    # ------------------------------------------------------------------------------ #
    # iterate over all combination
    # ------------------------------------------------------------------------------ #

    log.info(f"Reading from {args.input_base}")
    log.info(f"Writing to {output_path}")

    for layout in tqdm(conditions.keys(), desc="Layouts"):

        dataframes = dict()
        for key in [
            "bursts",
            "isis",
            "rij",
            "rij_paired",
            "mod_rij",
            "mod_rij_paired",
            "trials",
        ]:
            dataframes[key] = []
        if "sim" in args.etype:
            # we collect the correlation coefficients of synaptic resources for sim
            dataframes["drij"] = []

        for cdx, condition in enumerate(
            tqdm(conditions[layout], leave=False, desc="Conditions")
        ):

            # depending on the type of experiment, we have different naming conventions
            # where wildcards '*' should be completed
            if "exp" in args.etype:
                input_paths = glob.glob(f"{args.input_base}/{layout}/*")
            elif args.etype == "sim":
                input_paths = glob.glob(
                    f"{args.input_base}/stim=off_{layout}_kin=30_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate={condition}_rep=*.hdf5"
                )
            elif args.etype == "sim_partial":
                input_paths = glob.glob(
                    f"{args.input_base}/stim=02_{layout}_kin=30_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_stimrate={condition}_rep=*.hdf5"
                )
            elif args.etype == "sim_partial_no_inhib":
                input_paths = glob.glob(
                    f"{args.input_base}/stim=02_{layout}_kin=30_jA=45.0_jG=0.0_jM=15.0_tD=20.0_rate=80.0_stimrate={condition}_rep=*.hdf5"
                )

            log.debug(f"found {len(input_paths)} files for {layout} {condition}")

            # trials / realizations
            pbar = tqdm(input_paths, desc="Files", leave=False)
            for path in pbar:

                trial = os.path.basename(path)
                if "sim" in args.etype:
                    trial = trial.split("rep=")[-1].split(".")[0]

                log.info("------------")
                log.info(f"{args.etype} {layout} {condition} {trial}")
                pbar.set_description(f"{args.etype} {layout} {condition} {trial}")
                log.info("------------")

                # for the dataframes, we need to tidy up some labels
                if "exp" in args.etype:
                    condition_string = condition[2:]
                    stimulation_string = "On" if condition[0:2] == "2_" else "Off"
                elif "sim" in args.etype:
                    condition_string = f"{condition} Hz"
                    # here we should be a bit more careful, maybe
                    stimulation_string = "On" if cdx == 1 else "Off"

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

                # propagation delay: how long to go from peak to peak of the module-level
                # population rate
                ah.find_burst_core_delays(h5f)
                delays = np.array(
                    [np.mean(x) for x in h5f["ana.bursts.system_level.core_delays_mean"]]
                )

                df = pd.DataFrame(
                    {
                        "Duration": blen,
                        "Sequence length": slen,
                        "Core delay": delays,
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
                # neuron level correlation coefficients
                # ------------------------------------------------------------------------------ #

                # NxN matrix
                neuron_rij = ah.find_rij(
                    h5f, which="neurons", time_bin_size=time_bin_size_for_rij
                )
                np.fill_diagonal(neuron_rij, np.nan)
                neuron_rij_flat = neuron_rij.flatten()
                df = pd.DataFrame(
                    {
                        "Correlation Coefficient": neuron_rij_flat,
                        "Condition": condition_string,
                        "Trial": trial,
                        "Stimulation": stimulation_string,
                        "Type": args.etype,
                    }
                )
                # just the bunch of all rijs
                dataframes["rij"].append(df)
                neuron_rij_mean = np.nanmean(neuron_rij)
                neuron_rij_median = np.nanmedian(neuron_rij)

                # we also want to compare the correlation coefficients for different
                # combinations ("parings") of neurons from certain modules
                pair_descriptions = dict()
                pair_descriptions["across_groups_0_2"] = "within_stim"
                pair_descriptions["across_groups_1_3"] = "within_nonstim"
                pair_descriptions["across_groups_0_1"] = "across"
                pair_descriptions["across_groups_2_3"] = "across"
                pair_descriptions["all"] = "all"
                for pairing in pair_descriptions.keys():
                    neuron_rij_paired = ah.find_rij_pairs(
                        h5f, rij=neuron_rij, pairing=pairing, which="neurons"
                    )
                    df = pd.DataFrame(
                        {
                            "Correlation Coefficient": neuron_rij_paired,
                            "Condition": condition_string,
                            "Trial": trial,
                            "Pairing": pair_descriptions[pairing],
                            "Pair ID": np.arange(len(neuron_rij_paired)),
                            "Stimulation": stimulation_string,
                            "Type": args.etype,
                        }
                    )
                    dataframes["rij_paired"].append(df)

                # ------------------------------------------------------------------------------ #
                # module level correlation coefficients
                # ------------------------------------------------------------------------------ #

                # 4x4 matrix
                module_rij = ah.find_rij(h5f, which="modules")
                np.fill_diagonal(module_rij, np.nan)
                module_rij_flat = module_rij.flatten()
                df = pd.DataFrame(
                    {
                        "Correlation Coefficient": module_rij_flat,
                        "Condition": condition_string,
                        "Trial": trial,
                        "Stimulation": stimulation_string,
                        "Type": args.etype,
                    }
                )
                # just the bunch of all rijs
                dataframes["mod_rij"].append(df)
                module_rij_mean = np.nanmean(module_rij)
                module_rij_median = np.nanmedian(module_rij)

                # pair descriptions as above
                for pairing in pair_descriptions.keys():
                    module_rij_paired = ah.find_rij_pairs(
                        h5f,
                        rij=module_rij,
                        pairing=pairing,
                        which="modules",
                    )
                    df = pd.DataFrame(
                        {
                            "Correlation Coefficient": module_rij_paired,
                            "Condition": condition_string,
                            "Trial": trial,
                            "Pairing": pair_descriptions[pairing],
                            "Pair ID": np.arange(len(module_rij_paired)),
                            "Stimulation": stimulation_string,
                            "Type": args.etype,
                        }
                    )
                    dataframes["mod_rij_paired"].append(df)

                # ------------------------------------------------------------------------------ #
                # Correlation of the depletion variable, for simulations
                # ------------------------------------------------------------------------------ #

                if "sim" in args.etype:
                    drij = ah.find_rij(h5f, which="depletion")
                    np.fill_diagonal(drij, np.nan)
                    drij_flat = drij.flatten()
                    df = pd.DataFrame(
                        {
                            "Depletion rij": drij_flat,
                            "Condition": condition_string,
                            "Trial": trial,
                            "Stimulation": stimulation_string,
                            "Type": args.etype,
                        }
                    )
                    # just the bunch of all rijs
                    dataframes["drij"].append(df)

                # ------------------------------------------------------------------------------ #
                # and some summary statistics on the trial level
                # ------------------------------------------------------------------------------ #

                fc = ah._functional_complexity(neuron_rij)
                df = pd.DataFrame(
                    {
                        "Num Bursts": [len(blen)],
                        "Mean Neuron Correlation": [neuron_rij_mean],
                        "Median Neuron Correlation": [neuron_rij_median],
                        "Mean Module Correlation": [module_rij_mean],
                        "Median Module Correlation": [module_rij_median],
                        "Mean IBI": [np.nanmean(ibis)],
                        "Median IBI": [np.nanmedian(ibis)],
                        "Mean Rate": [np.nanmean(h5f["ana.rates.system_level"])],
                        "Mean Fraction": [np.nanmean(fracs)],
                        "Median Fraction": [np.nanmedian(fracs)],
                        "Mean Core delays": [np.nanmean(delays)],
                        "Median Core delays": [np.nanmedian(delays)],
                        "Functional Complexity": [fc],
                        "Condition": condition_string,
                        "Trial": trial,
                        "Stimulation": stimulation_string,
                        "Type": args.etype,
                    }
                )
                if "sim" in args.etype:
                    df["Mean Depletion rij"] = np.nanmean(drij)
                    df["Median Depletion rij"] = np.nanmedian(drij)
                dataframes["trials"].append(df)

                # ------------------------------------------------------------------------------ #
                # Finalize, and save a copy of the analyzed file for this trial
                # ------------------------------------------------------------------------------ #

                if "exp" in args.etype and save_analysed_h5f:
                    bnb.hi5.recursive_write(
                        filename=(
                            f"{output_path}/{layout}/{trial}/{condition}_analyzed.hdf5"
                        ),
                        h5_data=h5f,
                    )

                bnb.hi5.close_hot()
                del h5f

        # for every layout, join list of dataframes and save
        for key in dataframes.keys():
            dataframes[key] = pd.concat(dataframes[key], ignore_index=True)
            if key == "isis":
                dataframes[key]["logISI"] = dataframes[key].apply(
                    lambda row: np.log10(row["ISI"]), axis=1
                )

        # for the simulations we append a suffix because layotus `k=...` are not unique.
        if "sim" in args.etype:
            suffix = args.etype[3:]
        else:
            suffix = ""

        meta_data = dict()
        meta_data["remove_null_sequences"] = remove_null_sequences
        meta_data["time_bin_size_for_rij"] = time_bin_size_for_rij
        meta_data["bs_large"] = bs_large(args.etype)
        meta_data["threshold_factor"] = threshold_factor(args.etype)
        meta_data["etype"] = args.etype
        meta_data["created"] = datetime.datetime.now().isoformat()
        meta_data["input_base"] = args.input_base
        meta_data["output_path"] = output_path
        meta_data["save_analysed_h5f"] = save_analysed_h5f

        dict_of_dfs_to_hdf5(dataframes, f"{output_path}/{layout}{suffix}.hdf5", meta_data)


# ------------------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------------------ #


def prepare_file(etype, condition, path_prefix):
    if "exp" in etype:
        h5f = ah.load_experimental_files(
            path_prefix=f"{path_prefix}/", condition=condition
        )
    elif "sim" in etype:
        h5f = ah.prepare_file(path_prefix)

    ah.find_rates(h5f, bs_large=bs_large(etype))
    ah.find_system_bursts_from_global_rate(
        h5f,
        rate_threshold=threshold_factor(etype) * np.nanmax(h5f["ana.rates.system_level"]),
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


def dict_of_dfs_to_hdf5(df_dict, df_path, meta=dict()):
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    for key in df_dict.keys():
        df = df_dict[key]
        df.to_hdf(df_path, f"/data/df_{key}", complevel=6)

    # save some metadata
    import h5py

    with h5py.File(df_path, "a") as f:
        for key in meta.keys():
            f.create_dataset(f"/meta/{key}", data=meta[key])


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
