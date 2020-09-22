import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../ana/"))
import utility as ut

# Function to run logISI method
def logisi_pasq_method(spike_train, cutoff=0.1):

    if len(spike_train) > 3:
        # Calculates threshold as isi_low
        isi_low = logisi_break_calc(spike_train, cutoff)
        print("isi_low", isi_low)
        if isi_low is None or isi_low >= 1:
            print("A")
            # If no value for isi_low found, or isi_low above 1 second, find bursts using threshold equal to cutoff (default 100ms)
            result = logisi_find_burst(
                spike_train, min_ibi=0, min_durn=0, min_spikes=3, isi_low=cutoff
            )

        elif isi_low < 0:
            print("B")
            result = None

        elif isi_low > cutoff and isi_low < 1:
            print("C")
            # If isi_low >cutoff, find bursts using threshold equal to cutoff (default 100ms)
            bursts = logisi_find_burst(
                spike_train, min_ibi=isi_low, min_durn=0, min_spikes=3, isi_low=cutoff
            )
            if bursts is not None:
                # If bursts have been found, add burst related spikes using threshold of isi_low
                brs = logisi_find_burst(
                    spike_train, min_ibi=0, min_durn=0, min_spikes=3, isi_low=isi_low
                )
                result = add_brs(bursts, brs, spike_train)
            else:
                result = bursts

        else:
            print("D")
            # If isi_low<cutoff, find bursts using a threshold equal to isi_low
            result = logisi_find_burst(
                spike_train, min_ibi=0, min_durn=0, min_spikes=3, isi_low=isi_low
            )

    else:
        result = None

    return result

debug = None
# Function to find cutoff threshold.
def find_thresh(h, h_edges, ISITh=100.0):
    void_th = 0.7
    peak_idx, peak_props = find_peaks(h, height=0, distance=3)
    num_peaks = len(peak_idx)
    peak_pos = h_edges[peak_idx]
    peak_heights = peak_props["peak_heights"]
    # find peak position of the intra-burst-isi, if below the specified threshold
    if (peak_pos < ISITh).any():
        last_idx = np.where(peak_pos < ISITh)[0][-1]
        intra_idx = np.argmax(peak_heights[0:last_idx])
        intra_height = peak_heights[intra_idx]
    else:
        return -1000.0

    y1 = intra_height
    x1 = peak_idx[intra_idx]
    global debug
    debug = h
    print("peak_idx", peak_idx)
    print("peak_heights", peak_heights)

    print("x1:", intra_idx)
    print("y1:", intra_height)
    print("last_idx:", last_idx)
    num_peaks_after_burst = num_peaks - intra_idx

    if num_peaks_after_burst == 0:
        return None
    else:
        x_2s = peak_idx[intra_idx + 1 :]
        y_2s = peak_heights[intra_idx + 1 :]
        # x_2s = np.delete(x_2s, [3,6])
        # y_2s = np.delete(y_2s, [3,6])

        print("h", h)
        print("x1", x1)
        print("x_2s", x_2s)
        print("num_peaks", num_peaks)

        f = lambda x: np.amin(h[x1:x])
        ymins = np.vectorize(f)(x_2s)

        f = lambda x: np.argmin(h[x1:x])
        xmins = np.vectorize(f)(x_2s) + x1

        void_pars = 1 - (ymins / np.sqrt(y1 * y_2s))

        print("ymins: ", ymins)
        print("xmins: ", xmins)
        print("void_pars: ", void_pars)

    try:
        void_idx = np.where(void_pars >= void_th)[0][0]
        print("void idx", void_idx)
    except:
        void_idx = None

    if void_idx is None:
        return None
    else:
        print("ISImax: ", h_edges[xmins[void_idx]])
        return h_edges[xmins[void_idx]]


# Calculates cutoff for burst detection
def logisi_break_calc(st, cutoff):
    isi = np.diff(st) * 1000.0
    max_isi = np.ceil(np.log10(np.max(isi)))
    isi = isi[isi >= 1]
    br = np.logspace(0, max_isi, int(10 * max_isi))
    hist, edges = np.histogram(isi, bins=br)
    hist_smooth = lowess(
        endog=hist / np.sum(hist),
        exog=np.arange(len(hist)),
        frac=0.05,
        is_sorted=True,
        return_sorted=False,
    )
    thr = find_thresh(h=hist_smooth, h_edges=edges, ISITh=cutoff * 1000.0)
    if not thr is None:
        thr = thr / 1000.0

    return thr


###Function to add burst related spikes to edges of bursts
def add_brs(bursts, brs, spikes):
    def is_between(x, a, b):
        return (x >= a) & (x <= b)

    num_bursts = len(bursts["beg"])
    num_brs = len(brs["beg"])
    burst_adj = {"beg": np.zeros(num_bursts), "end": np.zeros(num_bursts)}

    for i in range(num_bursts):
        for j in range(num_brs):
            if is_between(bursts["beg"][i], brs["beg"][j], brs["end"][j]) or is_between(
                bursts["end"][i], brs["beg"][j], brs["end"][j]
            ):
                burst_adj["beg"][i] = np.fmin(bursts["beg"][i], brs["beg"][j])
                burst_adj["end"][i] = np.fmax(bursts["end"][i], brs["end"][j])
                break
            else:
                burst_adj["beg"][i] = bursts["beg"][i]
                burst_adj["end"][i] = bursts["end"][i]

            if brs["end"][j] > bursts["end"][i]:
                break

    diff_begs = np.diff(burst_adj["beg"])
    diff_ends = np.diff(burst_adj["end"])

    rejects = np.array([])
    rejects = np.concatenate((rejects, np.where(diff_begs == 0)[0]))
    rejects = np.concatenate((rejects, np.where(diff_ends == 0)[0] + 1))
    rejects = np.sort(np.unique(rejects))

    if len(rejects) > 0:
        for key in burst_adj.keys():
            np.delete(burst_adj[key], rejects)

    burst_adj["blen"] = burst_adj["end"] - burst_adj["beg"] + 1
    burst_adj["durn"] = spikes[burst_adj["end"]] - spikes[burst_adj["beg"]]
    burst_adj["mean_isis"] = burst_adj["durn"] / (burst_adj["blen"] - 1)

    ibis = spikes[burst_adj["beg"][1:]] - spikes[burst_adj["end"][:-1]]
    ibis = np.insert(ibis, 0, np.nan)
    burst_adj["IBI"] = ibis

    return burst_adj


##Function for finding bursts, taken from sjemea
def logisi_find_burst(spikes, min_ibi, min_durn, min_spikes, isi_low, debug=True):
    """
        Find bursts, taken from sjemea

        Parameters
        ----------
        spikes: array
            containing the spike times

        min_ibi : float
            minimal interval to separate two bursts, otherwise will be merged

        min_durn, min_spikes: floats
            minimal burst-duration and number of spikes required to count as burst


        Returns
        -------
        something: of_type
    """

    ## For one spike train, find the burst using log isi method.
    ## e.g.
    ## find.bursts(s$spikes[[5]])
    ## init.
    ##

    # value to return if no bursts found.
    no_bursts = None

    nspikes = len(spikes)

    ## Create a temp array for the storage of the bursts.  Assume that
    ## it will not be longer than Nspikes/2 since we need at least two
    ## spikes to be in a burst.

    max_bursts = np.floor(nspikes / 2)

    bursts = []

    # current burst number
    burst = 0

    ## Phase 1 -- burst detection. Each interspike interval of the data
    ## is compared with the threshold THRE. If the interval is greater
    ## than the threshold value, it can not be part of a burst; if the
    ## interval is smaller or equal to the threhold, the interval may be
    ## part of a burst.

    ## last_end is the time of the last spike in the previous burst.
    ## This is used to calculate the IBI.
    ## For the first burst, this is no previous IBI
    last_end = np.nan

    eps = 1e-10
    n = 1
    in_burst = False

    print("nspikes", nspikes)
    print("isi_low", isi_low)

    while n < nspikes-1:
        next_isi = spikes[n] - spikes[n - 1]
        if in_burst:
            if next_isi - isi_low > eps:
                ## end of burst
                end = n - 1
                in_burst = False

                ibi = spikes[beg] - last_end
                last_end = spikes[end]
                res = np.array([beg, end, ibi])
                burst = burst + 1

                assert burst <= max_bursts
                bursts.append(res)

        else:
            ## not yet in burst.
            if next_isi - isi_low <= eps:
                ## Found the start of a new burst.
                beg = n - 1
                in_burst = True

        n = n + 1

    ## At the end of the burst, check if we were in a burst when the
    ## train finished.
    if in_burst:
        end = nspikes
        ibi = spikes[beg] - last_end
        res = np.array([beg, end, ibi])
        burst = burst + 1
        assert burst <= max_bursts
        bursts.append(res)

    print("burst", burst)
    ## Check if any bursts were found.
    if burst > 0:
        # convert bursts into a dictionary of 1d arrays
        bursts = np.array(bursts)
        print(bursts)
        bursts = {"beg": bursts[:,0].astype(int), "end": bursts[:,1].astype(int), "IBI": bursts[:,2]}

    else:
        ## no bursts were found, so return an empty structure.
        return no_bursts

    if debug:
        print("End of phase1")
        print(bursts)

    ## Phase 2 -- merging of bursts.    Here we see if any pair of bursts
    ## have an IBI less than min_ibi; if so, we then merge the bursts.
    ## We specifically need to check when say three bursts are merged
    ## into one.

    ibis = bursts["IBI"]
    rejects = np.array([])

    if (ibis < min_ibi).any():
        ## Merge bursts efficiently.    Work backwards through the list, and
        ## then delete the merged lines afterwards.    This works when we
        ## have say 3+ consecutive bursts that merge into one.

        # remove these later
        merge_bursts = np.where(ibis < min_ibi)[0]
        rejects = merge_bursts

        for burst in reversed(merge_bursts):
            bursts["end"][burst - 1] = bursts["end"][burst]
            bursts["end"][burst] = np.nan  # not needed, but helpful.

    if debug:
        print("End of phase 2\n")
        print(bursts)
        print("num bursts", len(bursts["beg"]))

    ## Phase 3 -- remove small bursts: less than min duration (MIN_DURN), or
    ## having too few spikes (less than MIN_SPIKES).
    ## In this phase we have the possibility of deleting all spikes.

    ## BLEN = number of spikes in a burst.
    ## DURN = duration of burst.
    bursts["blen"] = bursts["end"] - bursts["beg"] + 1
    print(bursts["end"])
    bursts["durn"] = spikes[bursts["end"]] - spikes[bursts["beg"]]
    bursts["mean_isis"] = bursts["durn"] / (bursts["blen"] - 1)

    rejects = np.concatenate((rejects, np.where(bursts["durn"] < min_durn)[0]))
    rejects = np.concatenate((rejects, np.where(bursts["blen"] < min_spikes)[0]))
    rejects = np.sort(np.unique(rejects))

    if len(rejects) > 0:
        for key in bursts.keys():
            np.delete(bursts[key], rejects)

    if len(bursts["beg"]) == 0:
        ## All the bursts were removed during phase 3.
        bursts = no_bursts
    else:
        ## Recompute IBI (only needed if phase 3 deleted some cells).
        if len(bursts["beg"]) > 1:
            ibis = spikes[bursts["beg"][1:]] - spikes[bursts["end"][:-1]]
            ibis = np.insert(ibis, 0, np.nan)
        else:
            ibis = np.array([np.nan])

        bursts["IBI"] = ibis
        assert len(bursts["IBI"]) == len(bursts["beg"])

    ## End -- return burst structure.
    return bursts


# ------------------------------------------------------------------------------ #
#
# ------------------------------------------------------------------------------ #

src_prefix = "/Users/paul/mpi/simulation/brian_modular_cultures/_latest/dat/dyn/2x2merged_sparse/"
tar_prefix = "/Users/paul/Desktop/"
files = [
    "gampa=40.00_rate=0.0400_recovery=2.00_rep=00",
    "gampa=30.00_rate=0.0400_recovery=2.00_rep=00",
    "gampa=35.00_rate=0.0400_recovery=2.00_rep=00",
]


for f in files:
    neurons = []
    spikes = []

    spiketimes = ut.h5_load(src_prefix + f + ".hdf5", "/data/spiketimes", silent=True)

    for n in range(0, spiketimes.shape[0]):
        train = spiketimes[n]
        train = train[np.nonzero(train)]
        neurons.append([n] * len(train))
        spikes.append(train)

    spikes = np.concatenate(spikes)
    neurons = np.concatenate(neurons)

    np.savetxt(tar_prefix + f + ".csv", np.vstack((neurons, spikes)).T, fmt="%02d,%.6f")


f = files[0]
neurons, spikes = np.loadtxt(tar_prefix + f + ".csv", unpack=True, delimiter=",")
neurons = neurons.astype(int)

idx = np.argwhere(neurons == 50).flatten()
print(spikes[idx])

foo = logisi_pasq_method(spikes[idx])
# spikes[idx][foo[,"beg"]]
