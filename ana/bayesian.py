from pydoc import doc
import bitsandbobs as bnb
import numpy as np
import matplotlib.pyplot as plt

# import colorcet as cc
import bokeh
import bokeh.plotting as bkp
import pymc as pm
import xarray as xr
import arviz as az
import theano
import theano.tensor as tt
import pandas as pd

rng = np.random.default_rng(seed=42)


def best(y1, y2, n1=None, n2=None, **kwargs):
    """
    Function description

    # Parameters
    y1, y2 : 1d np arrays, data
    n1, n2 : str, names to use for the variables y1, y2 in the trace
    kwargs : passed to pm.sample
    """

    if n1 is None:
        n1 = "group1"
    if n2 is None:
        n2 = "group2"

    # combined data for prior
    y = np.concatenate((y1, y2))
    y_mean = np.mean(y)
    y_std = np.std(y)

    with pm.Model() as model:
        # we are agnostic about the difference, same priors for both groups
        # and 2 std is pretty wide.
        mean_1 = pm.Normal(f"{n1}_mean", mu=y_mean, sigma=y_std * 2)
        mean_2 = pm.Normal(f"{n2}_mean", mu=y_mean, sigma=y_std * 2)

        # try to be a bit informative about std
        # std_1 = pm.Uniform(f"{n1}_std", lower=1, upper=10)
        # std_2 = pm.Uniform(f"{n2}_std", lower=1, upper=10)
        std_1 = pm.HalfCauchy(f"{n1}_std", beta=1)
        std_2 = pm.HalfCauchy(f"{n2}_std", beta=1)

        # we assume both data to share the normality parameter, and want nu >= 1
        # JEP 2013 paper: “This prior was selected because it balances nearly normal
        # distributions (ν > 30) with heavy tailed distributions (ν < 30)”
        nu = pm.Exponential("nu_minus_one", 1 / 29.0) + 1

        # convert std to precision for pymcs student t
        lam_1 = std_1**-2
        lam_2 = std_2**-2

        # Likelihood (sampling distribution) of observations
        group1 = pm.StudentT("drug", nu=nu, mu=mean_1, lam=lam_1, observed=y1)
        group2 = pm.StudentT("placebo", nu=nu, mu=mean_2, lam=lam_2, observed=y2)

        # observables
        diff_of_means = pm.Deterministic("diff_of_means", mean_1 - mean_2)
        diff_of_stds = pm.Deterministic("diff_of_stds", std_1 - std_2)
        effect_size = pm.Deterministic(
            "effect_size", diff_of_means / np.sqrt((std_1**2 + std_2**2) / 2)
        )

        kwargs = kwargs.copy()
        kwargs.setdefault("draws", 1000)
        kwargs.setdefault("return_inferencedata", True)

        # sample
        trace = pm.sample(**kwargs)

    return trace


def best_paired(y1, y2, **kwargs):
    """
    Function description

    # Parameters
    y1, y2 : 1d np arrays, data
    kwargs : passed to pm.sample

    https://github.com/mikemeredith/BEST/blob/main/R/BESTmcmc.R
    https://github.com/rasmusab/bayesian_first_aid/blob/d80c0fded797cff623a5ec42fb2ad8ffbec8b441/R/bayes_t_test.R#L383

    paired_samples_t_model_string <- "model {
        for(i in 1:length(pair_diff)) {
            pair_diff[i] ~ dt( mu_diff , tau_diff , nu )  student t
            # every sample of the difference comes from student t?
        }
        diff_pred ~ dt( mu_diff , tau_diff , nu )
        # student dt

        eff_size <- (mu_diff - comp_mu) / sigma_diff
        # comp mu is probably the null hypothesis -> no change: 0

        mu_diff ~ dnorm( mean_mu , precision_mu )
        # normal distributed mu_differences?


        tau_diff <- 1/pow( sigma_diff , 2 )
        # precision of t

        sigma_diff ~ dunif( sigma_low , sigma_high )


        # A trick to get an exponentially distributed prior on nu that starts at 1.
        nu <- nuMinusOne + 1
        nuMinusOne ~ dexp(1/29)
        }"
    """

    mu_ref = 0

    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    observed_diff = y2 - y1

    with pm.Model() as model:
        # I guess here our prior is more important then when unpaired.
        # here, we operate on differences, so mu is the mean difference etc
        mu = pm.Normal(
            f"mean_of_diffs",
            mu=np.mean(observed_diff),
            sigma=2 * np.fmax(np.std(observed_diff), 1e-5),
        )
        # std = pm.Uniform(f"std_of_diffs", lower=0.01, upper=10)
        std = pm.HalfCauchy(f"std_of_diffs", beta=1)

        # we assume both data to share the normality parameter, and want nu >= 1
        # JEP 2013 paper: “This prior was selected because it balances nearly normal
        # distributions (ν > 30) with heavy tailed distributions (ν < 30)”
        nu = pm.Exponential("nu_minus_one", 1 / 29.0) + 1

        # convert std to precision for pymcs student t
        lam = std**-2

        # Likelihood (sampling distribution) of observations
        diffs = pm.StudentT("diffs", nu=nu, mu=mu, lam=lam, observed=observed_diff)

        # observables
        effect_size = pm.Deterministic("effect_size", (mu - mu_ref) / std)

        kwargs = kwargs.copy()
        kwargs.setdefault("draws", 1000)
        kwargs.setdefault("return_inferencedata", True)

        # sample
        trace = pm.sample(**kwargs)

    return trace


def probability_of_direction(
    posterior_samples, ref_func=np.greater, ref_val=0, pretty_print=False
):
    """

    # Parameters
    posterior_samples : xarray or nd array, posterior samples shape (chains, draws)

    # Example
    ```
    trace = best_paired(y1, y2)
    probability_direction(trace.posterior["mean_of_diffs"])
    ```
    """

    if isinstance(posterior_samples, xr.DataArray):
        posterior_samples = posterior_samples.to_numpy()

    size = np.prod(posterior_samples.shape)
    prob = np.sum(ref_func(posterior_samples, ref_val)) / size

    if not pretty_print:
        return prob
    else:
        if ref_func is np.greater:
            cp = "<"
        else:
            # the printout is misleading if we use np.less
            cp = "|"
        left = f"{(1 - prob)*100.0:.1f}%"
        right = f"{(prob)*100.0:.1f}%"
        center = f"{ref_val:.2f}" if isinstance(ref_val, float) else f"{ref_val}"

        return f"{left} {cp} {center} {cp} {right}"
