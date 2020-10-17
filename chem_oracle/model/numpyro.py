import logging
import os
from itertools import permutations
from typing import Dict

use_cpu = "ORACLE_USECPU" in os.environ

if not use_cpu:
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import ops
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro import deterministic, sample
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import Predictive, log_likelihood
from numpyro.util import set_platform

from ..util import indices
from .common import differential_disruptions, disruptions

if not use_cpu:
    # force GPU
    set_platform("gpu")

if jax.devices()[0].__class__.__name__ != "GpuDevice":
    devs = jax.devices()
    xla_flags = os.environ["XLA_FLAGS"]
    logging.warning(f"Not running on GPU!\nDevices: {devs}\nXLA_FLAGS={xla_flags}")

SAMPLED_RVS = [
    "mem_beta",
    "reactivities_norm",
]


def tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor):
    # 1 - (probability that any combination of two reagents react independently)
    tri_doesnt_react_binary = (
        jnp.prod(
            1
            - (
                M1[:, :, jnp.newaxis]
                * M2[:, jnp.newaxis, :]
                * react_matrix[jnp.newaxis, ...]
            ),
            axis=[1, 2],
        )
        * jnp.prod(
            1
            - (
                M1[:, :, jnp.newaxis]
                * M3[:, jnp.newaxis, :]
                * react_matrix[jnp.newaxis, ...]
            ),
            axis=[1, 2],
        )
        * jnp.prod(
            1
            - (
                M2[:, :, jnp.newaxis]
                * M3[:, jnp.newaxis, :]
                * react_matrix[jnp.newaxis, ...]
            ),
            axis=[1, 2],
        )
    )
    # 1 - (probability that genuine three-component reaction occurs)
    tri_doesnt_react_ternary = jnp.prod(
        1
        - (
            M1[:, :, jnp.newaxis, jnp.newaxis]
            * M2[:, jnp.newaxis, :, jnp.newaxis]
            * M3[:, jnp.newaxis, jnp.newaxis, :]
            * react_tensor[np.newaxis, ...]
        ),
        axis=[1, 2, 3],
    )

    return tri_doesnt_react_binary * tri_doesnt_react_ternary


def triangular_indices(N, ndims, shift=0):
    """
    N: is the number of properties len(v) =  N*(N-1)*...*(N-(ndim-1))/ndim!
    ndim: number of tensor dimensions
    """
    idx = indices(N, ndims)
    t = np.zeros(tuple(N for _ in range(ndims)), dtype="int")
    for i, ind in enumerate(idx):
        for perm in permutations(ind):
            t[perm] = i + shift + 1
    return t


def stick_breaking(beta, normalize=False):
    t1 = jnp.ones((*beta.shape[:-1], 1))
    t2 = jnp.cumprod(1 - beta, axis=-1)[..., :-1]
    portion_remaining = jnp.concatenate([t1, t2], axis=-1)
    res = beta * portion_remaining
    if normalize:
        return res / res.max(axis=1, keepdims=True)
    return res


class Model:
    def __init__(
        self, N_props: int, observe: bool, likelihood_sd: float,
    ):
        self.N_props = N_props
        self.observe = observe
        self.likelihood_sd = likelihood_sd

        self.N_bin = self.N_props * (self.N_props - 1) // 2
        self.N_tri = self.N_props * (self.N_props - 1) * (self.N_props - 2) // 6
        self.bi_idx = triangular_indices(self.N_props, 2)

        self.tri_idx = triangular_indices(self.N_props, 3, shift=self.N_bin)

    def sample(
        self,
        facts: pd.DataFrame,
        draws=500,
        tune=500,
        model_params=None,
        **sampler_params,
    ) -> Dict:
        nuts_kernel = NUTS(self._pyro_model)
        mcmc = MCMC(nuts_kernel, num_samples=draws, num_warmup=tune, **sampler_params)
        rng_key = PRNGKey(0)
        mcmc.run(
            rng_key, facts, *(model_params or []), extra_fields=("potential_energy",),
        )
        self.trace = {**mcmc.get_samples(), **mcmc.get_extra_fields()}
        # convert trace data to plain old numpy arrays
        self.trace = {k: np.array(v) for k, v in self.trace.items()}
        return self.trace

    def predict(
        self,
        facts: pd.DataFrame,
        knowledge_trace: Dict,
        draws=500,
        model_params=None,
        **sampler_params,
    ):
        # numpyro chokes on non-sampled vars
        sampled_vars_trace = {k: knowledge_trace[k] for k in SAMPLED_RVS}
        predictive = Predictive(self._pyro_model, sampled_vars_trace, num_samples=draws)
        self.trace = predictive(
            PRNGKey(0), facts, *(model_params or []), **sampler_params
        )
        self.trace = {k: np.array(v) for k, v in self.trace.items()}
        return self.trace

    def log_likelihoods(self, facts: pd.DataFrame, trace: Dict = None) -> Dict:
        trace = trace or self.trace
        trace = {
            k: v for k, v in trace.items() if k in SAMPLED_RVS
        }  # only keep sampled variables
        return log_likelihood(
            self._pyro_model, trace, facts, False  # do not impute - important!
        )

    def experiment_likelihoods(self, facts: pd.DataFrame, trace=None):
        trace = trace or self.trace
        likelihoods = self.log_likelihoods(facts, trace)
        bins = likelihoods["reacts_binary_NMR"]
        tris = likelihoods["reacts_ternary_NMR"]
        result = []
        bin_idx = 0
        tri_idx = 0
        for _, (_, _, _, c3) in enumerate(
            facts[["compound1", "compound2", "compound3"]].itertuples()
        ):
            if c3 == -1:
                # binary reaction
                result.append(bins[:, bin_idx])
                bin_idx += 1
            else:
                result.append(tris[:, tri_idx])
                tri_idx += 1
        return np.stack(result).T

    def condition(
        self,
        facts: pd.DataFrame,
        method_name: str = "NMR",
        differential: bool = True,
        trace=None,
        predict=True,
        **disruption_params,
    ) -> pd.DataFrame:
        """
        predict: Draw from posterior predictive rather than using `trace` directly.
            This is useful if conditioning is done on a chemical space different than
            one used during sampling.
        """
        # calculate reactivity for binary reactions
        trace = trace or self.trace
        if predict:
            prediction = self.predict(facts, trace)
        else:
            prediction = trace

        bin_avg = 1 - np.mean(prediction["bin_doesnt_react"], axis=0)
        bin_std = np.std(prediction["bin_doesnt_react"], axis=0)
        # calculate reactivity for three component reactions
        tri_avg = 1 - np.mean(prediction["tri_doesnt_react"], axis=0)
        tri_std = np.std(prediction["tri_doesnt_react"], axis=0)

        new_facts = facts.copy()
        # remove old disruption values
        new_facts.loc[:, ["reactivity_disruption", "uncertainty_disruption"]] = np.nan
        # update dataframe with calculated reactivities
        new_facts.loc[
            new_facts["compound3"] == -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack([bin_avg, bin_std]).T
        new_facts.loc[
            new_facts["compound3"] != -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack([tri_avg, tri_std]).T

        bin_missings = (new_facts["compound3"] == -1) & pd.isna(
            new_facts[f"{method_name}_reactivity"]
        )
        tri_missings = (new_facts["compound3"] != -1) & pd.isna(
            new_facts[f"{method_name}_reactivity"]
        )

        new_facts["likelihood"] = self.experiment_likelihoods(facts, trace).mean(axis=0)
        new_facts["likelihood_sd"] = self.experiment_likelihoods(facts, trace).std(
            axis=0
        )

        n_bin = bin_missings.sum()

        if differential:
            r, u = differential_disruptions(
                new_facts, prediction, method_name, **disruption_params
            )
        else:
            r, u = disruptions(new_facts, prediction, method_name)

        new_facts.loc[bin_missings, ["reactivity_disruption"]] = r[bin_missings]
        new_facts.loc[bin_missings, ["uncertainty_disruption"]] = u[bin_missings]
        new_facts.loc[tri_missings, ["reactivity_disruption"]] = r[tri_missings]
        new_facts.loc[tri_missings, ["uncertainty_disruption"]] = u[tri_missings]

        return new_facts


class NonstructuralModel(Model):
    def __init__(self, ncompounds, N_props=8, observe=True, likelihood_sd=0.25):
        super().__init__(
            N_props=N_props, observe=observe, likelihood_sd=likelihood_sd,
        )
        self.ncompounds = ncompounds

    def _pyro_model(self, facts, impute=True):
        N_bin = self.N_props * (self.N_props - 1) // 2
        N_tri = self.N_props * (self.N_props - 1) * (self.N_props - 2) // 6
        bin_facts = facts[facts["compound3"] == -1]
        bin_r1 = bin_facts.compound1.values
        bin_r2 = bin_facts.compound2.values
        bin_NMR = bin_facts.NMR_reactivity.values
        bin_HPLC = bin_facts.HPLC_reactivity.values
        bin_missing_NMR = jnp.isnan(bin_NMR).nonzero()[0]
        bin_missing_HPLC = jnp.isnan(bin_HPLC).nonzero()[0]
        impute_bin_HPLC = impute and bin_missing_HPLC.any()
        impute_bin_NMR = impute and bin_missing_NMR.any()

        tri_facts = facts[facts["compound3"] != -1]
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values
        tri_NMR = tri_facts.NMR_reactivity.values
        tri_HPLC = tri_facts.HPLC_reactivity.values
        tri_missing_NMR = jnp.isnan(tri_NMR).nonzero()[0]
        tri_missing_HPLC = jnp.isnan(tri_HPLC).nonzero()[0]
        impute_tri_HPLC = impute and tri_missing_HPLC.any()
        impute_tri_NMR = impute and tri_missing_NMR.any()

        bi_idx = jnp.array(triangular_indices(self.N_props, 2))
        tri_idx = jnp.array(triangular_indices(self.N_props, 3, shift=N_bin))

        mem_beta = sample(
            "mem_beta",
            dist.Beta(0.9, 0.9),
            sample_shape=(self.ncompounds, self.N_props + 1),
        )
        # the first property is non-reactive, so ignore that
        mem = deterministic("mem", stick_breaking(mem_beta, normalize=True)[..., 1:])

        reactivities_norm = sample(
            "reactivities_norm", dist.Beta(1.0, 3.0), sample_shape=(N_bin + N_tri,),
        )

        # add zero entry for self-reactivity for each reactivity mode
        reactivities_with_zero = jnp.concatenate([jnp.zeros((1,)), reactivities_norm],)

        react_matrix = deterministic("react_matrix", reactivities_with_zero[bi_idx],)
        react_tensor = reactivities_with_zero[tri_idx]

        m1, m2 = mem[bin_r1, :][:, :, np.newaxis], mem[bin_r2, :][:, np.newaxis, :]
        M1, M2, M3 = mem[tri_r1, :], mem[tri_r2, :], mem[tri_r3, :]

        bin_doesnt_react = deterministic(
            "bin_doesnt_react",
            jnp.prod(1 - (m1 * m2) * react_matrix[np.newaxis, ...], axis=[1, 2],),
        )

        tri_no_react = deterministic(
            "tri_doesnt_react",
            tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor),
        )

        if impute_bin_HPLC:
            hplc_impute_binary = sample(
                "reacts_binary_HPLC_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )
            bin_HPLC = ops.index_update(bin_HPLC, bin_missing_HPLC, hplc_impute_binary)

        if impute_tri_HPLC:
            hplc_impute_ternary = sample(
                "reacts_ternary_HPLC_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )
            tri_HPLC = ops.index_update(tri_HPLC, tri_missing_HPLC, hplc_impute_ternary)

        if impute_bin_NMR:
            nmr_impute_binary = sample(
                "reacts_binary_NMR_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )
            bin_NMR = ops.index_update(bin_NMR, bin_missing_NMR, nmr_impute_binary)

        if impute_tri_NMR:
            nmr_impute_ternary = sample(
                "reacts_ternary_NMR_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )
            tri_NMR = ops.index_update(tri_NMR, tri_missing_NMR, nmr_impute_ternary)

        nmr_obs_binary = sample(
            "reacts_binary_NMR",
            dist.Normal(loc=1 - bin_doesnt_react, scale=self.likelihood_sd),
            obs=bin_NMR,
        )

        nmr_obs_ternary = sample(
            "reacts_ternary_NMR",
            dist.Normal(loc=1 - tri_no_react, scale=self.likelihood_sd),
            obs=tri_NMR,
        )

        hplc_obs_binary = sample(
            "reacts_binary_HPLC",
            dist.Normal(loc=1 - bin_doesnt_react, scale=self.likelihood_sd),
            obs=bin_HPLC,
        )

        hplc_obs_ternary = sample(
            "reacts_ternary_HPLC",
            dist.Normal(loc=1 - tri_no_react, scale=self.likelihood_sd),
            obs=tri_HPLC,
        )


class StructuralModel(Model):
    def __init__(
        self,
        fingerprint_matrix: np.ndarray,
        N_props=8,
        observe=True,
        likelihood_sd=0.25,
    ):
        """Bayesian reactivity model informed by structural fingerprints.
        TODO: Update docs

        Args:
            fingerprint_matrix (n_compounds × fingerprint_length matrix):
                a numpy matrix row i of which contains the fingerprint bits
                for the i-th compound.
            N_props (int, optional): Number of abstract properties. Defaults to 4.
        """
        # """fingerprints: Matrix of Morgan fingerprints for reagents."""

        super().__init__(
            N_props=N_props, observe=observe, likelihood_sd=likelihood_sd,
        )

        self.fingerprints = fingerprint_matrix
        self.ncompounds, self.fingerprint_length = fingerprint_matrix.shape

    def _pyro_model(self, facts, impute=True):
        N_bin = self.N_props * (self.N_props - 1) // 2
        N_tri = self.N_props * (self.N_props - 1) * (self.N_props - 2) // 6
        bin_facts = facts[facts["compound3"] == -1]
        bin_r1 = bin_facts.compound1.values
        bin_r2 = bin_facts.compound2.values
        bin_NMR = bin_facts.NMR_reactivity.values
        bin_HPLC = bin_facts.HPLC_reactivity.values
        bin_missing_HPLC = jnp.isnan(bin_HPLC).nonzero()[0]
        bin_missing_NMR = jnp.isnan(bin_NMR).nonzero()[0]
        impute_bin_HPLC = impute and bin_missing_HPLC.any()
        impute_bin_NMR = impute and bin_missing_NMR.any()

        tri_facts = facts[facts["compound3"] != -1]
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values
        tri_NMR = tri_facts.NMR_reactivity.values
        tri_HPLC = tri_facts.HPLC_reactivity.values
        tri_missing_HPLC = jnp.isnan(tri_HPLC).nonzero()[0]
        tri_missing_NMR = jnp.isnan(tri_NMR).nonzero()[0]
        impute_tri_HPLC = impute and tri_missing_HPLC.any()
        impute_tri_NMR = impute and tri_missing_NMR.any()

        bi_idx = jnp.array(triangular_indices(self.N_props, 2))
        tri_idx = jnp.array(triangular_indices(self.N_props, 3, shift=N_bin))

        mem_beta = sample(
            "mem_beta",
            dist.Beta(0.9, 0.9),
            sample_shape=(self.fingerprint_length, self.N_props + 1),
        )
        # the first property is non-reactive, so ignore that
        mem = deterministic("mem", stick_breaking(mem_beta, normalize=True)[..., 1:])

        reactivities_norm = sample(
            "reactivities_norm", dist.Beta(1.0, 3.0), sample_shape=(N_bin + N_tri,),
        )

        # add zero entry for self-reactivity for each reactivity mode
        reactivities_with_zero = jnp.concatenate([jnp.zeros((1,)), reactivities_norm],)

        react_matrix = deterministic("react_matrix", reactivities_with_zero[bi_idx],)
        react_tensor = reactivities_with_zero[tri_idx]

        # Convert fingerprint memberships to molecule memberships
        fp1, fp2 = (
            self.fingerprints[bin_r1, :, jnp.newaxis],
            self.fingerprints[bin_r2, :, jnp.newaxis],
        )
        m1, m2 = (
            jnp.max(fp1 * mem, axis=1)[:, :, np.newaxis],
            jnp.max(fp2 * mem, axis=1)[:, np.newaxis, :],
        )
        FP1, FP2, FP3 = (
            self.fingerprints[tri_r1, :, jnp.newaxis],
            self.fingerprints[tri_r2, :, jnp.newaxis],
            self.fingerprints[tri_r3, :, jnp.newaxis],
        )
        M1, M2, M3 = (
            jnp.max(FP1 * mem, axis=1),
            jnp.max(FP2 * mem, axis=1),
            jnp.max(FP3 * mem, axis=1),
        )

        bin_doesnt_react = deterministic(
            "bin_doesnt_react",
            jnp.prod(1 - (m1 * m2) * react_matrix[np.newaxis, ...], axis=[1, 2],),
        )

        tri_no_react = deterministic(
            "tri_doesnt_react",
            tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor),
        )

        if impute_bin_HPLC:
            hplc_impute_binary = sample(
                "reacts_binary_HPLC_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )
            bin_HPLC = ops.index_update(bin_HPLC, bin_missing_HPLC, hplc_impute_binary)

        if impute_tri_HPLC:
            hplc_impute_ternary = sample(
                "reacts_ternary_HPLC_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )
            tri_HPLC = ops.index_update(tri_HPLC, tri_missing_HPLC, hplc_impute_ternary)

        if impute_bin_NMR:
            nmr_impute_binary = sample(
                "reacts_binary_NMR_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )
            bin_NMR = ops.index_update(bin_NMR, bin_missing_NMR, nmr_impute_binary)

        if impute_tri_NMR:
            nmr_impute_ternary = sample(
                "reacts_ternary_NMR_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )
            tri_NMR = ops.index_update(tri_NMR, tri_missing_NMR, nmr_impute_ternary)

        nmr_obs_binary = sample(
            "reacts_binary_NMR",
            dist.Normal(loc=1 - bin_doesnt_react, scale=self.likelihood_sd),
            obs=bin_NMR,
        )

        nmr_obs_ternary = sample(
            "reacts_ternary_NMR",
            dist.Normal(loc=1 - tri_no_react, scale=self.likelihood_sd),
            obs=tri_NMR,
        )

        hplc_obs_binary = sample(
            "reacts_binary_HPLC",
            dist.Normal(loc=1 - bin_doesnt_react, scale=self.likelihood_sd),
            obs=bin_HPLC,
        )

        hplc_obs_ternary = sample(
            "reacts_ternary_HPLC",
            dist.Normal(loc=1 - tri_no_react, scale=self.likelihood_sd),
            obs=tri_HPLC,
        )
