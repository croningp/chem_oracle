import logging
import math
import os
from itertools import combinations, permutations
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import ops
from jax.random import PRNGKey
from matplotlib import pyplot as plt

import numpyro.distributions as dist
from numpyro import deterministic, plate, sample
from numpyro.infer import MCMC, NUTS
from numpyro.util import set_platform

from ..util import indices

# force GPU
set_platform("gpu")

if jax.devices()[0].__class__.__name__ != "GpuDevice":
    devs = jax.devices()
    xla_flags = os.environ["XLA_FLAGS"]
    logging.warn(f"Not running on GPU!\nDevices: {devs}\nXLA_FLAGS={xla_flags}")


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
        self, facts: pd.DataFrame, draws=1000, tune=500, **sampler_params
    ) -> Dict:
        nuts_kernel = NUTS(self._pyro_model)
        mcmc = MCMC(nuts_kernel, num_samples=draws, num_warmup=tune, **sampler_params)
        rng_key = PRNGKey(0)
        run = mcmc.run(rng_key, facts, extra_fields=("potential_energy",),)
        self.trace = mcmc.get_samples()
        return self.trace

    def condition(
        self,
        facts: pd.DataFrame,
        method_name: str = "MS",
        differential: bool = True,
        **disruption_params,
    ) -> pd.DataFrame:
        # calculate reactivity for binary reactions
        bin_avg = 1 - np.mean(self.trace["bin_doesnt_react"], axis=0)
        bin_std = np.std(self.trace["bin_doesnt_react"], axis=0)
        # calculate reactivity for three component reactions
        tri_avg = 1 - np.mean(self.trace["tri_doesnt_react"], axis=0)
        tri_std = np.std(self.trace["tri_doesnt_react"], axis=0)

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
        n_bin = bin_missings.sum()
        if differential:
            r, u = differential_disruptions(
                new_facts, self.trace, method_name, **disruption_params
            )
        else:
            r, u = disruptions(new_facts, self.trace, method_name)
        new_facts.loc[bin_missings, ["reactivity_disruption"]] = r[:n_bin]
        new_facts.loc[bin_missings, ["uncertainty_disruption"]] = u[:n_bin]
        new_facts.loc[tri_missings, ["reactivity_disruption"]] = r[n_bin:]
        new_facts.loc[tri_missings, ["uncertainty_disruption"]] = u[n_bin:]
        return new_facts


class NonstructuralModel(Model):
    def __init__(
        self, ncompounds, N_props=8, observe=True, likelihood_sd=0.25,
    ):
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

        tri_facts = facts[facts["compound3"] != -1]
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values
        tri_NMR = tri_facts.NMR_reactivity.values
        tri_HPLC = tri_facts.HPLC_reactivity.values
        tri_missing_NMR = jnp.isnan(tri_NMR).nonzero()[0]
        tri_missing_HPLC = jnp.isnan(tri_HPLC).nonzero()[0]

        if not (
            bin_missing_HPLC.any()
            or bin_missing_NMR.any()
            or tri_missing_HPLC.any()
            or tri_missing_NMR.any()
        ):
            logging.debug("Nothing to impute, setting impute to False")
            impute = False

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

        if impute:
            hplc_impute_binary = sample(
                "reacts_binary_HPLC_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )

            hplc_impute_ternary = sample(
                "reacts_ternary_HPLC_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )

            nmr_impute_binary = sample(
                "reacts_binary_NMR_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )

            nmr_impute_ternary = sample(
                "reacts_ternary_NMR_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )

            bin_NMR = ops.index_update(bin_NMR, bin_missing_NMR, nmr_impute_binary)
            tri_NMR = ops.index_update(tri_NMR, tri_missing_NMR, nmr_impute_ternary)
            bin_HPLC = ops.index_update(bin_HPLC, bin_missing_HPLC, hplc_impute_binary)
            tri_HPLC = ops.index_update(tri_HPLC, tri_missing_HPLC, hplc_impute_ternary)

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
        bin_missing_NMR = jnp.isnan(bin_NMR).nonzero()[0]
        bin_missing_HPLC = jnp.isnan(bin_HPLC).nonzero()[0]

        tri_facts = facts[facts["compound3"] != -1]
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values
        tri_NMR = tri_facts.NMR_reactivity.values
        tri_HPLC = tri_facts.HPLC_reactivity.values
        tri_missing_NMR = jnp.isnan(tri_NMR).nonzero()[0]
        tri_missing_HPLC = jnp.isnan(tri_HPLC).nonzero()[0]

        if not (
            bin_missing_HPLC.any()
            or bin_missing_NMR.any()
            or tri_missing_HPLC.any()
            or tri_missing_NMR.any()
        ):
            logging.debug("Nothing to impute, setting impute to False")
            impute = False

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

        if impute:
            hplc_impute_binary = sample(
                "reacts_binary_HPLC_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )

            hplc_impute_ternary = sample(
                "reacts_ternary_HPLC_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_HPLC], scale=self.likelihood_sd
                ).mask(False),
            )

            nmr_impute_binary = sample(
                "reacts_binary_NMR_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )

            nmr_impute_ternary = sample(
                "reacts_ternary_NMR_missing",
                dist.Normal(
                    loc=1 - tri_no_react[tri_missing_NMR], scale=self.likelihood_sd
                ).mask(False),
            )

            bin_NMR = ops.index_update(bin_NMR, bin_missing_NMR, nmr_impute_binary)
            tri_NMR = ops.index_update(tri_NMR, tri_missing_NMR, nmr_impute_ternary)
            bin_HPLC = ops.index_update(bin_HPLC, bin_missing_HPLC, hplc_impute_binary)
            tri_HPLC = ops.index_update(tri_HPLC, tri_missing_HPLC, hplc_impute_ternary)

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
