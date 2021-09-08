import pdb
import os
from itertools import permutations
from typing import Dict

use_cpu = "ORACLE_USECPU" in os.environ

if not use_cpu and "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"

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

SAMPLED_RVS = [
    "mem_beta",
    "reactivities_norm",
]


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
        self,
        N_props: int,
        observe: bool,
        likelihood_sd: float,
        mem_beta_a: float,
        mem_beta_b: float,
        react_beta_a: float,
        react_beta_b: float,
    ):
        self.N_props = N_props
        self.observe = observe
        self.likelihood_sd = likelihood_sd
        self.mem_beta_a = mem_beta_a
        self.mem_beta_b = mem_beta_b
        self.react_beta_a = react_beta_a
        self.react_beta_b = react_beta_b

        self.N_bin = self.N_props * (self.N_props - 1) // 2
        self.N_tri = self.N_bin * (self.N_props - 2) // 3
        self.N_tet = self.N_tri * (self.N_props - 3) // 4

        self.bin_reactivity_idx = triangular_indices(self.N_props, 2)
        self.tri_reactivity_idx = triangular_indices(self.N_props, 3, shift=self.N_bin)
        self.tet_reactivity_idx = triangular_indices(
            self.N_props, 4, shift=self.N_bin + self.N_tri
        )

    def sample(
        self,
        facts: pd.DataFrame,
        draws=500,
        tune=500,
        model_params=None,
        rng_seed=0,
        **sampler_params,
    ) -> Dict:
        nuts_kernel = NUTS(self._pyro_model)
        mcmc = MCMC(nuts_kernel, num_samples=draws, num_warmup=tune, **sampler_params)
        rng_key = PRNGKey(rng_seed)
        mcmc.run(
            rng_key,
            facts,
            *(model_params or []),
            extra_fields=("potential_energy",),
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
        prediction = predictive(
            PRNGKey(0), facts, *(model_params or []), **sampler_params
        )
        # convert jax => numpy arrays
        prediction = {k: np.array(v) for k, v in prediction.items()}
        prediction.update(sampled_vars_trace)
        return prediction

    def log_likelihoods(self, facts: pd.DataFrame, trace: Dict = None) -> Dict:
        trace = trace or self.trace
        trace = {
            k: v for k, v in trace.items() if k in SAMPLED_RVS
        }  # only keep sampled variables
        return log_likelihood(
            self._pyro_model, trace, facts, False  # do not impute - important!
        )

    def experiment_likelihoods(self, facts: pd.DataFrame, trace=None, method="NMR"):
        trace = trace or self.trace
        likelihoods = self.log_likelihoods(facts, trace)
        bins = likelihoods[f"reacts_binary_{method}"]
        tris = likelihoods[f"reacts_ternary_{method}"]
        quats = likelihoods[f"reacts_quaternary_{method}"]
        assert bins.shape[0] == tris.shape[0] == quats.shape[0]
        assert len(facts) == bins.shape[1] + tris.shape[1] + quats.shape[1]

        result = np.zeros((bins.shape[0], len(facts)))
        result[:, facts.query("compound3==-1").index] = bins
        result[:, facts.query("compound3!=-1 and compound4==-1").index] = tris
        result[:, facts.query("compound4!=-1").index] = quats

        return result

    def condition(
        self,
        facts: pd.DataFrame,
        method_name: str = "NMR",
        differential: bool = True,
        trace=None,
        predict=False,
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

        # calculate reactivity for reactions
        bin_avg = 1 - np.mean(prediction["bin_doesnt_react"], axis=0)
        bin_std = np.std(prediction["bin_doesnt_react"], axis=0)
        tri_avg = 1 - np.mean(prediction["tri_doesnt_react"], axis=0)
        tri_std = np.std(prediction["tri_doesnt_react"], axis=0)
        tet_avg = 1 - np.mean(prediction["tet_doesnt_react"], axis=0)
        tet_std = np.std(prediction["tet_doesnt_react"], axis=0)

        new_facts = facts.copy()
        # remove old disruption values
        new_facts.loc[:, ["reactivity_disruption", "uncertainty_disruption"]] = np.nan

        # update dataframe with calculated reactivities
        new_facts.loc[
            new_facts["compound3"] == -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack([bin_avg, bin_std]).T
        new_facts.loc[
            (new_facts["compound3"] != -1) & (new_facts["compound4"] == -1),
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack([tri_avg, tri_std]).T
        new_facts.loc[
            new_facts["compound4"] != -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack([tet_avg, tet_std]).T

        bin_missings = (new_facts["compound3"] == -1) & pd.isna(
            new_facts[f"{method_name}_reactivity"]
        )
        tri_missings = (
            (new_facts["compound3"] != -1)
            & (new_facts["compound4"] == -1)
            & pd.isna(new_facts[f"{method_name}_reactivity"])
        )
        tet_missings = (new_facts["compound4"] != -1) & pd.isna(
            new_facts[f"{method_name}_reactivity"]
        )

        likelihoods = self.experiment_likelihoods(facts, prediction)

        new_facts["likelihood"] = likelihoods.mean(axis=0)
        new_facts["likelihood_sd"] = likelihoods.std(axis=0)

        n_bin = bin_missings.sum()
        n_tri = tri_missings.sum()

        if differential:
            r, u = differential_disruptions(
                new_facts, prediction, method_name, **disruption_params
            )
            new_facts.loc[bin_missings, ["reactivity_disruption"]] = r[bin_missings]
            new_facts.loc[bin_missings, ["uncertainty_disruption"]] = u[bin_missings]
            new_facts.loc[tri_missings, ["reactivity_disruption"]] = r[tri_missings]
            new_facts.loc[tri_missings, ["uncertainty_disruption"]] = u[tri_missings]
            new_facts.loc[tet_missings, ["reactivity_disruption"]] = r[tet_missings]
            new_facts.loc[tet_missings, ["uncertainty_disruption"]] = u[tet_missings]
        else:
            r, u = disruptions(new_facts, prediction, method_name)
            new_facts.loc[bin_missings, ["reactivity_disruption"]] = r[:n_bin]
            new_facts.loc[bin_missings, ["uncertainty_disruption"]] = u[:n_bin]
            new_facts.loc[tri_missings, ["reactivity_disruption"]] = r[
                n_bin : (n_bin + n_tri)
            ]
            new_facts.loc[tri_missings, ["uncertainty_disruption"]] = u[
                n_bin : (n_bin + n_tri)
            ]
            new_facts.loc[tet_missings, ["reactivity_disruption"]] = r[
                (n_bin + n_tri) :
            ]
            new_facts.loc[tet_missings, ["uncertainty_disruption"]] = u[
                (n_bin + n_tri) :
            ]

        return new_facts


class NonstructuralModel(Model):
    def __init__(
        self,
        ncompounds,
        N_props=8,
        observe=True,
        likelihood_sd=0.25,
        mem_beta_a=0.9,
        mem_beta_b=0.9,
        react_beta_a=1.0,
        react_beta_b=3.0,
    ):
        super().__init__(
            N_props=N_props,
            observe=observe,
            likelihood_sd=likelihood_sd,
            mem_beta_a=mem_beta_a,
            mem_beta_b=mem_beta_b,
            react_beta_a=react_beta_a,
            react_beta_b=react_beta_b,
        )
        self.ncompounds = ncompounds

    def _pyro_model(self, facts: pd.DataFrame, impute=True):
        observation_columns = [col for col in facts.columns if col.startswith("event")]
        N_event = len(observation_columns)

        N_bin = self.N_props * (self.N_props - 1) // 2
        N_tri = N_bin * (self.N_props - 2) // 3
        N_tet = N_tri * (self.N_props - 3) // 4
        bin_facts = facts[facts["compound3"] == -1]
        bin_r1 = bin_facts.compound1.values
        bin_r2 = bin_facts.compound2.values
        bin_obs = bin_facts[observation_columns].values
        bin_missing_obs = np.isnan(bin_obs).nonzero()
        impute_bin_obs = impute and bin_missing_obs[0].size > 0

        tri_facts = facts.query("compound3 != -1 and compound4 == -1")
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values
        tri_obs = tri_facts[observation_columns].values
        tri_missing_obs = np.isnan(tri_obs).nonzero()
        impute_tri_obs = impute and tri_missing_obs[0].size > 0

        tet_facts = facts[facts["compound4"] != -1]
        tet_r1 = tet_facts.compound1.values
        tet_r2 = tet_facts.compound2.values
        tet_r3 = tet_facts.compound3.values
        tet_r4 = tet_facts.compound4.values
        tet_obs = tet_facts[observation_columns].values
        tet_missing_obs = np.isnan(tet_obs).nonzero()
        impute_tet_obs = impute and tet_missing_obs[0].size > 0

        mem_beta = sample(
            "mem_beta",
            dist.Beta(self.mem_beta_a, self.mem_beta_b),
            sample_shape=(self.ncompounds, self.N_props + 1),
        )
        # the first property is non-reactive, so ignore that
        mem = deterministic(
            "mem", stick_breaking(mem_beta, normalize=True)[..., 1:]
        )  # ncompounds x N_props

        reactivities_norm = sample(
            "reactivities_norm",
            dist.Beta(self.react_beta_a, self.react_beta_b),
            sample_shape=(N_bin + N_tri + N_tet, N_event),
        )

        # add zero entry for self-reactivity for each reactivity mode
        reactivities_with_zero = jnp.concatenate(
            [jnp.zeros((1, N_event)), reactivities_norm],
        )

        react_tensors = [
            deterministic(f"react_tensor_rank{i+2}", reactivities_with_zero[idx, :])
            for i, idx in enumerate(
                [
                    self.bin_reactivity_idx,
                    self.tri_reactivity_idx,
                    self.tet_reactivity_idx,
                ]
            )
        ]

        bin_doesnt_react = deterministic(
            "bin_doesnt_react",
            jnp.prod(
                1
                - mem[bin_r1][:, :, np.newaxis, np.newaxis]
                * mem[bin_r2][:, np.newaxis, :, np.newaxis]
                * react_tensors[0][np.newaxis, :, :, :],
                axis=[1, 2],
            ),
        )

        tri_doesnt_react = deterministic(
            "tri_doesnt_react",
            (
                jnp.prod(
                    1
                    - mem[tri_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tri_r2][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tri_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tri_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tri_r2][:, :, np.newaxis, np.newaxis]
                    * mem[tri_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tri_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tri_r2][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tri_r3][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
            ),
        )

        tet_doesnt_react = deterministic(
            "tet_doesnt_react",
            (
                jnp.prod(
                    1
                    - mem[tet_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r2][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r2][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r2][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r3][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r2][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r2][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r2][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[2][np.newaxis, :, :, :, :, :],
                    axis=[1, 2, 3, 4],
                )
            ),
        )

        if impute_bin_obs:
            obs_impute_binary = sample(
                "reacts_binary_obs_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_obs], scale=self.likelihood_sd
                ).mask(False),
            )
            bin_obs = ops.index_update(
                bin_obs, bin_missing_obs, obs_impute_binary.clip(0.0, 1.0)
            )

        if impute_tri_obs:
            obs_impute_ternary = sample(
                "reacts_ternary_obs_missing",
                dist.Normal(
                    loc=1 - tri_doesnt_react[tri_missing_obs], scale=self.likelihood_sd
                ).mask(False),
            )
            tri_obs = ops.index_update(
                tri_obs, tri_missing_obs, obs_impute_ternary.clip(0.0, 1.0)
            )

        if impute_tet_obs:
            obs_impute_quaternary = sample(
                "reacts_quaternary_obs_missing",
                dist.Normal(
                    loc=1 - tet_doesnt_react[tet_missing_obs], scale=self.likelihood_sd
                ).mask(False),
            )
            tet_obs = ops.index_update(
                tet_obs, tet_missing_obs, obs_impute_quaternary.clip(0.0, 1.0)
            )


        event_obs_binary = sample(
            "reacts_binary_obs",
            dist.Normal(loc=1 - bin_doesnt_react, scale=self.likelihood_sd),
            obs=bin_obs,
        )

        event_obs_ternary = sample(
            "reacts_ternary_obs",
            dist.Normal(loc=1 - tri_doesnt_react, scale=self.likelihood_sd),
            obs=tri_obs,
        )

        event_obs_quaternary = sample(
            "reacts_quaternary_obs",
            dist.Normal(loc=1 - tet_doesnt_react, scale=self.likelihood_sd),
            obs=tet_obs,
        )


class StructuralModel(Model):
    def __init__(
        self,
        fingerprint_matrix: np.ndarray,
        N_props=8,
        observe=True,
        likelihood_sd=0.25,
        mem_beta_a=0.9,
        mem_beta_b=1.5,
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
            N_props=N_props,
            observe=observe,
            likelihood_sd=likelihood_sd,
            mem_beta_a=mem_beta_a,
            mem_beta_b=mem_beta_b,
        )

        self.fingerprints = fingerprint_matrix
        self.ncompounds, self.fingerprint_length = fingerprint_matrix.shape

    def _pyro_model(self, facts, impute=True):
        observation_columns = [col for col in facts.columns if col.startswith("event")]
        N_event = len(observation_columns)

        N_bin = self.N_props * (self.N_props - 1) // 2
        N_tri = N_bin * (self.N_props - 2) // 3
        N_tet = N_tri * (self.N_props - 3) // 4
        bin_facts = facts[facts["compound3"] == -1]
        bin_r1 = bin_facts.compound1.values
        bin_r2 = bin_facts.compound2.values
        bin_obs = bin_facts[observation_columns].values
        bin_missing_obs = np.isnan(bin_obs).nonzero()
        impute_bin_obs = impute and bin_missing_obs[0].size > 0

        tri_facts = facts.query("compound3 != -1 and compound4 == -1")
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values
        tri_obs = tri_facts[observation_columns].values
        tri_missing_obs = np.isnan(tri_obs).nonzero()
        impute_tri_obs = impute and tri_missing_obs[0].size > 0

        tet_facts = facts[facts["compound4"] != -1]
        tet_r1 = tet_facts.compound1.values
        tet_r2 = tet_facts.compound2.values
        tet_r3 = tet_facts.compound3.values
        tet_r4 = tet_facts.compound4.values
        tet_obs = tet_facts[observation_columns].values
        tet_missing_obs = np.isnan(tet_obs).nonzero()
        impute_tet_obs = impute and tet_missing_obs[0].size > 0

        mem_beta = sample(
            "mem_beta",
            dist.Beta(self.mem_beta_a, self.mem_beta_b),
            sample_shape=(self.fingerprint_length, self.N_props + 1),
        )

        # the first property is non-reactive, so ignore that
        fp_mem = deterministic(
            "fp_mem", stick_breaking(mem_beta, normalize=True)[..., 1:]
        )

        mem = deterministic(
            "mem", jnp.max(self.fingerprints[..., jnp.newaxis] * fp_mem, axis=1)
        )

        reactivities_norm = sample(
            "reactivities_norm",
            dist.Beta(self.react_beta_a, self.react_beta_b),
            sample_shape=(N_bin + N_tri + N_tet, N_event),
        )

        # add zero entry for self-reactivity for each reactivity mode
        reactivities_with_zero = jnp.concatenate(
            [jnp.zeros((1, N_event)), reactivities_norm],
        )

        react_tensors = [
            deterministic(f"react_tensor_rank{i+2}", reactivities_with_zero[idx, :])
            for i, idx in enumerate(
                [
                    self.bin_reactivity_idx,
                    self.tri_reactivity_idx,
                    self.tet_reactivity_idx,
                ]
            )
        ]

        bin_doesnt_react = deterministic(
            "bin_doesnt_react",
            jnp.prod(
                1
                - mem[bin_r1][:, :, np.newaxis, np.newaxis]
                * mem[bin_r2][:, np.newaxis, :, np.newaxis]
                * react_tensors[0][np.newaxis, :, :, :],
                axis=[1, 2],
            ),
        )

        tri_doesnt_react = deterministic(
            "tri_doesnt_react",
            (
                jnp.prod(
                    1
                    - mem[tri_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tri_r2][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tri_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tri_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tri_r2][:, :, np.newaxis, np.newaxis]
                    * mem[tri_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tri_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tri_r2][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tri_r3][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
            ),
        )

        tet_doesnt_react = deterministic(
            "tet_doesnt_react",
            (
                jnp.prod(
                    1
                    - mem[tet_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r2][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r2][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r2][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r3][:, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r2][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r2][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r2][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :, :],
                    axis=[1, 2, 3],
                )
                * jnp.prod(
                    1
                    - mem[tet_r1][:, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[tet_r3][:, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                    * mem[tet_r4][:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[2][np.newaxis, :, :, :, :, :],
                    axis=[1, 2, 3, 4],
                )
            ),
        )

        if impute_bin_obs:
            obs_impute_binary = sample(
                "reacts_binary_obs_missing",
                dist.Normal(
                    loc=1 - bin_doesnt_react[bin_missing_obs], scale=self.likelihood_sd
                ).mask(False),
            )
            bin_obs = ops.index_update(
                bin_obs, bin_missing_obs, obs_impute_binary.clip(0.0, 1.0)
            )

        if impute_tri_obs:
            obs_impute_ternary = sample(
                "reacts_ternary_obs_missing",
                dist.Normal(
                    loc=1 - tri_doesnt_react[tri_missing_obs], scale=self.likelihood_sd
                ).mask(False),
            )
            tri_obs = ops.index_update(
                tri_obs, tri_missing_obs, obs_impute_ternary.clip(0.0, 1.0)
            )

        if impute_tet_obs:
            obs_impute_quaternary = sample(
                "reacts_quaternary_obs_missing",
                dist.Normal(
                    loc=1 - tet_doesnt_react[tet_missing_obs], scale=self.likelihood_sd
                ).mask(False),
            )
            tet_obs = ops.index_update(
                tet_obs, tet_missing_obs, obs_impute_quaternary.clip(0.0, 1.0)
            )

        event_obs_binary = sample(
            "reacts_binary_obs",
            dist.Normal(loc=1 - bin_doesnt_react, scale=self.likelihood_sd),
            obs=bin_obs,
        )

        event_obs_ternary = sample(
            "reacts_ternary_obs",
            dist.Normal(loc=1 - tri_doesnt_react, scale=self.likelihood_sd),
            obs=tri_obs,
        )

        event_obs_quaternary = sample(
            "reacts_quaternary_obs",
            dist.Normal(loc=1 - tet_doesnt_react, scale=self.likelihood_sd),
            obs=tet_obs,
        )
