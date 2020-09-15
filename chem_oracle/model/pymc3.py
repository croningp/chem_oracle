import copy

import numpy as np
import pandas as pd
import theano.tensor as tt

import pymc3 as pm

from .common import disruptions, differential_disruptions
from ..util import indices, stick_breaking, tri_doesnt_react, triangular_indices


def log_likelihood(model: pm.Model, trace, var):
    # need to cache these for performance
    logp_func = model[
        var
    ].logp_elemwise  # [rv.logp_elemwise for rv in model.observed_RVs]

    point_likelihoods = np.array([logp_func(point) for point in trace.points()])
    return point_likelihoods


def log_likelihoods(model: pm.Model, trace):
    # need to cache these for performance
    rvs = model.observed_RVs

    return {rv.name: log_likelihood(model, trace, rv.name) for rv in rvs}


class Model:
    def __init__(
        self,
        N_props: int,
        N_react: int,
        dirichlet: bool,
        observe: bool,
        likelihood_sd: float,
    ):
        self.N_props = N_props
        self.N_react = N_react
        self.dirichlet = dirichlet
        self.observe = observe
        self.likelihood_sd = likelihood_sd

        self.N_bin = self.N_props * (self.N_props - 1) // 2
        self.N_tri = self.N_props * (self.N_props - 1) * (self.N_props - 2) // 6
        self.bi_idx = triangular_indices(self.N_props, 2)
        self.bi_idx.tag.test_value = np.random.randint(
            0, 1, size=(self.N_props, self.N_props)
        )
        self.tri_idx = triangular_indices(self.N_props, 3, shift=self.N_bin)
        self.tri_idx.tag.test_value = np.random.randint(
            0, 1, size=(self.N_props, self.N_props, self.N_props)
        )

    def sample(self, facts: pd.DataFrame, **sampler_params,) -> pm.sampling.MultiTrace:
        m = self._pymc3_model(facts)
        with m:
            self.trace = pm.sample(**sampler_params)
        return self.trace

    def load_trace(self, facts: pd.DataFrame, trace_dir: str):
        m = self._pymc3_model(facts)
        with m:
            self.trace = pm.load_trace(trace_dir)

    def log_likelihoods(self, facts: pd.DataFrame, trace=None):
        trace = trace or self.trace
        return log_likelihoods(self._pymc3_model(facts), trace)

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
        new_facts.loc[bin_missings, ["reactivity_disruption"]] = r[
            r.index[bin_missings][:n_bin]
        ]
        new_facts.loc[bin_missings, ["uncertainty_disruption"]] = u[
            u.index[bin_missings][:n_bin]
        ]
        new_facts.loc[tri_missings, ["reactivity_disruption"]] = r[
            r.index[tri_missings][n_bin:]
        ]
        new_facts.loc[tri_missings, ["uncertainty_disruption"]] = u[
            u.index[tri_missings][n_bin:]
        ]
        return new_facts


class NonstructuralModel(Model):
    def __init__(
        self,
        ncompounds,
        N_props=8,
        N_react=1,
        dirichlet=True,
        observe=True,
        likelihood_sd=0.25,
    ):
        super().__init__(
            N_props=N_props,
            N_react=N_react,
            dirichlet=dirichlet,
            observe=observe,
            likelihood_sd=likelihood_sd,
        )
        self.ncompounds = ncompounds

    def _pymc3_model(
        self, facts,
    ):
        bin_facts = facts[facts["compound3"] == -1]
        bin_r1 = bin_facts.compound1.values
        bin_r2 = bin_facts.compound2.values

        tri_facts = facts[facts["compound3"] != -1]
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values

        with pm.Model() as m:
            mem_beta = pm.Beta(
                "mem_beta", 0.9, 0.9, shape=(self.ncompounds, self.N_props + 1)
            )
            # the first property is non-reactive, so ignore that
            mem = pm.Deterministic(
                "mem", stick_breaking(mem_beta, normalize=True)[..., 1:]
            )

            if self.dirichlet:
                reactivities = pm.Dirichlet(
                    "reactivities",
                    a=0.4 * np.ones(self.N_bin + self.N_tri),
                    shape=(self.N_react, self.N_bin + self.N_tri),
                ).T

                # Normalize reactivities (useful with the Dirichlet prior)
                reactivities_norm = pm.Deterministic(
                    "reactivities_norm",
                    reactivities / reactivities.max(axis=0, keepdims=True),
                )
            else:
                reactivities_norm = pm.Beta(
                    "reactivities_norm",
                    alpha=1.0,
                    beta=3.0,
                    shape=(self.N_react, self.N_bin + self.N_tri),
                ).T

            # add zero entry for self-reactivity for each reactivity mode
            reactivities_with_zero = tt.concatenate(
                [tt.zeros((1, self.N_react)), reactivities_norm], axis=0
            )

            react_matrix = pm.Deterministic(
                "react_matrix", reactivities_with_zero[self.bi_idx, ...],
            )
            react_tensor = reactivities_with_zero[self.tri_idx, ...]

            m1, m2 = mem[bin_r1, :][:, :, np.newaxis], mem[bin_r2, :][:, np.newaxis, :]
            M1, M2, M3 = mem[tri_r1, :], mem[tri_r2, :], mem[tri_r3, :]

            bin_doesnt_react = pm.Deterministic(
                "bin_doesnt_react",
                tt.prod(
                    1
                    - tt.batched_dot(m1, m2)[..., np.newaxis]
                    * react_matrix[np.newaxis, ...],
                    axis=[1, 2, 3],
                ),
            )

            tri_no_react = pm.Deterministic(
                "tri_doesnt_react",
                tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor),
            )
            if self.observe:
                nmr_obs_binary = pm.Normal(
                    "reacts_binary_NMR",
                    mu=1 - bin_doesnt_react,
                    sd=self.likelihood_sd,
                    observed=bin_facts["NMR_reactivity"],
                )
                nmr_obs_ternary = pm.Normal(
                    "reacts_ternary_NMR",
                    mu=1 - tri_no_react,
                    sd=self.likelihood_sd,
                    observed=tri_facts["NMR_reactivity"],
                )
                hplc_obs_binary = pm.Normal(
                    "reacts_binary_HPLC",
                    mu=1 - bin_doesnt_react,
                    sd=self.likelihood_sd,
                    observed=bin_facts["HPLC_reactivity"],
                )
                hplc_obs_ternary = pm.Normal(
                    "reacts_ternary_HPLC",
                    mu=1 - tri_no_react,
                    sd=self.likelihood_sd,
                    observed=tri_facts["HPLC_reactivity"],
                )

        return m


class StructuralModel(Model):
    def __init__(
        self,
        fingerprint_matrix: np.ndarray,
        N_props=8,
        N_react=1,
        dirichlet=True,
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
            N_props=N_props,
            N_react=N_react,
            dirichlet=dirichlet,
            observe=observe,
            likelihood_sd=likelihood_sd,
        )

        self.fingerprints = tt._shared(fingerprint_matrix)
        self.ncompounds, self.fingerprint_length = fingerprint_matrix.shape

    def _pymc3_model(self, facts):
        bin_facts = facts[facts["compound3"] == -1]
        bin_r1 = bin_facts.compound1.values
        bin_r2 = bin_facts.compound2.values

        tri_facts = facts[facts["compound3"] != -1]
        tri_r1 = tri_facts.compound1.values
        tri_r2 = tri_facts.compound2.values
        tri_r3 = tri_facts.compound3.values

        with pm.Model() as m:
            mem_beta = pm.Beta(
                "mem_beta", 0.9, 0.9, shape=(self.fingerprint_length, self.N_props + 1)
            )
            # the first property is non-reactive, so ignore that
            mem = pm.Deterministic(
                "mem", stick_breaking(mem_beta, normalize=True)[..., 1:]
            )

            if self.dirichlet:
                reactivities = pm.Dirichlet(
                    "reactivities",
                    a=0.4 * np.ones(self.N_bin + self.N_tri),
                    shape=(self.N_react, self.N_bin + self.N_tri),
                ).T

                # Normalize reactivities (useful with the Dirichlet prior)
                reactivities_norm = pm.Deterministic(
                    "reactivities_norm",
                    reactivities / reactivities.max(axis=0, keepdims=True),
                )
            else:
                reactivities_norm = pm.Beta(
                    "reactivities_norm",
                    alpha=1.0,
                    beta=3.0,
                    shape=(self.N_react, self.N_bin + self.N_tri),
                ).T

            # add zero entry for self-reactivity for each reactivity mode
            reactivities_with_zero = tt.concatenate(
                [tt.zeros((1, self.N_react)), reactivities_norm], axis=0
            )

            react_matrix = pm.Deterministic(
                "react_matrix", reactivities_with_zero[self.bi_idx, :],
            )
            react_tensor = reactivities_with_zero[self.tri_idx, ...]

            # Convert fingerprint memberships to molecule memberships
            fp1, fp2 = self.fingerprints[bin_r1, :], self.fingerprints[bin_r2, :]
            m1, m2 = (
                tt.max(tt.mul(fp1[:, :, np.newaxis], mem), axis=1)[:, :, np.newaxis],
                tt.max(tt.mul(fp2[:, :, np.newaxis], mem), axis=1)[:, np.newaxis, :],
            )
            FP1, FP2, FP3 = (
                self.fingerprints[tri_r1, :],
                self.fingerprints[tri_r2, :],
                self.fingerprints[tri_r3, :],
            )
            M1, M2, M3 = (
                tt.max(tt.mul(FP1[:, :, np.newaxis], mem), axis=1),
                tt.max(tt.mul(FP2[:, :, np.newaxis], mem), axis=1),
                tt.max(tt.mul(FP3[:, :, np.newaxis], mem), axis=1),
            )

            bin_doesnt_react = pm.Deterministic(
                "bin_doesnt_react",
                tt.prod(
                    1
                    - tt.batched_dot(m1, m2)[..., np.newaxis]
                    * react_matrix[np.newaxis, ...],
                    axis=[1, 2, 3],
                ),
            )

            tri_no_react = pm.Deterministic(
                "tri_doesnt_react",
                tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor),
            )
            if self.observe:
                nmr_obs_binary = pm.Normal(
                    "reacts_binary_NMR",
                    mu=1 - bin_doesnt_react,
                    sd=self.likelihood_sd,
                    observed=bin_facts["NMR_reactivity"],
                )
                nmr_obs_ternary = pm.Normal(
                    "reacts_ternary_NMR",
                    mu=1 - tri_no_react,
                    sd=self.likelihood_sd,
                    observed=tri_facts["NMR_reactivity"],
                )
                hplc_obs_binary = pm.Normal(
                    "reacts_binary_HPLC",
                    mu=1 - bin_doesnt_react,
                    sd=self.likelihood_sd,
                    observed=bin_facts["HPLC_reactivity"],
                )
                hplc_obs_ternary = pm.Normal(
                    "reacts_ternary_HPLC",
                    mu=1 - tri_no_react,
                    sd=self.likelihood_sd,
                    observed=tri_facts["HPLC_reactivity"],
                )

        return m
