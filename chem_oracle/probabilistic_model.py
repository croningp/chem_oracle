import numpy as np
import pymc3 as pm
import pandas as pd
import theano.tensor as tt

from .util import triangular_tensor, tri_doesnt_react, indices


def beta_params(mu, sd):
    """
    Beta distribution parameters to match a given mean and SD.
    :param mu:
    :param sd:
    :return:
    """
    M = 1 / (1 - mu)
    b = (M - 1) / M ** 3 / sd ** 2 - 1 / M
    a = (M - 1) * b
    return a, b


def reactivity_disruption(observations, probabilities):
    pi = np.mean(probabilities, axis=1)
    n_exp = observations.shape[0]
    res = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        obs_pos = probabilities[:, observations[i, :] == 1]
        obs_neg = probabilities[:, observations[i, :] == 0]
        mu_ji = np.mean(obs_pos, axis=1)
        mu_ji_bar = np.mean(obs_neg, axis=1)
        for j in range(n_exp):
            res[i, j] = pi[i] * np.abs(mu_ji[j] - pi[j]) + (1 - pi[i]) * np.abs(
                mu_ji_bar[j] - pi[j]
            )

    np.fill_diagonal(res, 0.0)
    return res


def uncertainty_disruption(observations, probabilities):
    pi = np.mean(probabilities, axis=1)
    stdj = np.std(probabilities, axis=1)
    n_exp = observations.shape[0]
    res = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        obs_pos = probabilities[:, observations[i, :] == 1]
        obs_neg = probabilities[:, observations[i, :] == 0]
        stdji = np.std(obs_pos, axis=1)
        stdji_bar = np.std(obs_neg, axis=1)
        for j in range(n_exp):
            res[i, j] = pi[i] * np.abs(stdji[j] - stdj[j]) + (1 - pi[i]) * np.abs(
                stdji_bar[j] - stdj[j]
            )

    np.fill_diagonal(res, 0.0)
    return res


class Model:
    def __init__(self, N):
        self.N = N
        self.bin_indices = indices(N, 2)
        self.tri_indices = indices(N, 3)

    def sample(
        self, facts: pd.DataFrame, n_samples, variational, **pymc3_params
    ) -> pm.sampling.MultiTrace:
        m = self._pymc3_model(facts)
        with m:
            if variational:
                self.approx = pm.fit(**pymc3_params)
                self.trace = self.approx.sample(n_samples)
            else:
                self.trace = pm.sample(n_samples, **pymc3_params)

    def condition(self, facts: pd.DataFrame, method_name: str = "MS") -> pd.DataFrame:
        # calculate reactivity for binary reactions
        bin_avg = 1 - np.mean(self.trace["bin_doesnt_react"], axis=0)
        bin_std = np.std(self.trace["bin_doesnt_react"], axis=0)
        # calculate reactivity for three component reactions
        tri_avg = 1 - np.mean(self.trace["tri_doesnt_react"], axis=0)
        tri_std = np.std(self.trace["tri_doesnt_react"], axis=0)
        # binary reactions
        bins = facts[(facts["compound3"] == -1)]
        tris = facts[(facts["compound3"] != -1)]

        new_facts = facts.copy()
        # update dataframe with calculated reactivities
        new_facts.loc[
            new_facts["compound3"] == -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack([bin_avg, bin_std]).T
        new_facts.loc[
            new_facts["compound3"] != -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack([tri_avg, tri_std]).T
        observations = np.hstack(
            (
                self.trace[f"reacts_binary_{method_name}_missing"],
                self.trace[f"reacts_ternary_{method_name}_missing"],
            )
        ).T
        probabilities = np.hstack(
            (
                1
                - self.trace["bin_doesnt_react"][
                    :, pd.isna(bins[f"{method_name}_reactivity"])
                ],
                1
                - self.trace["tri_doesnt_react"][
                    :, pd.isna(tris[f"{method_name}_reactivity"])
                ],
            )
        ).T
        r = reactivity_disruption(observations, probabilities).sum(axis=0)
        u = uncertainty_disruption(observations, probabilities).sum(axis=0)
        bin_missings = (new_facts["compound3"] == -1) & pd.isna(
            new_facts[f"{method_name}_reactivity"]
        )
        tri_missings = (new_facts["compound3"] != -1) & pd.isna(
            new_facts[f"{method_name}_reactivity"]
        )
        n_bin = bin_missings.sum()
        new_facts.loc[bin_missings, ["reactivity_disruption"]] = r[:n_bin]
        new_facts.loc[bin_missings, ["uncertainty_disruption"]] = u[:n_bin]
        new_facts.loc[tri_missings, ["reactivity_disruption"]] = r[n_bin:]
        new_facts.loc[tri_missings, ["uncertainty_disruption"]] = u[n_bin:]
        return new_facts


class NonstructuralModel(Model):
    def __init__(self, ncompounds, N=4):
        super().__init__(N)
        self.ncompounds = ncompounds

    def _pymc3_model(self, facts):
        bin_facts = facts[facts["compound3"] == -1]
        tri_facts = facts[facts["compound3"] != -1]

        bin_r1, bin_r2 = (
            tt._shared(bin_facts["compound1"].values),
            tt._shared(bin_facts["compound2"].values),
        )
        tri_r1, tri_r2, tri_r3 = (
            tt._shared(tri_facts["compound1"].values),
            tt._shared(tri_facts["compound2"].values),
            tt._shared(tri_facts["compound3"].values),
        )

        with pm.Model() as m:
            mem = pm.Uniform(
                "mem", lower=0.0, upper=1.0, shape=(self.ncompounds, self.N)
            )
            bin_reactivities = pm.Uniform(
                "bin_reactivities",
                lower=0.0,
                upper=1.0,
                shape=self.N * (self.N - 1) // 2,
            )
            tri_reactivities = pm.Uniform(
                "tri_reactivities",
                lower=0.0,
                upper=1.0,
                shape=self.N * (self.N - 1) * (self.N - 2) // 6,
            )
            react_tensor = triangular_tensor(
                tri_reactivities, self.N, 3, self.tri_indices
            )
            react_matrix = triangular_tensor(
                bin_reactivities, self.N, 2, self.bin_indices
            )
            # memberships of binary reactions
            m1, m2 = mem[bin_r1, :][:, :, np.newaxis], mem[bin_r2, :][:, np.newaxis, :]
            bin_doesnt_react = pm.Deterministic(
                "bin_doesnt_react",
                tt.prod(
                    1 - tt.batched_dot(m1, m2) * react_matrix[np.newaxis, :, :],
                    axis=[1, 2],
                ),
            )
            # memberships of ternary reactions
            M1, M2, M3 = mem[tri_r1, :], mem[tri_r2, :], mem[tri_r3, :]
            tri_no_react = pm.Deterministic(
                "tri_doesnt_react",
                tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor),
            )
            # observations
            hplc_obs_binary = pm.Bernoulli(
                "reacts_binary_HPLC",
                p=1 - bin_doesnt_react,
                observed=bin_facts["HPLC_reactivity"],
            )
            ms_obs_binary = pm.Bernoulli(
                "reacts_binary_MS",
                p=1 - bin_doesnt_react,
                observed=bin_facts["MS_reactivity"],
            )
            hplc_obs_ternary = pm.Bernoulli(
                "reacts_ternary_HPLC",
                p=1 - tri_no_react,
                observed=tri_facts["HPLC_reactivity"],
            )
            ms_obs_ternary = pm.Bernoulli(
                "reacts_ternary_MS",
                p=1 - tri_no_react,
                observed=tri_facts["MS_reactivity"],
            )
        return m


class StructuralModel(Model):
    def __init__(self, fingerprint_matrix: np.ndarray, N: int = 4):
        """Bayesian reactivity model informed by structural fingerprints.

        Args:
            fingerprint_matrix (n_compounds × fingerprint_length matrix):
                a numpy matrix row i of which contains the fingerprint bits
                for the i-th compound.
            N (int, optional): Number of abstract properties. Defaults to 4.
        """
        # """fingerprints: Matrix of Morgan fingerprints for reagents."""
        super().__init__(N)
        self.fingerprints = tt._shared(fingerprint_matrix)
        self.ncompounds, self.fingerprint_length = fingerprint_matrix.shape

    def _pymc3_model(self, facts):
        bin_facts = facts[facts["compound3"] == -1]
        tri_facts = facts[facts["compound3"] != -1]

        bin_r1, bin_r2 = (
            tt._shared(bin_facts["compound1"].values),
            tt._shared(bin_facts["compound2"].values),
        )
        tri_r1, tri_r2, tri_r3 = (
            tt._shared(tri_facts["compound1"].values),
            tt._shared(tri_facts["compound2"].values),
            tt._shared(tri_facts["compound3"].values),
        )

        with pm.Model() as m:
            mem = pm.Beta(
                "mem", alpha=1.0, beta=3.0, shape=(self.fingerprint_length, self.N)
            )

            bin_reactivities = pm.Uniform(
                "bin_reactivities",
                lower=0.0,
                upper=1.0,
                shape=self.N * (self.N - 1) // 2,
            )
            tri_reactivities = pm.Uniform(
                "tri_reactivities",
                lower=0.0,
                upper=1.0,
                shape=self.N * (self.N - 1) * (self.N - 2) // 6,
            )

            react_tensor = triangular_tensor(
                tri_reactivities, self.N, 3, self.tri_indices
            )
            react_matrix = triangular_tensor(
                bin_reactivities, self.N, 2, self.bin_indices
            )

            # memberships of binary reactions
            fp1, fp2 = self.fingerprints[bin_r1, :], self.fingerprints[bin_r2, :]
            m1, m2 = (
                tt.max(tt.mul(fp1[:, :, np.newaxis], mem), axis=1)[:, :, np.newaxis],
                tt.max(tt.mul(fp2[:, :, np.newaxis], mem), axis=1)[:, np.newaxis, :],
            )
            bin_doesnt_react = pm.Deterministic(
                "bin_doesnt_react",
                tt.prod(
                    1 - tt.batched_dot(m1, m2) * react_matrix[np.newaxis, :, :],
                    axis=[1, 2],
                ),
            )
            # memberships of ternary reactions
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
            tri_no_react = pm.Deterministic(
                "tri_doesnt_react",
                tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor),
            )

            # TODO: Revert to Bernoulli?
            nmr_obs_binary = pm.Normal(
                "reacts_binary_HPLC",
                mu=1 - bin_doesnt_react,
                sd=0.05,
                observed=bin_facts["HPLC_reactivity"],
            )
            ms_obs_binary = pm.Normal(
                "reacts_binary_MS",
                mu=1 - bin_doesnt_react,
                sd=0.05,
                observed=bin_facts["MS_reactivity"],
            )
            nmr_obs_ternary = pm.Normal(
                "reacts_ternary_HPLC",
                mu=1 - tri_no_react,
                sd=0.05,
                observed=tri_facts["HPLC_reactivity"],
            )
            ms_obs_ternary = pm.Normal(
                "reacts_ternary_MS",
                mu=1 - tri_no_react,
                sd=0.05,
                observed=tri_facts["MS_reactivity"],
            )
        return m
