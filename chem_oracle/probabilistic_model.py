import numpy as np
import pymc3 as pm
import theano.tensor as tt

from .util import triangular_tensor, tri_doesnt_react, indices


class Model:
    def __init__(self, N):
        self.N = N
        self.bin_indices = indices(N, 2)
        self.tri_indices = indices(N, 3)

    def condition(self, facts, n_samples, variational, **pymc3_params):
        m = self._pymc3_model(facts)
        with m:
            if variational:
                self.approx = pm.fit(**pymc3_params)
                self.trace = self.approx.sample(n_samples)
            else:
                self.trace = pm.sample(n_samples, **pymc3_params)


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
            react_tensor = pm.Deterministic(
                "react_tensor",
                triangular_tensor(tri_reactivities, self.N, 3, self.tri_indices),
            )
            react_matrix = pm.Deterministic(
                "react_matrix",
                triangular_tensor(bin_reactivities, self.N, 2, self.bin_indices),
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

            nmr_obs_binary = pm.Normal(
                "reacts_binary_nmr",
                mu=1 - bin_doesnt_react,
                sd=0.1,
                observed=bin_facts["HPLC_reactivity"],
            )
            ms_obs_binary = pm.Normal(
                "reacts_binary_ms",
                mu=1 - bin_doesnt_react,
                sd=0.1,
                observed=bin_facts["MS_reactivity"],
            )
            nmr_obs_ternary = pm.Normal(
                "reacts_ternary_nmr",
                mu=1 - tri_no_react,
                sd=0.1,
                observed=tri_facts["HPLC_reactivity"],
            )
            ms_obs_ternary = pm.Normal(
                "reacts_ternary_ms",
                mu=1 - tri_no_react,
                sd=0.1,
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
            react_tensor = pm.Deterministic(
                "react_tensor",
                triangular_tensor(tri_reactivities, self.N, 3, self.tri_indices),
            )
            react_matrix = pm.Deterministic(
                "react_matrix",
                triangular_tensor(bin_reactivities, self.N, 2, self.bin_indices),
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

            nmr_obs_binary = pm.Normal(
                "reacts_binary_nmr",
                mu=1 - bin_doesnt_react,
                sd=0.05,
                observed=bin_facts["NMR_reactivity"],
            )
            ms_obs_binary = pm.Normal(
                "reacts_binary_ms",
                mu=1 - bin_doesnt_react,
                sd=0.05,
                observed=bin_facts["MS_reactivity"],
            )
            nmr_obs_ternary = pm.Normal(
                "reacts_ternary_nmr",
                mu=1 - tri_no_react,
                sd=0.05,
                observed=tri_facts["NMR_reactivity"],
            )
            ms_obs_ternary = pm.Normal(
                "reacts_ternary_ms",
                mu=1 - tri_no_react,
                sd=0.05,
                observed=tri_facts["MS_reactivity"],
            )
        return m
