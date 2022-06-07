from itertools import permutations
from typing import Dict

import jax

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro import deterministic, sample
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import Predictive, log_likelihood
from numpyro import handlers as handlers
from chem_oracle import util

from chem_oracle.util import indices


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
        mem_beta_a: float = 1.0,
        mem_beta_b: float = 2.0,
        likelihood_sd: float = 0.3,
    ):
        self.N_props = N_props
        self.mem_beta_a = mem_beta_a
        self.mem_beta_b = mem_beta_b
        self.likelihood_sd = likelihood_sd

        # Number of _unique_ reactivities for each reaction arity
        self.N_bin = self.N_props * (self.N_props - 1) // 2
        self.N_tri = self.N_bin * (self.N_props - 2) // 3
        self.N_tet = self.N_tri * (self.N_props - 3) // 4
        self.N = [self.N_bin, self.N_tri, self.N_tet]

        # Indices for 2-, 3-, etc. component reactivities within
        # the sampled vector of _unique_ reactivities
        self.reactivity_indices = [
            triangular_indices(self.N_props, i + 2, shift=sum(self.N[:i]))
            for i in range(len(self.N))
        ]

    def _pyro_model(self, facts: pd.DataFrame, observe=True):
        compounds = [col for col in facts.columns if col.startswith("compound")]
        observation_columns = [col for col in facts.columns if col.startswith("event")]
        facts = util.unique_reactions(facts, compounds, observation_columns)

        fact_sets = {
            2: facts.query("compound3 == -1"),  # 2-component
            3: facts.query("compound3 != -1 and compound4 == -1"),  # 3-component
            4: facts.query("compound4 != -1"),  # 4-component
        }
        reactants = [
            [fact_set[f"compound{j+1}"].values for j in range(n_components)]
            for n_components, fact_set in fact_sets.items()
        ]

        obs = [
            fact_set[observation_columns].sum(axis=1) for fact_set in fact_sets.values()
        ]

        present_inds = [(~np.isnan(o.values)).nonzero() for o in obs]

        obs = [o.dropna().values for o in obs]

        mem = self.mem()

        reactivities = jnp.concatenate(
            [
                sample(f"reactivities{i}", dist.Beta(1.0, 2.0), sample_shape=(n,))
                for i, n in zip(fact_sets, self.N)
            ]
        )

        reactivities_with_zero = jnp.concatenate(
            [jnp.zeros((1,)), reactivities],
        )

        react_tensors = [
            deterministic(f"react_tensor{i+2}", reactivities_with_zero[idx])
            for i, idx in enumerate(self.reactivity_indices)
        ]

        reacts = [
            deterministic(
                "reacts2",
                jnp.sum(
                    mem[reactants[0][0]][:, :, np.newaxis]
                    * mem[reactants[0][1]][:, np.newaxis, :]
                    * react_tensors[0][np.newaxis, :, :],
                    axis=(1, 2),
                ),
            ),
            deterministic(
                "reacts3",
                jnp.sum(
                    mem[reactants[1][0]][:, np.newaxis, np.newaxis, :]
                    * mem[reactants[1][1]][:, np.newaxis, :, np.newaxis]
                    * mem[reactants[1][2]][:, :, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :],
                    axis=(1, 2, 3),
                ),
            ),
            deterministic(
                "reacts4",
                jnp.sum(
                    mem[reactants[2][0]][:, np.newaxis, np.newaxis, np.newaxis, :]
                    * mem[reactants[2][1]][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[reactants[2][2]][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[reactants[2][3]][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[2][np.newaxis, :, :, :, :],
                    axis=(1, 2, 3, 4),
                ),
            ),
        ]

        if observe:
            event_obs = [
                sample(
                    f"reacts_obs{i+2}",
                    dist.Normal(loc=reacts[i][present_inds[i]], scale=self.likelihood_sd),
                    obs=o,
                )
                for i, o in enumerate(obs)
            ]
        else:
            event_obs = [
                sample(
                    f"reacts_obs{i+2}",
                    dist.Normal(loc=reacts[i][present_inds[i]], scale=self.likelihood_sd),
                )
                for i, o in enumerate(obs)
            ]

    def sample(
        self,
        facts: pd.DataFrame,
        draws=500,
        tune=500,
        model_params=None,
        rng_seed=0,
        nuts_kwargs={},
        sampler_kwargs={},
    ) -> Dict:
        def do_mcmc(rng_key):
            nuts_kernel = NUTS(self._pyro_model, **nuts_kwargs)
            mcmc = MCMC(
                nuts_kernel, num_samples=draws, num_warmup=tune, **sampler_kwargs
            )
            mcmc.run(
                rng_key,
                facts,
                *(model_params or []),
                extra_fields=("potential_energy",),
            )
            return {**mcmc.get_samples(), **mcmc.get_extra_fields()}

        n_parallel = jax.local_device_count()
        if n_parallel > 1:
            rng_keys = jax.random.split(PRNGKey(rng_seed), n_parallel)
            results = jax.pmap(do_mcmc)(rng_keys)
            # concatenate results along pmap'ed axis
            self.trace = {k: np.concatenate(v) for k, v in results.items()}
        else:
            self.trace = do_mcmc(PRNGKey(rng_seed))
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
        sites = self.sites(facts, knowledge_trace)
        sampled_vars_trace = {
            k: v
            for k, v in knowledge_trace.items()
            if k in sites and sites[k]["type"] == "sample"
        }
        predictive = Predictive(self._pyro_model, sampled_vars_trace, num_samples=draws)
        prediction = predictive(PRNGKey(0), facts, *(model_params or []), observe=False)
        # convert jax => numpy arrays
        prediction = {k: np.array(v) for k, v in prediction.items()}
        prediction.update(sampled_vars_trace)
        return prediction

    def sites(self, facts: pd.DataFrame, trace=None) -> Dict:
        trace = trace or self.trace
        mdl = handlers.substitute(
            self._pyro_model, data={k: v[0] for k, v in trace.items()}
        )
        return handlers.trace(mdl).get_trace(facts)

    def log_likelihoods(self, facts: pd.DataFrame, trace: Dict = None) -> Dict:
        trace = trace or self.trace
        sites = self.sites(facts, trace)
        trace = {
            k: v
            for k, v in trace.items()
            if k in sites and sites[k]["type"] == "sample"
        }  # only keep sampled variables
        return log_likelihood(self._pyro_model, trace, facts)

    def experiment_likelihoods(self, facts: pd.DataFrame, trace: Dict = None):
        trace = trace or self.trace
        likelihoods = self.log_likelihoods(facts, trace)
        reaction_likelihoods = [
            likelihoods[k] for k in sorted(likelihoods) if k.startswith("reacts_obs")
        ]

        result = np.zeros_like(np.concatenate(reaction_likelihoods, axis=1))
        result[:, facts.query("compound3==-1").index] = reaction_likelihoods[0]
        result[
            :, facts.query("compound3!=-1 and compound4==-1").index
        ] = reaction_likelihoods[1]
        result[:, facts.query("compound4!=-1").index] = reaction_likelihoods[2]

        return result

    def condition(
        self,
        facts: pd.DataFrame,
        trace=None,
        calculate_disruptions=True,
    ) -> pd.DataFrame:
        # calculate reactivity for binary reactions
        trace = trace or self.trace

        event_names = [col for col in facts.columns if col.startswith("event")]
        events = facts[event_names].sum(axis=1, skipna=False)
        n_samples = len(trace["mem"])
        masks = [
            facts["compound3"] == -1,  # 2-component
            (facts["compound3"] != -1) & (facts["compound4"] == -1),  # 3-component
            facts["compound4"] != -1,  # 4-component
        ]

        react_preds = np.zeros((n_samples, *(events.shape)))
        for i, mask in enumerate(masks):
            react_preds[:, mask, ...] = trace[f"reacts{i+2}"]

        likelihoods = self.experiment_likelihoods(facts, trace)

        conditioning_dfs = [
            pd.DataFrame(react_preds.mean(axis=0), columns=["avg_expected"]),
            pd.DataFrame(react_preds.std(axis=0), columns=["std_expected"]),
            pd.DataFrame(likelihoods.mean(axis=0), columns=["avg_likelihood"]),
            pd.DataFrame(likelihoods.std(axis=0), columns=["std_likelihood"]),
        ]

        if calculate_disruptions:
            conditioning_dfs.extend(
                [
                    pd.DataFrame(
                        reactivity_disruption(events, react_preds),
                        columns=["reactivity_disruption"],
                    )
                ]
            )

        conditioning_df = pd.concat(conditioning_dfs, axis=1)

        return pd.concat(
            [
                # Remove existing columns within facts
                facts.drop(columns=facts.columns.intersection(conditioning_df.columns)),
                conditioning_df,
            ],
            axis=1,
        )


class NonstructuralModel(Model):
    def __init__(self, ncompounds, **kwargs):
        super().__init__(**kwargs)
        self.ncompounds = ncompounds

    def mem(self):
        return sample(
            "mem",
            dist.Beta(self.mem_beta_a, self.mem_beta_b),
            sample_shape=(self.ncompounds, self.N_props),
        )


class StructuralModel(Model):
    def __init__(self, fingerprint_matrix: np.ndarray, **kwargs):
        """Bayesian reactivity model informed by structural fingerprints.

        Args:
            fingerprint_matrix (n_compounds × fingerprint_length matrix):
                a numpy matrix row i of which contains the fingerprint bits
                for the i-th compound.
            N_props (int, optional): Number of abstract properties. Defaults to 4.
        """
        # """fingerprints: Matrix of Morgan fingerprints for reagents."""

        super().__init__(**kwargs)
        self.fingerprints = fingerprint_matrix
        self.ncompounds, self.fingerprint_length = fingerprint_matrix.shape

    def mem(self):
        fp_mem = sample(
            "fp_mem",
            dist.Beta(self.mem_beta_a, self.mem_beta_b),
            sample_shape=(self.fingerprint_length, self.N_props),
        )

        return deterministic(
            "mem", jnp.max(self.fingerprints[..., jnp.newaxis] * fp_mem, axis=1)
        )


def differential_disruption(
    observations: pd.DataFrame,
    reactivities: np.ndarray,
    method,
    order: int,
    min_points: int = 7,
):
    observations = observations.copy()
    result = np.zeros(observations.shape[0])
    if order == 0 or reactivities.shape[0] < min_points:
        # reached bottom of tree or not enough points to do a partition
        return result

    disruptions = method(observations, reactivities)
    top = disruptions.argmax()
    result[top] = disruptions[top]
    predicted_reactivity = np.median(reactivities[:, top], axis=0)
    observations.loc[top] = predicted_reactivity
    outcomes = reactivities[:, top] > predicted_reactivity[None, :]
    cov = np.cov(outcomes.T)
    pivot = np.abs(cov).sum(axis=1).argmax()
    flip = cov[pivot, :] < 0
    consensus = np.logical_xor(outcomes, flip.T)
    selection_pos = consensus.all(axis=1)
    selection_neg = ~consensus.any(axis=1)
    res = np.maximum(
        result,
        differential_disruption(
            observations, reactivities[selection_pos], method, order - 1, min_points
        ),
        differential_disruption(
            observations, reactivities[selection_neg], method, order - 1, min_points
        ),
    )
    return res


def timeline_disruption(observations: pd.DataFrame, reactivities: np.ndarray):
    """
    Calculate the disruption to the expected outcome of future experiments
    by performing a given experiment.
    """
    N = reactivities.shape[0]
    result = np.zeros_like(observations)
    missing = observations.isna()
    reactivities = reactivities[:, missing]
    discrete_outcomes = reactivities.round()
    expected_outcome = np.median(discrete_outcomes, axis=0).round()
    expected_matches = discrete_outcomes == expected_outcome[None, :]
    num_matches = N - expected_matches.sum(axis=0)
    result[missing] = num_matches

    return result.sum(axis=-1)


def reactivity_disruption(observations: pd.DataFrame, reactivities: np.ndarray):
    """
    Calculate the disruption to the expected outcome of future experiments
    by performing a given experiment.
    """
    missing = observations.isna()
    reactivities = reactivities[:, missing]
    medians = np.median(reactivities, axis=0)
    flips = reactivities > medians
    result = np.zeros_like(observations)

    result[missing] = np.array(
        [
            np.abs(reactivities[f].mean(axis=0) - reactivities[~f].mean(axis=0)).sum()
            for f in flips.T
        ]
    )

    return np.abs(result)


def uncertainty_disruption(observations: pd.DataFrame, reactivities: np.ndarray):
    """
    Calculate the disruption to the outcome uncertainty of future experiments
    by performing a given experiment.
    """
    missing = observations.isna()
    reactivities = reactivities[:, missing]
    medians = np.median(reactivities, axis=0)
    flips = reactivities > medians
    result = np.zeros_like(observations)

    result[missing] = np.array(
        [
            np.abs(reactivities[f].std(axis=0) - reactivities[~f].std(axis=0)).sum()
            for f in flips.T
        ]
    )

    return np.abs(result).sum(axis=1)
