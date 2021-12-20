import os
from itertools import permutations
from typing import Dict

import jax

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
from numpyro import handlers as handlers

from ..util import indices
from .common import reactivity_disruption

if not use_cpu:
    # force GPU
    set_platform("gpu")


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
        likelihood_sd: float = 0.25,
        mem_a: float = 0.4,
        react_a: float = 0.4,
    ):
        self.N_props = N_props
        self.likelihood_sd = likelihood_sd
        self.mem_a = mem_a
        self.react_a = react_a

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

    def _pyro_model(self, facts, impute=True):
        observation_columns = [col for col in facts.columns if col.startswith("event")]
        N_event = len(observation_columns)

        fact_sets = [
            facts.query("compound3 == -1"),  # 2-component
            facts.query("compound3 != -1 and compound4 == -1"),  # 3-component
            facts.query("compound4 != -1"),  # 4-component
        ]
        reactants = [
            [fact_set[f"compound{j+1}"].values for j in range(i + 2)]
            for i, fact_set in enumerate(fact_sets)
        ]

        obs = [fact_set[observation_columns].values for fact_set in fact_sets]
        missing_obs = [np.isnan(o).nonzero() for o in obs]
        should_imputes = [impute and mo[0].size > 0 for mo in missing_obs]

        mem = self.mem()

        reactivities = [
            sample(
                f"reactivities_{i}",
                dist.Dirichlet(
                    self.react_a * jnp.ones((n, N_event + 1))
                ),  # Last event is "no event"
            )
            for i, n in zip(range(2, 5), self.N)
        ]

        # add zero entry for self-reactivity for each reactivity mode
        reactivities_with_zero = jnp.concatenate(
            [jnp.zeros((1, N_event))] +
            [
                (r / jnp.max(r, axis=1)[:, None])[:, 1:] for r in reactivities
            ]
        )

        react_tensors = [
            deterministic(f"react_tensor{i+2}", reactivities_with_zero[idx, :])
            for i, idx in enumerate(self.reactivity_indices)
        ]

        doesnt_react = [
            deterministic(
                "doesnt_react2",
                jnp.prod(
                    1
                    - mem[reactants[0][0]][:, :, np.newaxis, np.newaxis]
                    * mem[reactants[0][1]][:, np.newaxis, :, np.newaxis]
                    * react_tensors[0][np.newaxis, :, :, :],
                    axis=[1, 2],
                ),
            ),
            deterministic(
                "doesnt_react3",
                (
                    jnp.prod(
                        1
                        - mem[reactants[1][0]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[1][1]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[1][0]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[1][2]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[1][1]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[1][2]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[1][0]][:, np.newaxis, np.newaxis, :, np.newaxis]
                        * mem[reactants[1][1]][:, np.newaxis, :, np.newaxis, np.newaxis]
                        * mem[reactants[1][2]][:, :, np.newaxis, np.newaxis, np.newaxis]
                        * react_tensors[1][np.newaxis, :, :, :, :],
                        axis=[1, 2, 3],
                    )
                ),
            ),
            deterministic(
                "doesnt_react4",
                (
                    jnp.prod(
                        1
                        - mem[reactants[2][0]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][1]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][0]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][2]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][0]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][3]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][1]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][2]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][1]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][3]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][2]][:, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][3]][:, np.newaxis, :, np.newaxis]
                        * react_tensors[0][np.newaxis, :, :, :],
                        axis=[1, 2],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][0]][:, np.newaxis, np.newaxis, :, np.newaxis]
                        * mem[reactants[2][1]][:, np.newaxis, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][2]][:, :, np.newaxis, np.newaxis, np.newaxis]
                        * react_tensors[1][np.newaxis, :, :, :, :],
                        axis=[1, 2, 3],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][0]][:, np.newaxis, np.newaxis, :, np.newaxis]
                        * mem[reactants[2][1]][:, np.newaxis, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][3]][:, :, np.newaxis, np.newaxis, np.newaxis]
                        * react_tensors[1][np.newaxis, :, :, :, :],
                        axis=[1, 2, 3],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][0]][:, np.newaxis, np.newaxis, :, np.newaxis]
                        * mem[reactants[2][2]][:, np.newaxis, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][3]][:, :, np.newaxis, np.newaxis, np.newaxis]
                        * react_tensors[1][np.newaxis, :, :, :, :],
                        axis=[1, 2, 3],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][1]][:, np.newaxis, np.newaxis, :, np.newaxis]
                        * mem[reactants[2][2]][:, np.newaxis, :, np.newaxis, np.newaxis]
                        * mem[reactants[2][3]][:, :, np.newaxis, np.newaxis, np.newaxis]
                        * react_tensors[1][np.newaxis, :, :, :, :],
                        axis=[1, 2, 3],
                    )
                    * jnp.prod(
                        1
                        - mem[reactants[2][0]][
                            :, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis
                        ]
                        * mem[reactants[2][1]][
                            :, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis
                        ]
                        * mem[reactants[2][2]][
                            :, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis
                        ]
                        * mem[reactants[2][3]][
                            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
                        ]
                        * react_tensors[2][np.newaxis, :, :, :, :, :],
                        axis=[1, 2, 3, 4],
                    )
                ),
            ),
        ]

        for i, should_impute in enumerate(should_imputes):
            if should_impute:
                obs_impute = sample(
                    f"reacts_obs_missing{i+2}",
                    dist.Normal(
                        loc=1 - doesnt_react[i][missing_obs[i]],
                        scale=self.likelihood_sd,
                    ).mask(False),
                )
                obs[i] = ops.index_update(
                    obs[i], missing_obs[i], obs_impute.clip(0.0, 1.0)
                )

        event_obs = [
            sample(
                f"reacts_obs{i+2}",
                dist.Normal(loc=1 - doesnt_react[i], scale=self.likelihood_sd),
                obs=o,
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
        prediction = predictive(
            PRNGKey(0), facts, *(model_params or []), **sampler_params
        )
        # convert jax => numpy arrays
        prediction = {k: np.array(v) for k, v in prediction.items()}
        prediction.update(sampled_vars_trace)
        return prediction

    def sites(self, facts: pd.DataFrame, trace=None, impute=False) -> Dict:
        trace = trace or self.trace
        mdl = handlers.substitute(
            self._pyro_model, data={k: v[0] for k, v in trace.items()}
        )
        return handlers.trace(mdl).get_trace(facts, impute=impute)

    def log_likelihoods(self, facts: pd.DataFrame, trace: Dict = None) -> Dict:
        trace = trace or self.trace
        sites = self.sites(facts, trace)
        trace = {
            k: v
            for k, v in trace.items()
            if k in sites and sites[k]["type"] == "sample"
        }  # only keep sampled variables
        return log_likelihood(
            self._pyro_model, trace, facts, False  # do not impute - important!
        )

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
        events = facts[event_names]
        n_samples = len(trace["mem"])
        masks = [
            facts["compound3"] == -1,  # 2-component
            (facts["compound3"] != -1) & (facts["compound4"] == -1),  # 3-component
            facts["compound4"] != -1,  # 4-component
        ]

        react_preds = np.zeros((n_samples, *(events.shape)))
        for i, mask in enumerate(masks):
            react_preds[:, mask, ...] = 1 - trace[f"doesnt_react{i+2}"]

        likelihoods = self.experiment_likelihoods(facts, trace)

        conditioning_dfs = [
            pd.DataFrame(react_preds.mean(axis=0), columns=event_names).add_prefix(
                "avg_expected_"
            ),
            pd.DataFrame(react_preds.std(axis=0), columns=event_names).add_prefix(
                "std_expected_"
            ),
            pd.DataFrame(likelihoods.mean(axis=0), columns=event_names).add_prefix(
                "avg_likelihood_"
            ),
            pd.DataFrame(likelihoods.std(axis=0), columns=event_names).add_prefix(
                "std_likelihood_"
            ),
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
        mem_raw = sample(
            "mem_raw",
            dist.Dirichlet(self.mem_a * jnp.ones((self.ncompounds, self.N_props + 1))),
        )
        # the first property is non-reactive, so ignore that
        return deterministic(
            "mem", (mem_raw / jnp.max(mem_raw, axis=1)[:, None])[..., 1:]
        )  # ncompounds x N_props


class StructuralModel(Model):
    def __init__(self, fingerprint_matrix: np.ndarray, **kwargs):
        """Bayesian reactivity model informed by structural fingerprints.
        TODO: Update docs

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
        mem_raw = sample(
            "mem_raw",
            dist.Dirichlet(self.mem_a * jnp.ones((self.fingerprint_length, self.N_props + 1))),
        )

        # the first property is non-reactive, so ignore that
        fp_mem = deterministic(
            "fp_mem", (mem_raw / jnp.max(mem_raw, axis=1)[:, None])[..., 1:]
        )

        return deterministic(
            "mem", jnp.max(self.fingerprints[..., jnp.newaxis] * fp_mem, axis=1)
        )
