from copy import deepcopy

import numpy as np
import pandas as pd


def reactivity_disruption(observations, probabilities):
    pi = np.mean(probabilities, axis=1)
    n_exp = observations.shape[0]
    res = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        reacts = observations[i, :] > 0.5
        if reacts.all() or not reacts.any():
            # all observations positive/negative, won't cause any disruption
            res[i, :] = 0.0
            continue
        obs_pos = probabilities[:, reacts]
        obs_neg = probabilities[:, ~reacts]
        mu_ji = np.mean(obs_pos, axis=1)
        mu_ji_bar = np.mean(obs_neg, axis=1)
        for j in range(n_exp):
            res[i, j] = pi[i] * np.abs(mu_ji[j] - pi[j]) + (1 - pi[i]) * np.abs(
                mu_ji_bar[j] - pi[j]
            )

    np.fill_diagonal(res, 0.0)
    return np.nan_to_num(res, 0.0)


def uncertainty_disruption(observations, probabilities):
    pi = np.mean(probabilities, axis=1)
    stdj = np.std(probabilities, axis=1)
    n_exp = observations.shape[0]
    res = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        reacts = observations[i, :] > 0.5
        if reacts.all() or not reacts.any():
            # all observations positive/negative, won't cause any disruption
            res[i, :] = 0.0
            continue
        obs_pos = probabilities[:, reacts]
        obs_neg = probabilities[:, ~reacts]
        stdji = np.std(obs_pos, axis=1)
        stdji_bar = np.std(obs_neg, axis=1)
        for j in range(n_exp):
            res[i, j] = pi[i] * np.abs(stdji[j] - stdj[j]) + (1 - pi[i]) * np.abs(
                stdji_bar[j] - stdj[j]
            )

    np.fill_diagonal(res, 0.0)
    return np.nan_to_num(res, 0.0)


def disruptions(facts, trace, method_name: str):
    # binary reactions
    bins = facts[(facts["compound3"] == -1)]
    tris = facts[(facts["compound3"] != -1) & (facts["compound4"] == -1)]
    tets = facts[(facts["compound4"] != -1)]

    bin_impute = trace.get(
        f"reacts_binary_{method_name}_missing",
        np.zeros((trace["bin_doesnt_react"].shape[0], 0), dtype="float32"),
    )
    tri_impute = trace.get(
        f"reacts_ternary_{method_name}_missing",
        np.zeros((trace["tri_doesnt_react"].shape[0], 0), dtype="float32"),
    )
    tet_impute = trace.get(
        f"reacts_quaternary_{method_name}_missing",
        np.zeros((trace["tet_doesnt_react"].shape[0], 0), dtype="float32"),
    )

    observations = np.hstack((bin_impute, tri_impute, tet_impute)).T
    probabilities = np.hstack(
        (
            1
            - trace["bin_doesnt_react"][:, pd.isna(bins[f"{method_name}_reactivity"])],
            1
            - trace["tri_doesnt_react"][:, pd.isna(tris[f"{method_name}_reactivity"])],
            1
            - trace["tet_doesnt_react"][:, pd.isna(tets[f"{method_name}_reactivity"])],
        )
    ).T
    return (
        reactivity_disruption(observations, probabilities).sum(axis=1),
        uncertainty_disruption(observations, probabilities).sum(axis=1),
    )


def differential_disruptions(
    facts, trace, method_name: str, n: int = 5, sort_by_reactivity: bool = False
):
    # create a copy of dataframe and remove existing disruption values
    f = facts.copy()
    f["reactivity_disruption"] = np.nan
    f["uncertainty_disruption"] = np.nan
    # convert trace to dictionary so we can mutate it!
    t = (
        deepcopy(trace)
        if isinstance(trace, dict)
        else {v: trace[v] for v in trace.varnames}
    )  # latter case applies to PyMC3 traces
    for _ in range(n):
        bin_missings = (f["compound3"] == -1) & pd.isna(f[f"{method_name}_reactivity"])
        tri_missings = (f["compound3"] != -1) & (f["compound4"] == -1) & pd.isna(f[f"{method_name}_reactivity"])
        tet_missings = (f["compound4"] != -1) & pd.isna(f[f"{method_name}_reactivity"])
        
        n_bin = bin_missings.sum()
        n_tri = tri_missings.sum()
        n_tet = tet_missings.sum()

        if n_bin + n_tri + n_tet == 0:
            # no more missing experiments - we are done
            break

        rr, uu = disruptions(f, t, method_name)
        if sort_by_reactivity:
            top_rxn = np.argmax(rr)
        else:
            top_rxn = np.argmax(uu)
        max_r = rr[top_rxn]
        max_u = uu[top_rxn]
        if top_rxn < n_bin:
            var_name = f"reacts_binary_{method_name}_missing"
            index_name = f.index[bin_missings]
        elif n_bin < top_rxn < n_bin + n_tri:
            var_name = f"reacts_ternary_{method_name}_missing"
            index_name = f.index[tri_missings]
            top_rxn -= n_bin
        else:
            var_name = f"reacts_quaternary_{method_name}_missing"
            index_name = f.index[tet_missings]
            top_rxn -= n_bin + n_tri

        # assign top pick as reactive or unreactive
        outcome = t[var_name][:, top_rxn].mean() > 0.5 and 1.0 or 0.0
        f.loc[index_name[top_rxn], f"{method_name}_reactivity"] = outcome
        # assign disruption score in dataframe
        f.loc[index_name[top_rxn], "reactivity_disruption"] = max_r
        f.loc[index_name[top_rxn], "uncertainty_disruption"] = max_u
        # remove MCMC samples incompatible with `outcome` from MCMC trace
        valid_indices = np.abs(t[var_name][:, top_rxn] - outcome) < 0.5
        if not valid_indices.any():
            break
        for var in t:
            t[var] = t[var][valid_indices, ...]
        # remove experiment from MCMC trace
        t[var_name] = np.delete(t[var_name], top_rxn, axis=1)

    return (f["reactivity_disruption"], f["uncertainty_disruption"])
