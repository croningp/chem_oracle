import math
from itertools import combinations, permutations
from os import path
from typing import List

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from matplotlib import pyplot as plt
from rdkit.Chem import AllChem, Draw


def indices(N: int, ndims: int) -> np.ndarray:
    """
    Generates an array `A` of indices where the permutations of `A[n]` give the
    unique non-zero indices of a symmetric rank `ndims` tensor with dimensions
    `N`×`N`×...×`N` (`ndims` times).

    >>> A = indices(4, 3) # 4×4×4 tensor
    >>> A
    array([[0, 1, 2],
           [0, 1, 3],
           [0, 2, 3],
           [1, 2, 3]])
    
    Note that any permutation of the indices would correspond to the same linear
    index since the tensor is symmetric. `A[[0, 1, 2]] == A[[1, 2, 0]]` etc.
    """
    nindices = math.factorial(N) / math.factorial(ndims) / math.factorial(N - ndims)
    v = []
    for comb in combinations(range(N), ndims):
        v.append(" ".join(str(s) for s in comb))
    v.sort()
    v = [e.split(" ") for e in v]
    return np.array(v, dtype="int")


def symmteric_rank2(i, j):
    I, J = np.maximum(i, j), np.minimum(i, j)
    return I * (I + 1) / 2 + J


def symmetric_rank2_indices(N: int):
    return np.fromfunction(symmteric_rank2, (N, N)).astype("int")


def symmetric_rank3(*args):
    I, J, K = sorted(args)
    return I * (I + 1) * (I + 2) / 6 + J * (J + 1) / 2 + K


def symmetric_rank3_indices(N: int):
    return np.array(
        [
            [[symmetric_rank3(i, j, k) for k in range(N)] for j in range(N)]
            for i in range(N)
        ],
        dtype="int",
    )


def triangular_tensor(v, N, ndims, indices):
    """
    v: vector of unique reactivities
    N: is the number of properties len(v) =  N*(N-1)*...*(N-(ndim-1))/ndim!
    ndim: number of tensor dimensions
    indices: tensor indices of non-zero elements
    """
    t = tt.zeros(tuple(N for _ in range(ndims)))
    for i, ind in enumerate(indices):
        for perm in permutations(ind):
            t = tt.set_subtensor(t[perm], v[i])
    return t


def tri_doesnt_react(M1, M2, M3, react_matrix, react_tensor):
    # 1 - (probability that any combination of two reagents react independently)
    tri_doesnt_react_binary = (
        tt.prod(
            1
            - tt.batched_dot(M1[:, :, np.newaxis], M2[:, np.newaxis, :])
            * react_matrix[np.newaxis, :, :],
            axis=[1, 2],
        )
        * tt.prod(
            1
            - tt.batched_dot(M1[:, :, np.newaxis], M3[:, np.newaxis, :])
            * react_matrix[np.newaxis, :, :],
            axis=[1, 2],
        )
        * tt.prod(
            1
            - tt.batched_dot(M1[:, :, np.newaxis], M2[:, np.newaxis, :])
            * react_matrix[np.newaxis, :, :],
            axis=[1, 2],
        )
    )
    # 1 - (probability that genuine three-component reaction occurs)
    tri_doesnt_react_ternary = tt.prod(
        1
        - tt.batched_dot(
            tt.batched_dot(M1[:, :, np.newaxis], M2[:, np.newaxis, :])[
                :, :, :, np.newaxis
            ],
            M3[:, np.newaxis, :],
        )
        * react_tensor[np.newaxis, :, :, :],
        axis=[1, 2, 3],
    )
    return tri_doesnt_react_binary * tri_doesnt_react_ternary


def morgan_bits(mol, radius, nbits):
    result = np.zeros(nbits)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    result[morgan_fp.GetOnBits()] = 1.0
    return result


def morgan_matrix(mols, radius, nbits):
    return np.stack([morgan_bits(mol, radius, nbits) for mol in mols])


def fingerprint_motifs(mol, radius, nbits):
    bitset = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=bitset)
    return {
        bit: [
            Draw.DrawMorganBit(mol, bit, bitset, whichExample=i)
            for i, _ in enumerate(bitset[bit])
        ]
        for bit in bitset
    }


def split_bin_tri(facts):
    bin_facts = facts[facts["compound3"] == -1]
    tri_facts = facts[facts["compound3"] != -1]
    return bin_facts, tri_facts


def build_results(predictions, new_facts):
    tri_stds = np.std(predictions["tri_doesnt_react"], axis=0)
    tri_mean_pred = 1 - np.mean(predictions["tri_doesnt_react"], axis=0)
    bin_stds = np.std(predictions["bin_doesnt_react"], axis=0)
    bin_mean_pred = 1 - np.mean(predictions["bin_doesnt_react"], axis=0)
    bin_new_facts, tri_new_facts = split_bin_tri(new_facts)

    bin_new_facts["stdd"] = bin_stds
    bin_new_facts["prediction"] = bin_mean_pred
    bin_new_facts["rounded_pred"] = np.round(bin_new_facts["prediction"])
    tri_new_facts["stdd"] = tri_stds
    tri_new_facts["prediction"] = tri_mean_pred
    tri_new_facts["rounded_pred"] = np.round(tri_new_facts["prediction"])

    full_facts = pd.concat([tri_new_facts, bin_new_facts]).sort_index()

    return full_facts


def predict(trained_model, new_facts):
    prediction_model = trained_model.pymc3_model(new_facts)
    with prediction_model:
        predictions = pm.sample_posterior_predictive(
            trained_model.trace, vars=prediction_model.deterministics
        )
    full_facts = build_results(predictions, new_facts)
    return full_facts


def posterior_reactivity_plot(fact_number, posterior_trace, facts):
    f = plt.figure()
    bin_facts = facts[facts["compound3"] == -1]
    tri_facts = facts[facts["compound3"] != -1]
    active_facts = bin_facts if fact_number in bin_facts.index else tri_facts
    var_name = (
        "bin_doesnt_react" if fact_number in bin_facts.index else "tri_doesnt_react"
    )
    location = active_facts.index.get_loc(fact_number)
    reactant_names = active_facts.iloc[location][
        ["reagent_name1", "reagent_name2", "reagent_name3"]
    ]
    plt.hist(
        1 - posterior_trace[var_name][:, location],
        np.linspace(0, 1, 30),
        density=True,
        color="lightgrey",
        rwidth=0.8,
    )
    pm.plots.kdeplot(1 - posterior_trace[var_name][:, location])
    plt.title(" + ".join(rn for rn in reactant_names if rn != "N/A"))
    plt.xlim((0.0, 1.0))
    plt.yticks([])
    plt.xlabel("Reaction probability")
    return f, 1 - posterior_trace[var_name][:, location]


def reaction_number(experiment_dir: str) -> int:
    experiment_dir = path.basename(experiment_dir)
    return int(experiment_dir.split("_")[0])


def reaction_components(experiment_dir: str) -> List[int]:
    experiment_dir = path.basename(experiment_dir)
    components = experiment_dir.split("_")[1]
    return [int(s) for s in components.split("-")]
