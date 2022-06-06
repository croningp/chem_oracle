import itertools
from os import path
from typing import Iterable, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, MACCSkeys


def indices(N: int, ndims: int, allow_repeat=False) -> np.ndarray:
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
    if allow_repeat:
        combs = itertools.combinations_with_replacement(range(N), ndims)
    else:
        combs = itertools.combinations(range(N), ndims)
    return np.array(list(combs))


def morgan_bits(smiles: str, radius: int, nbits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    result = np.zeros(nbits)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    result[morgan_fp.GetOnBits()] = 1.0
    return result


def rdkit_bits(smiles: str, minpath: int, maxpath: int, nbits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    result = np.zeros(nbits)
    rdkit_fp = AllChem.RDKFingerprint(
        mol, minPath=minpath, maxPath=maxpath, fpSize=nbits, nBitsPerHash=1
    )
    result[rdkit_fp.GetOnBits()] = 1.0
    return result

def maccs_matrix(mols: Iterable[str]):
    return np.array([MACCSkeys.GenMACCSKeys(mol) for mol in mols])
    

def morgan_matrix(mols: Iterable[str], radius: int, nbits: int):
    return np.stack([morgan_bits(mol, radius, nbits) for mol in mols])


def rdkit_matrix(mols: Iterable[str], radius: int, nbits: int):
    return np.stack([rdkit_bits(mol, 1, radius, nbits) for mol in mols])


def split_bin_tri(facts):
    bin_facts = facts[facts["compound3"] == -1]
    tri_facts = facts[facts["compound3"] != -1]
    return bin_facts, tri_facts


def reaction_number(experiment_dir: str) -> int:
    experiment_dir = path.basename(experiment_dir)
    return int(experiment_dir.split("_")[0])


def reaction_components(experiment_dir: str) -> List[int]:
    experiment_dir = path.basename(experiment_dir)
    components = experiment_dir.split("_")[1]
    return [int(s) for s in components.split("-")]


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


def unique_reactions(df: pd.DataFrame, compound_cols, event_names):
    lookup = df.reset_index().set_index(compound_cols)
    for cmp in lookup.index:
        sub_combinations = []
        for size in range(2, len([i for i in cmp if i != -1])):
            for comb in itertools.combinations(cmp, size):
                if -1 not in comb:
                    comb = comb + (-1,) * (len(cmp) - len(comb))
                    sub_combinations.append(comb)
        sub_reactions = lookup.loc[sub_combinations, event_names].dropna()
        if sub_reactions.empty:
            continue
        lookup.loc[cmp, event_names] = (
            lookup.loc[cmp, event_names] - sub_reactions.max(axis=0)
        ).clip(lower=0.0)

    return lookup.reset_index().set_index("index")
