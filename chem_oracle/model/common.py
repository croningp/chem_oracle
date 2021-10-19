import numpy as np
import pandas as pd


def differential_disruption(
    observations: pd.DataFrame, reactivities: np.ndarray, method, order: int, min_points:int=7
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
        differential_disruption(observations, reactivities[selection_pos], method, order - 1, min_points),
        differential_disruption(observations, reactivities[selection_neg], method, order - 1, min_points)
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

    return np.abs(result).sum(axis=1)


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
