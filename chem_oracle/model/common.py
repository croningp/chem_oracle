import numpy as np
import pandas as pd


def differential_disruption(
    observations: pd.DataFrame, reactivities: np.ndarray, method, order: int
):
    observations = observations.copy()
    result = np.zeros(observations.shape[0])
    for _ in range(order):
        disruptions = method(observations, reactivities)
        top = disruptions.argmax()
        result[top] = disruptions[top]
        predicted_reactivity = reactivities[:, top].mean(axis=0).round()
        observations.loc[top] = predicted_reactivity
        selection = (reactivities[:, top].round() == predicted_reactivity[None, :]).all(
            axis=-1
        )
        reactivities = reactivities[selection]
        if not reactivities.size or disruptions[top] == 0.0:
            return result
    return result


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
