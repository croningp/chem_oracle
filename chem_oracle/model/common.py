from copy import deepcopy
from jax._src.numpy.lax_numpy import isnan

import numpy as np
import pandas as pd


def reactivity_disruption(observations: pd.DataFrame, probabilities: np.ndarray):
    """
    Calculate the disruption to the expected outcome of future experiments of
    upcoming experiments by performing a given experiment.
    """
    missing = observations.isna()
    res = np.zeros_like(observations)
    cov = np.abs(np.cov(probabilities[:, missing].T))
    np.fill_diagonal(cov, 0.0)
    res[missing] = np.sum(cov, axis=1)
    return res.sum(axis=1)
