"""
Metrics module for evaluating similarity and error between two numeric arrays.
Includes Pearson, Spearman, MSE, explained variance, and cosine similarity.
"""

from typing import Any
from param import Array
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
import scipy
import numpy as np
from numpy.typing import ArrayLike

from numpy.typing import NDArray
from numpy import number

def pearson_metric(x: NDArray[number], y: NDArray[number]) -> float:
    return pearsonr(x, y).statistic

def spearman_metric(x: NDArray[number], y: NDArray[number]) -> float:
    return spearmanr(x, y).statistic

def mse_metric(x: NDArray[number], y: NDArray[number]) -> float:
    return mean_squared_error(x, y)

def explained_variance_metric(x: NDArray[number], y: NDArray[number]) -> float:
    return explained_variance_score(x, y)

def cosine_similarity_metric(x: NDArray[number], y: NDArray[number]) -> float:
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]

def mae_metric(x: ArrayLike, y: ArrayLike) -> float:
    return float(np.mean(np.abs(x - y)))

def explained_variance_metric_v2(x: ArrayLike, y: ArrayLike) -> Any:
    if scipy.sparse.issparse(x):
        x = x.toarray()
    if scipy.sparse.issparse(y):
        y = y.toarray()
    mse = np.mean((x - y) ** 2)
    total_variance = np.var(x)
    if total_variance == 0:
        return 0.0
    explained_var = 1 - (mse / total_variance)
    return explained_var