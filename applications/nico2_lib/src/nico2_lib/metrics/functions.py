"""
Metrics module for evaluating similarity and error between two numeric arrays.
Includes Pearson, Spearman, MSE, explained variance, and cosine similarity.
"""

from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr

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