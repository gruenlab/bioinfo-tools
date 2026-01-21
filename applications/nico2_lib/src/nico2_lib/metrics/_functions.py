"""
Metrics module for evaluating similarity and error between two numeric arrays.
Includes Pearson, Spearman, MSE, explained variance, and cosine similarity.
"""

from typing import Any

import numpy as np
import scipy
from numpy import number
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def _to_dense_array(x: ArrayLike) -> NDArray[np.number]:
    if scipy.sparse.issparse(x):
        return np.asarray(x.toarray())
    return np.asarray(x)


def _ensure_1d_or_2d(
    x: NDArray[np.number], y: NDArray[np.number]
) -> tuple[NDArray[np.number], NDArray[np.number]]:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim == 1:
        return x, y
    if x.ndim == 2:
        return x, y
    raise ValueError("x and y must be 1d or 2d arrays")


def pearson_metric(
    x: NDArray[number], y: NDArray[number]
) -> float | NDArray[np.number]:
    x_arr, y_arr = _ensure_1d_or_2d(np.asarray(x), np.asarray(y))
    if x_arr.ndim == 1:
        return pearsonr(x_arr, y_arr).statistic
    return np.array(
        [pearsonr(x_arr[:, i], y_arr[:, i]).statistic for i in range(x_arr.shape[1])]
    )


def spearman_metric(
    x: NDArray[number], y: NDArray[number]
) -> float | NDArray[np.number]:
    x_arr, y_arr = _ensure_1d_or_2d(np.asarray(x), np.asarray(y))
    if x_arr.ndim == 1:
        return spearmanr(x_arr, y_arr).statistic
    return np.array(
        [spearmanr(x_arr[:, i], y_arr[:, i]).statistic for i in range(x_arr.shape[1])]
    )


def mse_metric(x: NDArray[number], y: NDArray[number]) -> float | NDArray[np.number]:
    x_arr, y_arr = _ensure_1d_or_2d(np.asarray(x), np.asarray(y))
    if x_arr.ndim == 1:
        return mean_squared_error(x_arr, y_arr)
    return mean_squared_error(x_arr, y_arr, multioutput="raw_values")


def explained_variance_metric(
    x: NDArray[number], y: NDArray[number]
) -> float | NDArray[np.number]:
    x_arr, y_arr = _ensure_1d_or_2d(np.asarray(x), np.asarray(y))
    if x_arr.ndim == 1:
        return explained_variance_score(x_arr, y_arr)
    return explained_variance_score(x_arr, y_arr, multioutput="raw_values")


def cosine_similarity_metric(
    x: NDArray[number], y: NDArray[number]
) -> float | NDArray[np.number]:
    x_arr, y_arr = _ensure_1d_or_2d(np.asarray(x), np.asarray(y))
    if x_arr.ndim == 1:
        return cosine_similarity(x_arr.reshape(1, -1), y_arr.reshape(1, -1))[0, 0]
    numerator = np.sum(x_arr * y_arr, axis=0)
    x_norm = np.linalg.norm(x_arr, axis=0)
    y_norm = np.linalg.norm(y_arr, axis=0)
    denom = x_norm * y_norm
    return np.divide(
        numerator, denom, out=np.zeros_like(numerator, dtype=float), where=denom != 0
    )


def mae_metric(x: NDArray[number], y: NDArray[number]) -> float | NDArray[np.number]:
    x_arr, y_arr = _ensure_1d_or_2d(_to_dense_array(x), _to_dense_array(y))
    if x_arr.ndim == 1:
        return float(np.mean(np.abs(x_arr - y_arr)))
    return np.mean(np.abs(x_arr - y_arr), axis=0)


def explained_variance_metric_v2(x: NDArray[number], y: NDArray[number]) -> Any:
    x_arr, y_arr = _ensure_1d_or_2d(_to_dense_array(x), _to_dense_array(y))
    if x_arr.ndim == 1:
        mse = np.mean((x_arr - y_arr) ** 2)
        total_variance = np.var(x_arr)
        if total_variance == 0:
            return 0.0
        return 1 - (mse / total_variance)
    mse = np.mean((x_arr - y_arr) ** 2, axis=0)
    total_variance = np.var(x_arr, axis=0)
    explained_var = np.ones_like(mse, dtype=float)
    nonzero = total_variance != 0
    explained_var[nonzero] = 1 - (mse[nonzero] / total_variance[nonzero])
    explained_var[~nonzero] = 0.0
    return explained_var
