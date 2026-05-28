"""Metric calculation functions for probeset evaluation.

Functions:
    calculate_mse: Computes mean squared error between matrices.
    calculate_explained_variance: Computes explained variance (R²).
    calculate_macro_mse: Computes macro-averaged MSE across cell types.
    calculate_macro_explained_variance: Computes macro-averaged explained variance.
    calculate_weighted_mse: Computes cell-count-weighted MSE across cell types.
    calculate_weighted_explained_variance: Computes cell-count-weighted explained variance.
    calculate_weighted_mse_baseline: Computes cell-count-weighted baseline MSE.
    calculate_weighted_explained_variance_baseline: Computes cell-count-weighted baseline explained variance.
    compute_standardized_score: Computes standardized score across a grouping dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse
from nico2_lib.metrics import explained_variance_metric_v2, mse_metric

__all__ = [
    "calculate_mse",
    "calculate_explained_variance",
    "calculate_macro_mse",
    "calculate_macro_explained_variance",
    "calculate_weighted_mse",
    "calculate_weighted_explained_variance",
    "calculate_weighted_mse_baseline",
    "calculate_weighted_explained_variance_baseline",
    "calculate_inverse_frequency_weighted_f1",
    "compute_standardized_score",
]


def calculate_mse(
    X_original: np.ndarray | scipy.sparse.spmatrix,
    X_reconstructed: np.ndarray | scipy.sparse.spmatrix,
) -> float:
    """Calculate mean squared error between original and reconstructed matrices.

    Args:
        X_original: Original data matrix.
        X_reconstructed: Reconstructed data matrix from NMF.

    Returns:
        Mean squared error value.
    """
    if scipy.sparse.issparse(X_original):
        X_original = X_original.toarray()
    if scipy.sparse.issparse(X_reconstructed):
        X_reconstructed = X_reconstructed.toarray()

    return float(mse_metric(np.asarray(X_original).ravel(), np.asarray(X_reconstructed).ravel()))


def calculate_explained_variance(
    X_original: np.ndarray | scipy.sparse.spmatrix,
    X_reconstructed: np.ndarray | scipy.sparse.spmatrix,
) -> float:
    """Calculate explained variance (R²) between original and reconstructed matrices.

    Formula: R² = 1 - (MSE / Variance)
    Where:
        - MSE = (1/n) * Σ(X_original - X_reconstructed)²
        - Variance = (1/n) * Σ(X_original - mean(X_original))²

    This measures the proportion of variance in the original data that is explained
    by the reconstruction. Values closer to 1 indicate better reconstruction.

    Args:
        X_original: Original data matrix.
        X_reconstructed: Reconstructed data matrix from NMF.

    Returns:
        Explained variance value (between 0 and 1, can be negative for poor fits).
    """
    if scipy.sparse.issparse(X_original):
        X_original = X_original.toarray()
    if scipy.sparse.issparse(X_reconstructed):
        X_reconstructed = X_reconstructed.toarray()

    return float(explained_variance_metric_v2(np.asarray(X_original).ravel(), np.asarray(X_reconstructed).ravel()))


def calculate_macro_explained_variance(
    celltype_results: dict[str, dict[str, Any]],
) -> float:
    """Calculate macro-averaged explained variance across cell types.

    Macro average gives equal weight to each cell type, regardless of cell count.

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with 'expvar_test_probe' values.

    Returns:
        Macro-averaged explained variance.
    """
    valid_expvar = [
        m["expvar_test_probe"]
        for m in celltype_results.values()
        if not m.get("skipped", False) and not np.isnan(m.get("expvar_test_probe", np.nan))
    ]
    return float(np.mean(valid_expvar)) if valid_expvar else np.nan


def calculate_macro_mse(celltype_results: dict[str, dict[str, Any]]) -> float:
    """Calculate macro-averaged MSE across cell types.

    Macro average gives equal weight to each cell type, regardless of cell count.

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with 'mse_test_probe' values.

    Returns:
        Macro-averaged MSE.
    """
    valid_mse = [
        m["mse_test_probe"]
        for m in celltype_results.values()
        if not m.get("skipped", False) and not np.isnan(m.get("mse_test_probe", np.nan))
    ]
    return float(np.mean(valid_mse)) if valid_mse else np.nan


def calculate_weighted_mse(celltype_results: dict[str, dict[str, Any]]) -> float:
    """Calculate weighted MSE across cell types (weighted by number of cells).

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with 'mse_test_probe' and 'n_cells' values.

    Returns:
        Weighted MSE.
    """
    valid = {
        ct: res
        for ct, res in celltype_results.items()
        if not res.get("skipped", False) and "mse_test_probe" in res and "n_cells" in res
    }
    if not valid:
        return np.nan
    total = sum(r["n_cells"] for r in valid.values())
    if total == 0:
        return np.nan
    return float(sum(r["mse_test_probe"] * r["n_cells"] for r in valid.values()) / total)


def calculate_weighted_explained_variance(
    celltype_results: dict[str, dict[str, Any]],
) -> float:
    """Calculate weighted explained variance across cell types (weighted by number of cells).

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with 'expvar_test_probe' and 'n_cells' values.

    Returns:
        Weighted explained variance.
    """
    valid = {
        ct: res
        for ct, res in celltype_results.items()
        if not res.get("skipped", False) and "expvar_test_probe" in res and "n_cells" in res
    }
    if not valid:
        return np.nan
    total = sum(r["n_cells"] for r in valid.values())
    if total == 0:
        return np.nan
    return float(sum(r["expvar_test_probe"] * r["n_cells"] for r in valid.values()) / total)


def calculate_weighted_mse_baseline(celltype_results: dict[str, dict[str, Any]]) -> float:
    """Calculate weighted baseline MSE across cell types (weighted by number of cells).

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with 'mse_test_baseline' and 'n_cells' values.

    Returns:
        Weighted baseline MSE.
    """
    valid = {
        ct: res
        for ct, res in celltype_results.items()
        if not res.get("skipped", False) and "mse_test_baseline" in res and "n_cells" in res
    }
    if not valid:
        return np.nan
    total = sum(r["n_cells"] for r in valid.values())
    if total == 0:
        return np.nan
    return float(sum(r["mse_test_baseline"] * r["n_cells"] for r in valid.values()) / total)


def calculate_weighted_explained_variance_baseline(
    celltype_results: dict[str, dict[str, Any]],
) -> float:
    """Calculate weighted baseline explained variance across cell types (weighted by number of cells).

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with 'expvar_test_baseline' and 'n_cells' values.

    Returns:
        Weighted baseline explained variance.
    """
    valid = {
        ct: res
        for ct, res in celltype_results.items()
        if not res.get("skipped", False) and "expvar_test_baseline" in res and "n_cells" in res
    }
    if not valid:
        return np.nan
    total = sum(r["n_cells"] for r in valid.values())
    if total == 0:
        return np.nan
    return float(sum(r["expvar_test_baseline"] * r["n_cells"] for r in valid.values()) / total)


def calculate_inverse_frequency_weighted_f1(
    per_celltype_f1: dict[str, float],
    per_celltype_n_cells: dict[str, int],
    weighting: str = "inverse",
) -> float:
    """Calculate inverse-frequency weighted F1, amplifying rare cell types.

    Standard macro F1 gives equal weight per cell type; weighted F1 upweights
    abundant cell types. This function inverts that: rare cell types receive
    higher weights so panels that identify them well score higher.

    Args:
        per_celltype_f1: F1 score per cell type (NaN values are skipped).
        per_celltype_n_cells: Number of cells per cell type in the full dataset.
        weighting: "inverse" for 1/n_cells weights, "log" for log(N_total/n_CT).

    Returns:
        Inverse-frequency weighted F1 score, or np.nan if no valid cell types.
    """
    valid_cts = [
        ct for ct in per_celltype_f1
        if ct in per_celltype_n_cells
        and not np.isnan(per_celltype_f1[ct])
        and per_celltype_n_cells[ct] > 0
    ]
    if not valid_cts:
        return float(np.nan)

    n_total = sum(per_celltype_n_cells[ct] for ct in valid_cts)
    if weighting == "log":
        raw_weights = {ct: np.log(n_total / per_celltype_n_cells[ct]) for ct in valid_cts}
    else:
        raw_weights = {ct: 1.0 / per_celltype_n_cells[ct] for ct in valid_cts}

    weight_sum = sum(raw_weights.values())
    return float(sum(raw_weights[ct] / weight_sum * per_celltype_f1[ct] for ct in valid_cts))


def compute_standardized_score(
    metrics_df: "pd.DataFrame",
    groupby_col: str | list[str],
    metric_cols: list[str] | None = None,
) -> "pd.DataFrame":
    """Compute standardized score across a grouping dimension.

    Args:
        metrics_df: DataFrame with aggregate metrics.
        groupby_col: Column(s) to group by (e.g., 'method', 'iteration', 'fold').
        metric_cols: List of metrics to standardize. Defaults to
            ['macro_mse', 'macro_expvar', 'weighted_mse', 'weighted_expvar'].

    Returns:
        DataFrame with additional 'standardized_score' column.

    Note:
        Standardized score is computed as the mean of z-scored metrics,
        with MSE metrics negated (lower is better) before averaging.
        Z-scores are computed within each group separately to avoid
        scale leakage between different methods or scenarios.
    """
    import pandas as pd

    if metric_cols is None:
        metric_cols = ["macro_mse", "macro_expvar", "weighted_mse", "weighted_expvar"]

    results = []
    groupby_cols = [groupby_col] if isinstance(groupby_col, str) else groupby_col

    for _group_val, group_df in metrics_df.groupby(groupby_cols, dropna=False):
        z_scores = []
        for metric in metric_cols:
            if metric not in group_df.columns:
                continue
            mean = group_df[metric].mean()
            std = group_df[metric].std(ddof=1)
            if std > 0:
                z = (group_df[metric] - mean) / std
                if "mse" in metric:
                    z = -z
                z_scores.append(z)

        group_df = group_df.copy()
        group_df["standardized_score"] = np.mean(z_scores, axis=0) if z_scores else np.nan
        results.append(group_df)

    return pd.concat(results, ignore_index=True)
