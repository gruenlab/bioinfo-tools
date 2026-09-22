"""Metric calculation functions for probeset evaluation.

Functions:
    calculate_mse: Computes mean squared error between matrices.
    calculate_explained_variance: Computes explained variance (R²), with a selectable
        gene-aggregation mode.
    calculate_macro_mse: Computes macro-averaged MSE across cell types.
    calculate_macro_explained_variance: Computes macro-averaged explained variance.
    calculate_weighted_mse: Computes cell-count-weighted MSE across cell types.
    calculate_weighted_explained_variance: Computes cell-count-weighted explained variance.
    calculate_weighted_mse_baseline: Computes cell-count-weighted baseline MSE.
    calculate_weighted_explained_variance_baseline: Computes cell-count-weighted baseline explained variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse
from nico2_lib.metrics import explained_variance_metric_v2, mse_metric

EXPVAR_MODES = (
    "global_mean",
    "variance_weighted_sum",
    "mean",
    "median",
    "expression_weighted_sum",
)

__all__ = [
    "calculate_mse",
    "calculate_explained_variance",
    "EXPVAR_MODES",
    "calculate_macro_mse",
    "calculate_macro_explained_variance",
    "calculate_weighted_mse",
    "calculate_weighted_explained_variance",
    "calculate_weighted_mse_baseline",
    "calculate_weighted_explained_variance_baseline",
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
    mode: str = "global_mean",
) -> float:
    """Calculate explained variance (R²) between original and reconstructed matrices.

    Five gene-aggregation modes are available (see ``EXPVAR_MODES``):

    - ``"global_mean"`` (default): pools every cell x gene entry into one 1D
      array and computes a single R² around one global scalar mean via
      ``explained_variance_metric_v2``'s 1D branch. Lets between-gene mean
      differences (e.g. genes with very different absolute expression levels)
      leak into the variance denominator; see
      docs/doc-pipeline/code-versions-documentation.md for the rationale and the
      trade-off vs. the per-gene-centred modes below. It is the default so
      two-argument call sites reproduce a single, stable convention.
    - ``"variance_weighted_sum"``: R² = 1 - sum_g(MSE_g) / sum_g(Var_g), i.e.
      the variance-weighted average of per-gene R² -- algebraically
      equivalent to summing per-gene MSE and per-gene variance separately
      before taking one ratio. Removes the between-gene term that
      ``"global_mean"`` is susceptible to.
    - ``"mean"`` / ``"median"``: unweighted, equal-footing average/median of
      per-gene R² across genes -- every gene counts equally regardless of its
      expression level (NOT expression-weighted).
    - ``"expression_weighted_sum"``: per-gene R² weighted by each gene's mean
      expression across cells, instead of by variance.

    All modes except ``"global_mean"`` delegate the per-gene R² to
    ``explained_variance_metric_v2``'s 2D branch (each gene's MSE and
    variance computed around its own mean, axis=0 over cells).

    Args:
        X_original: Original data matrix, shape (n_cells, n_genes).
        X_reconstructed: Reconstructed data matrix from NMF, same shape.
        mode: Aggregation mode, one of ``EXPVAR_MODES``. Defaults to
            ``"global_mean"``.

    Returns:
        Explained variance per the chosen mode; 0.0 if the relevant total
        weight (variance or expression) is exactly zero.

    Raises:
        ValueError: If ``mode`` is not one of ``EXPVAR_MODES``.
    """
    if scipy.sparse.issparse(X_original):
        X_original = X_original.toarray()
    if scipy.sparse.issparse(X_reconstructed):
        X_reconstructed = X_reconstructed.toarray()

    X_original = np.asarray(X_original)
    X_reconstructed = np.asarray(X_reconstructed)

    if mode == "global_mean":
        return float(explained_variance_metric_v2(X_original.ravel(), X_reconstructed.ravel()))

    if mode not in EXPVAR_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Must be one of {EXPVAR_MODES}.")

    per_gene_expvar = explained_variance_metric_v2(X_original, X_reconstructed)

    if mode == "mean":
        return float(np.mean(per_gene_expvar))
    if mode == "median":
        return float(np.median(per_gene_expvar))

    weights = np.var(X_original, axis=0) if mode == "variance_weighted_sum" else np.mean(X_original, axis=0)
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float(np.sum(weights * per_gene_expvar) / total_weight)


def calculate_macro_explained_variance(
    celltype_results: dict[str, dict[str, Any]],
    metric_key: str = "expvar_test_probe",
) -> float:
    """Calculate macro-averaged explained variance across cell types.

    Macro average gives equal weight to each cell type, regardless of cell count.

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with ``metric_key`` values.
        metric_key: Which per-celltype key to average (default
            ``"expvar_test_probe"``, i.e. all-genes). Pass e.g.
            ``"expvar_test_probe_panel_genes_only_variance_weighted_sum"`` to
            aggregate a specific gene-subset/mode combination instead.

    Returns:
        Macro-averaged explained variance.
    """
    valid_expvar = [
        m[metric_key]
        for m in celltype_results.values()
        if not m.get("skipped", False) and not np.isnan(m.get(metric_key, np.nan))
    ]
    return float(np.mean(valid_expvar)) if valid_expvar else np.nan


def calculate_macro_mse(
    celltype_results: dict[str, dict[str, Any]],
    metric_key: str = "mse_test_probe",
) -> float:
    """Calculate macro-averaged MSE across cell types.

    Macro average gives equal weight to each cell type, regardless of cell count.

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with ``metric_key`` values.
        metric_key: Which per-celltype key to average (default
            ``"mse_test_probe"``, i.e. all-genes). Pass e.g.
            ``"mse_test_probe_panel_genes_only"`` to aggregate a specific
            gene-subset instead (MSE has no aggregation-mode axis).

    Returns:
        Macro-averaged MSE.
    """
    valid_mse = [
        m[metric_key]
        for m in celltype_results.values()
        if not m.get("skipped", False) and not np.isnan(m.get(metric_key, np.nan))
    ]
    return float(np.mean(valid_mse)) if valid_mse else np.nan


def calculate_weighted_mse(
    celltype_results: dict[str, dict[str, Any]],
    metric_key: str = "mse_test_probe",
) -> float:
    """Calculate weighted MSE across cell types (weighted by number of cells).

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with ``metric_key`` and 'n_cells' values.
        metric_key: Which per-celltype key to weight-average (default
            ``"mse_test_probe"``, i.e. all-genes). Pass e.g.
            ``"mse_test_probe_panel_genes_only"`` to aggregate a specific
            gene-subset instead (MSE has no aggregation-mode axis).

    Returns:
        Weighted MSE.
    """
    valid = {
        ct: res
        for ct, res in celltype_results.items()
        if not res.get("skipped", False) and metric_key in res and "n_cells" in res
    }
    if not valid:
        return np.nan
    total = sum(r["n_cells"] for r in valid.values())
    if total == 0:
        return np.nan
    return float(sum(r[metric_key] * r["n_cells"] for r in valid.values()) / total)


def calculate_weighted_explained_variance(
    celltype_results: dict[str, dict[str, Any]],
    metric_key: str = "expvar_test_probe",
) -> float:
    """Calculate weighted explained variance across cell types (weighted by number of cells).

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with ``metric_key`` and 'n_cells' values.
        metric_key: Which per-celltype key to weight-average (default
            ``"expvar_test_probe"``, i.e. all-genes). Pass e.g.
            ``"expvar_test_probe_panel_genes_only_variance_weighted_sum"`` to
            aggregate a specific gene-subset/mode combination instead.

    Returns:
        Weighted explained variance.
    """
    valid = {
        ct: res
        for ct, res in celltype_results.items()
        if not res.get("skipped", False) and metric_key in res and "n_cells" in res
    }
    if not valid:
        return np.nan
    total = sum(r["n_cells"] for r in valid.values())
    if total == 0:
        return np.nan
    return float(sum(r[metric_key] * r["n_cells"] for r in valid.values()) / total)


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
    metric_key: str = "expvar_test_baseline",
) -> float:
    """Calculate weighted baseline explained variance across cell types (weighted by number of cells).

    Args:
        celltype_results: Dictionary containing cell type-specific results
            with ``metric_key`` and 'n_cells' values.
        metric_key: Which per-celltype key to weight-average (default
            ``"expvar_test_baseline"``, i.e. all-genes). Baseline is only ever
            computed on the all-genes scale (the baseline NMF fit doesn't
            depend on which panel is being reconstructed), so in practice this
            only varies by aggregation mode, e.g.
            ``"expvar_test_baseline_variance_weighted_sum"``.

    Returns:
        Weighted baseline explained variance.
    """
    valid = {
        ct: res
        for ct, res in celltype_results.items()
        if not res.get("skipped", False) and metric_key in res and "n_cells" in res
    }
    if not valid:
        return np.nan
    total = sum(r["n_cells"] for r in valid.values())
    if total == 0:
        return np.nan
    return float(sum(r[metric_key] * r["n_cells"] for r in valid.values()) / total)
