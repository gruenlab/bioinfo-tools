"""Tangram-based reconstruction check for probe panel evaluation.

This module provides an optional reconstruction check that uses Tangram
to map probe-panel expression back to the full transcriptome. It is
disabled by default and only runs when explicitly requested via the
``--include-tangram`` flag or ``EvaluationConfig.include_tangram = True``.

Note:
    Tangram operates on the **full dataset** (no train/test split). This is
    a method-level limitation rather than a pipeline choice.

Usage::

    from evaluation._reconstruction import run_tangram_reconstruction_check

    if config.include_tangram:
        tangram_results = run_tangram_reconstruction_check(
            adata_full=adata_reference,
            adata_subset=adata_panel,
            output_dir=config.output_dir / "Tangram-Evaluation",
            dataset_name=panel_name,
            celltype_col=config.celltype_col,
            num_epochs=config.tangram_n_epochs,
        )
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Import constants from the local _constants.py file (same directory as this file)
# Using importlib to avoid conflicts with _constants.py in other modules
import importlib.util
_recon_current_dir = Path(__file__).parent.absolute()
_recon_constants_path = _recon_current_dir / "_constants.py"
_recon_spec = importlib.util.spec_from_file_location("_eval_constants_recon", _recon_constants_path)
_eval_constants_recon = importlib.util.module_from_spec(_recon_spec)
_recon_spec.loader.exec_module(_eval_constants_recon)
MIN_CELLS_PER_CELLTYPE = _eval_constants_recon.MIN_CELLS_PER_CELLTYPE

from metrics import (
    calculate_mse,
    calculate_explained_variance,
    calculate_macro_mse,
    calculate_macro_explained_variance,
    calculate_weighted_mse,
    calculate_weighted_explained_variance,
)

logger = logging.getLogger(__name__)

_UTILITY_DIR = Path(__file__).parent.parent / "Utility-module"
if _UTILITY_DIR.exists():
    sys.path.insert(0, str(_UTILITY_DIR))
try:
    from _validation import is_anndata_raw_layer, is_anndata_raw  # type: ignore[import]
except ImportError:
    def is_anndata_raw_layer(adata, layer_name: str) -> bool:  # type: ignore[misc]
        return True
    def is_anndata_raw(adata) -> bool:  # type: ignore[misc]
        return True

__all__ = [
    "reconstruct_with_tangram",
    "run_tangram_reconstruction_check",
    "MIN_CELLS_PER_CELLTYPE",
]

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import tangram as tg
    _TANGRAM_AVAILABLE = True
except ImportError:
    tg = None
    _TANGRAM_AVAILABLE = False
    logger.debug("tangram not installed – reconstruction check will be unavailable.")


# ---------------------------------------------------------------------------
# Tangram helpers — imported from nico2_lib (identical unfiltered wrappers)
# ---------------------------------------------------------------------------

from nico2_lib.predictors._tangram._tangram_pred import (
    pp_adatas_unfiltered,
    map_cells_to_space as _map_cells_to_space_unfiltered,
    project_genes_unfiltered,
)


# ---------------------------------------------------------------------------
# Core reconstruction functions
# ---------------------------------------------------------------------------


def reconstruct_with_tangram(
    adata_full: sc.AnnData,
    adata_subset: sc.AnnData,
    num_epochs: int = 1000,
    strict_deps: bool = False,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    nmf_counts_input: str = "raw",
) -> sc.AnnData:
    """Reconstruct full transcriptome from probe genes using Tangram.

    When *train_idx* and *test_idx* are provided, the mapping is learned on
    the training cells and evaluated on the held-out test cells:

    - ``adata_sc`` (sc reference) = ``adata_full[train_idx]``
    - ``adata_sp`` (spatial target) = ``adata_subset[test_idx]``

    The returned ``AnnData`` then corresponds to the *test* cells.  When
    neither index is provided the full dataset is used (original behaviour).

    Args:
        adata_full: Full-transcriptome AnnData (all cells).
        adata_subset: Probe-panel AnnData (same cells, probe genes only).
        num_epochs: Number of Tangram optimisation epochs.
        strict_deps: If ``True``, raise ``ImportError`` when Tangram is not
            installed. If ``False`` (default), raise only if called directly.
        train_idx: Integer position array of training cells into *adata_full*.
            Must be provided together with *test_idx*.
        test_idx: Integer position array of test cells into *adata_full*.
            Must be provided together with *train_idx*.

    Returns:
        AnnData with reconstructed gene-expression values (test cells when
        *train_idx*/*test_idx* are given, all cells otherwise).

    Raises:
        ImportError: If Tangram is not installed.

    Example:
        >>> ad_ge = reconstruct_with_tangram(adata_ref, adata_panel, num_epochs=300)
        >>> ad_ge.shape
        (n_cells, n_full_genes)
    """
    if not _TANGRAM_AVAILABLE:
        raise ImportError(
            "Tangram is not installed. Install with: pip install tangram-sc"
        )

    import torch

    logger.info("Running Tangram reconstruction (num_epochs=%d)...", num_epochs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Determine which cells to use as sc reference and spatial target
    if train_idx is not None and test_idx is not None:
        logger.info(
            "Train/test split: %d training cells → sc reference, "
            "%d test cells → spatial target.",
            len(train_idx), len(test_idx),
        )
        ad_sc = adata_full[train_idx].copy()
        ad_sp = adata_subset[test_idx].copy()
    else:
        # Work on copies to avoid in-place modification of caller objects
        ad_sc = adata_full.copy()
        ad_sp = adata_subset.copy()

    if nmf_counts_input == "raw":
        if "counts" not in ad_sc.layers or "counts" not in ad_sp.layers:
            raise ValueError(
                "reconstruct_with_tangram requires 'counts' layer with raw data when "
                "nmf_counts_input='raw'. "
                f"Found layers: ad_sc={list(ad_sc.layers.keys())}, ad_sp={list(ad_sp.layers.keys())}"
            )
        if not is_anndata_raw_layer(ad_sc, "counts"):
            raise ValueError(
                "nmf_counts_input='raw': ad_sc.layers['counts'] does not contain raw integer counts"
            )
        if not is_anndata_raw_layer(ad_sp, "counts"):
            raise ValueError(
                "nmf_counts_input='raw': ad_sp.layers['counts'] does not contain raw integer counts"
            )
        logger.info("Extracting raw counts from layers['counts'] for Tangram (verified)")
        ad_sc.X = ad_sc.layers["counts"].copy()
        ad_sp.X = ad_sp.layers["counts"].copy()
    elif nmf_counts_input == "lognorm":
        if is_anndata_raw(ad_sc):
            raise ValueError(
                "nmf_counts_input='lognorm': ad_sc.X appears to contain raw integer counts, "
                "not log-normalized data."
            )
        if is_anndata_raw(ad_sp):
            raise ValueError(
                "nmf_counts_input='lognorm': ad_sp.X appears to contain raw integer counts, "
                "not log-normalized data."
            )
        logger.info("Using adata.X (log-normalized) for Tangram (verified)")
    else:
        raise ValueError(
            f"Unknown nmf_counts_input='{nmf_counts_input}'. Choose 'raw' or 'lognorm'."
        )

    logger.info("Preprocessing data (zero-count gene filtering disabled to preserve full gene set)...")
    pp_adatas_unfiltered(ad_sc, ad_sp, genes=None)

    # ad_map is a cell-by-voxel structure where ad_map.X[i, j] gives the probability for cell i to be in voxel j
    ad_map = _map_cells_to_space_unfiltered(
        ad_sc,
        ad_sp,
        mode="cells",
        density_prior="rna_count_based",
        num_epochs=num_epochs,
        device=device,
    )

    # ad_ge is a voxel-by-gene AnnData similar to spatial data ad_sp, but where gene expression has been projected from the single cells
    logger.info("Tangram mapping complete – projecting genes...")
    ad_ge = project_genes_unfiltered(adata_map=ad_map, adata_sc=ad_sc)
    logger.info("Tangram reconstruction complete.")
    return ad_ge


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _calculate_reconstruction_metrics(
    X_ref: np.ndarray,
    X_pred: np.ndarray,
) -> dict[str, float]:
    """Compute a standard suite of reconstruction quality metrics.

    Args:
        X_ref: Original data matrix (n_cells × n_genes).
        X_pred: Reconstructed data matrix (n_cells × n_genes).

    Returns:
        Dictionary with keys ``mse``, ``expvar``, ``rmse``, ``mae``,
        ``r2``, ``pearson``, ``n_cells``, ``n_genes``.

    Note:
        Uses shared metric helpers from _variability.py to ensure identical
        computation to NMF evaluation for cross-method comparability.
    """
    # Use shared helpers to ensure identical computation to NMF
    mse = calculate_mse(X_ref, X_pred)
    expvar = calculate_explained_variance(X_ref, X_pred)

    # Keep diagnostic variables for logging
    residual_var = float(np.var(X_ref - X_pred))
    total_var = float(np.var(X_ref))

    return {
        "mse": mse,
        "expvar": expvar,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(X_ref - X_pred))),
        "r2": float(r2_score(X_ref.flatten(), X_pred.flatten())),
        "pearson": float(pearsonr(X_ref.flatten(), X_pred.flatten())[0]),
        "n_cells": int(X_ref.shape[0]),
        "n_genes": int(X_ref.shape[1]),
        # Include diagnostic vars for transparency
        "residual_var": residual_var,
        "total_var": total_var,
    }


def _adata_to_dense(adata: sc.AnnData) -> np.ndarray:
    """Return the expression matrix as a dense float32 array.

    Args:
        adata: AnnData object.

    Returns:
        Dense (n_cells × n_genes) float32 array.
    """
    import scipy.sparse

    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _adata_to_dense_layer(adata: sc.AnnData, layer_name: str) -> np.ndarray:
    """Return a specific layer as a dense float32 array.

    Args:
        adata: AnnData object.
        layer_name: Name of the layer to extract.

    Returns:
        Dense (n_cells × n_genes) float32 array.

    Raises:
        ValueError: If layer not found in adata.layers.
    """
    import scipy.sparse

    if layer_name not in adata.layers:
        raise ValueError(
            f"Layer '{layer_name}' not found. Available: {list(adata.layers.keys())}"
        )

    X = adata.layers[layer_name]
    if scipy.sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-cell-type helpers
# ---------------------------------------------------------------------------


def _run_tangram_per_celltype(
    adata_full: sc.AnnData,
    adata_subset: sc.AnnData,
    celltype_col: str,
    num_epochs: int,
    min_cells: int = MIN_CELLS_PER_CELLTYPE,
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    nmf_counts_input: str = "raw",
) -> dict[str, dict[str, Any]]:
    """Run Tangram reconstruction separately for each cell type.

    Args:
        adata_full: Full-transcriptome AnnData.
        adata_subset: Probe-panel AnnData (same cells).
        celltype_col: obs column holding cell-type labels.
        num_epochs: Number of Tangram epochs per cell type.
        min_cells: Minimum cells needed to attempt reconstruction.
        per_celltype_splits: Pre-computed per-celltype train/test index splits,
            as returned by :func:`_variability.make_celltype_evaluation_splits`.
            When provided, training cells serve as the sc reference and test
            cells as the spatial target, so metrics are computed on held-out
            data only.  Pass ``None`` to use all cells (original behaviour).

    Returns:
        Mapping of ``{celltype: metrics_dict}`` where each value has the
        same structure as :func:`_calculate_reconstruction_metrics` plus
        a ``"skipped"`` boolean.
    """
    results: dict[str, dict[str, Any]] = {}

    if celltype_col not in adata_full.obs.columns:
        logger.warning(
            "Cell-type column '%s' not found – skipping per-cell-type Tangram.", celltype_col
        )
        return results

    celltypes = adata_full.obs[celltype_col].unique()
    logger.info("Running per-cell-type Tangram for %d cell types.", len(celltypes))

    for ct in celltypes:
        mask = adata_full.obs[celltype_col] == ct
        n_cells = int(mask.sum())

        if n_cells < min_cells:
            logger.warning(
                "Skipping '%s': only %d cells (minimum: %d).", ct, n_cells, min_cells
            )
            results[ct] = {"skipped": True, "skip_reason": "insufficient_cells", "n_cells": n_cells}
            continue

        try:
            # Determine train/test indices for this cell type.
            # Indices from per_celltype_splits are absolute positions into
            # adata_full, so we pass adata_full directly (not the celltype
            # subset) when a split is provided.
            if per_celltype_splits is not None and ct in per_celltype_splits:
                train_ct_idx, test_ct_idx = per_celltype_splits[ct]
                ad_ge = reconstruct_with_tangram(
                    adata_full, adata_subset, num_epochs=num_epochs,
                    train_idx=train_ct_idx, test_idx=test_ct_idx,
                    nmf_counts_input=nmf_counts_input,
                )
                ad_ct_ref = adata_full[test_ct_idx]
            else:
                ad_ct_full = adata_full[mask].copy()
                ad_ct_sub = adata_subset[mask].copy()
                ad_ge = reconstruct_with_tangram(
                    ad_ct_full, ad_ct_sub, num_epochs=num_epochs,
                    nmf_counts_input=nmf_counts_input,
                )
                ad_ct_ref = ad_ct_full

            # Align reference to the genes Tangram reconstructed.
            # Tangram lowercases var_names; build a case-insensitive lookup so
            # we can index adata_full (original case) with ad_ge gene names (lowercase).
            lower_to_orig = {g.lower(): g for g in adata_full.var_names}
            ref_genes = []
            ge_genes  = []
            for g in ad_ge.var_names:
                orig = lower_to_orig.get(g.lower())
                if orig is not None:
                    ref_genes.append(orig)
                    ge_genes.append(g)
            X_ref  = _adata_to_dense(ad_ct_ref[:, ref_genes])
            X_pred = _adata_to_dense(ad_ge[:, ge_genes])

            metrics = _calculate_reconstruction_metrics(X_ref, X_pred)
            metrics["skipped"] = False
            metrics["_mean_ref"]   = X_ref.mean(axis=0)
            metrics["_mean_pred"]  = X_pred.mean(axis=0)
            metrics["_gene_names"] = ref_genes
            results[ct] = metrics
            logger.info(
                "  %s: MSE=%.4f, ExpVar=%.4f", ct, metrics["mse"], metrics["expvar"]
            )

        except Exception as exc:
            logger.warning("Tangram failed for '%s': %s. Skipping.", ct, exc, exc_info=True)
            results[ct] = {"skipped": True, "skip_reason": str(exc), "n_cells": n_cells}

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_per_celltype_metrics(
    per_celltype_results: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Compute macro and weighted summary metrics over per-cell-type results.

    Args:
        per_celltype_results: Output of :func:`_run_tangram_per_celltype`.

    Returns:
        Dictionary with keys ``macro_mse``, ``macro_expvar``,
        ``weighted_mse``, ``weighted_expvar``, and ``n_celltypes_used``.

    Note:
        Uses shared aggregation helpers from _variability.py to ensure identical
        computation to NMF evaluation for cross-method comparability.
    """
    # Filter out skipped celltypes and convert to format expected by shared helpers
    # The shared helpers expect keys like "mse_test_probe" and "expvar_test_probe"
    valid_results = {}
    for ct, metrics in per_celltype_results.items():
        if not metrics.get("skipped", False) and "mse" in metrics and "expvar" in metrics:
            valid_results[ct] = {
                "mse_test_probe": metrics["mse"],
                "expvar_test_probe": metrics["expvar"],
                "n_cells": metrics["n_cells"],
                "skipped": False,
            }

    # Use shared aggregation helpers (identical to NMF)
    macro_mse_val = calculate_macro_mse(valid_results)
    macro_expvar_val = calculate_macro_explained_variance(valid_results)
    weighted_mse_val = calculate_weighted_mse(valid_results)
    weighted_expvar_val = calculate_weighted_explained_variance(valid_results)

    # Count valid cell types
    n_valid = len(valid_results)

    return {
        "macro_mse": macro_mse_val,
        "macro_expvar": macro_expvar_val,
        "weighted_mse": weighted_mse_val,
        "weighted_expvar": weighted_expvar_val,
        "n_celltypes_used": n_valid,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_tangram_results(
    output_dir: str | Path,
    dataset_name: str,
    global_metrics: dict[str, Any] | None,
    per_celltype_results: dict[str, dict[str, Any]] | None,
    per_celltype_summary: dict[str, float] | None = None,
) -> None:
    """Persist Tangram results to CSV files.

    Creates ``global/`` and ``per_celltype/`` sub-directories under
    *output_dir*.

    Args:
        output_dir: Root Tangram output directory.
        dataset_name: Panel / dataset identifier used as the filename stem.
        global_metrics: Metrics from global reconstruction, or ``None``.
        per_celltype_results: Per-cell-type metrics dict, or ``None``.
    """
    output_dir = Path(output_dir)

    if global_metrics is not None:
        global_dir = output_dir / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{**global_metrics, "dataset": dataset_name, "mode": "global"}])
        df.to_csv(global_dir / f"{dataset_name}.csv", index=False)
        logger.info("Saved global Tangram metrics to %s", global_dir / f"{dataset_name}.csv")

    if per_celltype_results:
        ct_dir = output_dir / "per_celltype"
        ct_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"celltype": ct, "dataset": dataset_name, "mode": "per_celltype",
             **{k: v for k, v in m.items() if not k.startswith("_")}}
            for ct, m in per_celltype_results.items()
        ]
        # Append summary row at the bottom
        if per_celltype_summary:
            rows.append({
                "celltype": "__summary__",
                "dataset": dataset_name,
                "mode": "per_celltype_summary",
                **per_celltype_summary,
            })
        df = pd.DataFrame(rows)
        df.to_csv(ct_dir / f"{dataset_name}.csv", index=False)
        logger.info("Saved per-cell-type Tangram metrics to %s", ct_dir / f"{dataset_name}.csv")


# ---------------------------------------------------------------------------
# Alignment heatmap
# ---------------------------------------------------------------------------


def _plot_tangram_alignment_heatmap(
    mean_ref: np.ndarray,
    mean_pred: np.ndarray,
    gene_names: list[str],
    row_labels: list[str],
    output_path: "Path",
    title: str,
    max_genes: int = None,
) -> None:
    """Save a 2-panel figure comparing original vs. reconstructed mean expression.

    Panels:
      1. Heatmap of original mean expression (rows = groups, cols = genes).
      2. Heatmap of Tangram-reconstructed mean expression (same layout).

    Args:
        mean_ref: Original mean expression matrix, shape ``(n_groups, n_genes)``.
        mean_pred: Reconstructed mean expression, same shape.
        gene_names: Gene names corresponding to columns of *mean_ref*.
        row_labels: Label for each row (cell types or ``["all cells"]``).
        output_path: File path for the saved PNG.
        title: Figure suptitle.
        max_genes: Deprecated parameter (kept for backwards compatibility but ignored).
            Now always shows all genes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_ref  = np.asarray(mean_ref,  dtype=float)
    mean_pred = np.asarray(mean_pred, dtype=float)

    # Use all genes (no filtering)
    n_genes = len(gene_names)
    n_rows = len(row_labels)

    # Shared colour scale for both heatmap panels
    vmin = float(min(mean_ref.min(), mean_pred.min()))
    vmax = float(max(mean_ref.max(), mean_pred.max()))

    # Calculate figure dimensions
    fig_h  = max(4.0, n_rows * 0.35 + 1.5)
    fig_w  = max(12.0, n_genes * 0.15)

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    _xt = list(range(n_genes))
    _yt = list(range(n_rows))

    # Determine font size based on number of genes
    gene_fontsize = 5 if n_genes > 50 else (6 if n_genes > 30 else 7)

    # Panel 1 — original
    im = axes[0].imshow(mean_ref, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Original", fontsize=10)
    axes[0].set_yticks(_yt)
    axes[0].set_yticklabels(row_labels, fontsize=7)
    axes[0].set_xticks(_xt)
    axes[0].set_xticklabels(gene_names, rotation=90, fontsize=gene_fontsize)
    axes[0].set_xlabel(f"All genes (n={n_genes})", fontsize=8)
    plt.colorbar(im, ax=axes[0], shrink=0.6)

    # Panel 2 — reconstructed
    im2 = axes[1].imshow(mean_pred, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Tangram reconstructed", fontsize=10)
    axes[1].set_yticks(_yt)
    axes[1].set_yticklabels(row_labels, fontsize=7)
    axes[1].set_xticks(_xt)
    axes[1].set_xticklabels(gene_names, rotation=90, fontsize=gene_fontsize)
    axes[1].set_xlabel(f"All genes (n={n_genes})", fontsize=8)
    plt.colorbar(im2, ax=axes[1], shrink=0.6)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved alignment heatmap to %s", output_path)


def _plot_tangram_correlation_heatmap(
    mean_ref: np.ndarray,
    mean_pred: np.ndarray,
    gene_names: list[str],
    row_labels: list[str],
    output_path: "Path",
    title: str,
) -> None:
    """Save a heatmap showing reconstruction quality metrics per cell type.

    Creates a cell-type × metric heatmap with:
      - Pearson correlation (original vs reconstructed mean expression)
      - MSE (mean squared error)
      - Explained variance

    Strategy: Average gene expression across cells of each cell type first,
    then compute metrics between original and reconstructed mean profiles.

    Args:
        mean_ref: Original mean expression matrix, shape ``(n_groups, n_genes)``.
        mean_pred: Reconstructed mean expression, same shape.
        gene_names: Gene names corresponding to columns of *mean_ref*.
        row_labels: Label for each row (cell types or ``["all cells"]``).
        output_path: File path for the saved PNG.
        title: Figure suptitle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr as _pearsonr

    mean_ref  = np.asarray(mean_ref,  dtype=float)
    mean_pred = np.asarray(mean_pred, dtype=float)

    n_celltypes = mean_ref.shape[0]
    n_genes = mean_ref.shape[1]

    # Compute metrics per cell type
    metrics = []
    for i in range(n_celltypes):
        ref_profile = mean_ref[i, :]
        pred_profile = mean_pred[i, :]

        # Pearson correlation
        if n_genes > 1:
            corr, _ = _pearsonr(ref_profile, pred_profile)
        else:
            corr = np.nan

        # MSE
        mse = float(np.mean((ref_profile - pred_profile) ** 2))

        # Explained variance (1 - residual_variance / total_variance)
        total_var = float(np.var(ref_profile))
        residual_var = float(np.var(ref_profile - pred_profile))
        expvar = 1.0 - (residual_var / total_var) if total_var > 0 else np.nan

        metrics.append({
            "celltype": row_labels[i],
            "Pearson R": corr,
            "MSE": mse,
            "Explained Var": expvar,
        })

    # Create DataFrame
    df = pd.DataFrame(metrics).set_index("celltype")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(6, max(4, n_celltypes * 0.4)))

    # Normalize MSE to 0-1 range for better color comparison
    # (Lower MSE is better, so we invert it: 1 - normalized_MSE)
    mse_max = df["MSE"].max()
    df["MSE (norm)"] = 1.0 - (df["MSE"] / mse_max) if mse_max > 0 else 0.0

    # Prepare data for heatmap (Pearson R, Explained Var, and normalized inverted MSE)
    heatmap_data = df[["Pearson R", "Explained Var", "MSE (norm)"]]

    # Create heatmap with annotations
    sns.heatmap(
        heatmap_data.T,  # Transpose so metrics are rows
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",  # Red-Yellow-Green: higher is better
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Quality (higher = better)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Cell Type", fontsize=10)
    ax.set_ylabel("Metric", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Add note about MSE normalization
    fig.text(
        0.5, 0.01,
        f"Note: MSE shown as 1 - (MSE / max_MSE) for visualization (higher = better). Max MSE = {mse_max:.2f}",
        ha="center",
        fontsize=7,
        style="italic",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", output_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_tangram_reconstruction_check(
    adata_full: sc.AnnData,
    adata_subset: sc.AnnData,
    output_dir: str | Path,
    dataset_name: str,
    celltype_col: str = "cluster",
    num_epochs: int = 1000,
    run_global: bool = True,
    run_per_celltype: bool = True,
    min_cells_per_celltype: int = MIN_CELLS_PER_CELLTYPE,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    plot_heatmaps: bool = True,
    nmf_counts_input: str = "raw",
) -> dict[str, Any]:
    """Run the Tangram reconstruction check for a single panel.

    This function is the primary entry point called by ``run_evaluation.py``
    when ``--include-tangram`` is active. It skips gracefully if Tangram
    is not installed.

    Args:
        adata_full: Full-transcriptome AnnData reference.
        adata_subset: Probe-panel AnnData (subset of genes).
        output_dir: Directory where results are written.
        dataset_name: Panel identifier (used in output file names).
        celltype_col: obs column holding cell-type labels.
        num_epochs: Tangram optimisation epochs.
        run_global: Whether to run global-mode reconstruction.
        run_per_celltype: Whether to run per-cell-type reconstruction.
        min_cells_per_celltype: Minimum cells for per-cell-type mode.
        train_idx: Integer position array of global training cells.  Passed to
            global reconstruction so the mapping is learned on training cells
            and evaluated on *test_idx* cells.  Pass ``None`` to use all cells.
        test_idx: Integer position array of global test cells (paired with
            *train_idx*).  Pass ``None`` to use all cells.
        per_celltype_splits: Pre-computed per-celltype train/test splits, as
            returned by :func:`_variability.make_celltype_evaluation_splits`.
            Passed to per-cell-type reconstruction so the same cells are used
            here as in the NMF per-cell-type evaluation.  Pass ``None`` to use
            all cells per cell type (original behaviour).
        plot_heatmaps: If ``True`` (default), save alignment heatmaps to
            ``output_dir/plots/`` after each reconstruction stage.  Each
            heatmap compares original vs. reconstructed mean expression per
            cell type and includes a per-gene Pearson R panel.

    Returns:
        Dictionary with keys ``"global"`` and/or ``"per_celltype"`` holding
        the corresponding metrics, plus ``"skipped": True`` if Tangram
        was unavailable.

    Example:
        >>> results = run_tangram_reconstruction_check(
        ...     adata_full=adata_ref,
        ...     adata_subset=adata_panel,
        ...     output_dir="results/Tangram-Evaluation",
        ...     dataset_name="Scanpy-Filter_All-Genes_deg_only_100",
        ... )
    """
    if not _TANGRAM_AVAILABLE:
        logger.warning(
            "Tangram not installed – skipping reconstruction check for '%s'. "
            "Install with: pip install tangram-sc",
            dataset_name,
        )
        return {"skipped": True, "skip_reason": "tangram_not_installed"}

    output_dir = Path(output_dir)
    result: dict[str, Any] = {"skipped": False}

    # Global reconstruction
    if run_global:
        logger.info("=== Tangram global reconstruction: %s ===", dataset_name)
        try:
            ad_ge = reconstruct_with_tangram(
                adata_full, adata_subset, num_epochs=num_epochs,
                train_idx=train_idx, test_idx=test_idx,
                nmf_counts_input=nmf_counts_input,
            )

            # Tangram's pp_adatas() lowercases var_names; build a case-insensitive
            # mapping so we can index adata_full (original case) with ad_ge gene names.
            lower_to_orig = {g.lower(): g for g in adata_full.var_names}
            ref_genes: list[str] = []
            ge_genes: list[str] = []
            for g in ad_ge.var_names:
                orig = lower_to_orig.get(g.lower())
                if orig is not None:
                    ref_genes.append(orig)
                    ge_genes.append(g)

            if not ref_genes:
                raise ValueError(
                    "No genes matched between Tangram output and reference AnnData "
                    "(case-insensitive). Check that adata_full and adata_subset share genes."
                )

            logger.info(
                "Global alignment: %d/%d ad_ge genes matched to adata_full.",
                len(ref_genes), len(ad_ge.var_names),
            )

            # Reference data: test cells when a split is provided, all cells otherwise
            adata_ref = adata_full[test_idx] if test_idx is not None else adata_full
            X_ref  = _adata_to_dense(adata_ref[:, ref_genes])
            X_pred = _adata_to_dense(ad_ge[:, ge_genes])
            global_metrics = _calculate_reconstruction_metrics(X_ref, X_pred)
            result["global"] = global_metrics
            logger.info(
                "Global Tangram complete: MSE=%.4f, ExpVar=%.4f",
                global_metrics["mse"],
                global_metrics["expvar"],
            )

            if plot_heatmaps:
                try:
                    heatmap_path = output_dir / "plots" / f"{dataset_name}_global_heatmap.png"
                    corr_heatmap_path = output_dir / "plots" / f"{dataset_name}_global_correlation_heatmap.png"
                    if celltype_col in adata_ref.obs.columns:
                        ct_labels = adata_ref.obs[celltype_col].values
                        cell_types_sorted = sorted(np.unique(ct_labels))
                        mean_ref_by_ct  = np.array([X_ref [ct_labels == ct].mean(axis=0) for ct in cell_types_sorted])
                        mean_pred_by_ct = np.array([X_pred[ct_labels == ct].mean(axis=0) for ct in cell_types_sorted])
                        _plot_tangram_alignment_heatmap(
                            mean_ref_by_ct, mean_pred_by_ct, ref_genes, cell_types_sorted,
                            output_path=heatmap_path,
                            title=f"{dataset_name} — Global Tangram alignment",
                        )
                        _plot_tangram_correlation_heatmap(
                            mean_ref_by_ct, mean_pred_by_ct, ref_genes, cell_types_sorted,
                            output_path=corr_heatmap_path,
                            title=f"{dataset_name} — Per-celltype reconstruction quality",
                        )
                    else:
                        _plot_tangram_alignment_heatmap(
                            X_ref.mean(axis=0, keepdims=True),
                            X_pred.mean(axis=0, keepdims=True),
                            ref_genes, ["all cells"],
                            output_path=heatmap_path,
                            title=f"{dataset_name} — Global Tangram alignment",
                        )
                        _plot_tangram_correlation_heatmap(
                            X_ref.mean(axis=0, keepdims=True),
                            X_pred.mean(axis=0, keepdims=True),
                            ref_genes, ["all cells"],
                            output_path=corr_heatmap_path,
                            title=f"{dataset_name} — Reconstruction quality",
                        )
                except Exception as plot_exc:
                    logger.warning("Could not save global alignment heatmap: %s", plot_exc)

        except Exception as exc:
            logger.warning("Global Tangram failed for '%s': %s", dataset_name, exc, exc_info=True)
            result["global"] = {"skipped": True, "skip_reason": str(exc)}
    else:
        global_metrics = None

    # Per-cell-type reconstruction
    per_celltype_results: dict[str, dict[str, Any]] | None = None
    if run_per_celltype:
        logger.info("=== Tangram per-cell-type reconstruction: %s ===", dataset_name)
        per_celltype_results = _run_tangram_per_celltype(
            adata_full, adata_subset, celltype_col, num_epochs, min_cells_per_celltype,
            per_celltype_splits=per_celltype_splits,
            nmf_counts_input=nmf_counts_input,
        )
        result["per_celltype"] = per_celltype_results
        summary = _aggregate_per_celltype_metrics(per_celltype_results)
        result["per_celltype_summary"] = summary
        logger.info(
            "Per-celltype summary: macro_mse=%.4f  macro_expvar=%.4f  "
            "weighted_mse=%.4f  weighted_expvar=%.4f  (%d cell types)",
            summary["macro_mse"], summary["macro_expvar"],
            summary["weighted_mse"], summary["weighted_expvar"],
            summary["n_celltypes_used"],
        )

        if plot_heatmaps:
            try:
                valid_cts = [
                    ct for ct, m in per_celltype_results.items()
                    if not m.get("skipped") and "_mean_ref" in m
                ]
                if len(valid_cts) >= 2:
                    # Gene lists may differ per cell type (tg.pp_adatas filters
                    # zero-expression genes per run).  Use the intersection so
                    # all rows have the same length before stacking.
                    gene_sets = [set(per_celltype_results[ct]["_gene_names"]) for ct in valid_cts]
                    common_genes = sorted(gene_sets[0].intersection(*gene_sets[1:]))
                    if common_genes:
                        mean_ref_rows, mean_pred_rows = [], []
                        for ct in valid_cts:
                            gene_idx = {g: i for i, g in enumerate(per_celltype_results[ct]["_gene_names"])}
                            idx = [gene_idx[g] for g in common_genes]
                            mean_ref_rows.append(per_celltype_results[ct]["_mean_ref"][idx])
                            mean_pred_rows.append(per_celltype_results[ct]["_mean_pred"][idx])
                        _plot_tangram_alignment_heatmap(
                            np.array(mean_ref_rows), np.array(mean_pred_rows),
                            common_genes, valid_cts,
                            output_path=output_dir / "plots" / f"{dataset_name}_per_celltype_heatmap.png",
                            title=f"{dataset_name} — Per-cell-type Tangram alignment",
                        )
                        _plot_tangram_correlation_heatmap(
                            np.array(mean_ref_rows), np.array(mean_pred_rows),
                            common_genes, valid_cts,
                            output_path=output_dir / "plots" / f"{dataset_name}_per_celltype_correlation_heatmap.png",
                            title=f"{dataset_name} — Per-celltype reconstruction quality",
                        )
                    else:
                        logger.warning("No common genes across cell types — skipping per-celltype heatmap.")
            except Exception as plot_exc:
                logger.warning("Could not save per-celltype alignment heatmap: %s", plot_exc)

    # Save results
    _save_tangram_results(
        output_dir,
        dataset_name,
        global_metrics=global_metrics if run_global and not result.get("global", {}).get("skipped") else None,
        per_celltype_results=per_celltype_results,
        per_celltype_summary=result.get("per_celltype_summary"),
    )

    return result