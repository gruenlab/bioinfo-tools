"""Tangram-based reconstruction check for probe panel evaluation.

This module provides an optional reconstruction check that uses Tangram
to map probe-panel expression back to the full transcriptome. It is
disabled by default and only runs when explicitly requested via the
``--include_tangram`` flag or ``EvaluationConfig.include_tangram = True``.

When ``train_idx`` / ``test_idx`` (or ``per_celltype_splits``) are supplied the
mapping is fit on the train cells and scored on the held-out test cells; without
them Tangram maps the full dataset.

Scoring conventions (fixed 2026-09-20; earlier Tangram results were invalid):
  * the reference matrix is taken from the SAME space Tangram mapped on
    (``nmf_counts_input="raw"`` -> ``layers["counts"]``), not from ``adata.X``;
  * the projected expression is divided by ``n_sc / n_sp`` (see
    ``reconstruct_with_tangram``) so one spot corresponds to one cell.

Usage::

    from _tangram import run_tangram_reconstruction_check

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
sys.path.insert(0, str(_UTILITY_DIR))
from _validation import is_anndata_raw_layer, is_anndata_raw  # type: ignore[import]

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
        nmf_counts_input: Which matrix Tangram maps on — ``"raw"`` (default, uses
            ``layers["counts"]``) or ``"lognorm"`` (uses ``.X``). Validated against
            the actual content of both AnnData objects.

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

    # Put the projection on a per-cell scale. Tangram's mapping matrix is a softmax over
    # spots for every sc cell (each sc cell's mass sums to 1), and the projection is
    # M.T @ X_sc, so every spot receives on average n_sc / n_sp cells' worth of
    # expression (~4x with a 5-fold split), not one cell's. Dividing by n_sc / n_sp
    # makes the mean prediction per spot equal the mean training expression ("one cell
    # per spot"), so MSE / ExpVar against a single held-out cell are comparable with
    # NMF / Ridge / scVI. (Tangram's own tutorials only use scale-invariant cosine
    # similarity, which is why this never shows up there.) Without a split, n_sc == n_sp
    # and the divisor is 1.
    import scipy.sparse

    scale_divisor = ad_sc.n_obs / ad_sp.n_obs
    X_ge = ad_ge.X.toarray() if scipy.sparse.issparse(ad_ge.X) else np.asarray(ad_ge.X)
    ad_ge.X = (X_ge / scale_divisor).astype(np.float32)
    ad_ge.uns["scale_divisor"] = float(scale_divisor)
    logger.info(
        "Tangram reconstruction complete (per-cell scale divisor n_sc/n_sp = %.3f).",
        scale_divisor,
    )
    return ad_ge


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


# Gene-subset x expvar-mode grid written alongside the legacy ``mse`` / ``expvar``
# keys. Same subsets/modes (and naming, minus the ``test_probe`` token) as the
# NMF/PCA/ICA/Ridge evaluations, so the comparison plots can read all methods alike.
_GRID_SUBSETS = ("all_genes", "panel_genes_only", "non_panel_genes_only")
_GRID_MODES = ("global_mean", "variance_weighted_sum")


def _panel_gene_mask(ref_genes: list[str], adata_subset: sc.AnnData) -> np.ndarray:
    """Boolean mask over *ref_genes* marking the probe-panel genes.

    Tangram lowercases var_names, so the match is case-insensitive.
    """
    panel_lower = {g.lower() for g in adata_subset.var_names}
    return np.array([g.lower() in panel_lower for g in ref_genes], dtype=bool)


def _calculate_reconstruction_metrics(
    X_ref: np.ndarray,
    X_pred: np.ndarray,
    panel_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a standard suite of reconstruction quality metrics.

    Args:
        X_ref: Original data matrix (n_cells × n_genes).
        X_pred: Reconstructed data matrix (n_cells × n_genes).
        panel_mask: Optional boolean mask over the gene axis marking probe-panel
            genes. When given, ``mse_<subset>`` / ``expvar_<subset>_<mode>`` are
            also written for ``panel_genes_only`` and ``non_panel_genes_only``
            (``all_genes`` is always written). ``expvar`` / ``mse`` remain the
            all-genes, global-mean values, unchanged.

    Returns:
        Dictionary with keys ``mse``, ``expvar``, ``rmse``, ``mae``,
        ``r2``, ``pearson``, ``n_cells``, ``n_genes``.

    Note:
        Uses the shared ``metrics`` helpers so the numbers are computed identically
        to the NMF evaluation, for cross-method comparability.
    """
    # Use shared helpers to ensure identical computation to NMF
    mse = calculate_mse(X_ref, X_pred)
    expvar = calculate_explained_variance(X_ref, X_pred)

    # Keep diagnostic variables for logging
    residual_var = float(np.var(X_ref - X_pred))
    total_var = float(np.var(X_ref))

    grid: dict[str, float] = {}
    masks: dict[str, np.ndarray | None] = {"all_genes": None}
    if panel_mask is not None:
        masks["panel_genes_only"] = panel_mask
        masks["non_panel_genes_only"] = ~panel_mask
    for subset, mask in masks.items():
        if mask is not None and not mask.any():
            continue
        Xr = X_ref if mask is None else X_ref[:, mask]
        Xp = X_pred if mask is None else X_pred[:, mask]
        grid[f"mse_{subset}"] = calculate_mse(Xr, Xp)
        for mode in _GRID_MODES:
            grid[f"expvar_{subset}_{mode}"] = calculate_explained_variance(Xr, Xp, mode=mode)

    return {
        **grid,
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
        # Scale sanity check: mean per-cell total expression, reference vs prediction.
        # After the per-cell scale correction these should be of the same order.
        "mean_total_ref": float(X_ref.sum(axis=1).mean()),
        "mean_total_pred": float(X_pred.sum(axis=1).mean()),
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


def _reference_matrix(
    adata: sc.AnnData,
    genes: list[str],
    nmf_counts_input: str = "raw",
) -> np.ndarray:
    """Dense reference matrix in the SAME expression space Tangram mapped on.

    ``reconstruct_with_tangram`` swaps ``layers["counts"]`` into ``.X`` only on its
    own private copies, so the caller's ``adata_full`` keeps whatever ``.X`` it had
    (log-normalised in the preprocessed evaluation files). Scoring the prediction
    against ``adata_full.X`` therefore compared raw-count-scale predictions with
    log-scale targets (MSE ~120, ExpVar ~ -1000). The reference must be taken from
    the matrix matching *nmf_counts_input* instead.

    Args:
        adata: Reference AnnData (test cells already selected).
        genes: Gene names (original case) to extract.
        nmf_counts_input: ``"raw"`` -> ``layers["counts"]``; ``"lognorm"`` -> ``.X``.

    Returns:
        Dense (n_cells × len(genes)) float32 array.

    Raises:
        ValueError: If ``"raw"`` is requested but ``layers["counts"]`` is missing, or
            *nmf_counts_input* is unknown.
    """
    import scipy.sparse

    sub = adata[:, genes]
    if nmf_counts_input == "raw":
        if "counts" not in sub.layers:
            raise ValueError(
                "nmf_counts_input='raw' requires adata.layers['counts'] for the "
                "reference matrix, but it is missing."
            )
        M = sub.layers["counts"]
    elif nmf_counts_input == "lognorm":
        M = sub.X
    else:
        raise ValueError(
            f"Unknown nmf_counts_input='{nmf_counts_input}'. Choose 'raw' or 'lognorm'."
        )
    if scipy.sparse.issparse(M):
        M = M.toarray()
    return np.asarray(M, dtype=np.float32)


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
        per_celltype_splits: Pre-computed per-celltype train/test index splits
            (the ``per_celltype_splits`` element of a
            :func:`_splits.generate_evaluation_splits` fold). When provided,
            training cells serve as the sc reference and test cells as the
            spatial target, so metrics are computed on held-out data only.
            Pass ``None`` to use all cells.
        nmf_counts_input: Matrix Tangram maps on — ``"raw"`` or ``"lognorm"`` —
            forwarded to :func:`reconstruct_with_tangram`.

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
                mask_sub = (
                    adata_subset.obs[celltype_col] == ct
                    if celltype_col in adata_subset.obs.columns
                    else mask
                )
                ad_ct_sub = adata_subset[mask_sub].copy()
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
            X_ref  = _reference_matrix(ad_ct_ref, ref_genes, nmf_counts_input)
            X_pred = _adata_to_dense(ad_ge[:, ge_genes])

            metrics = _calculate_reconstruction_metrics(
                X_ref, X_pred, panel_mask=_panel_gene_mask(ref_genes, adata_subset)
            )
            metrics["scale_divisor"] = float(ad_ge.uns.get("scale_divisor", 1.0))
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
        Uses the shared ``metrics`` aggregation helpers so the numbers match the
        NMF evaluation, for cross-method comparability.
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

    summary = {
        "macro_mse": macro_mse_val,
        "macro_expvar": macro_expvar_val,
        "weighted_mse": weighted_mse_val,
        "weighted_expvar": weighted_expvar_val,
        "n_celltypes_used": n_valid,
    }

    # Gene-subset x mode grid (variance-weighted etc.): same helpers, other keys.
    grid_keys = sorted({
        k for m in per_celltype_results.values() if not m.get("skipped", False)
        for k in m if k.startswith(("mse_", "expvar_")) and k.split("_", 1)[1] in _grid_suffixes()
    })
    for key in grid_keys:
        valid = {
            ct: {**m, key: m[key], "n_cells": m["n_cells"], "skipped": False}
            for ct, m in per_celltype_results.items()
            if not m.get("skipped", False) and key in m
        }
        if key.startswith("mse_"):
            summary[f"macro_{key}"] = calculate_macro_mse(valid, metric_key=key)
            summary[f"weighted_{key}"] = calculate_weighted_mse(valid, metric_key=key)
        else:
            summary[f"macro_{key}"] = calculate_macro_explained_variance(valid, metric_key=key)
            summary[f"weighted_{key}"] = calculate_weighted_explained_variance(valid, metric_key=key)
    return summary


def _grid_suffixes() -> set[str]:
    """Key suffixes (after ``mse_`` / ``expvar_``) written by the metric grid."""
    return set(_GRID_SUBSETS) | {f"{s}_{m}" for s in _GRID_SUBSETS for m in _GRID_MODES}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_tangram_results(
    output_dir: str | Path,
    dataset_name: str,
    global_metrics: dict[str, Any] | None,
    per_celltype_results: dict[str, dict[str, Any]] | None,
    per_celltype_summary: dict[str, float] | None = None,
    fold: int | None = None,
) -> None:
    """Persist Tangram results to CSV files.

    When *fold* is ``None`` results are written to ``global/`` and
    ``per_celltype/`` directly under *output_dir* (aggregated output).
    When *fold* is an integer they are written to ``per_fold/global/`` and
    ``per_fold/per_celltype/`` with ``_fold{fold}`` appended to the stem.

    Args:
        output_dir: Root Tangram output directory.
        dataset_name: Panel / dataset identifier used as the filename stem.
        global_metrics: Metrics from global reconstruction, or ``None``.
        per_celltype_results: Per-cell-type metrics dict, or ``None``.
        per_celltype_summary: Summary metrics (macro/weighted) to append as a
            ``__summary__`` row, or ``None``.
        fold: When set, save to a ``per_fold/`` subdirectory with fold-indexed
            filenames.  ``None`` saves to the standard (aggregated) locations.
    """
    output_dir = Path(output_dir)

    if fold is not None:
        save_dir = output_dir / "per_fold"
        stem = f"{dataset_name}_fold{fold}"
    else:
        save_dir = output_dir
        stem = dataset_name

    if global_metrics is not None:
        global_dir = save_dir / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{**global_metrics, "dataset": dataset_name, "mode": "global"}])
        df.to_csv(global_dir / f"{stem}.csv", index=False)
        logger.info("Saved global Tangram metrics to %s", global_dir / f"{stem}.csv")

    if per_celltype_results:
        ct_dir = save_dir / "per_celltype"
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
        df.to_csv(ct_dir / f"{stem}.csv", index=False)
        logger.info("Saved per-cell-type Tangram metrics to %s", ct_dir / f"{stem}.csv")

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
    nmf_counts_input: str = "raw",
    fold: int | None = None,
) -> dict[str, Any]:
    """Run the Tangram reconstruction check for a single panel.

    This function is the primary entry point called by ``run_evaluation.py``
    when ``--include_tangram`` is active. It skips gracefully if Tangram
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
        per_celltype_splits: Pre-computed per-celltype train/test splits (the
            ``per_celltype_splits`` element of a
            :func:`_splits.generate_evaluation_splits` fold). Passed to
            per-cell-type reconstruction so the same cells are used here as in
            the NMF per-cell-type evaluation.  Pass ``None`` to use all cells
            per cell type.
        nmf_counts_input: Matrix Tangram maps on — ``"raw"`` (default) or
            ``"lognorm"`` — forwarded to :func:`reconstruct_with_tangram`.
        fold: When provided, passed to :func:`_save_tangram_results` so
            results are written to ``per_fold/`` with a fold-indexed filename
            instead of overwriting the main output file.

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
            X_ref  = _reference_matrix(adata_ref, ref_genes, nmf_counts_input)
            X_pred = _adata_to_dense(ad_ge[:, ge_genes])
            global_metrics = _calculate_reconstruction_metrics(
                X_ref, X_pred, panel_mask=_panel_gene_mask(ref_genes, adata_subset)
            )
            global_metrics["scale_divisor"] = float(ad_ge.uns.get("scale_divisor", 1.0))
            result["global"] = global_metrics
            logger.info(
                "Global Tangram complete: MSE=%.4f, ExpVar=%.4f",
                global_metrics["mse"],
                global_metrics["expvar"],
            )

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

    # Save results
    _save_tangram_results(
        output_dir,
        dataset_name,
        global_metrics=global_metrics if run_global and not result.get("global", {}).get("skipped") else None,
        per_celltype_results=per_celltype_results,
        per_celltype_summary=result.get("per_celltype_summary"),
        fold=fold,
    )

    return result