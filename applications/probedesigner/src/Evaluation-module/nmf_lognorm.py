"""NMF variability evaluation with log-normalized reconstruction comparison.

Fits NMF on raw counts (same as nmf.py), but evaluates reconstruction quality
in log-normalized space: both the reference and the reconstructed matrix are
log-normalized using sc.pp.normalize_total + sc.pp.log1p with the *same*
target_sum so the comparison is on a consistent scale.

Functions:
    nmf_reconstruction_lognorm: Global NMF evaluation in lognorm space.
    nmf_reconstruction_by_celltype_lognorm: Per-cell-type version.
"""

from __future__ import annotations

import gc
import logging
import sys
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse
from nico2_lib.predictors._nmf._nmf_pred import NmfPredictor
from tqdm import tqdm

# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import NMF_PREDICTOR_FIXED_KWARGS
from metrics import calculate_explained_variance, calculate_mse

logger = logging.getLogger(__name__)

__all__ = [
    "nmf_reconstruction_lognorm",
    "nmf_reconstruction_by_celltype_lognorm",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lognormalize(A_raw: np.ndarray, target_sum: float) -> np.ndarray:
    """Apply sc.pp.normalize_total + sc.pp.log1p with an explicit target_sum.

    Using an explicit ``target_sum`` (derived from the reference data) ensures
    that multiple matrices (reference and reconstruction) are normalized to the
    same scale so their comparison via MSE / explained variance is meaningful.

    Args:
        A_raw: Dense raw count matrix, shape (cells × genes).
        target_sum: Each cell's counts are scaled to sum to this value before
            log1p. Derive once from the reference and reuse for both the
            reference and the reconstruction.

    Returns:
        Log-normalized float32 array of the same shape.
    """
    # np.array() always copies — np.asarray() would not copy when dtype already
    # matches, letting sc.pp.normalize_total mutate the caller's array in-place.
    tmp = ad.AnnData(X=np.array(A_raw, dtype=np.float64))
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)
    result = tmp.X
    if scipy.sparse.issparse(result):
        result = result.toarray()
    del tmp
    return np.array(result, dtype=np.float32)


def _to_dense_f64(X: Any, idx: np.ndarray) -> np.ndarray:
    """Slice X at ``idx`` and return a dense float64 array."""
    s = X[idx]
    if scipy.sparse.issparse(s):
        s = s.toarray()
    return np.asarray(s, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nmf_reconstruction_lognorm(
    adata: Any,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    target_sum_global: float,
    n_components: int = 5,
    random_state: int = 42,
    cached_full_nmf: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate probe reconstruction quality in log-normalized space (global).

    Fits NMF on raw counts, reconstructs the full transcriptome from probe
    genes only, then log-normalizes both the raw reference and the raw
    reconstruction using ``target_sum_global``. MSE and explained variance are
    computed in this shared log-normalized space.

    Args:
        adata: AnnData with full transcriptome (used for gene names only).
        probeset_genes: Genes in the probe panel.
        A_train: Training data, raw counts (cells × genes), float64.
        A_test: Test data, raw counts (cells × genes), float64.
        target_sum_global: Normalization target (median per-cell total from the
            full raw count matrix). Applied identically to the reference and
            the reconstruction so both are on the same scale.
        n_components: Number of NMF components.
        random_state: Random seed.
        cached_full_nmf: Pre-computed full NMF (W, H, recon) structured as
            ``{"training": {...}, "testing": {...}}``. Reused across panels
            within the same fold to avoid redundant baseline fits.

    Returns:
        Dict with keys ``lognorm_mse_test_baseline``,
        ``lognorm_expvar_test_baseline``, ``lognorm_mse_test_probe``,
        ``lognorm_expvar_test_probe``, ``probeset_size``,
        ``probeset_genes_found``. There are no ``lognorm_*_ratio`` keys —
        compare probe vs. baseline directly.

    Example:
        >>> result = nmf_reconstruction_lognorm(
        ...     adata, panel_genes, A_train, A_test,
        ...     target_sum_global=2500.0, n_components=5,
        ... )
        >>> result["lognorm_mse_test_probe"]
        0.42
    """
    logger.info("Lognorm NMF reconstruction for %d genes", len(probeset_genes))

    probeset_mask = np.array([g in probeset_genes for g in adata.var_names])
    if not probeset_mask.any():
        logger.error("No probeset genes found in dataset")
        return {}

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info("Found %d / %d probeset genes", len(probeset_genes_found), len(probeset_genes))

    A_train = A_train.astype(np.float64)
    A_test = A_test.astype(np.float64)
    A_P_train = A_train[:, probe_indices]
    A_P_test = A_test[:, probe_indices]

    # Pinned NMF config shared with Selection-module (see _constants.NMF_*).
    _nmf_kwargs = dict(
        n_components=n_components, embedding_size=n_components,
        random_state=random_state, **NMF_PREDICTOR_FIXED_KWARGS,
    )

    # ── Training NMF ─────────────────────────────────────────────────────────
    _use_train_cache = (
        cached_full_nmf is not None
        and "training" in cached_full_nmf
        and cached_full_nmf["training"].get("A_train") is not None
        and cached_full_nmf["training"]["A_train"].shape == A_train.shape
        and np.allclose(cached_full_nmf["training"]["A_train"], A_train)
    )
    if _use_train_cache:
        tc = cached_full_nmf["training"]
        H_full_train = tc["H_full_train"].astype(np.float64)
        W_full_train = tc["W_full_train"].astype(np.float64)
        A_train_baseline = tc["A_train_baseline"]
        train_pred = NmfPredictor(
            **_nmf_kwargs, h_reference=H_full_train, ref_embedding=W_full_train
        )
        logger.info("Using cached training NMF")
    else:
        train_pred = NmfPredictor(**_nmf_kwargs).fit(A_train)
        W_full_train = train_pred.ref_embedding
        H_full_train = train_pred.h_reference
        A_train_baseline = W_full_train @ H_full_train
        logger.info("Computed fresh training NMF")

    _, A_train_recon = train_pred.predict(A_P_train, indexer=probe_indices)

    # ── Testing NMF ──────────────────────────────────────────────────────────
    _use_test_cache = (
        cached_full_nmf is not None
        and "testing" in cached_full_nmf
        and cached_full_nmf["testing"].get("A_test") is not None
        and cached_full_nmf["testing"]["A_test"].shape == A_test.shape
        and np.allclose(cached_full_nmf["testing"]["A_test"], A_test)
    )
    if _use_test_cache:
        tc = cached_full_nmf["testing"]
        W_full_test = tc["W_full_test"].astype(np.float64)
        H_full_test = tc["H_full_test"].astype(np.float64)
        A_test_baseline = tc["A_test_baseline"]
        logger.info("Using cached testing NMF")
    else:
        test_pred = NmfPredictor(**_nmf_kwargs).fit(A_test)
        W_full_test = test_pred.ref_embedding
        H_full_test = test_pred.h_reference
        A_test_baseline = W_full_test @ H_full_test
        logger.info("Computed fresh testing NMF")

    _, A_test_recon = train_pred.predict(A_P_test, indexer=probe_indices)

    # ── Log-normalize with shared target_sum ─────────────────────────────────
    logger.info("Log-normalizing (target_sum=%.2f)", target_sum_global)
    A_test_lognorm_ref = _lognormalize(A_test, target_sum_global)
    A_test_recon_lognorm = _lognormalize(A_test_recon, target_sum_global)
    A_test_baseline_lognorm = _lognormalize(A_test_baseline, target_sum_global)

    # ── Metrics ──────────────────────────────────────────────────────────────
    lognorm_mse_test_probe = calculate_mse(A_test_lognorm_ref, A_test_recon_lognorm)
    lognorm_expvar_test_probe = calculate_explained_variance(A_test_lognorm_ref, A_test_recon_lognorm)
    lognorm_mse_test_baseline = calculate_mse(A_test_lognorm_ref, A_test_baseline_lognorm)
    lognorm_expvar_test_baseline = calculate_explained_variance(
        A_test_lognorm_ref, A_test_baseline_lognorm
    )

    logger.info(
        "Lognorm MSE    baseline=%.6f  probe=%.6f",
        lognorm_mse_test_baseline, lognorm_mse_test_probe,
    )
    logger.info(
        "Lognorm ExpVar baseline=%.6f  probe=%.6f",
        lognorm_expvar_test_baseline, lognorm_expvar_test_probe,
    )

    # Absolute metrics only (no probe/baseline ratios); the baseline numbers drive
    # the reference lines in the plots.
    result: dict[str, Any] = {
        "lognorm_mse_test_baseline": lognorm_mse_test_baseline,
        "lognorm_expvar_test_baseline": lognorm_expvar_test_baseline,
        "lognorm_mse_test_probe": lognorm_mse_test_probe,
        "lognorm_expvar_test_probe": lognorm_expvar_test_probe,
        "probeset_size": len(probeset_genes),
        "probeset_genes_found": len(probeset_genes_found),
    }

    # Expose freshly computed NMF so the caller can cache it for other panels
    if not _use_train_cache or not _use_test_cache:
        result["computed_full_nmf"] = {
            "training": {
                "A_train": A_train,
                "W_full_train": W_full_train,
                "H_full_train": H_full_train,
                "A_train_baseline": A_train_baseline,
            },
            "testing": {
                "A_test": A_test,
                "W_full_test": W_full_test,
                "H_full_test": H_full_test,
                "A_test_baseline": A_test_baseline,
            },
        }

    gc.collect()
    return result


def nmf_reconstruction_by_celltype_lognorm(
    adata: Any,
    probeset_genes: list[str],
    celltype_column: str,
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    ct_target_sums: dict[str, float],
    n_components: int = 5,
    random_state: int = 42,
    cached_full_nmf_by_celltype: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Evaluate probe reconstruction quality in log-normalized space per cell type.

    For each cell type the normalization target (``target_sum_ct``) is derived
    from **all** cells of that type so that both the reference raw counts and
    the raw reconstruction are normalized to the same scale before comparison.

    Args:
        adata: AnnData with full transcriptome; must have ``layers['counts']``.
        probeset_genes: Genes in the probe panel.
        celltype_column: obs column with cell-type labels.
        per_celltype_splits: ``{celltype: (train_global_idx, test_global_idx)}``
            as returned by :func:`_splits.generate_evaluation_splits`. Indices
            are integer positions into ``adata``.
        ct_target_sums: ``{celltype: target_sum}`` — median per-cell total
            computed from all cells of each cell type. Pre-computed once before
            the fold loop for efficiency.
        n_components: Number of NMF components.
        random_state: Random seed.
        cached_full_nmf_by_celltype: Cache of NMF W/H matrices per cell type
            (mutated in place).

    Returns:
        Dict with ``"celltype_results"`` (per-CT metrics) and ``"summary"``
        (macro/weighted aggregates).

    Raises:
        ValueError: If ``layers['counts']`` is not present in ``adata``.

    Example:
        >>> result = nmf_reconstruction_by_celltype_lognorm(
        ...     adata, panel_genes, "celltype", ct_splits, ct_target_sums,
        ...     n_components=5,
        ... )
        >>> result["summary"]["macro_lognorm_mse_test_probe"]
        0.45
    """
    if "counts" not in adata.layers:
        raise ValueError(
            "nmf_reconstruction_by_celltype_lognorm: adata.layers['counts'] not found. "
            "Raw counts are required for lognorm NMF evaluation."
        )

    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Celltype column '{celltype_column}' not found in adata.obs")

    logger.info(
        "Lognorm NMF reconstruction by cell type for %d genes", len(probeset_genes)
    )

    probeset_mask = np.array([g in probeset_genes for g in adata.var_names])
    if not probeset_mask.any():
        raise ValueError("No probeset genes found in dataset")

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info("Found %d / %d probeset genes", len(probeset_genes_found), len(probeset_genes))

    if cached_full_nmf_by_celltype is None:
        cached_full_nmf_by_celltype = {}

    celltype_results: dict[str, dict[str, Any]] = {}
    # Pinned NMF config shared with Selection-module (see _constants.NMF_*).
    _nmf_kwargs = dict(
        n_components=n_components, embedding_size=n_components,
        random_state=random_state, **NMF_PREDICTOR_FIXED_KWARGS,
    )

    for celltype in tqdm(per_celltype_splits, desc="Processing cell types (lognorm)"):
        train_global_idx, test_global_idx = per_celltype_splits[celltype]
        n_cells_ct = int(
            (adata.obs[celltype_column] == celltype).sum()
        )

        if n_cells_ct < n_components * 2:
            logger.warning(
                "Skipping %s: %d cells < minimum %d", celltype, n_cells_ct, n_components * 2
            )
            celltype_results[celltype] = _skipped_ct_row(
                len(probeset_genes), len(probeset_genes_found), n_cells_ct,
                reason="insufficient_cells",
            )
            continue

        try:
            # Raw counts for train / test cells (dense float64)
            A_train_ct = _to_dense_f64(adata.layers["counts"], train_global_idx)
            A_test_ct = _to_dense_f64(adata.layers["counts"], test_global_idx)
            A_P_test_ct = A_test_ct[:, probe_indices]

            # Per-CT target_sum (pre-computed from ALL cells of this CT)
            target_sum_ct = ct_target_sums.get(celltype)
            if target_sum_ct is None:
                # Fallback: compute now from train+test cells
                all_raw = np.concatenate([A_train_ct, A_test_ct], axis=0)
                target_sum_ct = float(np.median(all_raw.sum(axis=1)))
                logger.warning(
                    "target_sum_ct not found for %s; computed on-the-fly as %.2f",
                    celltype, target_sum_ct,
                )

            # ── NMF ──────────────────────────────────────────────────────────
            cached_ct = cached_full_nmf_by_celltype.get(celltype, {})
            _has_cache = (
                "training" in cached_ct and "testing" in cached_ct
                and all(k in cached_ct["training"] for k in ["W_full_train", "H_full_train"])
                and all(k in cached_ct["testing"] for k in ["W_full_test", "H_full_test"])
            )

            if _has_cache:
                logger.debug("Using cached NMF for %s", celltype)
                W_full_train = cached_ct["training"]["W_full_train"].astype(np.float64)
                H_full_train = cached_ct["training"]["H_full_train"].astype(np.float64)
                mse_train_baseline = cached_ct["training"]["mse_train_baseline"]
                expvar_train_baseline = cached_ct["training"]["expvar_train_baseline"]
                W_full_test = cached_ct["testing"]["W_full_test"].astype(np.float64)
                H_full_test = cached_ct["testing"]["H_full_test"].astype(np.float64)
                mse_test_baseline = cached_ct["testing"]["mse_test_baseline"]
                expvar_test_baseline = cached_ct["testing"]["expvar_test_baseline"]
                A_train_baseline = W_full_train @ H_full_train
                A_test_baseline = W_full_test @ H_full_test
            else:
                logger.info("Computing NMF for %s", celltype)
                _train_pred = NmfPredictor(**_nmf_kwargs).fit(A_train_ct)
                W_full_train = _train_pred.ref_embedding
                H_full_train = _train_pred.h_reference
                A_train_baseline = W_full_train @ H_full_train
                # Raw-space baseline metrics (kept for cache, not returned)
                from metrics import calculate_mse as _mse, calculate_explained_variance as _ev
                mse_train_baseline = _mse(A_train_ct, A_train_baseline)
                expvar_train_baseline = _ev(A_train_ct, A_train_baseline)

                _test_pred = NmfPredictor(**_nmf_kwargs).fit(A_test_ct)
                W_full_test = _test_pred.ref_embedding
                H_full_test = _test_pred.h_reference
                A_test_baseline = W_full_test @ H_full_test
                mse_test_baseline = _mse(A_test_ct, A_test_baseline)
                expvar_test_baseline = _ev(A_test_ct, A_test_baseline)

                cached_full_nmf_by_celltype[celltype] = {
                    "training": {
                        "W_full_train": W_full_train,
                        "H_full_train": H_full_train,
                        "mse_train_baseline": mse_train_baseline,
                        "expvar_train_baseline": expvar_train_baseline,
                    },
                    "testing": {
                        "W_full_test": W_full_test,
                        "H_full_test": H_full_test,
                        "mse_test_baseline": mse_test_baseline,
                        "expvar_test_baseline": expvar_test_baseline,
                    },
                }

            _pred_ct = NmfPredictor(
                **_nmf_kwargs, h_reference=H_full_train, ref_embedding=W_full_train
            )
            _, A_test_ct_recon = _pred_ct.predict(A_P_test_ct, indexer=probe_indices)

            # ── Log-normalize reference and reconstruction with same target_sum ──
            A_test_ct_lognorm_ref = _lognormalize(A_test_ct, target_sum_ct)
            A_test_ct_recon_lognorm = _lognormalize(A_test_ct_recon, target_sum_ct)
            A_test_ct_baseline_lognorm = _lognormalize(A_test_baseline, target_sum_ct)

            lognorm_mse_test_probe = calculate_mse(A_test_ct_lognorm_ref, A_test_ct_recon_lognorm)
            lognorm_expvar_test_probe = calculate_explained_variance(
                A_test_ct_lognorm_ref, A_test_ct_recon_lognorm
            )
            lognorm_mse_test_baseline = calculate_mse(
                A_test_ct_lognorm_ref, A_test_ct_baseline_lognorm
            )
            lognorm_expvar_test_baseline = calculate_explained_variance(
                A_test_ct_lognorm_ref, A_test_ct_baseline_lognorm
            )

            logger.info(
                "%s — lognorm MSE baseline=%.4f probe=%.4f",
                celltype, lognorm_mse_test_baseline, lognorm_mse_test_probe,
            )

            celltype_results[celltype] = {
                "lognorm_mse_test_baseline": lognorm_mse_test_baseline,
                "lognorm_expvar_test_baseline": lognorm_expvar_test_baseline,
                "lognorm_mse_test_probe": lognorm_mse_test_probe,
                "lognorm_expvar_test_probe": lognorm_expvar_test_probe,
                "probeset_size": len(probeset_genes),
                "probeset_genes_found": len(probeset_genes_found),
                "n_cells": n_cells_ct,
                "skipped": False,
            }

        except Exception as exc:
            logger.error("Lognorm NMF failed for %s: %s", celltype, exc, exc_info=True)
            celltype_results[celltype] = _skipped_ct_row(
                len(probeset_genes), len(probeset_genes_found), n_cells_ct,
                reason="evaluation_failed",
            )

    # ── Aggregate across cell types ───────────────────────────────────────────
    valid = {ct: r for ct, r in celltype_results.items() if not r.get("skipped", False)}
    summary: dict[str, Any] = {}

    if valid:
        total_cells = sum(r["n_cells"] for r in valid.values())
        mse_vals = [r["lognorm_mse_test_probe"] for r in valid.values()]
        ev_vals = [r["lognorm_expvar_test_probe"] for r in valid.values()]
        mse_base_vals = [r["lognorm_mse_test_baseline"] for r in valid.values()]
        ev_base_vals = [r["lognorm_expvar_test_baseline"] for r in valid.values()]
        n_cells_vals = [r["n_cells"] for r in valid.values()]

        summary = {
            "macro_lognorm_mse_test_probe": float(np.nanmean(mse_vals)),
            "macro_lognorm_expvar_test_probe": float(np.nanmean(ev_vals)),
            "macro_lognorm_mse_test_baseline": float(np.nanmean(mse_base_vals)),
            "macro_lognorm_expvar_test_baseline": float(np.nanmean(ev_base_vals)),
            "weighted_lognorm_mse_test_probe": float(
                np.sum(np.array(mse_vals) * np.array(n_cells_vals)) / total_cells
            ),
            "weighted_lognorm_expvar_test_probe": float(
                np.sum(np.array(ev_vals) * np.array(n_cells_vals)) / total_cells
            ),
            "weighted_lognorm_mse_test_baseline": float(
                np.sum(np.array(mse_base_vals) * np.array(n_cells_vals)) / total_cells
            ),
            "weighted_lognorm_expvar_test_baseline": float(
                np.sum(np.array(ev_base_vals) * np.array(n_cells_vals)) / total_cells
            ),
            "total_cells": total_cells,
            "n_celltypes_processed": len(valid),
            "n_celltypes_skipped": len(celltype_results) - len(valid),
        }
    else:
        summary = {k: np.nan for k in [
            "macro_lognorm_mse_test_probe", "macro_lognorm_expvar_test_probe",
            "macro_lognorm_mse_test_baseline", "macro_lognorm_expvar_test_baseline",
            "weighted_lognorm_mse_test_probe", "weighted_lognorm_expvar_test_probe",
            "weighted_lognorm_mse_test_baseline", "weighted_lognorm_expvar_test_baseline",
        ]}
        summary.update({"total_cells": 0, "n_celltypes_processed": 0,
                        "n_celltypes_skipped": len(celltype_results)})

    return {"celltype_results": celltype_results, "summary": summary}


def _skipped_ct_row(
    probeset_size: int,
    probeset_genes_found: int,
    n_cells: int,
    reason: str,
) -> dict[str, Any]:
    """Return a NaN-filled result dict for a skipped cell type."""
    return {
        "lognorm_mse_test_baseline": np.nan,
        "lognorm_expvar_test_baseline": np.nan,
        "lognorm_mse_test_probe": np.nan,
        "lognorm_expvar_test_probe": np.nan,
        "probeset_size": probeset_size,
        "probeset_genes_found": probeset_genes_found,
        "n_cells": n_cells,
        "skipped": True,
        "skip_reason": reason,
    }
