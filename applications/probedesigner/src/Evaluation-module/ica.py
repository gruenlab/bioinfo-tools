"""ICA-based representation evaluation for probeset evaluation.

Expects pre-normalised input (e.g. ``adata.X`` after ``sc.pp.normalize_total``
+ ``sc.pp.log1p``), exactly like ``pca.py``. The probe evaluation fixes the
mixing matrix H (mapped from ``FastICA.mixing_``) learned from a full-gene ICA
fit on the training split and solves for sources W using only probe genes via
least squares — the same "fixed basis, least-squares partial-feature solve"
pattern as ``pca.py`` and (with NNLS instead of unconstrained lstsq) ``nmf.py``.

Unlike PCA, ICA components are **not** ranked by explained variance — there is
no notion of "top-k of a nested sequence". ``n_components`` is therefore a hard
hyperparameter of the fit, not a truncation point, and is not directly
comparable to a PCA run's ``n_components`` of the same value. Component
sign/scale indeterminacy (a well-known ICA property) does not affect
reconstruction fidelity: it cancels out in ``S @ H``, since both S and H come
from the same fit.

FastICA can fail to converge on expression data (it's an iterative,
gradient-based algorithm, unlike PCA's closed-form SVD). ``max_iter``/``tol``
are pinned explicitly (rather than relying on sklearn's version-dependent
defaults) and set more generously than sklearn's defaults. Per-cell-type
fits — smaller sample sizes, more prone to non-convergence — treat a
non-converged fit as a skip (``skip_reason="ica_failed_to_converge"``), the
same way an insufficient-cell-count cell type is skipped elsewhere in this
module.

The full-gene baseline is an *oracle* reference computed independently on each
split, matching ``pca.py``'s (and ``nmf.py``'s) convention: on train, ICA is
fit and evaluated on the same training data; on test, a **separate** ICA is
fit directly on the test data (not a transform of the train-fitted basis).

Functions:
    ica_reconstruction: Global probe evaluation via ICA.
    ica_reconstruction_by_celltype: Same as above, per cell type.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import scipy.sparse
from tqdm import tqdm

from nico2_lib.predictors._fastica._fastica import FastIcaPredictor
from metrics import calculate_explained_variance, calculate_mse
from metrics import (
    calculate_macro_explained_variance,
    calculate_macro_mse,
    calculate_weighted_explained_variance,
    calculate_weighted_explained_variance_baseline,
    calculate_weighted_mse,
    calculate_weighted_mse_baseline,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ica_reconstruction",
    "ica_reconstruction_by_celltype",
    "_compute_full_ica_baseline",
]

# Pinned explicitly rather than relying on sklearn's (version-dependent)
# defaults — FastICA is iterative and can need more headroom than the sklearn
# default (max_iter=200) to converge on expression data.
_ICA_MAX_ITER = 1000
_ICA_TOL = 1e-4
_ICA_WHITEN = "unit-variance"
_ICA_ALGORITHM = "parallel"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_ica(A: np.ndarray, n_components: int, random_state: int) -> FastIcaPredictor:
    """Fit FastICA via nico2_lib's ``FastIcaPredictor``.

    Convergence checking is delegated to ``FastIcaPredictor.fit()``, which
    raises ``RuntimeError`` on non-convergence (identical semantics to this
    module's previous hand-rolled check).

    Args:
        A: Data to fit (cells × genes), float64.
        n_components: Number of independent components.
        random_state: Random seed.

    Returns:
        Fitted ``FastIcaPredictor`` instance.

    Raises:
        RuntimeError: If FastICA does not converge within ``_ICA_MAX_ITER``
            iterations. Callers decide how to handle this (skip a cell type,
            or let it propagate for the global fit).
    """
    return FastIcaPredictor(
        n_components=n_components,
        algorithm=_ICA_ALGORITHM,
        whiten=_ICA_WHITEN,
        max_iter=_ICA_MAX_ITER,
        tol=_ICA_TOL,
        random_state=random_state,
    ).fit(A)


def _compute_full_ica_baseline(
    A_train_ln: np.ndarray,
    A_test_ln: np.ndarray,
    n_components: int,
    random_state: int,
) -> dict[str, Any]:
    """Fit ICA on log-normalised training data and compute baselines.

    See module docstring for the oracle test-baseline convention (an
    independent ICA fit on test data, matching ``pca.py``/``nmf.py``).

    Args:
        A_train_ln: Log-normalised training data (cells × genes).
        A_test_ln: Log-normalised test data (cells × genes).
        n_components: Number of independent components.
        random_state: Random seed.

    Returns:
        Dict with ``"training"`` and ``"testing"`` sub-dicts.
    """
    train_predictor = _fit_ica(A_train_ln, n_components, random_state)
    _, A_train_baseline = train_predictor.predict(A_train_ln, indexer=np.arange(A_train_ln.shape[1]))
    mse_train_bl = calculate_mse(A_train_ln, A_train_baseline)
    expvar_train_bl = calculate_explained_variance(A_train_ln, A_train_baseline)

    # Independent oracle fit on test data (not a transform of the train-fitted ICA).
    test_predictor = _fit_ica(A_test_ln, n_components, random_state)
    _, A_test_baseline = test_predictor.predict(A_test_ln, indexer=np.arange(A_test_ln.shape[1]))
    mse_test_bl = calculate_mse(A_test_ln, A_test_baseline)
    expvar_test_bl = calculate_explained_variance(A_test_ln, A_test_baseline)

    return {
        "training": {
            "predictor": train_predictor,
            "mse_train_baseline": mse_train_bl,
            "expvar_train_baseline": expvar_train_bl,
            "A_train_baseline": A_train_baseline,
        },
        "testing": {
            "mse_test_baseline": mse_test_bl,
            "expvar_test_baseline": expvar_test_bl,
            "A_test_baseline": A_test_baseline,
        },
        # Store raw arrays only for cache-validation in ica_reconstruction
        "_A_train": A_train_ln,
        "_A_test": A_test_ln,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ica_reconstruction(
    adata: Any,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    n_components: int = 5,
    random_state: int = 42,
    cached_full_ica: dict[str, Any] | None = None,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> dict[str, Any] | None:
    """Evaluate probe panel quality via ICA full-transcriptome reconstruction.

    Expects pre-normalised input (e.g. slices of ``adata.X`` after
    ``sc.pp.normalize_total`` + ``sc.pp.log1p``).  No normalisation is applied
    internally.  The mixing matrix H and per-gene mean are fixed from the
    training fit; sources S are solved via least squares using only probe genes.
    The full-gene reconstruction ``S_probe @ H + mean`` is compared to the ICA
    baseline (see module docstring for the oracle baseline convention).

    Args:
        adata: AnnData with ``var_names`` covering all genes.
        probeset_genes: Probe gene names to evaluate.
        A_train: Pre-normalised training data (cells × genes), float64.
        A_test: Pre-normalised test data (cells × genes), float64.
        n_components: Number of independent components.
        random_state: Random seed for ICA.
        cached_full_ica: Pre-computed baseline dict returned by a previous
            call's ``"computed_full_ica"`` key.
        expvar_mode: Explained-variance aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``).
            Defaults to ``"global_mean"``.
        expvar_modes: Optional list of explained-variance aggregation modes to
            additionally report (e.g. ``["global_mean", "variance_weighted_sum",
            "mean"]``). When given, adds ``expvar_test_probe_{subset}_{mode}``
            (and the ``expvar_test_baseline_{mode}`` counterpart) to the result
            for every ``(subset, mode)`` pair, on top of the unsuffixed keys
            (which are always computed from ``expvar_mode`` alone, unchanged).
            Defaults to ``None``, i.e. ``[expvar_mode]`` — no extra columns.
        gene_subsets: Optional list of gene subsets to score the probe
            reconstruction against: ``"all_genes"`` (default),
            ``"panel_genes_only"``, ``"non_panel_genes_only"``. This does
            **not** change what is reconstructed (always the full
            transcriptome) — it is a scoring-time column mask applied to the
            already-computed reconstruction, matching ``nmf.py``'s convention.
            Baseline metrics are only ever computed on the all-genes scale.
            Defaults to ``None``, i.e. ``["all_genes"]``.

    Returns:
        Metrics dict, or ``None`` if no probe genes are found in the dataset.
    """
    logger.info("Evaluating ICA representation with %d genes", len(probeset_genes))

    probeset_mask = np.array([g in probeset_genes for g in adata.var_names])
    if probeset_mask.sum() == 0:
        logger.error("No probeset genes found in the dataset")
        return None

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info(
        "Found %d / %d probeset genes", len(probeset_genes_found), len(probeset_genes)
    )
    if len(probe_indices) < n_components:
        logger.warning(
            "Probe genes (%d) < n_components (%d); lstsq solve is underdetermined",
            len(probe_indices), n_components,
        )

    A_train = A_train.astype(np.float64)
    A_test = A_test.astype(np.float64)

    A_P_train = A_train[:, probe_indices]
    A_P_test = A_test[:, probe_indices]

    # ── Full ICA baseline (cached or computed) ────────────────────────────────
    baseline_cached = False
    if cached_full_ica is not None:
        tr = cached_full_ica.get("training", {})
        cached_A = cached_full_ica.get("_A_train")
        if (
            cached_A is not None
            and cached_A.shape == A_train.shape
            and np.allclose(cached_A, A_train)
        ):
            predictor = tr["predictor"]
            mse_train_baseline = tr["mse_train_baseline"]
            expvar_train_baseline = tr["expvar_train_baseline"]
            te = cached_full_ica["testing"]
            mse_test_baseline = te["mse_test_baseline"]
            expvar_test_baseline = te["expvar_test_baseline"]
            A_test_baseline = te["A_test_baseline"]
            logger.info("Reusing cached ICA baseline")
            baseline_cached = True
        else:
            logger.warning("Cached ICA baseline does not match current data — recomputing")

    if not baseline_cached:
        logger.info("Computing full ICA baseline (n_components=%d)", n_components)
        baseline = _compute_full_ica_baseline(A_train, A_test, n_components, random_state)
        predictor = baseline["training"]["predictor"]
        mse_train_baseline = baseline["training"]["mse_train_baseline"]
        expvar_train_baseline = baseline["training"]["expvar_train_baseline"]
        mse_test_baseline = baseline["testing"]["mse_test_baseline"]
        expvar_test_baseline = baseline["testing"]["expvar_test_baseline"]
        A_test_baseline = baseline["testing"]["A_test_baseline"]

    # ── Probe-based reconstruction ────────────────────────────────────────────
    S_train, A_train_recon = predictor.predict(A_P_train, indexer=probe_indices)
    S_test, A_test_recon = predictor.predict(A_P_test, indexer=probe_indices)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mse_train_probe = calculate_mse(A_train, A_train_recon)
    expvar_train_probe = calculate_explained_variance(A_train, A_train_recon)
    mse_test_probe = calculate_mse(A_test, A_test_recon)
    expvar_test_probe = calculate_explained_variance(A_test, A_test_recon)

    logger.info(
        "Train: MSE baseline=%.6f probe=%.6f | Test: MSE baseline=%.6f probe=%.6f "
        "ExpVar baseline=%.3f probe=%.3f",
        mse_train_baseline, mse_train_probe,
        mse_test_baseline, mse_test_probe, expvar_test_baseline, expvar_test_probe,
    )

    # Absolute metrics only (no probe/baseline ratios); baseline numbers drive the
    # reference lines in the plots.
    result: dict[str, Any] = {
        "mse_train_baseline": mse_train_baseline,
        "mse_test_baseline": mse_test_baseline,
        "mse_train_probe": mse_train_probe,
        "mse_test_probe": mse_test_probe,
        "expvar_train_baseline": expvar_train_baseline,
        "expvar_test_baseline": expvar_test_baseline,
        "expvar_train_probe": expvar_train_probe,
        "expvar_test_probe": expvar_test_probe,
        "probeset_size": len(probeset_genes),
        "probeset_genes_found": len(probeset_genes_found),
    }

    # ── Gene-subset x expvar-mode grid (additive on top of the keys above) ────
    # Does not change what is reconstructed (A_test_recon is always the full
    # transcriptome, unchanged) -- purely a scoring-time column mask, same
    # convention as nmf.py::nmf_reconstruction.
    modes = list(expvar_modes) if expvar_modes else [expvar_mode]
    subsets = list(gene_subsets) if gene_subsets else ["all_genes"]

    panel_mask = np.zeros(A_test.shape[1], dtype=bool)
    panel_mask[probe_indices] = True
    _subset_masks: dict[str, np.ndarray | None] = {
        "all_genes": None,
        "panel_genes_only": panel_mask,
        "non_panel_genes_only": ~panel_mask,
    }

    # Baseline is only ever computed on the all-genes scale -- one value per mode.
    for m in modes:
        result[f"expvar_test_baseline_{m}"] = calculate_explained_variance(
            A_test, A_test_baseline, mode=m
        )

    for subset in subsets:
        mask = _subset_masks[subset]
        X_true = A_test if mask is None else A_test[:, mask]
        X_recon = A_test_recon if mask is None else A_test_recon[:, mask]
        result[f"mse_test_probe_{subset}"] = calculate_mse(X_true, X_recon)
        for m in modes:
            result[f"expvar_test_probe_{subset}_{m}"] = calculate_explained_variance(
                X_true, X_recon, mode=m
            )

    if not baseline_cached:
        result["computed_full_ica"] = baseline

    gc.collect()
    return result


def ica_reconstruction_by_celltype(
    adata: Any,
    probeset_genes: list[str],
    celltype_column: str = "cluster",  # = _constants.DEFAULT_CELLTYPE_COLUMN
    n_components: int = 5,
    random_state: int = 42,
    cached_full_ica_by_celltype: dict[str, Any] | None = None,
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate ICA representation for each cell type separately.

    Expects pre-normalised input from ``adata.X`` (e.g. after
    ``sc.pp.normalize_total`` + ``sc.pp.log1p``).  Each cell type uses its
    own ICA fit on the pre-normalised per-CT subset. FastICA is more prone to
    non-convergence than PCA on small sample sizes; a cell type whose ICA fit
    does not converge (train or test split) is skipped with
    ``skip_reason="ica_failed_to_converge"`` rather than failing the whole run.

    Args:
        adata: AnnData with ``var_names``, ``obs[celltype_column]``, and
            pre-normalised expression in ``adata.X``.
        probeset_genes: Probe gene names to evaluate.
        celltype_column: obs column holding cell-type labels.
        n_components: Number of independent components.
        random_state: Random seed.
        cached_full_ica_by_celltype: Per-cell-type ICA cache (mutated in
            place by this function).
        per_celltype_splits: Required. Per-cell-type train/test index arrays
            from :func:`_splits.generate_evaluation_splits`.
        expvar_mode: Explained-variance aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``).
            Defaults to ``"global_mean"``.
        expvar_modes: Optional list of explained-variance aggregation modes to
            additionally report per celltype (see :func:`ica_reconstruction`
            for the full explanation). Defaults to ``None``, i.e. ``[expvar_mode]``.
        gene_subsets: Optional list of gene subsets to score against:
            ``"all_genes"`` (default), ``"panel_genes_only"``,
            ``"non_panel_genes_only"``. See :func:`ica_reconstruction`.
            Defaults to ``None``, i.e. ``["all_genes"]``.

    Returns:
        Dict with ``"celltype_results"`` and ``"summary"`` keys.

    Raises:
        ValueError: If ``per_celltype_splits`` is ``None``.
    """
    if per_celltype_splits is None:
        raise ValueError(
            "per_celltype_splits is required. Generate splits first with "
            "generate_evaluation_splits() from _splits.py."
        )
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Celltype column '{celltype_column}' not found in adata.obs")

    logger.info("Evaluating ICA by cell type with %d probe genes", len(probeset_genes))

    probeset_mask = np.array([g in probeset_genes for g in adata.var_names])
    if probeset_mask.sum() == 0:
        raise ValueError("None of the probeset genes were found in the dataset")

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info(
        "Found %d / %d probeset genes", len(probeset_genes_found), len(probeset_genes)
    )

    if cached_full_ica_by_celltype is None:
        cached_full_ica_by_celltype = {}

    def _to_f64(ad: Any) -> np.ndarray:
        m = ad.X
        return (m.toarray() if scipy.sparse.issparse(m) else np.asarray(m)).astype(np.float64)

    # ── Gene-subset x expvar-mode grid setup (see ica_reconstruction for the
    # full explanation) — additive on top of the unsuffixed per-celltype keys. ──
    _modes = list(expvar_modes) if expvar_modes else [expvar_mode]
    _subsets = list(gene_subsets) if gene_subsets else ["all_genes"]
    _panel_mask = np.zeros(len(adata.var_names), dtype=bool)
    _panel_mask[probe_indices] = True
    _subset_masks: dict[str, np.ndarray | None] = {
        "all_genes": None,
        "panel_genes_only": _panel_mask,
        "non_panel_genes_only": ~_panel_mask,
    }
    # Full set of extra (subset, mode) keys -- used both when computing a
    # celltype's real results and when filling in NaN for skipped/failed ones,
    # so every celltype row has the same DataFrame columns.
    _extra_nan_keys: dict[str, float] = {}
    for _subset in _subsets:
        _extra_nan_keys[f"mse_train_probe_{_subset}"] = np.nan
        _extra_nan_keys[f"mse_test_probe_{_subset}"] = np.nan
        for _m in _modes:
            _extra_nan_keys[f"expvar_train_probe_{_subset}_{_m}"] = np.nan
            _extra_nan_keys[f"expvar_test_probe_{_subset}_{_m}"] = np.nan
    for _m in _modes:
        _extra_nan_keys[f"expvar_train_baseline_{_m}"] = np.nan
        _extra_nan_keys[f"expvar_test_baseline_{_m}"] = np.nan

    celltypes = adata.obs[celltype_column].unique()
    celltype_results: dict[str, dict[str, Any]] = {}
    min_cells = n_components * 2

    for celltype in tqdm(celltypes, desc="ICA by cell type"):
        logger.info("Processing cell type: %s", celltype)
        adata_ct = adata[adata.obs[celltype_column] == celltype]

        if celltype not in per_celltype_splits:
            logger.debug("Skipping '%s': not in per_celltype_splits", celltype)
            continue

        if adata_ct.shape[0] < min_cells:
            logger.warning(
                "Skipping '%s': %d cells < %d required",
                celltype, adata_ct.shape[0], min_cells,
            )
            celltype_results[celltype] = _skipped_result(
                probeset_genes, probeset_genes_found, adata_ct.shape[0],
                "insufficient_cells", extra_nan_keys=_extra_nan_keys,
            )
            continue

        try:
            train_pos, test_pos = per_celltype_splits[celltype]
            A_train_ct = _to_f64(adata[train_pos])
            A_test_ct = _to_f64(adata[test_pos])

            A_P_train_ct = A_train_ct[:, probe_indices]
            A_P_test_ct = A_test_ct[:, probe_indices]

            # ── Cache check ────────────────────────────────────────────────────
            cached_ct = cached_full_ica_by_celltype.get(celltype, {})
            if (
                "training" in cached_ct
                and all(k in cached_ct["training"] for k in
                        ["predictor", "mse_train_baseline", "expvar_train_baseline",
                         "A_train_baseline"])
                and "testing" in cached_ct
                and "A_test_baseline" in cached_ct["testing"]
            ):
                logger.info("Reusing cached ICA for '%s'", celltype)
                predictor = cached_ct["training"]["predictor"]
                mse_train_baseline = cached_ct["training"]["mse_train_baseline"]
                expvar_train_baseline = cached_ct["training"]["expvar_train_baseline"]
                A_train_baseline = cached_ct["training"]["A_train_baseline"]
                mse_test_baseline = cached_ct["testing"]["mse_test_baseline"]
                expvar_test_baseline = cached_ct["testing"]["expvar_test_baseline"]
                A_test_baseline = cached_ct["testing"]["A_test_baseline"]
            else:
                logger.info("Computing ICA for '%s'", celltype)
                bl = _compute_full_ica_baseline(A_train_ct, A_test_ct, n_components, random_state)
                predictor = bl["training"]["predictor"]
                mse_train_baseline = bl["training"]["mse_train_baseline"]
                expvar_train_baseline = bl["training"]["expvar_train_baseline"]
                A_train_baseline = bl["training"]["A_train_baseline"]
                mse_test_baseline = bl["testing"]["mse_test_baseline"]
                expvar_test_baseline = bl["testing"]["expvar_test_baseline"]
                A_test_baseline = bl["testing"]["A_test_baseline"]
                cached_full_ica_by_celltype[celltype] = {
                    "training": {
                        "predictor": predictor,
                        "mse_train_baseline": mse_train_baseline,
                        "expvar_train_baseline": expvar_train_baseline,
                        "A_train_baseline": A_train_baseline,
                    },
                    "testing": {
                        "mse_test_baseline": mse_test_baseline,
                        "expvar_test_baseline": expvar_test_baseline,
                        "A_test_baseline": A_test_baseline,
                    },
                }

            # ── Probe solve and metrics ────────────────────────────────────────
            S_train, A_train_recon = predictor.predict(A_P_train_ct, indexer=probe_indices)
            S_test, A_test_recon = predictor.predict(A_P_test_ct, indexer=probe_indices)

            mse_train_probe = calculate_mse(A_train_ct, A_train_recon)
            expvar_train_probe = calculate_explained_variance(A_train_ct, A_train_recon)
            mse_test_probe = calculate_mse(A_test_ct, A_test_recon)
            expvar_test_probe = calculate_explained_variance(A_test_ct, A_test_recon)

            # ── Gene-subset x expvar-mode grid (additive; see ica_reconstruction) ──
            _extra_metrics: dict[str, float] = {}
            for _m in _modes:
                _extra_metrics[f"expvar_train_baseline_{_m}"] = calculate_explained_variance(
                    A_train_ct, A_train_baseline, mode=_m
                )
                _extra_metrics[f"expvar_test_baseline_{_m}"] = calculate_explained_variance(
                    A_test_ct, A_test_baseline, mode=_m
                )
            for _subset in _subsets:
                _mask = _subset_masks[_subset]
                _Xtr_true = A_train_ct if _mask is None else A_train_ct[:, _mask]
                _Xtr_recon = A_train_recon if _mask is None else A_train_recon[:, _mask]
                _Xte_true = A_test_ct if _mask is None else A_test_ct[:, _mask]
                _Xte_recon = A_test_recon if _mask is None else A_test_recon[:, _mask]

                _extra_metrics[f"mse_train_probe_{_subset}"] = calculate_mse(_Xtr_true, _Xtr_recon)
                _extra_metrics[f"mse_test_probe_{_subset}"] = calculate_mse(_Xte_true, _Xte_recon)
                for _m in _modes:
                    _extra_metrics[f"expvar_train_probe_{_subset}_{_m}"] = calculate_explained_variance(
                        _Xtr_true, _Xtr_recon, mode=_m
                    )
                    _extra_metrics[f"expvar_test_probe_{_subset}_{_m}"] = calculate_explained_variance(
                        _Xte_true, _Xte_recon, mode=_m
                    )

            celltype_results[celltype] = {
                "mse_train_baseline": mse_train_baseline,
                "mse_test_baseline": mse_test_baseline,
                "mse_train_probe": mse_train_probe,
                "mse_test_probe": mse_test_probe,
                "expvar_train_baseline": expvar_train_baseline,
                "expvar_test_baseline": expvar_test_baseline,
                "expvar_train_probe": expvar_train_probe,
                "expvar_test_probe": expvar_test_probe,
                "probeset_size": len(probeset_genes),
                "probeset_genes_found": len(probeset_genes_found),
                "n_cells": adata_ct.shape[0],
                "skipped": False,
                **_extra_metrics,
            }

        except RuntimeError as exc:
            logger.warning("ICA failed to converge for '%s': %s", celltype, exc)
            celltype_results[celltype] = _skipped_result(
                probeset_genes, probeset_genes_found, adata_ct.shape[0],
                "ica_failed_to_converge", extra_nan_keys=_extra_nan_keys,
            )
        except Exception as exc:
            logger.error("ICA evaluation failed for '%s': %s", celltype, exc)
            celltype_results[celltype] = _skipped_result(
                probeset_genes, probeset_genes_found, adata_ct.shape[0],
                "evaluation_failed", extra_nan_keys=_extra_nan_keys,
            )

        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = {ct: r for ct, r in celltype_results.items() if not r.get("skipped", False)}
    if valid:
        summary: dict[str, Any] = {
            "weighted_mse_test_probe": calculate_weighted_mse(valid),
            "weighted_mse_test_baseline": calculate_weighted_mse_baseline(valid),
            "weighted_expvar_test_probe": calculate_weighted_explained_variance(valid),
            "weighted_expvar_test_baseline": calculate_weighted_explained_variance_baseline(valid),
            "macro_mse_test_probe": calculate_macro_mse(valid),
            "macro_mse_test_baseline": float(np.mean([r["mse_test_baseline"] for r in valid.values()])),
            "macro_expvar_test_probe": calculate_macro_explained_variance(valid),
            "macro_expvar_test_baseline": float(np.mean([r["expvar_test_baseline"] for r in valid.values()])),
            "total_cells": sum(r["n_cells"] for r in valid.values()),
            "n_celltypes_processed": len(valid),
            "n_celltypes_skipped": len(celltype_results) - len(valid),
        }

        # ── Gene-subset x expvar-mode grid summary (additive) ──────────────
        for _m in _modes:
            _bkey = f"expvar_test_baseline_{_m}"
            summary[f"weighted_expvar_test_baseline_{_m}"] = calculate_weighted_explained_variance_baseline(
                valid, metric_key=_bkey
            )
            summary[f"macro_expvar_test_baseline_{_m}"] = np.mean([r[_bkey] for r in valid.values()])
        for _subset in _subsets:
            _mse_key = f"mse_test_probe_{_subset}"
            summary[f"weighted_mse_test_probe_{_subset}"] = calculate_weighted_mse(valid, metric_key=_mse_key)
            summary[f"macro_mse_test_probe_{_subset}"] = calculate_macro_mse(valid, metric_key=_mse_key)
            for _m in _modes:
                _ekey = f"expvar_test_probe_{_subset}_{_m}"
                summary[f"weighted_expvar_test_probe_{_subset}_{_m}"] = calculate_weighted_explained_variance(
                    valid, metric_key=_ekey
                )
                summary[f"macro_expvar_test_probe_{_subset}_{_m}"] = calculate_macro_explained_variance(
                    valid, metric_key=_ekey
                )
    else:
        summary = {
            "weighted_mse_test_probe": np.nan,
            "weighted_mse_test_baseline": np.nan,
            "weighted_expvar_test_probe": np.nan,
            "weighted_expvar_test_baseline": np.nan,
            "macro_mse_test_probe": np.nan,
            "macro_mse_test_baseline": np.nan,
            "macro_expvar_test_probe": np.nan,
            "macro_expvar_test_baseline": np.nan,
            "total_cells": 0,
            "n_celltypes_processed": 0,
            "n_celltypes_skipped": len(celltype_results),
        }
        for _m in _modes:
            summary[f"weighted_expvar_test_baseline_{_m}"] = np.nan
            summary[f"macro_expvar_test_baseline_{_m}"] = np.nan
        for _subset in _subsets:
            summary[f"weighted_mse_test_probe_{_subset}"] = np.nan
            summary[f"macro_mse_test_probe_{_subset}"] = np.nan
            for _m in _modes:
                summary[f"weighted_expvar_test_probe_{_subset}_{_m}"] = np.nan
                summary[f"macro_expvar_test_probe_{_subset}_{_m}"] = np.nan

    return {"celltype_results": celltype_results, "summary": summary}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _skipped_result(
    probeset_genes: list[str],
    probeset_genes_found: Any,
    n_cells: int,
    skip_reason: str,
    extra_nan_keys: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build an all-NaN per-cell-type result row for a skipped cell type.

    Args:
        probeset_genes: Panel genes (used only for ``probeset_size``).
        probeset_genes_found: Count (or list) of panel genes present in the data.
        n_cells: Number of cells of this cell type.
        skip_reason: Short reason string recorded in the row.
        extra_nan_keys: Optional gene-subset x expvar-mode grid NaN keys (see
            ``ica_reconstruction_by_celltype``) merged into the result so every
            celltype row has the same DataFrame columns.

    Returns:
        Result dict with NaN metrics, ``skipped=True`` and ``skip_reason``.
    """
    return {
        "mse_train_baseline": np.nan,
        "mse_test_baseline": np.nan,
        "mse_train_probe": np.nan,
        "mse_test_probe": np.nan,
        "expvar_train_baseline": np.nan,
        "expvar_test_baseline": np.nan,
        "expvar_train_probe": np.nan,
        "expvar_test_probe": np.nan,
        "probeset_size": len(probeset_genes),
        "probeset_genes_found": len(probeset_genes_found),
        "n_cells": n_cells,
        "skipped": True,
        "skip_reason": skip_reason,
        **(extra_nan_keys or {}),
    }
