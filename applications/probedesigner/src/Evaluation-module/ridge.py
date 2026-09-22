"""Ridge regression evaluation for probeset evaluation.

Supervised approach: fit a multioutput Ridge regressor from probe-gene
expression to full-transcriptome expression.  Both raw count space and
pre-normalised space are evaluated in a single call so the two spaces can
be compared directly.  The caller is responsible for providing both matrices;
no normalisation is applied internally.  The trivial baseline in each space
is the per-gene training mean (constant predictor); it is reported as an
absolute reference (``mse_mean_baseline_*`` / ``expvar_mean_baseline_*``).
There are no probe/baseline ratio keys — compare the absolute
``mse_test_probe_*`` against the baseline directly.

Result keys use ``_raw`` / ``_lognorm`` suffixes to distinguish the two spaces.

Functions:
    ridge_reconstruction: Global probe evaluation via ridge regression.
    ridge_reconstruction_by_celltype: Same as above, per cell type.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import scipy.sparse
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from metrics import (
    calculate_explained_variance,
    calculate_macro_explained_variance,
    calculate_macro_mse,
    calculate_mse,
    calculate_weighted_explained_variance,
    calculate_weighted_mse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ridge_reconstruction",
    "ridge_reconstruction_by_celltype",
]

_DEFAULT_ALPHA_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
_SAMPLE_SIZE_FOR_CHECK = 5_000
_CV_N_SPLITS = 5
_CV_N_REPEATS = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_raw_counts(A: np.ndarray, name: str = "A_train") -> None:
    """Warn if *A* does not appear to contain raw integer counts.

    Samples up to ``_SAMPLE_SIZE_FOR_CHECK`` non-zero values and checks
    that they are close to their rounded values, mirroring the logic of
    ``is_anndata_raw_layer`` in the Utility-module.

    Args:
        A: Dense float64 array to check (cells × genes).
        name: Variable name used in the warning message.
    """
    flat = A.ravel()
    nz_idx = np.flatnonzero(flat)
    if len(nz_idx) == 0:
        return
    if len(nz_idx) > _SAMPLE_SIZE_FOR_CHECK:
        nz_idx = np.random.choice(nz_idx, _SAMPLE_SIZE_FOR_CHECK, replace=False)
    sample = flat[nz_idx]
    if not np.allclose(sample, np.round(sample)):
        logger.warning(
            "%s does not appear to contain raw integer counts — "
            "raw-space ridge metrics may be unreliable. "
            "Pass adata.layers['counts'] or adata.raw.X.",
            name,
        )


def _fit_ridge(
    A_P_train: np.ndarray,
    A_train: np.ndarray,
    alpha: float,
    use_ridgecv: bool,
    alpha_values: tuple[float, ...] | list[float],
    random_state: int = 42,
) -> tuple[Any, StandardScaler, float]:
    """Fit StandardScaler + ridge regressor; return (model, scaler, effective_alpha).

    Probe-gene features are z-score standardised before fitting so that the
    L2 penalty is applied uniformly regardless of per-gene expression scale.
    Targets (all genes) are left un-scaled so reconstruction errors stay in
    the original expression space and are comparable with PCA/NMF metrics.

    When *use_ridgecv* is ``True``, alpha is selected via
    ``RepeatedKFold`` (``_CV_N_SPLITS`` × ``_CV_N_REPEATS``) which gives
    more reliable alpha estimates for multioutput problems than the analytic
    GCV approximation (``cv=None``), which is derived for single-output
    regression only.
    """
    scaler: StandardScaler = StandardScaler()
    A_P_scaled = scaler.fit_transform(A_P_train)

    if use_ridgecv:
        cv = RepeatedKFold(
            n_splits=_CV_N_SPLITS,
            n_repeats=_CV_N_REPEATS,
            random_state=random_state,
        )
        reg = RidgeCV(alphas=list(alpha_values), cv=cv)
        reg.fit(A_P_scaled, A_train)
        effective_alpha = float(reg.alpha_)
    else:
        reg = Ridge(alpha=alpha)
        reg.fit(A_P_scaled, A_train)
        effective_alpha = alpha
    return reg, scaler, effective_alpha


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ridge_reconstruction(
    adata: Any,
    probeset_genes: list[str],
    A_train_raw: np.ndarray,
    A_test_raw: np.ndarray,
    A_train_ln: np.ndarray,
    A_test_ln: np.ndarray,
    alpha: float = 1.0,
    use_ridgecv: bool = False,
    alpha_values: list[float] | None = None,
    random_state: int = 42,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> dict[str, Any] | None:
    """Evaluate probe panel quality via ridge regression reconstruction.

    A multioutput Ridge regressor is fitted from probe-gene expression to
    full-transcriptome expression on the training set, independently in raw
    count space and pre-normalised space.  The caller must supply both
    matrices; no normalisation is applied internally.  Performance on the
    held-out test set is compared to the trivial per-gene mean predictor in
    each space.

    Raw-count validation follows ``is_anndata_raw_layer`` from the
    Utility-module: a warning is logged if ``A_train_raw`` does not appear to
    contain integer counts.

    Args:
        adata: AnnData with ``var_names`` covering all genes.
        probeset_genes: Probe gene names to evaluate.
        A_train_raw: Raw training counts (cells × genes), float64.
        A_test_raw: Raw test counts (cells × genes), float64.
        A_train_ln: Pre-normalised training data (cells × genes), float64.
        A_test_ln: Pre-normalised test data (cells × genes), float64.
        alpha: Ridge regularisation strength (ignored when *use_ridgecv*
            is ``True``).
        use_ridgecv: If ``True``, use cross-validation to select the best
            ``alpha`` from *alpha_values* (applied independently in each
            space).
        alpha_values: Candidate alphas for ``RidgeCV`` (default:
            ``(0.01, 0.1, 1.0, 10.0, 100.0)``).
        random_state: Random seed passed to ``RepeatedKFold`` when
            *use_ridgecv* is ``True``.
        expvar_mode: Explained-variance aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``).
            Defaults to ``"global_mean"``.
        expvar_modes: Optional list of explained-variance aggregation modes to
            additionally report, on top of the unsuffixed keys. Adds
            ``expvar_test_probe_{subset}_{mode}_{space}`` (and the
            ``expvar_mean_baseline_{mode}_{space}`` counterpart) for every
            ``(subset, mode)`` pair, in both raw and lognorm space. Defaults to
            ``None``, i.e. ``[expvar_mode]`` — no extra columns.
        gene_subsets: Optional list of gene subsets to score the probe
            reconstruction against: ``"all_genes"`` (default),
            ``"panel_genes_only"``, ``"non_panel_genes_only"``. This does
            **not** change what is reconstructed — it is a scoring-time column
            mask applied to the already-computed reconstruction, matching
            ``nmf.py``'s convention. Baseline metrics are only ever computed on
            the all-genes scale. Defaults to ``None``, i.e. ``["all_genes"]``.

    Returns:
        Metrics dict with ``_raw`` and ``_lognorm`` suffixed keys, or ``None``
        if no probe genes are found in the dataset.
    """
    logger.info("Evaluating Ridge reconstruction with %d probe genes", len(probeset_genes))

    probeset_mask = np.array([g in probeset_genes for g in adata.var_names])
    if probeset_mask.sum() == 0:
        logger.error("No probeset genes found in the dataset")
        return None

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info(
        "Found %d / %d probeset genes", len(probeset_genes_found), len(probeset_genes)
    )

    avs = tuple(alpha_values) if alpha_values is not None else _DEFAULT_ALPHA_VALUES

    A_train_raw = A_train_raw.astype(np.float64)
    A_test_raw = A_test_raw.astype(np.float64)
    A_train_ln = A_train_ln.astype(np.float64)
    A_test_ln = A_test_ln.astype(np.float64)
    _check_raw_counts(A_train_raw, "A_train_raw")

    A_P_train_raw = A_train_raw[:, probe_indices]
    A_P_test_raw = A_test_raw[:, probe_indices]
    A_P_train_ln = A_train_ln[:, probe_indices]
    A_P_test_ln = A_test_ln[:, probe_indices]

    logger.info(
        "Fitting %s (alpha=%s) in raw and log-normalised space",
        "RidgeCV" if use_ridgecv else "Ridge",
        list(avs) if use_ridgecv else alpha,
    )

    # ── Raw space ──────────────────────────────────────────────────────────────
    mean_train_raw = A_train_raw.mean(axis=0)
    A_mean_pred_raw = np.broadcast_to(mean_train_raw, A_test_raw.shape)
    mse_mean_baseline_raw = calculate_mse(A_test_raw, A_mean_pred_raw)
    expvar_mean_baseline_raw = calculate_explained_variance(A_test_raw, A_mean_pred_raw)

    reg_raw, scaler_raw, alpha_raw = _fit_ridge(A_P_train_raw, A_train_raw, alpha, use_ridgecv, avs, random_state)
    A_test_recon_raw = reg_raw.predict(scaler_raw.transform(A_P_test_raw))
    mse_test_probe_raw = calculate_mse(A_test_raw, A_test_recon_raw)
    expvar_test_probe_raw = calculate_explained_variance(A_test_raw, A_test_recon_raw)

    logger.info(
        "[raw]     MSE: baseline=%.6f probe=%.6f | expvar=%.3f",
        mse_mean_baseline_raw, mse_test_probe_raw, expvar_test_probe_raw,
    )

    # ── Log-normalised space ───────────────────────────────────────────────────
    mean_train_ln = A_train_ln.mean(axis=0)
    A_mean_pred_ln = np.broadcast_to(mean_train_ln, A_test_ln.shape)
    mse_mean_baseline_lognorm = calculate_mse(A_test_ln, A_mean_pred_ln)
    expvar_mean_baseline_lognorm = calculate_explained_variance(A_test_ln, A_mean_pred_ln)

    reg_ln, scaler_ln, alpha_ln = _fit_ridge(A_P_train_ln, A_train_ln, alpha, use_ridgecv, avs, random_state)
    if use_ridgecv:
        logger.info("RidgeCV: raw alpha=%.4g, lognorm alpha=%.4g", alpha_raw, alpha_ln)
    A_test_recon_ln = reg_ln.predict(scaler_ln.transform(A_P_test_ln))
    mse_test_probe_lognorm = calculate_mse(A_test_ln, A_test_recon_ln)
    expvar_test_probe_lognorm = calculate_explained_variance(A_test_ln, A_test_recon_ln)

    logger.info(
        "[lognorm] MSE: baseline=%.6f probe=%.6f | expvar=%.3f",
        mse_mean_baseline_lognorm, mse_test_probe_lognorm, expvar_test_probe_lognorm,
    )

    result: dict[str, Any] = {
        # Raw space
        "mse_mean_baseline_raw": mse_mean_baseline_raw,
        "expvar_mean_baseline_raw": expvar_mean_baseline_raw,
        "mse_test_probe_raw": mse_test_probe_raw,
        "expvar_test_probe_raw": expvar_test_probe_raw,
        "alpha_raw": alpha_raw,
        # Log-normalised space
        "mse_mean_baseline_lognorm": mse_mean_baseline_lognorm,
        "expvar_mean_baseline_lognorm": expvar_mean_baseline_lognorm,
        "mse_test_probe_lognorm": mse_test_probe_lognorm,
        "expvar_test_probe_lognorm": expvar_test_probe_lognorm,
        "alpha_lognorm": alpha_ln,
        # Common
        "probeset_size": len(probeset_genes),
        "probeset_genes_found": len(probeset_genes_found),
    }

    # ── Gene-subset x expvar-mode grid (additive on top of the keys above) ────
    # Does not change what is reconstructed -- purely a scoring-time column
    # mask, same convention as nmf.py::nmf_reconstruction. Composes with
    # ridge's own _raw/_lognorm space axis, kept as the outermost suffix
    # (matches every existing ridge column) so filtering by space stays a
    # simple trailing-suffix match regardless of subset/mode.
    modes = list(expvar_modes) if expvar_modes else [expvar_mode]
    subsets = list(gene_subsets) if gene_subsets else ["all_genes"]

    panel_mask = np.zeros(A_test_raw.shape[1], dtype=bool)
    panel_mask[probe_indices] = True
    _subset_masks: dict[str, np.ndarray | None] = {
        "all_genes": None,
        "panel_genes_only": panel_mask,
        "non_panel_genes_only": ~panel_mask,
    }

    # Baseline is only ever computed on the all-genes scale (no train/test axis
    # either -- ridge's baseline is scored only against the test split).
    for m in modes:
        result[f"expvar_mean_baseline_{m}_raw"] = calculate_explained_variance(
            A_test_raw, A_mean_pred_raw, mode=m
        )
        result[f"expvar_mean_baseline_{m}_lognorm"] = calculate_explained_variance(
            A_test_ln, A_mean_pred_ln, mode=m
        )

    for subset in subsets:
        mask = _subset_masks[subset]
        Xr_true = A_test_raw if mask is None else A_test_raw[:, mask]
        Xr_recon = A_test_recon_raw if mask is None else A_test_recon_raw[:, mask]
        Xl_true = A_test_ln if mask is None else A_test_ln[:, mask]
        Xl_recon = A_test_recon_ln if mask is None else A_test_recon_ln[:, mask]

        result[f"mse_test_probe_{subset}_raw"] = calculate_mse(Xr_true, Xr_recon)
        result[f"mse_test_probe_{subset}_lognorm"] = calculate_mse(Xl_true, Xl_recon)
        for m in modes:
            result[f"expvar_test_probe_{subset}_{m}_raw"] = calculate_explained_variance(
                Xr_true, Xr_recon, mode=m
            )
            result[f"expvar_test_probe_{subset}_{m}_lognorm"] = calculate_explained_variance(
                Xl_true, Xl_recon, mode=m
            )

    gc.collect()
    # Absolute metrics only (no probe/mean-baseline ratios); the mean-predictor
    # baseline numbers stay as an absolute reference.
    return result


def ridge_reconstruction_by_celltype(
    adata: Any,
    probeset_genes: list[str],
    celltype_column: str = "cluster",  # = _constants.DEFAULT_CELLTYPE_COLUMN
    alpha: float = 1.0,
    use_ridgecv: bool = False,
    alpha_values: list[float] | None = None,
    random_state: int = 42,
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate Ridge regression representation for each cell type separately.

    A fresh Ridge regressor is fitted per cell type in both raw and
    pre-normalised space.  The caller must ensure ``adata.layers['counts']``
    holds raw counts and ``adata.X`` holds pre-normalised data.  No caching
    is needed because ridge fitting is fast.

    Args:
        adata: AnnData with ``var_names``, ``obs[celltype_column]``,
            ``layers['counts']`` (raw), and ``X`` (pre-normalised).
        probeset_genes: Probe gene names to evaluate.
        celltype_column: obs column holding cell-type labels.
        alpha: Ridge regularisation strength.
        use_ridgecv: Use cross-validation to select alpha per cell type
            (independently for raw and lognorm spaces).
        alpha_values: Candidate alphas for ``RidgeCV``.
        random_state: Random seed passed to ``RepeatedKFold`` when
            *use_ridgecv* is ``True``.
        per_celltype_splits: Required. Per-cell-type train/test index arrays
            from :func:`_splits.generate_evaluation_splits`.
        expvar_mode: Explained-variance aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``).
            Defaults to ``"global_mean"``.
        expvar_modes: Optional list of explained-variance aggregation modes to
            additionally report per celltype (see :func:`ridge_reconstruction`
            for the full explanation). Defaults to ``None``, i.e. ``[expvar_mode]``.
        gene_subsets: Optional list of gene subsets to score against:
            ``"all_genes"`` (default), ``"panel_genes_only"``,
            ``"non_panel_genes_only"``. See :func:`ridge_reconstruction`.
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

    logger.info("Evaluating Ridge by cell type with %d probe genes", len(probeset_genes))

    probeset_mask = np.array([g in probeset_genes for g in adata.var_names])
    if probeset_mask.sum() == 0:
        raise ValueError("None of the probeset genes were found in the dataset")

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info(
        "Found %d / %d probeset genes", len(probeset_genes_found), len(probeset_genes)
    )

    avs = tuple(alpha_values) if alpha_values is not None else _DEFAULT_ALPHA_VALUES

    def _to_f64_raw(ad: Any) -> np.ndarray:
        m = ad.layers["counts"]
        return (m.toarray() if scipy.sparse.issparse(m) else np.asarray(m)).astype(np.float64)

    def _to_f64_ln(ad: Any) -> np.ndarray:
        m = ad.X
        return (m.toarray() if scipy.sparse.issparse(m) else np.asarray(m)).astype(np.float64)

    # ── Gene-subset x expvar-mode grid setup (see ridge_reconstruction for the
    # full explanation) — additive on top of the unsuffixed per-celltype keys.
    # The panel mask is shared by both spaces (raw and lognorm matrices share
    # the same gene axis). ──
    _modes = list(expvar_modes) if expvar_modes else [expvar_mode]
    _subsets = list(gene_subsets) if gene_subsets else ["all_genes"]
    _panel_mask = np.zeros(len(adata.var_names), dtype=bool)
    _panel_mask[probe_indices] = True
    _subset_masks: dict[str, np.ndarray | None] = {
        "all_genes": None,
        "panel_genes_only": _panel_mask,
        "non_panel_genes_only": ~_panel_mask,
    }
    # Full set of extra (subset, mode, space) / (subset, space) keys -- used
    # both when computing a celltype's real results and when filling in NaN
    # for skipped/failed ones, so every celltype row has the same columns.
    _extra_nan_keys: dict[str, float] = {}
    for _subset in _subsets:
        for _space in ("raw", "lognorm"):
            _extra_nan_keys[f"mse_train_probe_{_subset}_{_space}"] = np.nan
            _extra_nan_keys[f"mse_test_probe_{_subset}_{_space}"] = np.nan
            for _m in _modes:
                _extra_nan_keys[f"expvar_train_probe_{_subset}_{_m}_{_space}"] = np.nan
                _extra_nan_keys[f"expvar_test_probe_{_subset}_{_m}_{_space}"] = np.nan
    for _m in _modes:
        for _space in ("raw", "lognorm"):
            _extra_nan_keys[f"expvar_mean_baseline_{_m}_{_space}"] = np.nan

    celltypes = adata.obs[celltype_column].unique()
    celltype_results: dict[str, dict[str, Any]] = {}
    min_cells = max(5, len(probe_indices) + 2)

    for celltype in tqdm(celltypes, desc="Ridge by cell type"):
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
            A_train_ct_raw = _to_f64_raw(adata[train_pos])
            A_test_ct_raw = _to_f64_raw(adata[test_pos])
            A_train_ln = _to_f64_ln(adata[train_pos])
            A_test_ln = _to_f64_ln(adata[test_pos])

            A_P_train_raw = A_train_ct_raw[:, probe_indices]
            A_P_test_raw = A_test_ct_raw[:, probe_indices]
            A_P_train_ln = A_train_ln[:, probe_indices]
            A_P_test_ln = A_test_ln[:, probe_indices]

            # ── Raw space ─────────────────────────────────────────────────────
            mean_train_raw = A_train_ct_raw.mean(axis=0)
            A_mean_pred_raw = np.broadcast_to(mean_train_raw, A_test_ct_raw.shape)
            mse_mean_baseline_raw = calculate_mse(A_test_ct_raw, A_mean_pred_raw)
            expvar_mean_baseline_raw = calculate_explained_variance(A_test_ct_raw, A_mean_pred_raw)

            reg_raw, scaler_raw, eff_alpha_raw = _fit_ridge(A_P_train_raw, A_train_ct_raw, alpha, use_ridgecv, avs, random_state)
            A_test_recon_raw = reg_raw.predict(scaler_raw.transform(A_P_test_raw))
            A_train_recon_raw = reg_raw.predict(scaler_raw.transform(A_P_train_raw))
            mse_test_probe_raw = calculate_mse(A_test_ct_raw, A_test_recon_raw)
            expvar_test_probe_raw = calculate_explained_variance(A_test_ct_raw, A_test_recon_raw)
            mse_train_probe_raw = calculate_mse(A_train_ct_raw, A_train_recon_raw)
            expvar_train_probe_raw = calculate_explained_variance(A_train_ct_raw, A_train_recon_raw)

            # ── Log-normalised space ───────────────────────────────────────────
            mean_train_ln = A_train_ln.mean(axis=0)
            A_mean_pred_ln = np.broadcast_to(mean_train_ln, A_test_ln.shape)
            mse_mean_baseline_lognorm = calculate_mse(A_test_ln, A_mean_pred_ln)
            expvar_mean_baseline_lognorm = calculate_explained_variance(A_test_ln, A_mean_pred_ln)

            reg_ln, scaler_ln, eff_alpha_ln = _fit_ridge(A_P_train_ln, A_train_ln, alpha, use_ridgecv, avs, random_state)
            A_test_recon_ln = reg_ln.predict(scaler_ln.transform(A_P_test_ln))
            A_train_recon_ln = reg_ln.predict(scaler_ln.transform(A_P_train_ln))
            mse_test_probe_lognorm = calculate_mse(A_test_ln, A_test_recon_ln)
            expvar_test_probe_lognorm = calculate_explained_variance(A_test_ln, A_test_recon_ln)
            mse_train_probe_lognorm = calculate_mse(A_train_ln, A_train_recon_ln)
            expvar_train_probe_lognorm = calculate_explained_variance(A_train_ln, A_train_recon_ln)

            # ── Gene-subset x expvar-mode grid (additive; see ridge_reconstruction) ──
            _extra_metrics: dict[str, float] = {}
            for _m in _modes:
                _extra_metrics[f"expvar_mean_baseline_{_m}_raw"] = calculate_explained_variance(
                    A_test_ct_raw, A_mean_pred_raw, mode=_m
                )
                _extra_metrics[f"expvar_mean_baseline_{_m}_lognorm"] = calculate_explained_variance(
                    A_test_ln, A_mean_pred_ln, mode=_m
                )
            for _subset in _subsets:
                _mask = _subset_masks[_subset]
                _Xtr_true_raw = A_train_ct_raw if _mask is None else A_train_ct_raw[:, _mask]
                _Xtr_recon_raw = A_train_recon_raw if _mask is None else A_train_recon_raw[:, _mask]
                _Xte_true_raw = A_test_ct_raw if _mask is None else A_test_ct_raw[:, _mask]
                _Xte_recon_raw = A_test_recon_raw if _mask is None else A_test_recon_raw[:, _mask]
                _Xtr_true_ln = A_train_ln if _mask is None else A_train_ln[:, _mask]
                _Xtr_recon_ln = A_train_recon_ln if _mask is None else A_train_recon_ln[:, _mask]
                _Xte_true_ln = A_test_ln if _mask is None else A_test_ln[:, _mask]
                _Xte_recon_ln = A_test_recon_ln if _mask is None else A_test_recon_ln[:, _mask]

                _extra_metrics[f"mse_train_probe_{_subset}_raw"] = calculate_mse(_Xtr_true_raw, _Xtr_recon_raw)
                _extra_metrics[f"mse_test_probe_{_subset}_raw"] = calculate_mse(_Xte_true_raw, _Xte_recon_raw)
                _extra_metrics[f"mse_train_probe_{_subset}_lognorm"] = calculate_mse(_Xtr_true_ln, _Xtr_recon_ln)
                _extra_metrics[f"mse_test_probe_{_subset}_lognorm"] = calculate_mse(_Xte_true_ln, _Xte_recon_ln)
                for _m in _modes:
                    _extra_metrics[f"expvar_train_probe_{_subset}_{_m}_raw"] = calculate_explained_variance(
                        _Xtr_true_raw, _Xtr_recon_raw, mode=_m
                    )
                    _extra_metrics[f"expvar_test_probe_{_subset}_{_m}_raw"] = calculate_explained_variance(
                        _Xte_true_raw, _Xte_recon_raw, mode=_m
                    )
                    _extra_metrics[f"expvar_train_probe_{_subset}_{_m}_lognorm"] = calculate_explained_variance(
                        _Xtr_true_ln, _Xtr_recon_ln, mode=_m
                    )
                    _extra_metrics[f"expvar_test_probe_{_subset}_{_m}_lognorm"] = calculate_explained_variance(
                        _Xte_true_ln, _Xte_recon_ln, mode=_m
                    )

            celltype_results[celltype] = {
                # Raw space
                "mse_mean_baseline_raw": mse_mean_baseline_raw,
                "expvar_mean_baseline_raw": expvar_mean_baseline_raw,
                "mse_train_probe_raw": mse_train_probe_raw,
                "mse_test_probe_raw": mse_test_probe_raw,
                "expvar_train_probe_raw": expvar_train_probe_raw,
                "expvar_test_probe_raw": expvar_test_probe_raw,
                "alpha_raw": eff_alpha_raw,
                # Log-normalised space
                "mse_mean_baseline_lognorm": mse_mean_baseline_lognorm,
                "expvar_mean_baseline_lognorm": expvar_mean_baseline_lognorm,
                "mse_train_probe_lognorm": mse_train_probe_lognorm,
                "mse_test_probe_lognorm": mse_test_probe_lognorm,
                "expvar_train_probe_lognorm": expvar_train_probe_lognorm,
                "expvar_test_probe_lognorm": expvar_test_probe_lognorm,
                "alpha_lognorm": eff_alpha_ln,
                # Common
                "probeset_size": len(probeset_genes),
                "probeset_genes_found": len(probeset_genes_found),
                "n_cells": adata_ct.shape[0],
                "skipped": False,
                **_extra_metrics,
            }

        except Exception as exc:
            logger.error("Ridge evaluation failed for '%s': %s", celltype, exc)
            celltype_results[celltype] = _skipped_result(
                probeset_genes, probeset_genes_found, adata_ct.shape[0],
                "evaluation_failed", extra_nan_keys=_extra_nan_keys,
            )

        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = {ct: r for ct, r in celltype_results.items() if not r.get("skipped", False)}
    if valid:
        total_cells = sum(r["n_cells"] for r in valid.values())

        summary: dict[str, Any] = {
            # Raw space weighted
            "weighted_mse_test_probe_raw": calculate_weighted_mse(valid, metric_key="mse_test_probe_raw"),
            "weighted_expvar_test_probe_raw": calculate_weighted_explained_variance(valid, metric_key="expvar_test_probe_raw"),
            "weighted_mse_mean_baseline_raw": calculate_weighted_mse(valid, metric_key="mse_mean_baseline_raw"),
            "weighted_expvar_mean_baseline_raw": calculate_weighted_explained_variance(valid, metric_key="expvar_mean_baseline_raw"),
            # Raw space macro
            "macro_mse_test_probe_raw": calculate_macro_mse(valid, metric_key="mse_test_probe_raw"),
            "macro_expvar_test_probe_raw": calculate_macro_explained_variance(valid, metric_key="expvar_test_probe_raw"),
            "macro_mse_mean_baseline_raw": calculate_macro_mse(valid, metric_key="mse_mean_baseline_raw"),
            "macro_expvar_mean_baseline_raw": calculate_macro_explained_variance(valid, metric_key="expvar_mean_baseline_raw"),
            # Lognorm space weighted
            "weighted_mse_test_probe_lognorm": calculate_weighted_mse(valid, metric_key="mse_test_probe_lognorm"),
            "weighted_expvar_test_probe_lognorm": calculate_weighted_explained_variance(valid, metric_key="expvar_test_probe_lognorm"),
            "weighted_mse_mean_baseline_lognorm": calculate_weighted_mse(valid, metric_key="mse_mean_baseline_lognorm"),
            "weighted_expvar_mean_baseline_lognorm": calculate_weighted_explained_variance(valid, metric_key="expvar_mean_baseline_lognorm"),
            # Lognorm space macro
            "macro_mse_test_probe_lognorm": calculate_macro_mse(valid, metric_key="mse_test_probe_lognorm"),
            "macro_expvar_test_probe_lognorm": calculate_macro_explained_variance(valid, metric_key="expvar_test_probe_lognorm"),
            "macro_mse_mean_baseline_lognorm": calculate_macro_mse(valid, metric_key="mse_mean_baseline_lognorm"),
            "macro_expvar_mean_baseline_lognorm": calculate_macro_explained_variance(valid, metric_key="expvar_mean_baseline_lognorm"),
            # Counts
            "total_cells": total_cells,
            "n_celltypes_processed": len(valid),
            "n_celltypes_skipped": len(celltype_results) - len(valid),
        }

        # ── Gene-subset x expvar-mode grid summary (additive) ──────────────
        # Doubles relative to nmf/pca/ica's summary grid: every column is
        # computed once per space (raw, lognorm) on top of (subset, mode).
        for _m in _modes:
            for _space in ("raw", "lognorm"):
                _bkey = f"expvar_mean_baseline_{_m}_{_space}"
                summary[f"weighted_{_bkey}"] = calculate_weighted_explained_variance(valid, metric_key=_bkey)
                summary[f"macro_{_bkey}"] = calculate_macro_explained_variance(valid, metric_key=_bkey)
        for _subset in _subsets:
            for _space in ("raw", "lognorm"):
                _mse_key = f"mse_test_probe_{_subset}_{_space}"
                summary[f"weighted_{_mse_key}"] = calculate_weighted_mse(valid, metric_key=_mse_key)
                summary[f"macro_{_mse_key}"] = calculate_macro_mse(valid, metric_key=_mse_key)
                for _m in _modes:
                    _ekey = f"expvar_test_probe_{_subset}_{_m}_{_space}"
                    summary[f"weighted_{_ekey}"] = calculate_weighted_explained_variance(valid, metric_key=_ekey)
                    summary[f"macro_{_ekey}"] = calculate_macro_explained_variance(valid, metric_key=_ekey)
    else:
        nan = float("nan")
        summary = {
            "weighted_mse_test_probe_raw": nan,
            "weighted_expvar_test_probe_raw": nan,
            "weighted_mse_mean_baseline_raw": nan,
            "weighted_expvar_mean_baseline_raw": nan,
            "macro_mse_test_probe_raw": nan,
            "macro_expvar_test_probe_raw": nan,
            "macro_mse_mean_baseline_raw": nan,
            "macro_expvar_mean_baseline_raw": nan,
            "weighted_mse_test_probe_lognorm": nan,
            "weighted_expvar_test_probe_lognorm": nan,
            "weighted_mse_mean_baseline_lognorm": nan,
            "weighted_expvar_mean_baseline_lognorm": nan,
            "macro_mse_test_probe_lognorm": nan,
            "macro_expvar_test_probe_lognorm": nan,
            "macro_mse_mean_baseline_lognorm": nan,
            "macro_expvar_mean_baseline_lognorm": nan,
            "total_cells": 0,
            "n_celltypes_processed": 0,
            "n_celltypes_skipped": len(celltype_results),
        }
        for _m in _modes:
            for _space in ("raw", "lognorm"):
                _bkey = f"expvar_mean_baseline_{_m}_{_space}"
                summary[f"weighted_{_bkey}"] = nan
                summary[f"macro_{_bkey}"] = nan
        for _subset in _subsets:
            for _space in ("raw", "lognorm"):
                _mse_key = f"mse_test_probe_{_subset}_{_space}"
                summary[f"weighted_{_mse_key}"] = nan
                summary[f"macro_{_mse_key}"] = nan
                for _m in _modes:
                    _ekey = f"expvar_test_probe_{_subset}_{_m}_{_space}"
                    summary[f"weighted_{_ekey}"] = nan
                    summary[f"macro_{_ekey}"] = nan

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
    """Build an all-NaN per-cell-type result row (raw + lognorm keys) for a
    skipped cell type.

    Args:
        probeset_genes: Panel genes (used only for ``probeset_size``).
        probeset_genes_found: Count (or list) of panel genes present in the data.
        n_cells: Number of cells of this cell type.
        skip_reason: Short reason string recorded in the row.
        extra_nan_keys: Optional gene-subset x expvar-mode grid NaN keys (see
            ``ridge_reconstruction_by_celltype``) merged into the result so
            every celltype row has the same DataFrame columns.

    Returns:
        Result dict with NaN metrics, ``skipped=True`` and ``skip_reason``.
    """
    nan = float("nan")
    return {
        # Raw space
        "mse_mean_baseline_raw": nan,
        "expvar_mean_baseline_raw": nan,
        "mse_train_probe_raw": nan,
        "mse_test_probe_raw": nan,
        "expvar_train_probe_raw": nan,
        "expvar_test_probe_raw": nan,
        "alpha_raw": nan,
        # Log-normalised space
        "mse_mean_baseline_lognorm": nan,
        "expvar_mean_baseline_lognorm": nan,
        "mse_train_probe_lognorm": nan,
        "mse_test_probe_lognorm": nan,
        "expvar_train_probe_lognorm": nan,
        "expvar_test_probe_lognorm": nan,
        "alpha_lognorm": nan,
        # Common
        "probeset_size": len(probeset_genes),
        "probeset_genes_found": len(probeset_genes_found),
        "n_cells": n_cells,
        "skipped": True,
        "skip_reason": skip_reason,
        **(extra_nan_keys or {}),
    }
