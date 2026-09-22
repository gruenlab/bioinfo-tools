"""NMF-based representation evaluation for probeset evaluation.

Functions:
    nmf_reconstruction: Evaluates probe patterns for global reconstruction.
    nmf_reconstruction_by_celltype: Same as above, per cell type.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from metrics import (
    calculate_macro_explained_variance,
    calculate_macro_mse,
    calculate_weighted_explained_variance,
    calculate_weighted_explained_variance_baseline,
    calculate_weighted_mse,
    calculate_weighted_mse_baseline,
)

logger = logging.getLogger(__name__)

_UTILITY_DIR = Path(__file__).parent.parent / "Utility-module"
sys.path.insert(0, str(_UTILITY_DIR))
from _validation import is_anndata_raw_layer, is_anndata_raw  # type: ignore[import]
from _nmf_objective import resolve_nmf_objective

__all__ = [
    "nmf_reconstruction",
    "nmf_reconstruction_by_celltype",
]


def nmf_reconstruction(
    adata,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    n_components: int = 5,
    random_state: int = 42,
    cached_full_nmf: dict[str, Any] | None = None,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
    nmf_counts_input: str = "raw",
    nmf_objective: str = "auto",
) -> dict[str, Any]:
    """Evaluate how well a probe gene subset can represent the full transcriptome.

    This method evaluates if the patterns/factors learned from just the probe genes
    (a small subset) can be used to reconstruct the full transcriptome when combined
    with those genes' activities in new samples.

    **Implementation equivalence**: The constrained W-solve used here is identical to
    ``NmfPredictor.predict()`` in ``nico2_lib``:

    .. code-block:: python

        # This function (both train and test solve):
        W, _, _ = non_negative_factorization(
            A_P, H=H_P, init="custom", update_H=False
        )
        A_recon = W @ H_full_train

        # NmfPredictor.predict():
        w_query, _ = non_negative_factorization(
            X=X, H=h_reference[:, indexer], init="custom", update_H=False
        )
        return w_query @ h_reference

    H is **always fixed** to the probe-gene slice from the training fit: the solve is
    constrained (only W is estimated), never a fresh unconstrained NMF on the probe subset.

    Following scikit-learn convention:
        - A (samples/cells × features/genes): Data matrix
        - W (samples/cells × components): Sample factor matrix
        - H (components × features/genes): Feature factor matrix

    Args:
        adata: Annotated data matrix with full transcriptome.
        probeset_genes: List of genes in the probeset to evaluate.
        A_train: Pre-split training data (cells × genes).
        A_test: Pre-split testing data (cells × genes).
        n_components: Number of NMF components to use.
        random_state: Random state for reproducibility.
        cached_full_nmf: Pre-computed full NMF results to avoid recomputation.
        expvar_mode: Explained-variance aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``).
            Defaults to ``"global_mean"``.
        expvar_modes: Optional list of explained-variance aggregation modes to
            additionally report (e.g. ``["global_mean", "variance_weighted_sum",
            "mean"]``). When given, adds ``expvar_test_probe_{subset}_{mode}``
            (and the ``expvar_*_baseline_{mode}`` counterpart) to the result for
            every ``(subset, mode)`` pair, on top of the unsuffixed keys (which are
            always computed from ``expvar_mode`` alone, unchanged). Defaults to
            ``None``, i.e. ``[expvar_mode]`` — no extra columns.
        gene_subsets: Optional list of gene subsets to score the probe
            reconstruction against: ``"all_genes"`` (default),
            ``"panel_genes_only"``, ``"non_panel_genes_only"``. This does
            **not** change what is reconstructed (always the full
            transcriptome, exactly as before) — it is a scoring-time column
            mask applied to the already-computed reconstruction, identical to
            ``Analysis-scripts/pipeline/run_expvar_aggregation_test.py``'s
            convention. Baseline metrics are only ever computed on the
            all-genes scale (the baseline NMF fit doesn't depend on which
            panel is being reconstructed). Defaults to ``None``, i.e.
            ``["all_genes"]`` — matches today's implicit all-genes-only
            scoring.
        nmf_counts_input: Which count matrix ``A_train``/``A_test`` were derived from
            (``"raw"`` or ``"lognorm"``) — used only to resolve ``nmf_objective``, not
            to select the matrix (already fixed by the caller).
        nmf_objective: NMF factorization objective — ``"auto"`` (default) derives the
            solver/beta_loss from ``nmf_counts_input``; ``"frobenius"``/``"kl"`` force
            that objective regardless of input. See ``_nmf_objective.py``.

    Returns:
        Dictionary with MSE and explained variance metrics.
    """
    logger.info(f"Evaluating NMF representation with {len(probeset_genes)} genes")

    # Get indices of probe genes
    probeset_mask = np.array([gene in probeset_genes for gene in adata.var_names])
    if sum(probeset_mask) == 0:
        logger.error(f"No probeset genes found in the dataset")
        return None

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info(
        f"Found {len(probeset_genes_found)} out of {len(probeset_genes)} probeset genes in the dataset"
    )

    # Use pre-split data and ensure float64 dtype for NMF compatibility
    A_train = A_train.astype(np.float64)
    A_test = A_test.astype(np.float64)
    A_P_train = A_train[:, probe_indices]  # (cells × probe_genes)
    A_P_test = A_test[:, probe_indices]  # (cells × probe_genes)

    logger.info(f"Training: {A_train.shape[0]} cells × {A_train.shape[1]} genes")
    logger.info(f"Testing: {A_test.shape[0]} cells × {A_test.shape[1]} genes")
    logger.info(f"Probe subset: {len(probe_indices)} genes")

    # Validate dimensions
    if A_P_train.shape[1] != len(probe_indices):
        logger.error(
            f"Dimension mismatch: A_P_train columns ({A_P_train.shape[1]}) != probe_indices ({len(probe_indices)})"
        )
    if A_P_test.shape[1] != len(probe_indices):
        logger.error(
            f"Dimension mismatch: A_P_test columns ({A_P_test.shape[1]}) != probe_indices ({len(probe_indices)})"
        )

    # ================================================================
    # TRAINING PHASE
    # ================================================================
    logger.info("--- Training Phase ---")

    # Step 1: Full NMF on training data
    if cached_full_nmf is not None and "training" in cached_full_nmf:
        logger.info("Using cached full NMF results for training data")

        training_cache = cached_full_nmf["training"]
        if training_cache["A_train"].shape == A_train.shape:
            if np.allclose(training_cache["A_train"], A_train):
                H_full_train = training_cache["H_full_train"].astype(np.float64)
                W_full_train = training_cache["W_full_train"].astype(np.float64)
                A_train_baseline = training_cache["A_train_baseline"]
                mse_train_baseline = training_cache["mse_train_baseline"]
                expvar_train_baseline = training_cache["expvar_train_baseline"]
                logger.info("Successfully reused cached training NMF data")
            else:
                logger.warning("Cached training data doesn't match current split, recomputing...")
                cached_full_nmf = None
        else:
            logger.warning("Cached training data shape doesn't match, recomputing...")
            cached_full_nmf = None

    # solver/beta_loss/init/max_iter derived from the count-input choice (raw -> mu/KL,
    # lognorm -> cd/Frobenius) unless nmf_objective forces one; shared with Selection-module.
    _obj_kwargs = resolve_nmf_objective(nmf_counts_input, nmf_objective)
    logger.info(
        f"Global NMF objective: nmf_objective={nmf_objective} "
        f"(nmf_counts_input={nmf_counts_input}) -> {_obj_kwargs}"
    )
    _nmf_kwargs = dict(
        n_components=n_components, embedding_size=n_components,
        random_state=random_state, **_obj_kwargs,
    )

    if cached_full_nmf is None or "training" not in cached_full_nmf:
        logger.info("Computing Full NMF on training data")
        train_predictor = NmfPredictor(**_nmf_kwargs).fit(A_train)
        W_full_train = train_predictor.ref_embedding
        H_full_train = train_predictor.h_reference
        A_train_baseline = W_full_train @ H_full_train
        mse_train_baseline = calculate_mse(A_train, A_train_baseline)
        expvar_train_baseline = calculate_explained_variance(A_train, A_train_baseline, mode=expvar_mode)
    else:
        # Reconstruct a pre-fitted predictor from cached H so predict() can be used
        train_predictor = NmfPredictor(**_nmf_kwargs, h_reference=H_full_train, ref_embedding=W_full_train)

    # Step 2: Extract gene subset patterns (for logging only — computed internally by predict())
    logger.info("Step 1: Extract probe gene patterns")
    H_P = H_full_train[:, probe_indices]

    logger.info(f"H_full_train shape: {H_full_train.shape} (components × genes)")
    logger.info(f"H_P shape: {H_P.shape} (components × probe_genes)")
    logger.info(f"W_full_train shape: {W_full_train.shape} (cells × components)")

    if H_P.shape[1] != len(probe_indices):
        logger.error(
            f"Dimension mismatch: H_P columns ({H_P.shape[1]}) != probe_indices ({len(probe_indices)})"
        )

    logger.info("Step 2: Iterative solve for train sample factors")
    W_train, A_train_recon = train_predictor.predict(A_P_train, indexer=probe_indices)
    logger.info(f"W_train shape: {W_train.shape} (cells × components)")

    # ================================================================
    # TESTING PHASE
    # ================================================================
    logger.info("--- Testing Phase ---")

    W_full_test = None
    A_test_baseline = None
    mse_test_baseline = None
    expvar_test_baseline = None

    if cached_full_nmf is not None and "testing" in cached_full_nmf:
        logger.info("Using cached full NMF results for testing data")

        testing_cache = cached_full_nmf["testing"]
        if testing_cache["A_test"].shape == A_test.shape:
            if np.allclose(testing_cache["A_test"], A_test):
                H_full_test = testing_cache["H_full_test"].astype(np.float64)
                W_full_test = testing_cache["W_full_test"].astype(np.float64)
                A_test_baseline = testing_cache["A_test_baseline"]
                mse_test_baseline = testing_cache["mse_test_baseline"]
                expvar_test_baseline = testing_cache["expvar_test_baseline"]
                logger.info("Successfully reused cached testing NMF data")
            else:
                logger.warning("Cached testing data doesn't match current split, recomputing...")
                cached_full_nmf["testing"] = None
        else:
            logger.warning("Cached testing data shape doesn't match, recomputing...")
            cached_full_nmf["testing"] = None

    if cached_full_nmf is None or "testing" not in cached_full_nmf:
        logger.info("Computing Full NMF on testing data")
        _test_pred = NmfPredictor(**_nmf_kwargs).fit(A_test)
        W_full_test = _test_pred.ref_embedding
        H_full_test = _test_pred.h_reference
        A_test_baseline = W_full_test @ H_full_test
        mse_test_baseline = calculate_mse(A_test, A_test_baseline)
        expvar_test_baseline = calculate_explained_variance(A_test, A_test_baseline, mode=expvar_mode)

    logger.info("Step 2: Iterative solve for test sample factors")
    W_test, A_test_recon = train_predictor.predict(A_P_test, indexer=probe_indices)
    logger.info(f"W_test shape: {W_test.shape} (cells × components)")

    # ================================================================
    # RECONSTRUCTION AND EVALUATION
    # ================================================================
    logger.info("--- Reconstruction and Evaluation ---")

    mse_train_probe = calculate_mse(A_train, A_train_recon)
    expvar_train_probe = calculate_explained_variance(A_train, A_train_recon, mode=expvar_mode)

    mse_test_probe = calculate_mse(A_test, A_test_recon)
    expvar_test_probe = calculate_explained_variance(A_test, A_test_recon, mode=expvar_mode)

    logger.info(f"Training MSE (baseline): {mse_train_baseline:.6f}")
    logger.info(f"Training MSE (probe): {mse_train_probe:.6f}")
    logger.info(f"Test MSE (baseline): {mse_test_baseline:.6f}")
    logger.info(f"Test MSE (probe): {mse_test_probe:.6f}")
    logger.info(f"Test ExpVar (baseline): {expvar_test_baseline:.4f}")
    logger.info(f"Test ExpVar (probe): {expvar_test_probe:.4f}")

    # Reported quantities are all absolute — the probe reconstruction and the
    # full-gene "oracle" baseline are each scored directly (no probe/baseline ratios);
    # the baseline numbers drive the reference lines in the plots.
    result = {
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
    # convention as run_expvar_aggregation_test.py's gene-subset breakdown.
    modes = list(expvar_modes) if expvar_modes else [expvar_mode]
    subsets = list(gene_subsets) if gene_subsets else ["all_genes"]

    panel_mask = np.zeros(A_test.shape[1], dtype=bool)
    panel_mask[probe_indices] = True
    _subset_masks: dict[str, np.ndarray | None] = {
        "all_genes": None,
        "panel_genes_only": panel_mask,
        "non_panel_genes_only": ~panel_mask,
    }

    # Baseline is only ever computed on the all-genes scale (the baseline NMF fit
    # doesn't depend on which panel is being reconstructed) -- one value per mode.
    expvar_baseline_by_mode = {
        m: calculate_explained_variance(A_test, A_test_baseline, mode=m) for m in modes
    }
    for m, v in expvar_baseline_by_mode.items():
        result[f"expvar_test_baseline_{m}"] = v

    for subset in subsets:
        mask = _subset_masks[subset]
        X_true = A_test if mask is None else A_test[:, mask]
        X_recon = A_test_recon if mask is None else A_test_recon[:, mask]
        result[f"mse_test_probe_{subset}"] = calculate_mse(X_true, X_recon)
        for m in modes:
            result[f"expvar_test_probe_{subset}_{m}"] = calculate_explained_variance(
                X_true, X_recon, mode=m
            )

    if cached_full_nmf is None:
        result["computed_full_nmf"] = {
            "training": {
                "A_train": A_train,
                "W_full_train": W_full_train,
                "H_full_train": H_full_train,
                "A_train_baseline": A_train_baseline,
                "mse_train_baseline": mse_train_baseline,
                "expvar_train_baseline": expvar_train_baseline,
            },
            "testing": {
                "A_test": A_test,
                "W_full_test": W_full_test,
                "H_full_test": H_full_test,
                "A_test_baseline": A_test_baseline,
                "mse_test_baseline": mse_test_baseline,
                "expvar_test_baseline": expvar_test_baseline,
            },
        }

    gc.collect()

    return result


def nmf_reconstruction_by_celltype(
    adata,
    probeset_genes: list[str],
    celltype_column: str = "cluster",  # = _constants.DEFAULT_CELLTYPE_COLUMN
    n_components: int = 5,
    random_state: int = 42,
    cached_full_nmf_by_celltype: dict[str, Any] | None = None,
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    nmf_counts_input: str = "raw",
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
    nmf_objective: str = "auto",
) -> dict[str, Any]:
    """Evaluate NMF representation for each celltype separately.

    Uses train-test split approach for each cell type. The constrained W-solve
    is identical to ``NmfPredictor.predict()`` — H is fixed to the probe-gene
    slice from the training fit and is never updated during the solve.

    Args:
        adata: Annotated data matrix with full transcriptome.
        probeset_genes: List of genes in the probeset to evaluate.
        celltype_column: Column name in adata.obs containing celltype information.
        n_components: Number of NMF components to use.
        random_state: Random state for reproducibility.
        cached_full_nmf_by_celltype: Pre-computed full NMF results by celltype (W/H matrices only).
        per_celltype_splits: Required. Per-celltype train/test index splits as returned by
            :func:`_splits.generate_evaluation_splits`. Must be provided so that the exact same
            cell partitions are used across NMF and Tangram evaluation.
        nmf_counts_input: Count matrix to use as NMF input. ``"raw"`` (default): raw integer
            counts from ``adata.layers['counts']``. ``"lognorm"``: log-normalised counts from
            ``adata.X``.
        expvar_mode: Explained-variance aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``).
            Defaults to ``"global_mean"``.
        expvar_modes: Optional list of explained-variance aggregation modes to
            additionally report per celltype (e.g. ``["global_mean",
            "variance_weighted_sum", "mean"]``), on top of the unsuffixed keys
            (unchanged, still computed from ``expvar_mode`` alone). Defaults to
            ``None``, i.e. ``[expvar_mode]``. See :func:`nmf_reconstruction`
            for the full explanation.
        gene_subsets: Optional list of gene subsets to score against:
            ``"all_genes"`` (default), ``"panel_genes_only"``,
            ``"non_panel_genes_only"``. Scoring-time column mask only, does
            not change what is reconstructed. Defaults to ``None``, i.e.
            ``["all_genes"]``. See :func:`nmf_reconstruction` for the full
            explanation.
        nmf_objective: NMF factorization objective — ``"auto"`` (default) derives the
            solver/beta_loss from ``nmf_counts_input``; ``"frobenius"``/``"kl"`` force
            that objective regardless of input. See ``_nmf_objective.py``.

    Returns:
        Dictionary with MSE and explained variance metrics by celltype.

    Raises:
        ValueError: If ``per_celltype_splits`` is ``None``.
    """
    if per_celltype_splits is None:
        raise ValueError(
            "per_celltype_splits is required. Generate splits first with "
            "generate_evaluation_splits() from _splits.py and pass the "
            "per_celltype_splits dict from the returned tuple."
        )

    logger.info(f"Evaluating NMF representation by celltype with {len(probeset_genes)} genes")

    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Celltype column '{celltype_column}' not found in adata.obs")

    celltypes = adata.obs[celltype_column].unique()
    logger.info(f"Found {len(celltypes)} celltypes: {list(celltypes)}")

    probeset_mask = np.array([gene in probeset_genes for gene in adata.var_names])
    if sum(probeset_mask) == 0:
        raise ValueError(f"None of the probeset genes were found in the dataset")

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info(
        f"Found {len(probeset_genes_found)} out of {len(probeset_genes)} probeset genes in the dataset"
    )

    # ── Gene-subset x expvar-mode grid setup (see nmf_reconstruction for the
    # full explanation) — additive on top of the unsuffixed per-celltype keys. ──
    _modes = list(expvar_modes) if expvar_modes else [expvar_mode]
    _subsets = list(gene_subsets) if gene_subsets else ["all_genes"]

    # solver/beta_loss/init/max_iter derived from the count-input choice (raw -> mu/KL,
    # lognorm -> cd/Frobenius) unless nmf_objective forces one; shared with Selection-module.
    _obj_kwargs_ct = resolve_nmf_objective(nmf_counts_input, nmf_objective)
    logger.info(
        f"Per-celltype NMF objective: nmf_objective={nmf_objective} "
        f"(nmf_counts_input={nmf_counts_input}) -> {_obj_kwargs_ct}"
    )
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

    celltype_results = {}

    if cached_full_nmf_by_celltype is None:
        cached_full_nmf_by_celltype = {}

    def _to_float64(ad):
        if nmf_counts_input == "lognorm":
            if is_anndata_raw(ad):
                raise ValueError(
                    "nmf_counts_input='lognorm': ad.X appears to contain raw integer counts, "
                    "not log-normalized data."
                )
            m = ad.X
        else:  # raw
            if "counts" in ad.layers:
                if not is_anndata_raw_layer(ad, "counts"):
                    raise ValueError(
                        "nmf_counts_input='raw': ad.layers['counts'] does not contain raw integer counts."
                    )
                m = ad.layers["counts"]
            else:
                raise ValueError(
                    "nmf_counts_input='raw': ad.layers['counts'] not found. "
                    "Provide raw counts in layers['counts']."
                )
        return (m.toarray() if scipy.sparse.issparse(m) else np.asarray(m)).astype(np.float64)

    for celltype in tqdm(celltypes, desc="Processing celltypes"):
        logger.info(f"Processing celltype: {celltype}")

        adata_celltype = adata[adata.obs[celltype_column] == celltype]

        logger.info(
            f"Celltype {celltype}: {adata_celltype.shape[0]:,} cells × {adata_celltype.shape[1]:,} genes"
        )

        # Check if this celltype has pre-computed splits
        if celltype not in per_celltype_splits:
            logger.debug(
                "Skipping celltype '%s': not present in per_celltype_splits "
                "(likely filtered out due to insufficient cells).",
                celltype,
            )
            continue

        # Skip celltypes with too few cells for NMF
        min_cells_needed = n_components * 2
        if adata_celltype.shape[0] < min_cells_needed:
            logger.warning(
                f"Skipping celltype {celltype}: only {adata_celltype.shape[0]} cells (need at least {min_cells_needed})"
            )
            celltype_results[celltype] = {
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
                "n_cells": adata_celltype.shape[0],
                "skipped": True,
                "skip_reason": "insufficient_cells",
                **_extra_nan_keys,
            }
            continue

        try:
            # Resolve train/test from pre-computed splits (always required)
            train_pos, test_pos = per_celltype_splits[celltype]
            A_train_ct = _to_float64(adata[train_pos])
            A_test_ct = _to_float64(adata[test_pos])
            A_P_train_ct = A_train_ct[:, probe_indices]
            A_P_test_ct = A_test_ct[:, probe_indices]
            logger.info(
                f"Training: {A_train_ct.shape[0]} cells × {A_train_ct.shape[1]} genes "
                f"(from pre-computed split)"
            )
            logger.info(
                f"Testing:  {A_test_ct.shape[0]} cells × {A_test_ct.shape[1]} genes "
                f"(from pre-computed split)"
            )

            # Check cache for NMF W/H matrices
            cached_ct = cached_full_nmf_by_celltype.get(celltype, {})
            if (
                "training" in cached_ct
                and all(k in cached_ct["training"] for k in ["W_full_train", "H_full_train", "mse_train_baseline", "expvar_train_baseline"])
                and "testing" in cached_ct
                and all(k in cached_ct["testing"] for k in ["W_full_test", "H_full_test", "mse_test_baseline", "expvar_test_baseline"])
            ):
                logger.info(f"Using cached NMF results for celltype {celltype}")
                W_full_train = cached_ct["training"]["W_full_train"].astype(np.float64)
                H_full_train = cached_ct["training"]["H_full_train"].astype(np.float64)
                mse_train_baseline = cached_ct["training"]["mse_train_baseline"]
                expvar_train_baseline = cached_ct["training"]["expvar_train_baseline"]
                W_full_test = cached_ct["testing"]["W_full_test"].astype(np.float64)
                H_full_test = cached_ct["testing"]["H_full_test"].astype(np.float64)
                mse_test_baseline = cached_ct["testing"]["mse_test_baseline"]
                expvar_test_baseline = cached_ct["testing"]["expvar_test_baseline"]
            else:
                logger.info(f"Computing NMF for celltype {celltype}")

                # Objective kwargs resolved above (data-space-driven, see _nmf_objective.py).
                _nmf_kwargs_ct = dict(
                    n_components=n_components, embedding_size=n_components,
                    random_state=random_state,
                    **_obj_kwargs_ct,
                )

                logger.info("Computing Full NMF on training data")
                _train_pred_ct = NmfPredictor(**_nmf_kwargs_ct).fit(A_train_ct)
                W_full_train = _train_pred_ct.ref_embedding
                H_full_train = _train_pred_ct.h_reference
                A_train_baseline = W_full_train @ H_full_train
                mse_train_baseline = calculate_mse(A_train_ct, A_train_baseline)
                expvar_train_baseline = calculate_explained_variance(A_train_ct, A_train_baseline, mode=expvar_mode)

                logger.info("Computing Full NMF on testing data")
                _test_pred_ct = NmfPredictor(**_nmf_kwargs_ct).fit(A_test_ct)
                W_full_test = _test_pred_ct.ref_embedding
                H_full_test = _test_pred_ct.h_reference
                A_test_baseline = W_full_test @ H_full_test
                mse_test_baseline = calculate_mse(A_test_ct, A_test_baseline)
                expvar_test_baseline = calculate_explained_variance(A_test_ct, A_test_baseline, mode=expvar_mode)

                # Cache W/H matrices only (no raw data arrays)
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

            # Reconstructed baseline matrices, needed below for the extra expvar-mode
            # grid -- cheap (W @ H matmul reusing already-fit W/H), recomputed
            # unconditionally so this also works on the cache-hit branch above (which
            # only caches W/H, not the reconstructed matrix).
            A_train_baseline = W_full_train @ H_full_train
            A_test_baseline = W_full_test @ H_full_test

            # Apply NMF representation approach
            logger.info(f"Running NMF representation for celltype {celltype}")

            # H_full_train is set above (either from cache or fresh fit)
            _train_pred_ct = NmfPredictor(
                n_components=n_components, embedding_size=n_components,
                random_state=random_state,
                **_obj_kwargs_ct,
                h_reference=H_full_train, ref_embedding=W_full_train,
            )

            W_train, A_train_recon = _train_pred_ct.predict(A_P_train_ct, indexer=probe_indices)
            W_test, A_test_recon = _train_pred_ct.predict(A_P_test_ct, indexer=probe_indices)

            # ── Gene-subset x expvar-mode grid (additive; see nmf_reconstruction) ──
            _extra_metrics: dict[str, float] = {}
            _baseline_train_by_mode = {
                m: calculate_explained_variance(A_train_ct, A_train_baseline, mode=m) for m in _modes
            }
            _baseline_test_by_mode = {
                m: calculate_explained_variance(A_test_ct, A_test_baseline, mode=m) for m in _modes
            }
            for _m, _v in _baseline_train_by_mode.items():
                _extra_metrics[f"expvar_train_baseline_{_m}"] = _v
            for _m, _v in _baseline_test_by_mode.items():
                _extra_metrics[f"expvar_test_baseline_{_m}"] = _v
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

            mse_train_probe = calculate_mse(A_train_ct, A_train_recon)
            expvar_train_probe = calculate_explained_variance(A_train_ct, A_train_recon, mode=expvar_mode)

            mse_test_probe = calculate_mse(A_test_ct, A_test_recon)
            expvar_test_probe = calculate_explained_variance(A_test_ct, A_test_recon, mode=expvar_mode)

            logger.info(
                f"Celltype {celltype} - Test MSE (baseline): {mse_test_baseline:.6f}, (probe): {mse_test_probe:.6f}"
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
                "n_cells": adata_celltype.shape[0],
                "skipped": False,
                **_extra_metrics,
            }

        except Exception as e:
            logger.error(f"Failed NMF evaluation for celltype {celltype}: {e}")
            celltype_results[celltype] = {
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
                "n_cells": adata_celltype.shape[0],
                "skipped": True,
                "skip_reason": "evaluation_failed",
                **_extra_nan_keys,
            }

    # Calculate summary statistics across celltypes
    valid_results = {
        ct: res for ct, res in celltype_results.items() if not res.get("skipped", False)
    }

    if valid_results:
        total_cells = sum(res["n_cells"] for res in valid_results.values())

        weighted_mse_test_probe = calculate_weighted_mse(valid_results)
        weighted_expvar_test_probe = calculate_weighted_explained_variance(valid_results)
        weighted_mse_test_baseline = calculate_weighted_mse_baseline(valid_results)
        weighted_expvar_test_baseline = calculate_weighted_explained_variance_baseline(
            valid_results
        )

        macro_mse_test_probe = calculate_macro_mse(valid_results)
        macro_expvar_test_probe = calculate_macro_explained_variance(valid_results)
        macro_mse_test_baseline = np.mean([res["mse_test_baseline"] for res in valid_results.values()])
        macro_expvar_test_baseline = np.mean(
            [res["expvar_test_baseline"] for res in valid_results.values()]
        )

        # ── Gene-subset x expvar-mode grid summary (additive) ──────────────
        _extra_summary: dict[str, float] = {}
        for _m in _modes:
            _bkey = f"expvar_test_baseline_{_m}"
            _extra_summary[f"weighted_expvar_test_baseline_{_m}"] = (
                calculate_weighted_explained_variance_baseline(valid_results, metric_key=_bkey)
            )
            _extra_summary[f"macro_expvar_test_baseline_{_m}"] = np.mean(
                [res[_bkey] for res in valid_results.values()]
            )
        for _subset in _subsets:
            _mse_key = f"mse_test_probe_{_subset}"
            _extra_summary[f"weighted_mse_test_probe_{_subset}"] = calculate_weighted_mse(
                valid_results, metric_key=_mse_key
            )
            _extra_summary[f"macro_mse_test_probe_{_subset}"] = calculate_macro_mse(
                valid_results, metric_key=_mse_key
            )
            for _m in _modes:
                _ekey = f"expvar_test_probe_{_subset}_{_m}"
                _extra_summary[f"weighted_expvar_test_probe_{_subset}_{_m}"] = (
                    calculate_weighted_explained_variance(valid_results, metric_key=_ekey)
                )
                _extra_summary[f"macro_expvar_test_probe_{_subset}_{_m}"] = (
                    calculate_macro_explained_variance(valid_results, metric_key=_ekey)
                )

        summary_results = {
            "celltype_results": celltype_results,
            "summary": {
                "weighted_mse_test_probe": weighted_mse_test_probe,
                "weighted_mse_test_baseline": weighted_mse_test_baseline,
                "weighted_expvar_test_probe": weighted_expvar_test_probe,
                "weighted_expvar_test_baseline": weighted_expvar_test_baseline,
                "macro_mse_test_probe": macro_mse_test_probe,
                "macro_mse_test_baseline": macro_mse_test_baseline,
                "macro_expvar_test_probe": macro_expvar_test_probe,
                "macro_expvar_test_baseline": macro_expvar_test_baseline,
                "total_cells": total_cells,
                "n_celltypes_processed": len(valid_results),
                "n_celltypes_skipped": len(celltype_results) - len(valid_results),
                **_extra_summary,
            },
        }
    else:
        _extra_summary_nan: dict[str, float] = {}
        for _m in _modes:
            _extra_summary_nan[f"weighted_expvar_test_baseline_{_m}"] = np.nan
            _extra_summary_nan[f"macro_expvar_test_baseline_{_m}"] = np.nan
        for _subset in _subsets:
            _extra_summary_nan[f"weighted_mse_test_probe_{_subset}"] = np.nan
            _extra_summary_nan[f"macro_mse_test_probe_{_subset}"] = np.nan
            for _m in _modes:
                _extra_summary_nan[f"weighted_expvar_test_probe_{_subset}_{_m}"] = np.nan
                _extra_summary_nan[f"macro_expvar_test_probe_{_subset}_{_m}"] = np.nan

        summary_results = {
            "celltype_results": celltype_results,
            "summary": {
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
                **_extra_summary_nan,
            },
        }

    return summary_results
