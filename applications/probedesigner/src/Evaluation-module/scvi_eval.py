"""scVI-based representation evaluation for probeset evaluation (global reconstruction only).

Expects RAW counts (``adata.layers['counts']``), unlike ``pca.py``/``ica.py`` — scVI is a
negative-binomial count model (``nico2_lib``'s ``SCVI.setup_anndata`` registers ``.X`` via
``is_count_data=True``), not a pre-normalised-expression method.

No per-celltype variant exists here: ``nico2_lib``'s ``ScviPredictor`` retrains a full VAE
from scratch on **every single** ``.predict()`` call — ``.fit()`` is nearly a no-op (it just
wraps/stores the reference data and, only if ``n_factors`` is left unset, runs a hidden
KMeans/kneedle search), and ``feature_embedding`` always returns ``None`` (nothing is ever
cached for reuse). A per-celltype loop would multiply an already-expensive full-VAE-training
cost by the number of cell types; global-only + 5-fold CV is the deliberate cost-bounding
choice made for this evaluation method.

Cost model: per fold, 2 SCVI trainings for the oracle baseline (train-oracle + test-oracle,
computed once and cached across panels within that fold, mirroring ``pca.py``'s
``cached_full_pca`` pattern) plus **2 more SCVI trainings per panel** (train-probe +
test-probe, not cacheable across panels since the probe ``indexer`` differs per panel). For
``N`` panels across ``K`` folds that is ``K * (2 + 2*N)`` independent VAE trainings at
``max_epochs`` each — by far the most expensive of the five reconstruction methods in this
pipeline. Size ``--max_epochs``/``--n_splits`` accordingly.

KNOWN LIMITATION vs. ``pca.py``/``ica.py``/``nmf.py``: because ``ScviPredictor.predict()``
retrains independently on **every** call — even repeated calls with the same ``indexer`` on
the same fitted predictor — the train-probe and test-probe reconstructions below come from
two statistically independent VAE fits (same architecture/reference data, but no shared
"fixed basis" the way PCA/ICA/NMF/Ridge give). There is also no seed/determinism field
anywhere in ``ScviPredictor``, so results are not reproducible run-to-run. This is a property
of the current ``nico2_lib`` API, not a bug introduced here — do not rely on run-to-run
determinism for scVI results.

Functions:
    scvi_reconstruction: Global probe evaluation via scVI.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from nico2_lib.predictors._scvi._scvi_pred import ScviPredictor
from metrics import calculate_explained_variance, calculate_mse

logger = logging.getLogger(__name__)

__all__ = [
    "scvi_reconstruction",
    "_compute_full_scvi_baseline",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_full_scvi_baseline(
    A_train: np.ndarray,
    A_test: np.ndarray,
    n_components: int,
    max_epochs: int,
) -> dict[str, Any]:
    """Fit scVI on raw-count training data and compute baselines.

    2 SCVI trainings (train-oracle, test-oracle) — see module docstring for the
    cost model and the "no shared fixed basis" limitation this implies relative
    to ``pca.py``/``ica.py``/``nmf.py``'s oracle-baseline convention.

    Args:
        A_train: Raw-count training data (cells × genes), float32.
        A_test: Raw-count test data (cells × genes), float32.
        n_components: scVI latent dimension (``ScviPredictor.n_factors``).
        max_epochs: Training epochs per VAE fit.

    Returns:
        Dict with ``"training"`` and ``"testing"`` sub-dicts.
    """
    train_predictor = ScviPredictor(n_factors=n_components, max_epochs=max_epochs).fit(A_train)
    _, A_train_baseline = train_predictor.predict(A_train, indexer=np.arange(A_train.shape[1]))
    mse_train_bl = calculate_mse(A_train, A_train_baseline)
    expvar_train_bl = calculate_explained_variance(A_train, A_train_baseline)

    # Independent oracle fit on test data (not a transform of the train-fitted model) —
    # matches pca.py/ica.py/nmf.py's test-baseline convention.
    test_predictor = ScviPredictor(n_factors=n_components, max_epochs=max_epochs).fit(A_test)
    _, A_test_baseline = test_predictor.predict(A_test, indexer=np.arange(A_test.shape[1]))
    mse_test_bl = calculate_mse(A_test, A_test_baseline)
    expvar_test_bl = calculate_explained_variance(A_test, A_test_baseline)

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
        # Store raw arrays only for cache-validation in scvi_reconstruction
        "_A_train": A_train,
        "_A_test": A_test,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scvi_reconstruction(
    adata: Any,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    n_components: int = 10,
    max_epochs: int = 200,
    cached_full_scvi: dict[str, Any] | None = None,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> dict[str, Any] | None:
    """Evaluate probe panel quality via scVI full-transcriptome reconstruction.

    Global evaluation only — see module docstring for why there is no
    ``scvi_reconstruction_by_celltype`` counterpart.

    IMPORTANT: unlike ``pca_reconstruction``/``ica_reconstruction``, the ``predictor``
    resolved from the baseline (cached or freshly computed) is **not** a reusable fixed
    basis for the probe-based reconstruction below — ``ScviPredictor.predict()`` retrains
    a fresh VAE on every call regardless of whether the predictor or its ``indexer`` were
    seen before (see module docstring). The two ``.predict(..., indexer=probe_indices)``
    calls below each trigger a full training run.

    Args:
        adata: AnnData with ``var_names`` covering all genes.
        probeset_genes: Probe gene names to evaluate.
        A_train: Raw-count training data (cells × genes).
        A_test: Raw-count test data (cells × genes).
        n_components: scVI latent dimension (``ScviPredictor.n_factors``). Left as an
            explicit int (not ``None``) by design — ``None`` triggers ``ScviPredictor``'s
            own hidden KMeans/kneedle elbow search, an extra cost this evaluation
            deliberately avoids defaulting into.
        max_epochs: Training epochs per VAE fit (each ``.predict()`` call retrains from
            scratch — see module docstring for the cost model).
        cached_full_scvi: Pre-computed baseline dict returned by a previous call's
            ``"computed_full_scvi"`` key.
        expvar_mode: Explained-variance aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``).
            Defaults to ``"global_mean"``.
        expvar_modes: Optional list of explained-variance aggregation modes to
            additionally report. Defaults to ``None``, i.e. ``[expvar_mode]``.
        gene_subsets: Optional list of gene subsets to score the probe reconstruction
            against: ``"all_genes"`` (default), ``"panel_genes_only"``,
            ``"non_panel_genes_only"``. Pure post-hoc column masking on the
            already-computed reconstruction — no extra SCVI runs. Defaults to ``None``,
            i.e. ``["all_genes"]``.

    Returns:
        Metrics dict, or ``None`` if no probe genes are found in the dataset.
    """
    logger.info("Evaluating scVI representation with %d genes", len(probeset_genes))

    probeset_mask = np.array([g in probeset_genes for g in adata.var_names])
    if probeset_mask.sum() == 0:
        logger.error("No probeset genes found in the dataset")
        return None

    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logger.info(
        "Found %d / %d probeset genes", len(probeset_genes_found), len(probeset_genes)
    )

    A_train = np.asarray(A_train, dtype=np.float32)
    A_test = np.asarray(A_test, dtype=np.float32)

    A_P_train = A_train[:, probe_indices]
    A_P_test = A_test[:, probe_indices]

    # ── Full scVI baseline (cached or computed) — 2 SCVI trainings if not cached ──
    baseline_cached = False
    if cached_full_scvi is not None:
        tr = cached_full_scvi.get("training", {})
        cached_A = cached_full_scvi.get("_A_train")
        if (
            cached_A is not None
            and cached_A.shape == A_train.shape
            and np.allclose(cached_A, A_train)
        ):
            predictor = tr["predictor"]
            mse_train_baseline = tr["mse_train_baseline"]
            expvar_train_baseline = tr["expvar_train_baseline"]
            te = cached_full_scvi["testing"]
            mse_test_baseline = te["mse_test_baseline"]
            expvar_test_baseline = te["expvar_test_baseline"]
            A_test_baseline = te["A_test_baseline"]
            logger.info("Reusing cached scVI baseline")
            baseline_cached = True
        else:
            logger.warning("Cached scVI baseline does not match current data — recomputing")

    if not baseline_cached:
        logger.info(
            "Computing full scVI baseline (n_components=%d, max_epochs=%d)",
            n_components, max_epochs,
        )
        baseline = _compute_full_scvi_baseline(A_train, A_test, n_components, max_epochs)
        predictor = baseline["training"]["predictor"]
        mse_train_baseline = baseline["training"]["mse_train_baseline"]
        expvar_train_baseline = baseline["training"]["expvar_train_baseline"]
        mse_test_baseline = baseline["testing"]["mse_test_baseline"]
        expvar_test_baseline = baseline["testing"]["expvar_test_baseline"]
        A_test_baseline = baseline["testing"]["A_test_baseline"]

    # ── Probe-based reconstruction — 2 more SCVI trainings (train-probe, test-probe);
    # NOT the same fixed basis as `predictor` above — see module/function docstring. ──
    _, A_train_recon = predictor.predict(A_P_train, indexer=probe_indices)
    _, A_test_recon = predictor.predict(A_P_test, indexer=probe_indices)

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
    # convention as pca.py::pca_reconstruction / ica.py::ica_reconstruction.
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
        result["computed_full_scvi"] = baseline

    gc.collect()
    return result
