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
import pandas as pd
import scipy.sparse
from sklearn.decomposition import non_negative_factorization  # still used by cNMF path
from nico2_lib.predictors._nmf._nmf_pred import NmfPredictor
from tqdm import tqdm

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
    "nmf_reconstruction",
    "nmf_reconstruction_by_celltype",
]


def nmf_reconstruction(
    adata,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    n_components: int = 5,
    max_iter: int = 1000,
    random_state: int = 42,
    cached_full_nmf: dict[str, Any] | None = None,
    # cNMF options
    use_consensus_nmf: bool = False,
    cnmf_k_values: list[int] | None = None,
    cnmf_n_iter: int = 100,
    cnmf_max_iter: int = 1000,
    cnmf_k_selection_method: str = "elbow",
    cnmf_density_threshold: float = 0.5,
    cnmf_local_neighborhood_size: float = 0.30,
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

    H is **always fixed** to the probe-gene slice from the training fit. The solve is
    constrained, not free — this is the key distinction from the pre-refactor bug.

    Following scikit-learn convention:
        - A (samples/cells × features/genes): Data matrix
        - W (samples/cells × components): Sample factor matrix
        - H (components × features/genes): Feature factor matrix

    Args:
        adata: Annotated data matrix with full transcriptome.
        probeset_genes: List of genes in the probeset to evaluate.
        A_train: Pre-split training data (cells × genes).
        A_test: Pre-split testing data (cells × genes).
        n_components: Number of NMF components to use (ignored if use_consensus_nmf=True).
        max_iter: Maximum number of iterations for standard NMF optimization.
        random_state: Random state for reproducibility.
        cached_full_nmf: Pre-computed full NMF results to avoid recomputation.
        use_consensus_nmf: If True, use consensus NMF instead of standard NMF (more stable but slower).
        cnmf_k_values: List of K values to test for cNMF. If None, uses range around n_components.
        cnmf_n_iter: Number of independent NMF runs for consensus (default: 100).
        cnmf_max_iter: Max optimization iterations per cNMF run (default: 1000, matches standard NMF).
        cnmf_k_selection_method: Method for automatic K selection ("silhouette" or "elbow").
        cnmf_density_threshold: Density threshold for filtering outlier spectra (default: 0.5).
        cnmf_local_neighborhood_size: Local neighborhood size for density calculation (default: 0.30).

    Returns:
        Dictionary with MSE and explained variance metrics, plus cNMF stability metrics if enabled.
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
    # CONSENSUS NMF PATH (if requested)
    # ================================================================
    if use_consensus_nmf:
        logger.info("=== Using Consensus NMF for evaluation ===")
        try:
            from pathlib import Path

            utility_module_path = Path(__file__).parent.parent / 'Utility-module'

            if str(utility_module_path) not in sys.path:
                sys.path.insert(0, str(utility_module_path))

            import _constants
            import _consensus_nmf

            run_consensus_nmf_global = _consensus_nmf.run_consensus_nmf_global
            select_optimal_k = _consensus_nmf.select_optimal_k

            import anndata as _ad

            if cnmf_k_values is None:
                k_min = max(2, n_components - 3)
                k_max = n_components + 5
                cnmf_k_values = list(range(k_min, k_max + 1, 2))
            logger.info(f"Testing K values: {cnmf_k_values}")

            train_adata = _ad.AnnData(X=A_train, var=pd.DataFrame(index=adata.var_names))
            train_adata.layers['counts'] = A_train

            logger.info(f"Running consensus NMF on training data ({cnmf_n_iter} iterations per K)")
            cnmf_result = run_consensus_nmf_global(
                train_adata,
                k_values=cnmf_k_values,
                n_iter=cnmf_n_iter,
                random_state=random_state,
                density_threshold=cnmf_density_threshold,
                max_iter=cnmf_max_iter,
            )

            if not cnmf_result:
                logger.error("Consensus NMF failed! Returning None.")
                return None

            cnmf_obj = cnmf_result["cnmf_object"]
            consensus_by_k = cnmf_result["consensus_by_k"]

            optimal_k = select_optimal_k(cnmf_obj, method=cnmf_k_selection_method)
            logger.info(f"✓ cNMF selected optimal K={optimal_k} (method: {cnmf_k_selection_method})")

            H_consensus = consensus_by_k[optimal_k]["consensus_H"]
            W_consensus_train = consensus_by_k[optimal_k]["consensus_W"]

            filtered_gene_names = consensus_by_k[optimal_k]["gene_names"]
            n_genes_filtered = len(filtered_gene_names)
            n_genes_original = adata.n_vars
            n_genes_removed = n_genes_original - n_genes_filtered

            logger.info(f"Gene filtering: {n_genes_original} original → {n_genes_filtered} filtered ({n_genes_removed} removed)")

            filtered_gene_indices = np.array([
                np.where(adata.var_names == gene)[0][0]
                for gene in filtered_gene_names
            ])

            A_train_filtered = A_train[:, filtered_gene_indices]
            A_test_filtered = A_test[:, filtered_gene_indices]

            logger.info(f"A_train shape: {A_train.shape} → A_train_filtered: {A_train_filtered.shape}")
            logger.info(f"A_test shape: {A_test.shape} → A_test_filtered: {A_test_filtered.shape}")

            probe_gene_names = adata.var_names[probe_indices]
            probe_indices_filtered = []
            missing_probes = []

            for probe_gene in probe_gene_names:
                matches = np.where(filtered_gene_names == probe_gene)[0]
                if len(matches) > 0:
                    probe_indices_filtered.append(matches[0])
                else:
                    missing_probes.append(probe_gene)

            probe_indices_filtered = np.array(probe_indices_filtered)

            if missing_probes:
                logger.warning(
                    f"{len(missing_probes)} probe genes were filtered out (zero variance): "
                    f"{missing_probes[:5]}{'...' if len(missing_probes) > 5 else ''}"
                )

            logger.info(
                f"Probe indices: {len(probe_indices)} original → {len(probe_indices_filtered)} filtered "
                f"({len(missing_probes)} missing)"
            )

            A_P_train_filtered = A_train_filtered[:, probe_indices_filtered]
            A_P_test_filtered = A_test_filtered[:, probe_indices_filtered]

            H_P_consensus = H_consensus[:, probe_indices_filtered]

            logger.info(f"Consensus H shape: {H_consensus.shape} (components × genes_filtered)")
            logger.info(f"Consensus H_P shape: {H_P_consensus.shape} (components × probe_genes_filtered)")

            A_train_baseline = W_consensus_train @ H_consensus
            mse_train_baseline = calculate_mse(A_train_filtered, A_train_baseline)
            expvar_train_baseline = calculate_explained_variance(A_train_filtered, A_train_baseline)

            W_train_probe, _, _ = non_negative_factorization(
                A_P_train_filtered, H=H_P_consensus, n_components=optimal_k, init="custom",
                update_H=False, max_iter=cnmf_max_iter, random_state=random_state,
            )
            A_train_probe_recon = W_train_probe @ H_consensus
            mse_train_probe = calculate_mse(A_train_filtered, A_train_probe_recon)
            expvar_train_probe = calculate_explained_variance(A_train_filtered, A_train_probe_recon)

            logger.info(f"Training baseline: MSE={mse_train_baseline:.4f}, ExpVar={expvar_train_baseline:.4f}")
            logger.info(f"Training probe: MSE={mse_train_probe:.4f}, ExpVar={expvar_train_probe:.4f}")

            W_test_probe, _, _ = non_negative_factorization(
                A_P_test_filtered, H=H_P_consensus, n_components=optimal_k, init="custom",
                update_H=False, max_iter=cnmf_max_iter, random_state=random_state,
            )

            A_test_probe_recon = W_test_probe @ H_consensus
            mse_test_probe = calculate_mse(A_test_filtered, A_test_probe_recon)
            expvar_test_probe = calculate_explained_variance(A_test_filtered, A_test_probe_recon)

            W_test_baseline, H_test_baseline, _ = non_negative_factorization(
                A_test_filtered, n_components=optimal_k, max_iter=cnmf_max_iter, random_state=random_state
            )
            A_test_baseline = W_test_baseline @ H_test_baseline
            mse_test_baseline = calculate_mse(A_test_filtered, A_test_baseline)
            expvar_test_baseline = calculate_explained_variance(A_test_filtered, A_test_baseline)

            logger.info(f"Testing baseline: MSE={mse_test_baseline:.4f}, ExpVar={expvar_test_baseline:.4f}")
            logger.info(f"Testing probe: MSE={mse_test_probe:.4f}, ExpVar={expvar_test_probe:.4f}")

            stability_metrics = {}
            if hasattr(cnmf_obj, 'stability_metrics') and optimal_k in cnmf_obj.stability_metrics:
                stability_metrics = cnmf_obj.stability_metrics[optimal_k]

            return {
                "mse_train_baseline": mse_train_baseline,
                "expvar_train_baseline": expvar_train_baseline,
                "mse_train_probe": mse_train_probe,
                "expvar_train_probe": expvar_train_probe,
                "mse_test_baseline": mse_test_baseline,
                "expvar_test_baseline": expvar_test_baseline,
                "mse_test_probe": mse_test_probe,
                "expvar_test_probe": expvar_test_probe,
                "probeset_size": len(probe_indices),
                "n_components": optimal_k,
                "n_components_requested": n_components,
                "method": "consensus_nmf",
                "cnmf_k_values_tested": cnmf_k_values,
                "cnmf_optimal_k": optimal_k,
                "cnmf_k_selection_method": cnmf_k_selection_method,
                "cnmf_n_iter": cnmf_n_iter,
                "cnmf_stability": stability_metrics,
            }

        except Exception as exc:
            logger.error(f"Consensus NMF evaluation failed: {exc}", exc_info=True)
            logger.warning("Falling back to standard NMF evaluation")
            use_consensus_nmf = False

    # ================================================================
    # STANDARD NMF PATH
    # ================================================================
    if not use_consensus_nmf:
        logger.info("=== Using Standard NMF for evaluation ===")

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

    _nmf_kwargs = dict(
        embedding_size=n_components, seed=random_state, max_iter=max_iter,
        beta_loss="frobenius", init=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
    )

    if cached_full_nmf is None or "training" not in cached_full_nmf:
        logger.info("Computing Full NMF on training data")
        train_predictor = NmfPredictor(**_nmf_kwargs).fit(A_train)
        W_full_train = train_predictor.ref_embedding
        H_full_train = train_predictor.h_reference
        A_train_baseline = W_full_train @ H_full_train
        mse_train_baseline = calculate_mse(A_train, A_train_baseline)
        expvar_train_baseline = calculate_explained_variance(A_train, A_train_baseline)
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
        expvar_test_baseline = calculate_explained_variance(A_test, A_test_baseline)

    logger.info("Step 2: Iterative solve for test sample factors")
    W_test, A_test_recon = train_predictor.predict(A_P_test, indexer=probe_indices)
    logger.info(f"W_test shape: {W_test.shape} (cells × components)")

    # ================================================================
    # RECONSTRUCTION AND EVALUATION
    # ================================================================
    logger.info("--- Reconstruction and Evaluation ---")

    mse_train_probe = calculate_mse(A_train, A_train_recon)
    expvar_train_probe = calculate_explained_variance(A_train, A_train_recon)
    mse_ratio_train = (
        mse_train_probe / mse_train_baseline if mse_train_baseline > 0 else float("inf")
    )
    expvar_ratio_train = (
        expvar_train_probe / expvar_train_baseline if expvar_train_baseline > 0 else 0
    )

    mse_test_probe = calculate_mse(A_test, A_test_recon)
    expvar_test_probe = calculate_explained_variance(A_test, A_test_recon)
    mse_ratio = mse_test_probe / mse_test_baseline if mse_test_baseline > 0 else float("inf")
    expvar_ratio = expvar_test_probe / expvar_test_baseline if expvar_test_baseline > 0 else 0

    logger.info(f"Training MSE (baseline): {mse_train_baseline:.6f}")
    logger.info(f"Training MSE (probe): {mse_train_probe:.6f}")
    logger.info(f"Test MSE (baseline): {mse_test_baseline:.6f}")
    logger.info(f"Test MSE (probe): {mse_test_probe:.6f}")
    logger.info(f"MSE ratio train (probe/baseline): {mse_ratio_train:.3f}")
    logger.info(f"MSE ratio test (probe/baseline): {mse_ratio:.3f}")
    logger.info(f"ExpVar ratio train (probe/baseline): {expvar_ratio_train:.3f}")
    logger.info(f"ExpVar ratio test (probe/baseline): {expvar_ratio:.3f}")

    result = {
        "mse_train_baseline": mse_train_baseline,
        "mse_test_baseline": mse_test_baseline,
        "mse_test_probe": mse_test_probe,
        "expvar_train_baseline": expvar_train_baseline,
        "expvar_test_baseline": expvar_test_baseline,
        "expvar_test_probe": expvar_test_probe,
        "mse_ratio": mse_ratio,
        "expvar_ratio": expvar_ratio,
        "probeset_size": len(probeset_genes),
        "probeset_genes_found": len(probeset_genes_found),
    }

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
    celltype_column: str = "celltypes_v2",
    n_components: int = 5,
    max_iter: int = 1000,
    random_state: int = 42,
    cached_full_nmf_by_celltype: dict[str, Any] | None = None,
    # cNMF options
    use_consensus_nmf: bool = False,
    cnmf_k_values: list[int] | None = None,
    cnmf_n_iter: int = 100,
    cnmf_max_iter: int = 1000,
    cnmf_k_selection_method: str = "elbow",
    cnmf_density_threshold: float = 0.5,
    cnmf_local_neighborhood_size: float = 0.30,
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    nmf_counts_input: str = "raw",
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
        max_iter: Maximum number of iterations for NMF.
        random_state: Random state for reproducibility.
        cached_full_nmf_by_celltype: Pre-computed full NMF results by celltype (W/H matrices only).
        per_celltype_splits: Required. Per-celltype train/test index splits as returned by
            :func:`_splits.generate_evaluation_splits`. Must be provided so that the exact same
            cell partitions are used across NMF and Tangram evaluation.
        nmf_counts_input: Count matrix to use as NMF input. ``"raw"`` (default): raw integer
            counts from ``adata.layers['counts']``. ``"lognorm"``: log-normalised counts from
            ``adata.X``.

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
                "mse_ratio_train": np.nan,
                "mse_ratio_test": np.nan,
                "expvar_ratio_train": np.nan,
                "expvar_ratio_test": np.nan,
                "generalization_gap": np.nan,
                "probeset_size": len(probeset_genes),
                "probeset_genes_found": len(probeset_genes_found),
                "n_cells": adata_celltype.shape[0],
                "skipped": True,
                "skip_reason": "insufficient_cells",
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

                _nmf_kwargs_ct = dict(
                    embedding_size=n_components, seed=random_state, max_iter=max_iter,
                    beta_loss="frobenius", init=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
                )

                logger.info("Computing Full NMF on training data")
                _train_pred_ct = NmfPredictor(**_nmf_kwargs_ct).fit(A_train_ct)
                W_full_train = _train_pred_ct.ref_embedding
                H_full_train = _train_pred_ct.h_reference
                A_train_baseline = W_full_train @ H_full_train
                mse_train_baseline = calculate_mse(A_train_ct, A_train_baseline)
                expvar_train_baseline = calculate_explained_variance(A_train_ct, A_train_baseline)

                logger.info("Computing Full NMF on testing data")
                _test_pred_ct = NmfPredictor(**_nmf_kwargs_ct).fit(A_test_ct)
                W_full_test = _test_pred_ct.ref_embedding
                H_full_test = _test_pred_ct.h_reference
                A_test_baseline = W_full_test @ H_full_test
                mse_test_baseline = calculate_mse(A_test_ct, A_test_baseline)
                expvar_test_baseline = calculate_explained_variance(A_test_ct, A_test_baseline)

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

            # Apply NMF representation approach
            logger.info(f"Running NMF representation for celltype {celltype}")

            # H_full_train is set above (either from cache or fresh fit)
            _train_pred_ct = NmfPredictor(
                embedding_size=n_components, seed=random_state, max_iter=max_iter,
                beta_loss="frobenius", init=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
                h_reference=H_full_train, ref_embedding=W_full_train,
            )

            W_train, A_train_recon = _train_pred_ct.predict(A_P_train_ct, indexer=probe_indices)
            W_test, A_test_recon = _train_pred_ct.predict(A_P_test_ct, indexer=probe_indices)

            mse_train_probe = calculate_mse(A_train_ct, A_train_recon)
            expvar_train_probe = calculate_explained_variance(A_train_ct, A_train_recon)

            mse_test_probe = calculate_mse(A_test_ct, A_test_recon)
            expvar_test_probe = calculate_explained_variance(A_test_ct, A_test_recon)

            mse_ratio = (
                mse_test_probe / mse_test_baseline if mse_test_baseline > 0 else float("inf")
            )
            expvar_ratio = (
                expvar_test_probe / expvar_test_baseline if expvar_test_baseline > 0 else 0
            )

            logger.info(
                f"Celltype {celltype} - Test MSE (baseline): {mse_test_baseline:.6f}, (probe): {mse_test_probe:.6f}, ratio: {mse_ratio:.3f}"
            )

            mse_ratio_train = (
                mse_train_probe / mse_train_baseline if mse_train_baseline > 0 else float("inf")
            )
            mse_ratio_test = mse_ratio
            expvar_ratio_train = (
                expvar_train_probe / expvar_train_baseline if expvar_train_baseline > 0 else 0
            )
            expvar_ratio_test = expvar_ratio
            generalization_gap = mse_ratio_test - mse_ratio_train

            celltype_results[celltype] = {
                "mse_train_baseline": mse_train_baseline,
                "mse_test_baseline": mse_test_baseline,
                "mse_train_probe": mse_train_probe,
                "mse_test_probe": mse_test_probe,
                "expvar_train_baseline": expvar_train_baseline,
                "expvar_test_baseline": expvar_test_baseline,
                "expvar_train_probe": expvar_train_probe,
                "expvar_test_probe": expvar_test_probe,
                "mse_ratio_train": mse_ratio_train,
                "mse_ratio_test": mse_ratio_test,
                "expvar_ratio_train": expvar_ratio_train,
                "expvar_ratio_test": expvar_ratio_test,
                "generalization_gap": generalization_gap,
                "probeset_size": len(probeset_genes),
                "probeset_genes_found": len(probeset_genes_found),
                "n_cells": adata_celltype.shape[0],
                "skipped": False,
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
                "mse_ratio_train": np.nan,
                "mse_ratio_test": np.nan,
                "expvar_ratio_train": np.nan,
                "expvar_ratio_test": np.nan,
                "generalization_gap": np.nan,
                "probeset_size": len(probeset_genes),
                "probeset_genes_found": len(probeset_genes_found),
                "n_cells": adata_celltype.shape[0],
                "skipped": True,
                "skip_reason": "evaluation_failed",
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
            },
        }
    else:
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
            },
        }

    return summary_results
