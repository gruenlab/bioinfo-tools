"""Clustering quality, neighborhood preservation, and cell-type classification metrics.

This module contains streamlined functions for evaluating probeset performance
across multiple clustering and classification metrics. It computes neighborhood
preservation (k-NN overlap), clustering quality (ARI/NMI), and celltype
identification accuracy (decision tree classification).

Functions:
    compute_neighborhood_preservation: Compares k-NN overlap between full and reduced gene sets.
    evaluate_neighborhood_preservation: Evaluates preservation across multiple probesets.
    compute_clustering_similarity: Computes ARI and NMI between clustering results.
    evaluate_clustering_quality: Evaluates clustering across probesets.
    split_train_test_sets: Splits data into train and test sets.
    uniform_samples: Creates uniform samples across cell types.
    evaluate_celltype_identification: Trains classifiers and evaluates accuracy.
"""

from __future__ import annotations

import gc
import logging
import sys
from typing import Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import constants from the local _constants.py file (same directory as this file)
# Using importlib to avoid conflicts with _constants.py in other modules
import importlib.util
_current_dir = Path(__file__).parent.absolute()
_constants_path = _current_dir / "_constants.py"
_spec = importlib.util.spec_from_file_location("_eval_constants", _constants_path)
_eval_constants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_constants)

DEFAULT_KNN_K_VALUES = _eval_constants.DEFAULT_KNN_K_VALUES
DEFAULT_SUBSAMPLE_SIZE = _eval_constants.DEFAULT_SUBSAMPLE_SIZE
DEFAULT_SPLIT_RATIO = _eval_constants.DEFAULT_SPLIT_RATIO
DEFAULT_CELLTYPE_CLF_MAX_DEPTH = _eval_constants.DEFAULT_CELLTYPE_CLF_MAX_DEPTH
DEFAULT_CELLTYPE_COLUMN = _eval_constants.DEFAULT_CELLTYPE_COLUMN

logger = logging.getLogger(__name__)

__all__ = [
    "compute_neighborhood_preservation",
    "evaluate_neighborhood_preservation",
    "compute_clustering_similarity",
    "compute_celltype_jaccard_vs_leiden",
    "evaluate_clustering_quality",
    "split_train_test_sets",
    "uniform_samples",
    "evaluate_celltype_identification",
]

# Ensure the Utility-module directory is on sys.path so its files are importable
# as flat modules — consistent with how run_evaluation.py sets up the path.
_utility_dir = _current_dir.parent / "Utility-module"
if str(_utility_dir) not in sys.path:
    sys.path.insert(0, str(_utility_dir))

from _validation import is_anndata_raw  # noqa: E402

# Ensure the evaluation module directory is on sys.path for sibling imports
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))


def compute_neighborhood_preservation(
    ref_data: sc.AnnData,
    reduced_data: sc.AnnData,
    k_values: list[int] | None = None,
    cell_annotations: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute how well k-nearest neighbors are preserved between datasets.

    For each cell, the k nearest neighbours are read from the per-k directed kNN
    graph (``obsp["neighbors_k{k}_distances"]``) ranked by ascending distance, on
    both the reference and the reduced-panel embedding, and the two neighbour sets
    are compared by Jaccard index; the mean over cells (and per cell type) is the
    preservation score. Falls back to weight-ranked ``_connectivities`` only when a
    distances matrix is absent.

    Args:
        ref_data: Reference AnnData object (full transcriptome).
        reduced_data: Reduced gene set AnnData object.
        k_values: List of k values to evaluate. Defaults to [5, 10, 15, 20, 30, 50].
        cell_annotations: Optional array of cell type labels aligned with the common
            cells (post-subset order). When provided, per-cell-type mean Jaccard scores
            are stored under "celltype_scores" in the return dict.

    Returns:
        Dictionary with preservation scores for each k value and optimal k.
        When cell_annotations is provided, each rep result also has a "celltype_scores"
        key: {k: {celltype: mean_jaccard}}.
    """
    if k_values is None:
        k_values = DEFAULT_KNN_K_VALUES

    # Match cell barcodes between datasets
    common_cells = list(set(ref_data.obs_names).intersection(set(reduced_data.obs_names)))

    if len(common_cells) == 0:
        logger.info("No common cells found between reference and test data")
        return {"optimal_k": None, "scores": {}, "best_score": np.nan}

    # Get reference and test data subsets for common cells
    ref_subset = ref_data[common_cells]
    reduced_subset = reduced_data[common_cells]
    n_cells = len(common_cells)

    # Run garbage collection to ensure memory is available
    gc.collect()

    # Always use PCA representation
    representations = ["pca"]

    # Process each representation
    all_results = {}

    for rep in representations:
        if len(representations) > 1:
            logger.info(f"\n--- Evaluating {rep.upper()} representation ---")

        knn_overlap_scores: dict[int, float] = {}
        knn_celltype_scores: dict[int, dict[str, float]] = {}

        # For each k value, compute the preservation score
        for k in k_values:
            logger.info(
                f"Computing neighborhood preservation for k={k} ({rep.upper() if len(representations) > 1 else ''})..."
            )

            # Read each cell's k nearest neighbours from the *directed* kNN graph
            # (obsp["neighbors_k{k}_distances"] — k+1 explicit entries per row: the cell
            # itself at distance 0 plus its k neighbours, NOT symmetrised), ranked by
            # ascending distance and with the self-entry dropped. The `_connectivities`
            # graph is symmetrised (rows routinely hold > k nonzeros) and its `.indices`
            # are column order, not similarity rank — taking `.indices[:k]` off it is
            # not a kNN quantity and strongly attenuates the score. Every k reads its
            # own explicit neighbors_k{k} graph; the plain obsp["connectivities"] is the
            # clustering/UMAP graph at the CLI --n_neighbors and must not be used here.
            dist_keys = [
                f"neighbors_k{k}_distances",
                f"neighbors_pca_k{k}_distances",
            ]
            conn_keys = [
                f"neighbors_k{k}_connectivities",
                f"neighbors_pca_k{k}_connectivities",
            ]
            graph_key = None
            use_distance_rank = True
            for key in dist_keys:
                if key in ref_subset.obsp and key in reduced_subset.obsp:
                    graph_key = key
                    break
            if graph_key is None:
                # Fallback for graphs preprocessed without a distances matrix:
                # rank the symmetrised connectivities by weight (higher = closer).
                for key in conn_keys:
                    if key in ref_subset.obsp and key in reduced_subset.obsp:
                        graph_key = key
                        use_distance_rank = False
                        break
                if graph_key is not None:
                    logger.warning(
                        "k=%d: no neighbors_k%d_distances found; falling back to "
                        "weight-ranked connectivities (approximate).", k, k,
                    )

            if graph_key is None:
                logger.info(f"Warning: kNN graph for k={k} not found in one or both datasets")
                logger.info(f"  Looking for one of: {dist_keys + conn_keys}")
                logger.info(f"  Available in ref: {list(ref_subset.obsp.keys())}")
                logger.info(f"  Available in reduced: {list(reduced_subset.obsp.keys())}")
                knn_overlap_scores[k] = np.nan
                continue

            ref_graph = ref_subset.obsp[graph_key]
            reduced_graph = reduced_subset.obsp[graph_key]

            def _knn_indices(cell_idx, row):
                """Top-k neighbour cell indices for one CSR row, similarity-ranked.

                scanpy's ``neighbors_k{k}_distances`` stores the query cell itself as an
                explicit distance-0 entry (each row holds k+1 entries), so it always
                sorts first. Drop it before truncating to k — otherwise the guaranteed
                self-match inflates the Jaccard, most severely at small k (a spurious
                high point at k=5 fading through k=10-20). The ``_connectivities``
                fallback has no stored diagonal, so the filter is a harmless no-op there.
                """
                if row.data.size == 0:
                    return row.indices
                # distances: nearest = smallest; connectivities: nearest = largest
                order = np.argsort(row.data) if use_distance_rank else np.argsort(-row.data)
                ranked = row.indices[order]
                ranked = ranked[ranked != cell_idx]
                return ranked[:k]

            # Calculate neighborhood overlap for each cell
            knn_overlap_sum = 0
            ct_overlap_accumulator: dict[str, list[float]] = {}

            for i in range(n_cells):
                ref_neighbors = _knn_indices(i, ref_graph[i])
                reduced_neighbors = _knn_indices(i, reduced_graph[i])

                # Compute overlap (Jaccard index)
                intersection = len(set(ref_neighbors).intersection(set(reduced_neighbors)))
                union = len(set(ref_neighbors).union(set(reduced_neighbors)))
                overlap = intersection / union if union > 0 else 0
                knn_overlap_sum += overlap

                # Accumulate per-cell-type scores
                if cell_annotations is not None:
                    ct = str(cell_annotations[i])
                    if ct not in ct_overlap_accumulator:
                        ct_overlap_accumulator[ct] = []
                    ct_overlap_accumulator[ct].append(overlap)

            # Store average overlap for this k
            knn_overlap_scores[k] = knn_overlap_sum / n_cells
            if cell_annotations is not None:
                knn_celltype_scores[k] = {
                    ct: float(np.mean(scores)) for ct, scores in ct_overlap_accumulator.items()
                }
            logger.info(f"Average neighborhood preservation for k={k}: {knn_overlap_scores[k]:.4f}")

        # Find the k value with the best preservation score
        valid_scores = {k: score for k, score in knn_overlap_scores.items() if not np.isnan(score)}
        if valid_scores:
            optimal_k = max(valid_scores.items(), key=lambda x: x[1])[0]
            best_score = valid_scores[optimal_k]
        else:
            optimal_k = None
            best_score = np.nan

        # Store results for this representation
        all_results[rep] = {
            "optimal_k": optimal_k,
            "scores": knn_overlap_scores,
            "best_score": best_score,
            "celltype_scores": knn_celltype_scores,
        }

    # Return nested dict if multiple representations, flat dict if single representation
    if len(representations) > 1:
        return all_results
    else:
        return all_results[representations[0]]


def evaluate_neighborhood_preservation(
    sets: dict[str, sc.AnnData],
    reference_key: str = "full_transcriptome",
    celltype_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate neighborhood preservation across all genesets.

    Args:
        sets: Dictionary of AnnData objects for different genesets.
        reference_key: Key for the reference dataset in the sets dictionary.
        celltype_col: Optional column in obs with cell type labels. When provided,
            per-cell-type mean kNN Jaccard scores are appended to the result DataFrame
            as rows with a "celltype" column.

    Returns:
        DataFrame with neighborhood preservation metrics for all genesets.
    """
    # Initialize results storage
    results = []

    # Reference data
    reference_data = sets[reference_key]

    # Evaluate each geneset
    for dataset_name, dataset in sets.items():
        logger.info(f"Processing dataset: {dataset_name}")

        # For h5ad files, use the data directly with process_data
        if isinstance(dataset, sc.AnnData):
            logger.info(f"Dataset {dataset_name} is already an AnnData object")

            # Ensure a 'counts' layer exists (nothing downstream here reads a chosen
            # layer name — this block only guarantees the layer is populated).
            if "counts" not in dataset.layers:
                logger.info(f"No 'counts' layer found, storing X as 'counts' for {dataset_name}")
                dataset.layers["counts"] = dataset.X.copy()

            # Skip processing if the dataset has already been processed
            has_embeddings = "X_pca" in dataset.obsm
            # Check for neighbor graphs - can be 'neighbors_pca_k5', 'neighbors_nmf_k10', etc.
            has_neighbors = any("neighbors" in k and "_k" in k for k in dataset.uns.keys())

            if has_embeddings and has_neighbors:
                logger.info(f"Dataset {dataset_name} is preprocessed and ready for evaluation")
            else:
                # Dataset needs processing - this should have been done in preprocessing step
                logger.info(f"WARNING: Dataset {dataset_name} may not be fully preprocessed.")
                logger.info(f"         Expected preprocessing to include embeddings and neighbor graphs.")
                logger.info(
                    f"         Has embeddings: {has_embeddings}, Has neighbors: {has_neighbors}"
                )
                logger.info(f"         Attempting to proceed with evaluation anyway...")

        else:
            logger.info(f"Unknown dataset type for {dataset_name}, skipping")

        # Run garbage collection after processing each dataset to free memory
        gc.collect()

        if dataset_name == reference_key:
            continue

        logger.info(f"\nEvaluating neighborhood preservation for: {dataset_name}")

        # Build cell_annotations aligned with common cells (post-subset order)
        cell_annotations: np.ndarray | None = None
        if celltype_col is not None and celltype_col in reference_data.obs.columns:
            common_cells = list(
                set(reference_data.obs_names).intersection(set(dataset.obs_names))
            )
            cell_annotations = reference_data[common_cells].obs[celltype_col].values

        # Compute neighborhood preservation across different k values
        # Pass the dimensionality reduction method
        preservation_results = compute_neighborhood_preservation(
            reference_data,
            dataset,
            cell_annotations=cell_annotations,
        )

        # Flat results (PCA representation)
        for k, score in preservation_results["scores"].items():
            results.append(
                {
                    "dataset": dataset_name,
                    "representation": "pca",
                    "k": k,
                    "preservation_score": score,
                    "is_optimal": k == preservation_results["optimal_k"],
                }
            )

            # Add a summary row
            results.append(
                {
                    "dataset": dataset_name,
                    "representation": "pca",
                    "k": "optimal",
                    "preservation_score": preservation_results["best_score"],
                    "optimal_k": preservation_results["optimal_k"],
                }
            )

        # Emit per-cell-type kNN Jaccard rows from celltype_scores
        ct_scores = preservation_results.get("celltype_scores", {})
        for k, ct_dict in ct_scores.items():
            for ct, score in ct_dict.items():
                results.append(
                    {
                        "dataset": dataset_name,
                        "representation": "pca",
                        "k": k,
                        "preservation_score": score,
                        "celltype": ct,
                    }
                )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def compute_clustering_similarity(
    reference_data: sc.AnnData,
    test_data: sc.AnnData,
    celltype_col: str | None = None,
) -> dict[str, Any]:
    """Compute similarity between clustering results using ARI and NMI.

    Args:
        reference_data: Reference AnnData object with Leiden clustering results.
        test_data: Test AnnData object with Leiden clustering results.
        celltype_col: Optional column in reference_data.obs with true cell type labels.
            When provided, also computes per-cell-type Jaccard similarity vs Leiden clusters.

    Returns:
        Dictionary with ARI, NMI, and (if celltype_col given) celltype_jaccard scores.
    """
    # Match cell barcodes between datasets
    common_cells = list(set(reference_data.obs_names).intersection(set(test_data.obs_names)))

    if len(common_cells) == 0:
        logger.info("No common cells found between reference and test data")
        return {}

    # Get reference and test data subsets for common cells
    ref_subset = reference_data[common_cells]
    test_subset = test_data[common_cells]

    # Get leiden cluster columns (PCA-only format: leiden_{n}_clusters)
    test_leiden_cols = [
        col for col in test_subset.obs.columns if col.startswith("leiden_") and "_clusters" in col
    ]

    # Initialize results dictionary
    ari_scores = {}
    nmi_scores = {}

    # Always evaluate PCA representation
    representations = [None]  # None = no suffix (leiden_{n}_clusters format)

    for rep in representations:
        rep_results_ari = ari_scores
        rep_results_nmi = nmi_scores

        # Get leiden cluster columns — unsuffixed names first, then the ``_pca``-suffixed
        # variant as a fallback.
        ref_leiden_cols = [
            col
            for col in ref_subset.obs.columns
            if col.startswith("leiden_") and "_clusters" in col and "_pca" not in col and "_nmf" not in col
        ]
        test_leiden_cols_rep = [
            col for col in test_leiden_cols if "_pca" not in col and "_nmf" not in col
        ]
        if not ref_leiden_cols:
            ref_leiden_cols = [
                col for col in ref_subset.obs.columns
                if col.startswith("leiden_") and "_clusters_pca" in col
            ]
            test_leiden_cols_rep = [col for col in test_leiden_cols if "_clusters_pca" in col]

        # Pair reference and panel Leiden columns by the ACTUAL number of clusters each
        # column contains, not by the integer in its name. That integer is only the
        # *target*: the resolution search keeps the target name even when it falls back
        # to the closest resolution it found (see _preprocessing.py), and the reference
        # and each panel run that search independently — so `leiden_42_clusters` on the
        # two sides can hold e.g. 40 vs 45 clusters. Pairing on the realized count (and
        # skipping when no panel column has a matching count) keeps every ARI/NMI row an
        # apples-to-apples comparison.
        test_by_actual: dict[int, str] = {}
        for col in test_leiden_cols_rep:
            n_actual = test_subset.obs[col].astype(str).nunique()
            # On a collision prefer the column whose target name matches its actual count.
            if n_actual not in test_by_actual or f"leiden_{n_actual}_clusters" in col:
                test_by_actual[n_actual] = col

        for ref_col in ref_leiden_cols:
            ref_n_clusters = int(ref_subset.obs[ref_col].astype(str).nunique())

            test_col = test_by_actual.get(ref_n_clusters)
            if test_col is None:
                logger.debug(
                    "No panel Leiden column with %d clusters to match reference '%s'; skipping row",
                    ref_n_clusters, ref_col,
                )
                continue

            # Calculate Adjusted Rand Index and Normalized Mutual Information
            ari = adjusted_rand_score(
                ref_subset.obs[ref_col].astype(str), test_subset.obs[test_col].astype(str)
            )
            nmi = normalized_mutual_info_score(
                ref_subset.obs[ref_col].astype(str), test_subset.obs[test_col].astype(str)
            )

            rep_results_ari[ref_n_clusters] = ari
            rep_results_nmi[ref_n_clusters] = nmi
            logger.info(f"ARI for {ref_n_clusters} clusters: {ari:.4f}")
            logger.info(f"NMI for {ref_n_clusters} clusters: {nmi:.4f}")

        # Store results for this representation
        if rep is not None:
            ari_scores[rep] = rep_results_ari
            nmi_scores[rep] = rep_results_nmi

    # Also calculate ARI and NMI for default leiden clustering
    if "leiden" in ref_subset.obs and "leiden" in test_subset.obs:
        default_ari = adjusted_rand_score(
            ref_subset.obs["leiden"].astype(str), test_subset.obs["leiden"].astype(str)
        )
        default_nmi = normalized_mutual_info_score(
            ref_subset.obs["leiden"].astype(str), test_subset.obs["leiden"].astype(str)
        )
        ari_scores["default"] = default_ari
        nmi_scores["default"] = default_nmi
        logger.info(f"Default leiden ARI: {default_ari:.4f}")
        logger.info(f"Default leiden NMI: {default_nmi:.4f}")

    # Compute per-cell-type Jaccard/purity against best-matching Leiden clusters.
    celltype_jaccard: dict[str, Any] = {}
    if celltype_col is not None and celltype_col in ref_subset.obs.columns:
        true_labels = ref_subset.obs[celltype_col].values
        for rep in representations:
            if rep is not None:
                cluster_col_candidates = [
                    col for col in test_subset.obs.columns
                    if col.startswith("leiden_") and "_clusters" in col and f"_{rep}" in col
                ]
            else:
                cluster_col_candidates = [
                    col for col in test_subset.obs.columns
                    if col.startswith("leiden_") and "_clusters" in col
                    and "_pca" not in col and "_nmf" not in col
                ]
            if cluster_col_candidates:
                def _extract_n(col: str) -> int:
                    try:
                        return int(col.split("_")[1])
                    except (IndexError, ValueError):
                        return 0
                best_col = max(cluster_col_candidates, key=_extract_n)
                jaccard_results = compute_celltype_jaccard_vs_leiden(
                    true_labels, test_subset.obs[best_col].values
                )
                rep_key = rep if rep is not None else "default"
                celltype_jaccard[rep_key] = jaccard_results

    return {"ari": ari_scores, "nmi": nmi_scores, "celltype_jaccard": celltype_jaccard}


def compute_celltype_jaccard_vs_leiden(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Compute per-cell-type Jaccard similarity and purity against Leiden clusters.

    For each true cell type T, finds the best-matching Leiden cluster (maximum
    intersection), then computes Jaccard(T, C_best) and purity(T, C_best).

    This complements global ARI/NMI by giving per-class clustering quality,
    exposing poor recovery of rare cell types that global metrics mask.

    Args:
        true_labels: Ground-truth cell type annotation array (n_cells,).
        cluster_labels: Leiden cluster label array (n_cells,) aligned with true_labels.

    Returns:
        Dictionary keyed by cell type name, each containing:
            - jaccard: |T ∩ C_best| / |T ∪ C_best|
            - purity: |T ∩ C_best| / |T|
            - n_cells: number of cells belonging to this cell type
            - best_cluster: label of the best-matching Leiden cluster
    """
    true_labels = np.asarray(true_labels)
    cluster_labels = np.asarray(cluster_labels)

    results: dict[str, dict[str, Any]] = {}
    for ct in np.unique(true_labels):
        ct_mask = true_labels == ct
        ct_indices = set(np.where(ct_mask)[0])
        n_ct = len(ct_indices)

        best_cluster = None
        best_overlap = -1
        for cl in np.unique(cluster_labels):
            cl_indices = set(np.where(cluster_labels == cl)[0])
            overlap = len(ct_indices & cl_indices)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = cl
                best_cl_indices = cl_indices

        union = len(ct_indices | best_cl_indices)
        jaccard = best_overlap / union if union > 0 else 0.0
        purity = best_overlap / n_ct if n_ct > 0 else 0.0

        results[str(ct)] = {
            "jaccard": float(jaccard),
            "purity": float(purity),
            "n_cells": int(n_ct),
            "best_cluster": str(best_cluster),
        }

    return results


def evaluate_clustering_quality(
    sets: dict[str, sc.AnnData],
    reference_key: str = "full_transcriptome",
    celltype_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate clustering quality across all genesets.

    Args:
        sets: Dictionary of AnnData objects for different genesets.
        reference_key: Key for the reference dataset in the sets dictionary.
        celltype_col: Optional column in obs with true cell type labels. When provided,
            per-cell-type Jaccard and purity rows are appended to the result DataFrame.

    Returns:
        DataFrame with clustering similarity metrics for all genesets.
        When celltype_col is given, additional rows have a "celltype" column with
        "jaccard" and "purity" values.
    """
    # Initialize results storage
    results = []

    # Reference data
    reference_data = sets[reference_key]

    # Evaluate each geneset
    for dataset_name, dataset in sets.items():
        if dataset_name == reference_key:
            continue

        logger.info(f"\nEvaluating clustering quality for: {dataset_name}")

        # Compute clustering similarity (returns dict with 'ari', 'nmi', 'celltype_jaccard')
        clustering_scores = compute_clustering_similarity(
            reference_data, dataset, celltype_col=celltype_col
        )
        ari_scores = clustering_scores["ari"]
        nmi_scores = clustering_scores["nmi"]
        celltype_jaccard_scores = clustering_scores.get("celltype_jaccard", {})

        # Store clustering results (PCA representation)
        for n_clusters in ari_scores.keys():
            results.append(
                {
                    "dataset": dataset_name,
                    "n_clusters": n_clusters,
                    "ARI": ari_scores[n_clusters],
                    "NMI": nmi_scores[n_clusters],
                    "representation": "pca",
                }
            )

        # Emit per-cell-type Jaccard/purity rows (one row per CT per representation)
        for rep_key, ct_dict in celltype_jaccard_scores.items():
            for ct, ct_metrics in ct_dict.items():
                results.append(
                    {
                        "dataset": dataset_name,
                        "representation": rep_key,
                        "celltype": ct,
                        "jaccard": ct_metrics["jaccard"],
                        "purity": ct_metrics["purity"],
                        "n_cells": ct_metrics["n_cells"],
                        "best_cluster": ct_metrics["best_cluster"],
                    }
                )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def split_train_test_sets(
    adata: sc.AnnData,
    split: int = 4,
    seed: int = 2020,
    verbose: bool = True,
    obs_key: str | None = None,
) -> None:
    """Split data to train and test set.

    This function was copied from the Spapros package (Kuemmerle, Nature Methods (2024)).

    Args:
        adata: An already preprocessed annotated data matrix. Typically log normalised data.
        split: Number of splits (train:test ratio will be split:1).
        seed: Random number seed.
        verbose: Verbosity level > 1.
        obs_key: Provide a column name of adata.obs. If provided, each group is split
            with the defined ratio.
    """
    if not obs_key:
        n_train = (adata.n_obs // (split + 1)) * split
        np.random.seed(seed=seed)
        train_obs = np.random.choice(adata.n_obs, n_train, replace=False)
        test_obs = np.array([True for i in range(adata.n_obs)])
        test_obs[train_obs] = False
        train_obs = np.invert(test_obs)
        if verbose:
            logger.info(f"Split data to ratios {split}:1 (train:test)")
            logger.info(f"datapoints: {adata.n_obs}")
            logger.info(f"train data: {np.sum(train_obs)}")
            logger.info(f"test data: {np.sum(test_obs)}")
        adata.obs["train_set"] = train_obs
        adata.obs["test_set"] = test_obs
    else:
        adata.obs["train_set"] = False
        adata.obs["test_set"] = False
        for group in adata.obs[obs_key].unique():
            df = adata.obs.loc[adata.obs[obs_key] == group]
            n_obs = len(df)
            n_train = (n_obs // (split + 1)) * split
            np.random.seed(seed=seed)
            train_obs = np.random.choice(n_obs, n_train, replace=False)
            test_obs = np.array([True for i in range(n_obs)])
            test_obs[train_obs] = False
            train_obs = np.invert(test_obs)
            if verbose:
                logger.info(f"Split data for group {group}")
                logger.info(f"to ratios {split}:1 (train:test)")
                logger.info(f"datapoints: {n_obs}")
                logger.info(f"train data: {np.sum(train_obs)}")
                logger.info(f"test data: {np.sum(test_obs)}")
            adata.obs.loc[df.index, "train_set"] = train_obs
            adata.obs.loc[df.index, "test_set"] = test_obs


def uniform_samples(
    adata: sc.AnnData,
    ct_key: str,
    set_key: str = "train_set",
    subsample: int = 500,
    seed: int = 2020,
    celltypes: Union[list[str], str] = "all",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Subsample cells per celltype uniformly.

    This function was copied from the Spapros package (Kuemmerle, Nature Methods (2024)).
    If the number of cells of a celltype is lower we're oversampling that celltype.

    Args:
        adata: An already preprocessed annotated data matrix. Typically log normalised data.
        ct_key: Column of `adata.obs` with cell type annotation.
        set_key: Column of `adata.obs` indicating the train set.
        subsample: Number of random choices.
        seed: Random number seed.
        celltypes: List of celltypes to consider or `all`.

    Returns:
        Tuple containing:
            - X: expression matrix (n_samples x n_genes)
            - y: dict mapping each celltype to binary labels (celltype vs "other")
            - cts: actual celltype labels for each sample
    """
    a = adata[adata.obs[set_key], :]
    if celltypes == "all":
        celltypes = list(a.obs[ct_key].unique())

    # Get subsample for each celltype
    all_obs = []
    for ct in celltypes:
        df = a.obs.loc[a.obs[ct_key] == ct]
        n_obs = len(df)
        np.random.seed(seed=seed)
        if n_obs > subsample:
            obs = np.random.choice(n_obs, subsample, replace=False)
            all_obs += list(df.iloc[obs].index.values)
        else:
            obs = np.random.choice(n_obs, subsample, replace=True)
            all_obs += list(df.iloc[obs].index.values)

    if scipy.sparse.issparse(a.X):
        X = a[all_obs, :].X.toarray()
    else:
        X = a[all_obs, :].X.copy()

    y = {}
    for ct in celltypes:
        y[ct] = np.where(a[all_obs, :].obs[ct_key] == ct, ct, "other")

    cts = a[all_obs].obs[ct_key].values

    return X, y, cts


def evaluate_celltype_identification(
    sets: dict[str, sc.AnnData],
    reference_key: str = "full_transcriptome",
    celltype_col: str = DEFAULT_CELLTYPE_COLUMN,
    output_dir: str | None = None,
    celltype_clf_max_depth: int = DEFAULT_CELLTYPE_CLF_MAX_DEPTH,
) -> pd.DataFrame:
    """Evaluate celltype identification accuracy using a decision tree classifier.

    For each geneset, an internal 80/20 stratified split (seed 42) is drawn, class-balanced
    train/test sets are sampled (``DEFAULT_SUBSAMPLE_SIZE`` cells per celltype), and a single
    ``DecisionTreeClassifier(max_depth=celltype_clf_max_depth, random_state=42)`` is fit on
    the train split and scored on the held-out test split. The depth is the same for every
    panel in one call — it is not tuned per panel — but is an explicit, overridable
    parameter (defaulting to ``DEFAULT_CELLTYPE_CLF_MAX_DEPTH``) rather than a hidden
    constant, so a caller can deliberately compare depths across separate calls.

    Args:
        sets: Dictionary of AnnData objects for different genesets.
        reference_key: Key for the reference dataset in the sets dictionary.
        celltype_col: Name of the column in adata.obs containing celltype labels.
        output_dir: Optional output directory for saving results (not used for evaluation).
        celltype_clf_max_depth: Fixed ``max_depth`` for the classifier, applied identically
            to every panel in this call. Changing it changes the measuring instrument, so
            hold it constant across any set of panels being compared.

    Returns:
        DataFrame with classification accuracy metrics for all genesets.
    """
    # Initialize results storage
    results = []

    # Get reference data
    reference_data = sets[reference_key]

    # Check if celltype column exists in reference data
    if celltype_col not in reference_data.obs.columns:
        logger.error(f"Error: celltype column '{celltype_col}' not found in reference data")
        logger.info(f"Available columns: {list(reference_data.obs.columns)}")
        return pd.DataFrame()

    # Iterate through all genesets
    for dataset_name, dataset in sets.items():
        if dataset_name == reference_key:
            continue

        logger.info(f"\nEvaluating celltype identification for: {dataset_name}")

        # Find common cells between reference and test dataset
        common_cells = list(set(reference_data.obs_names).intersection(set(dataset.obs_names)))

        if len(common_cells) == 0:
            logger.warning(f"No common cells found between reference and {dataset_name}")
            continue

        # Get reference and test data subsets for common cells
        ref_subset = reference_data[common_cells].copy()
        test_subset = (
            dataset[common_cells].copy()
        )  # called test_subset but it is actually the original anndata of the evaluated panel subsetted to common cells with the reference data

        # Check if adata.X is raw
        is_raw_X = is_anndata_raw(test_subset)

        if not is_raw_X:
            logger.info(f"Using log normalized data for testing cell type classification accuracy")
        else:
            logger.info(f"No log normalized data available, using raw data instead")

        # Get celltype labels from reference data and add to test_subset
        try:
            test_subset.obs[celltype_col] = ref_subset.obs[celltype_col].values
        except KeyError as e:
            logger.error(f"KeyError accessing celltype column: {e}")
            continue

        # Apply train/test split (like the reference method)
        # Use 4:1 split ratio (80% train, 20% test) stratified by celltype
        split_train_test_sets(
            test_subset, split=DEFAULT_SPLIT_RATIO, seed=42, verbose=False, obs_key=celltype_col
        )

        # Check if we have both train and test cells for all celltypes
        celltypes = test_subset.obs[celltype_col].unique()
        celltypes_with_train = test_subset.obs.loc[test_subset.obs["train_set"], celltype_col].unique()
        celltypes_with_test = test_subset.obs.loc[test_subset.obs["test_set"], celltype_col].unique()

        valid_celltypes = [
            ct for ct in celltypes if ct in celltypes_with_train and ct in celltypes_with_test
        ]

        if len(valid_celltypes) == 0:
            logger.warning(f"No celltypes with both train and test samples in {dataset_name}")
            continue

        try:
            # Use uniform sampling to get balanced train and test sets
            subsample_train = DEFAULT_SUBSAMPLE_SIZE  # Number of cells per celltype for training
            subsample_test = DEFAULT_SUBSAMPLE_SIZE  # Number of cells per celltype for testing

            # Get uniformly sampled training data
            # Note: uniform_samples returns (X, y_dict, cts) where:
            #   - X: expression matrix
            #   - y_dict: binary labels per celltype (for one-vs-rest, not used here)
            #   - cts: actual celltype labels (used for multi-class classification)
            X_train, _, y_train = uniform_samples(
                test_subset,  # called test_subset but it is actually the original anndata of the evaluated panel subsetted to common cells with the reference data
                ct_key=celltype_col,
                set_key="train_set",
                subsample=subsample_train,
                seed=42,
                celltypes=valid_celltypes,
            )

            # Get uniformly sampled test data
            X_test, _, y_test = uniform_samples(
                test_subset,
                ct_key=celltype_col,
                set_key="test_set",
                subsample=subsample_test,
                seed=42,
                celltypes=valid_celltypes,
            )

            # Train the cell-type classifier. max_depth is FIXED to
            # celltype_clf_max_depth for every panel in this call — it is not tuned per
            # panel, which keeps the classifier a simple, capacity-matched readout of the
            # panel's cell-type signal.
            dt_classifier = DecisionTreeClassifier(
                max_depth=celltype_clf_max_depth, random_state=42
            )
            dt_classifier.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = dt_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Store dataset-level results (overall accuracy + macro/weighted F1)
            results.append(
                {
                    "dataset": dataset_name,
                    "accuracy": accuracy,
                    "classifier_max_depth": celltype_clf_max_depth,
                    "n_genes": len(test_subset.var_names),
                    "macro_f1": class_report["macro avg"]["f1-score"],
                    "weighted_f1": class_report["weighted avg"]["f1-score"],
                }
            )

            # Also store per-class metrics
            for celltype in dt_classifier.classes_:
                if celltype in class_report:
                    results.append(
                        {
                            "dataset": dataset_name,  # Keep dataset name separate (don't append celltype)
                            "celltype": celltype,
                            "accuracy": accuracy,  # Overall accuracy
                            "precision": class_report[celltype]["precision"],
                            "recall": class_report[celltype]["recall"],
                            "f1-score": class_report[celltype]["f1-score"],
                            "support": class_report[celltype]["support"],
                        }
                    )

            logger.info(f"Celltype identification accuracy for {dataset_name}: {accuracy:.4f}")
            logger.info(f"Classifier max_depth (fixed): {celltype_clf_max_depth}")
            logger.info(f"Macro F1 score: {class_report['macro avg']['f1-score']:.4f}")

        except Exception as e:
            logger.error(f"Error in model training for {dataset_name}: {e}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Log a ranking summary over the panels (dataset-level rows only). Not saved —
    # the calling script persists results_df; this block is a console convenience.
    if not results_df.empty:
        if "celltype" in results_df.columns:
            dataset_results = results_df[pd.isna(results_df.get("celltype"))].copy()
        else:
            dataset_results = results_df.copy()

        dataset_results = dataset_results.sort_values("macro_f1", ascending=False)
        dataset_results["accuracy_rank"] = dataset_results["accuracy"].rank(
            ascending=False, method="min"
        ).astype(int)
        dataset_results["macro_f1_rank"] = dataset_results["macro_f1"].rank(
            ascending=False, method="min"
        ).astype(int)
        dataset_results["average_rank"] = dataset_results[
            ["accuracy_rank", "macro_f1_rank"]
        ].mean(axis=1)
        dataset_results = dataset_results.sort_values("average_rank")

        logger.info("\n=== PERFORMANCE RANKING SUMMARY ===")
        logger.info("Ranked by mean of accuracy_rank and macro_f1_rank (lower is better):")
        for idx, row in dataset_results.iterrows():
            logger.info(
                f"  {row['dataset']}: Avg Rank={row['average_rank']:.1f}, "
                f"Accuracy={row['accuracy']:.4f} (rank {row['accuracy_rank']}), "
                f"Macro F1={row['macro_f1']:.4f} (rank {row['macro_f1_rank']})"
            )

    return results_df
