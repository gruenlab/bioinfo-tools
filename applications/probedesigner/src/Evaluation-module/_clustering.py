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
import os
import sys
import warnings
import traceback
from typing import Any, Union, List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import NMF

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
    "compute_rare_celltype_marker_coverage",
]

# Ensure the Utility-module directory is on sys.path so its files are importable
# as flat modules — consistent with how run_evaluation.py sets up the path.
_utility_dir = _current_dir.parent / "Utility-module"
if str(_utility_dir) not in sys.path:
    sys.path.insert(0, str(_utility_dir))

from _validation import is_anndata_raw, is_anndata_raw_layer, X_is_raw  # noqa: E402

# Ensure the evaluation module directory is on sys.path for sibling imports
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from metrics import calculate_inverse_frequency_weighted_f1  # noqa: E402


def compute_neighborhood_preservation(
    ref_data: sc.AnnData,
    reduced_data: sc.AnnData,
    k_values: list[int] | None = None,
    dimensionality_reduction: str = "pca",
    cell_annotations: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute how well k-nearest neighbors are preserved between datasets.

    Compares k-NN overlap between full transcriptome and reduced gene set across
    multiple k values using Jaccard index.

    Args:
        ref_data: Reference AnnData object (full transcriptome).
        reduced_data: Reduced gene set AnnData object.
        k_values: List of k values to evaluate. Defaults to [5, 10, 15, 20, 30, 50].
        dimensionality_reduction: Which dimensionality reduction was used: "pca",
            "nmf", or "both" (default: "pca").
        cell_annotations: Optional array of cell type labels aligned with the common
            cells (post-subset order). When provided, per-cell-type mean Jaccard scores
            are stored under "celltype_scores" in the return dict.

    Returns:
        Dictionary with preservation scores for each k value and optimal k.
        If dimensionality_reduction="both", returns nested dict with 'pca' and 'nmf' keys.
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

    # Determine which representations to evaluate
    if dimensionality_reduction == "both":
        representations = ["pca", "nmf"]
        logger.info("Computing neighborhood preservation for both PCA and NMF representations")
    elif dimensionality_reduction == "nmf":
        representations = ["nmf"]
    else:  # pca only (default)
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

            # Construct the correct keys based on representation
            # For single-mode runs (pca or nmf only), the keys don't have the rep suffix
            # For "both" mode runs, they do have the suffix
            if rep == "nmf":
                # Try with suffix first (from "both" mode), then without (from "nmf" only mode)
                possible_keys = [
                    (f"neighbors_nmf_k{k}_connectivities" if k != 15 else "neighbors_nmf_connectivities"),
                    (f"neighbors_k{k}_connectivities" if k != 15 else "connectivities"),
                ]
            else:  # pca
                # Try with suffix first (from "both" mode), then without (from "pca" only mode)
                possible_keys = [
                    (f"neighbors_pca_k{k}_connectivities" if k != 15 else "neighbors_pca_connectivities"),
                    (f"neighbors_k{k}_connectivities" if k != 15 else "connectivities"),
                ]

            # Find the correct keys
            ref_conn_key = None
            reduced_conn_key = None
            for key in possible_keys:
                if key in ref_subset.obsp and key in reduced_subset.obsp:
                    ref_conn_key = key
                    reduced_conn_key = key
                    break

            # Check if both datasets have the required connectivities
            if ref_conn_key is None or reduced_conn_key is None:
                logger.info(f"Warning: Connectivities for k={k} not found in one or both datasets")
                logger.info(f"  Looking for one of: {possible_keys}")
                logger.info(f"  Available in ref: {list(ref_subset.obsp.keys())}")
                logger.info(f"  Available in reduced: {list(reduced_subset.obsp.keys())}")
                knn_overlap_scores[k] = np.nan
                continue

            ref_conn = ref_subset.obsp[ref_conn_key]
            reduced_conn = reduced_subset.obsp[reduced_conn_key]

            # Calculate neighborhood overlap for each cell
            knn_overlap_sum = 0
            ct_overlap_accumulator: dict[str, list[float]] = {}

            for i in range(n_cells):
                # Get indices of k nearest neighbors in reference
                ref_neighbors = (
                    ref_conn[i].indices[:k]
                    if ref_conn[i].indices.size >= k
                    else ref_conn[i].indices
                )

                # Get indices of k nearest neighbors in reduced
                reduced_neighbors = (
                    reduced_conn[i].indices[:k]
                    if reduced_conn[i].indices.size >= k
                    else reduced_conn[i].indices
                )

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
    dimensionality_reduction: str = "pca",
    celltype_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate neighborhood preservation across all genesets.

    Args:
        sets: Dictionary of AnnData objects for different genesets.
        reference_key: Key for the reference dataset in the sets dictionary.
        dimensionality_reduction: Which dimensionality reduction to use ("pca", "nmf", or "both").
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

            # Check if we have a counts layer
            if "counts" in dataset.layers:
                logger.info(f"Using 'counts' layer for {dataset_name}")
                layer = "counts"
            else:
                logger.info(f"No 'counts' layer found, using X for {dataset_name}")
                # Store raw data in counts layer if not present
                dataset.layers["counts"] = dataset.X.copy()
                layer = "counts"

            # Skip processing if the dataset has already been processed
            # Check for either PCA or NMF embeddings AND any neighbor graph
            has_embeddings = "X_pca" in dataset.obsm or "X_nmf" in dataset.obsm
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
            dimensionality_reduction=dimensionality_reduction,
            cell_annotations=cell_annotations,
        )

        # Handle both flat (single representation) and nested (both representations) results
        if dimensionality_reduction == "both":
            # Nested results with 'pca' and 'nmf' keys
            for rep in ["pca", "nmf"]:
                if rep in preservation_results:
                    # Store results for each k value
                    for k, score in preservation_results[rep]["scores"].items():
                        results.append(
                            {
                                "dataset": dataset_name,
                                "representation": rep,
                                "k": k,
                                "preservation_score": score,
                                "is_optimal": k == preservation_results[rep]["optimal_k"],
                            }
                        )

                    # Add a summary row
                    results.append(
                        {
                            "dataset": dataset_name,
                            "representation": rep,
                            "k": "optimal",
                            "preservation_score": preservation_results[rep]["best_score"],
                            "optimal_k": preservation_results[rep]["optimal_k"],
                        }
                    )
        else:
            # Flat results (single representation)
            # Store results for each k value
            for k, score in preservation_results["scores"].items():
                results.append(
                    {
                        "dataset": dataset_name,
                        "representation": dimensionality_reduction,
                        "k": k,
                        "preservation_score": score,
                        "is_optimal": k == preservation_results["optimal_k"],
                    }
                )

            # Add a summary row
            results.append(
                {
                    "dataset": dataset_name,
                    "representation": dimensionality_reduction,
                    "k": "optimal",
                    "preservation_score": preservation_results["best_score"],
                    "optimal_k": preservation_results["optimal_k"],
                }
            )

        # Emit per-cell-type kNN Jaccard rows from celltype_scores
        if dimensionality_reduction == "both":
            reps_to_check = ["pca", "nmf"]
            rep_results_map = preservation_results
        else:
            reps_to_check = [dimensionality_reduction]
            rep_results_map = {dimensionality_reduction: preservation_results}

        for rep in reps_to_check:
            if rep not in rep_results_map:
                continue
            ct_scores = rep_results_map[rep].get("celltype_scores", {})
            for k, ct_dict in ct_scores.items():
                for ct, score in ct_dict.items():
                    results.append(
                        {
                            "dataset": dataset_name,
                            "representation": rep,
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

    # Check if we have both PCA and NMF results (indicated by _pca and _nmf suffixes)
    test_leiden_cols = [
        col for col in test_subset.obs.columns if col.startswith("leiden_") and "_clusters" in col
    ]
    has_pca_results = any("_pca" in col for col in test_leiden_cols)
    has_nmf_results = any("_nmf" in col for col in test_leiden_cols)

    # Initialize results dictionary for both metrics
    ari_scores = {}
    nmi_scores = {}

    # Determine representations to evaluate
    if has_pca_results and has_nmf_results:
        representations = ["pca", "nmf"]
        logger.info("Detected both PCA and NMF clustering results - evaluating both")
    elif has_nmf_results:
        representations = ["nmf"]
    elif has_pca_results:
        representations = ["pca"]
    else:
        representations = [None]  # Old format without suffix

    for rep in representations:
        if rep is not None:
            logger.info(f"\n--- Evaluating {rep.upper()} clustering ---")
            rep_results_ari = {}
            rep_results_nmi = {}
        else:
            rep_results_ari = ari_scores
            rep_results_nmi = nmi_scores

        # Get leiden cluster columns for this representation
        if rep is not None:
            ref_leiden_cols = [
                col
                for col in ref_subset.obs.columns
                if col.startswith("leiden_") and "_clusters" in col and f"_{rep}" in col
            ]
            test_leiden_cols_rep = [col for col in test_leiden_cols if f"_{rep}" in col]
        else:
            ref_leiden_cols = [
                col
                for col in ref_subset.obs.columns
                if col.startswith("leiden_") and "_clusters" in col and "_pca" not in col and "_nmf" not in col
            ]
            test_leiden_cols_rep = [
                col for col in test_leiden_cols if "_pca" not in col and "_nmf" not in col
            ]

        # For each reference cluster resolution
        for ref_col in ref_leiden_cols:
            # Extract cluster count
            try:
                parts = ref_col.split("_")
                # Handle both formats: leiden_15_clusters_pca and leiden_15_clusters
                if rep is not None:
                    ref_n_clusters = int(parts[1])
                else:
                    ref_n_clusters = int(parts[1])
            except (IndexError, ValueError):
                continue

            # Find matching test column with same number of clusters
            if rep is not None:
                pattern = f"leiden_{ref_n_clusters}_clusters_{rep}"
            else:
                pattern = f"leiden_{ref_n_clusters}_clusters"

            matching_col = [col for col in test_leiden_cols_rep if pattern in col]

            if not matching_col:
                continue

            test_col = matching_col[0]

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
        if has_pca_results and has_nmf_results:
            # In both mode, store default separately
            ari_scores["default"] = default_ari
            nmi_scores["default"] = default_nmi
        else:
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
    dimensionality_reduction: str = "pca",
    celltype_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate clustering quality across all genesets.

    Args:
        sets: Dictionary of AnnData objects for different genesets.
        reference_key: Key for the reference dataset in the sets dictionary.
        dimensionality_reduction: Which dimensionality reduction was used.
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

        # Determine which representation(s) were used
        # Check if we have both PCA and NMF embeddings to know if we used "both" mode
        has_pca = "X_pca" in dataset.obsm
        has_nmf = "X_nmf" in dataset.obsm

        # Check if ari_scores is nested (both mode) or flat (single mode)
        if isinstance(ari_scores, dict) and any(k in ari_scores for k in ["pca", "nmf"]):
            # Both mode - results are nested by representation
            for rep in ["pca", "nmf"]:
                if rep in ari_scores:
                    for n_clusters in ari_scores[rep].keys():
                        ari = ari_scores[rep][n_clusters]
                        nmi = nmi_scores[rep][n_clusters]
                        results.append(
                            {
                                "dataset": dataset_name,
                                "n_clusters": n_clusters,
                                "ARI": ari,
                                "NMI": nmi,
                                "representation": rep,
                            }
                        )
            # Handle default if present
            if "default" in ari_scores:
                results.append(
                    {
                        "dataset": dataset_name,
                        "n_clusters": "default",
                        "ARI": ari_scores["default"],
                        "NMI": nmi_scores["default"],
                        "representation": "both",
                    }
                )
        else:
            # Single mode - results are flat
            if has_nmf:
                representation = "nmf"
            elif has_pca:
                representation = "pca"
            else:
                representation = "unknown"

            # Store results
            for n_clusters in ari_scores.keys():
                ari = ari_scores[n_clusters]
                nmi = nmi_scores[n_clusters]
                results.append(
                    {
                        "dataset": dataset_name,
                        "n_clusters": n_clusters,
                        "ARI": ari,
                        "NMI": nmi,
                        "representation": representation,
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
    celltype_col: str = "new_annot",
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Evaluate celltype identification accuracy using decision tree classifier.

    Trains on complete dataset and predicts celltype labels for each geneset.

    Args:
        sets: Dictionary of AnnData objects for different genesets.
        reference_key: Key for the reference dataset in the sets dictionary.
        celltype_col: Name of the column in adata.obs containing celltype labels.
        output_dir: Optional output directory for saving results (not used for evaluation).

    Returns:
        DataFrame with classification accuracy metrics for all genesets.
    """
    # Initialize results storage
    results = []
    feature_importances_list = []  # Store feature importances for each dataset

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

            # Initialize and train the decision tree classifier
            # Use different max_depths to find the best performing model based on macro F1-score
            max_depths = [None, 5, 10, 15, 20]
            best_macro_f1 = 0
            best_model = None
            best_depth = None

            for depth in max_depths:
                dt_classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
                dt_classifier.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = dt_classifier.predict(X_test)

                # Calculate macro F1-score (equal weight per celltype)
                macro_f1 = f1_score(y_test, y_pred, average="macro")

                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_model = dt_classifier
                    best_depth = depth

            # Use the best model for final evaluation
            dt_classifier = best_model

            # Make predictions on the test set
            y_pred = dt_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Extract and store feature importances
            feature_names = list(test_subset.var_names)
            importances = dt_classifier.feature_importances_

            # Create DataFrame with feature importances, sorted by importance
            feature_importance_df = pd.DataFrame(
                {"gene": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            # Store features for this dataset
            for idx, row in feature_importance_df.iterrows():
                feature_importances_list.append(
                    {
                        "dataset": dataset_name,
                        "gene": row["gene"],
                        "importance": row["importance"],
                        "rank": idx + 1,
                    }
                )

            logger.info(
                f"Top 5 important genes for {dataset_name}: {', '.join(feature_importance_df['gene'].head(5).tolist())}"
            )

            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Extract macro F1-score for display
            macro_f1 = class_report["macro avg"]["f1-score"]

            # Compute inverse-frequency weighted F1 (amplifies rare cell types)
            per_ct_f1 = {
                ct: class_report[ct]["f1-score"]
                for ct in dt_classifier.classes_
                if ct in class_report
            }
            full_ct_counts = test_subset.obs[celltype_col].value_counts().to_dict()
            inv_freq_f1 = calculate_inverse_frequency_weighted_f1(per_ct_f1, full_ct_counts)

            # Store results - include both macro and weighted F1 scores
            results.append(
                {
                    "dataset": dataset_name,
                    "accuracy": accuracy,
                    "best_max_depth": best_depth,
                    "n_genes": len(test_subset.var_names),
                    "macro_f1": class_report["macro avg"]["f1-score"],
                    "weighted_f1": class_report["weighted avg"]["f1-score"],
                    "weighted_precision": class_report["weighted avg"]["precision"],
                    "weighted_recall": class_report["weighted avg"]["recall"],
                    "inverse_freq_weighted_f1": inv_freq_f1,
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
            logger.info(f"Best max_depth parameter: {best_depth}")
            logger.info(f"Macro F1 score: {class_report['macro avg']['f1-score']:.4f}")
            logger.info(f"Weighted F1 score: {class_report['weighted avg']['f1-score']:.4f}")

        except Exception as e:
            logger.error(f"Error in model training for {dataset_name}: {e}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate average feature importance per dataset (but don't save separately)
    if feature_importances_list:
        feature_importances_df = pd.DataFrame(feature_importances_list)
        avg_importance_per_dataset = (
            feature_importances_df.groupby("dataset")["importance"].mean().to_dict()
        )
    else:
        avg_importance_per_dataset = {}

    # Return results to calling function for saving
    # Note: The calling script (Run-Evaluation.py) handles saving and plotting
    if not results_df.empty:
        # Check if we have celltype-specific results (which contain celltype column)
        if "celltype" in results_df.columns:
            # Filter for only the dataset-level results (not per-celltype)
            dataset_results = results_df[pd.isna(results_df.get("celltype"))].copy()
            # Filter for per-celltype results (for heatmap)
            celltype_results = results_df[results_df["dataset"].str.contains("_")].copy()
        else:
            # If no celltype column, assume all rows are dataset-level
            dataset_results = results_df.copy()

        # Add average feature importance to dataset_results
        if avg_importance_per_dataset:
            dataset_results["avg_feature_importance"] = dataset_results["dataset"].map(
                avg_importance_per_dataset
            )
        else:
            dataset_results["avg_feature_importance"] = np.nan

        # Sort dataset results by macro F1-score (descending) for ranking
        dataset_results = dataset_results.sort_values("macro_f1", ascending=False)

        # Add ranking columns
        dataset_results["accuracy_rank"] = dataset_results["accuracy"].rank(
            ascending=False, method="min"
        ).astype(int)
        dataset_results["macro_f1_rank"] = dataset_results["macro_f1"].rank(
            ascending=False, method="min"
        ).astype(int)
        dataset_results["weighted_f1_rank"] = dataset_results["weighted_f1"].rank(
            ascending=False, method="min"
        ).astype(int)
        dataset_results["avg_feature_importance_rank"] = dataset_results[
            "avg_feature_importance"
        ].rank(ascending=False, method="min").astype(int)

        # Calculate average rank across all metrics (including feature importance)
        dataset_results["average_rank"] = dataset_results[
            ["accuracy_rank", "macro_f1_rank", "weighted_f1_rank", "avg_feature_importance_rank"]
        ].mean(axis=1)
        dataset_results = dataset_results.sort_values("average_rank")

        # Print ranking summary (but don't save - calling script handles saving)
        logger.info("\n=== PERFORMANCE RANKING SUMMARY ===")
        logger.info("Ranked by average rank across all metrics (lower is better):")
        for idx, row in dataset_results.iterrows():
            logger.info(
                f"  {row['dataset']}: Avg Rank={row['average_rank']:.1f}, "
                f"Accuracy={row['accuracy']:.4f} (rank {row['accuracy_rank']}), "
                f"Macro F1={row['macro_f1']:.4f} (rank {row['macro_f1_rank']}), "
                f"Weighted F1={row['weighted_f1']:.4f} (rank {row['weighted_f1_rank']}), "
                f"Avg Feature Importance={row['avg_feature_importance']:.4f} (rank {row['avg_feature_importance_rank']})"
            )

    return results_df


def compute_rare_celltype_marker_coverage(
    adata_ref: sc.AnnData,
    panel_genes: list[str],
    celltype_col: str,
    rare_threshold_fraction: float = 0.01,
    rare_threshold_absolute: int = 50,
    top_n_deg: int = 50,
    log2fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    deg_key: str = "rank_genes_groups",
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Compute how many exclusive DEG markers for rare cell types are in the panel.

    For each rare cell type (below frequency or absolute-count thresholds), counts
    how many panel genes are exclusive markers (top DEGs with sufficient FC and
    significance). Returns both per-CT counts and a panel-level coverage fraction.

    Args:
        adata_ref: Reference AnnData with expression in .X and cell type labels in obs.
        panel_genes: List of gene names in the evaluated panel.
        celltype_col: Column in adata_ref.obs with cell type labels.
        rare_threshold_fraction: Cell types with < this fraction of total cells are rare.
        rare_threshold_absolute: Cell types with fewer than this many cells are rare.
        top_n_deg: Maximum number of top DEGs per cell type to consider.
        log2fc_threshold: Minimum log2 fold-change for a gene to be an exclusive marker.
        pval_threshold: Maximum adjusted p-value for a gene to be an exclusive marker.
        deg_key: Key in adata_ref.uns where rank_genes_groups results are stored.
        force_recompute: If True, recompute DEGs even if deg_key exists in adata_ref.uns.

    Returns:
        Dictionary with:
            - rare_celltypes: list of rare cell type names
            - n_rare_celltypes: count of rare cell types
            - per_rare_ct: dict mapping each rare CT to coverage stats
            - panel_fraction_rare_ct_covered: fraction of rare CTs with >=1 panel marker
            - panel_n_rare_ct_covered: count of rare CTs with >=1 panel marker
    """
    if celltype_col not in adata_ref.obs.columns:
        raise ValueError(f"celltype_col '{celltype_col}' not found in adata_ref.obs")

    ct_counts = adata_ref.obs[celltype_col].value_counts()
    n_total = int(ct_counts.sum())

    rare_celltypes = [
        ct for ct, n in ct_counts.items()
        if n < rare_threshold_absolute or n < n_total * rare_threshold_fraction
    ]

    if not rare_celltypes:
        logger.info("No rare cell types found with current thresholds.")
        return {
            "rare_celltypes": [],
            "n_rare_celltypes": 0,
            "per_rare_ct": {},
            "panel_fraction_rare_ct_covered": float("nan"),
            "panel_n_rare_ct_covered": 0,
        }

    # Obtain DEG results — compute on a copy if missing or forced
    if deg_key not in adata_ref.uns or force_recompute:
        logger.warning(
            f"DEG key '{deg_key}' not found in adata_ref.uns (or force_recompute=True). "
            "Computing DEGs via wilcoxon on a copy — this may be slow."
        )
        adata_copy = adata_ref.copy()
        sc.tl.rank_genes_groups(
            adata_copy,
            groupby=celltype_col,
            method="wilcoxon",
            key_added=deg_key,
            use_raw=False,
        )
        deg_uns = adata_copy.uns[deg_key]
    else:
        deg_uns = adata_ref.uns[deg_key]

    panel_gene_set = set(panel_genes)
    per_rare_ct: dict[str, dict[str, Any]] = {}

    for ct in rare_celltypes:
        ct_str = str(ct)
        n_cells = int(ct_counts[ct])

        try:
            names = np.asarray(deg_uns["names"][ct_str])
            logfcs = np.asarray(deg_uns["logfoldchanges"][ct_str])
            pvals_adj = np.asarray(deg_uns["pvals_adj"][ct_str])
        except (KeyError, TypeError):
            logger.warning(f"DEG results not found for cell type '{ct_str}', skipping.")
            per_rare_ct[ct_str] = {
                "n_exclusive_markers_total": 0,
                "n_exclusive_markers_in_panel": 0,
                "n_cells": n_cells,
                "fraction_cells": n_cells / n_total,
            }
            continue

        # Select top-N DEGs passing FC and p-value thresholds
        passing = (logfcs >= log2fc_threshold) & (pvals_adj < pval_threshold)
        exclusive_markers = list(names[:top_n_deg][passing[:top_n_deg]])

        n_exclusive_total = len(exclusive_markers)
        n_in_panel = len([g for g in exclusive_markers if g in panel_gene_set])

        per_rare_ct[ct_str] = {
            "n_exclusive_markers_total": n_exclusive_total,
            "n_exclusive_markers_in_panel": n_in_panel,
            "n_cells": n_cells,
            "fraction_cells": n_cells / n_total,
        }

    n_covered = sum(1 for v in per_rare_ct.values() if v["n_exclusive_markers_in_panel"] >= 1)
    n_rare = len(rare_celltypes)

    return {
        "rare_celltypes": [str(ct) for ct in rare_celltypes],
        "n_rare_celltypes": n_rare,
        "per_rare_ct": per_rare_ct,
        "panel_fraction_rare_ct_covered": n_covered / n_rare if n_rare > 0 else float("nan"),
        "panel_n_rare_ct_covered": n_covered,
    }
