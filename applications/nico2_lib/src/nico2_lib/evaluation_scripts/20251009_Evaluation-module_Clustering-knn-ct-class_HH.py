# STREAMLINED Assessment Functions for Benchmarking Evaluation
# ==============================================================================
# This is a streamlined function library containing ONLY the functions used by 
# benchmarking evaluation scripts (e.g., 20250901_Evaluation_*.py)
#
# - Script written by: Helene Hemmer
# - Date: 09.10.2025
# - Original script: 20250424_Assess-clustering-neighborhood-classification_v2_HH.py
# 
# - Python environment: SpaprosProbeDesign
# - Python container: python.sif
#
# ==============================================================================
# INCLUDED FUNCTIONS (organized by purpose):
# ==============================================================================
#
# 2. NEIGHBORHOOD PRESERVATION METRICS
#    - compute_neighborhood_preservation(ref_data, reduced_data, k_values, ...)
#      → Compares k-NN overlap between full and reduced gene sets
#    - evaluate_neighborhood_preservation(sets, reference_key)
#      → Evaluates preservation across multiple probesets, creates visualizations
#
# 3. CLUSTERING QUALITY METRICS  
#    - compute_clustering_similarity(reference_data, test_data)
#      → Computes ARI and NMI between clustering results
#    - evaluate_clustering_quality(sets, reference_key)
#      → Evaluates clustering across probesets, creates visualizations
#
# 4. CELLTYPE IDENTIFICATION METRICS
#    - split_train_test_sets(adata, split, seed, obs_key)
#      → Splits data into train/test sets
#    - uniform_samples(adata, ct_key, set_key, subsample, seed, celltypes)
#      → Creates uniform samples across cell types
#    - evaluate_celltype_identification(sets, reference_key, celltype_col)
#      → Trains decision tree classifiers, evaluates celltype prediction accuracy
#
# ==============================================================================

# #### Load data packages
# For single cell analysis
import scanpy as sc # type: ignore

# For linear regression model
import statsmodels.api as sm # type: ignore
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances
import scipy
from scipy.stats import ks_2samp # type: ignore

# Basic packages
import numpy as np
import pandas as pd
import os
import gc
import sys
import warnings
import traceback
import logging
import psutil
import importlib.util
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path

# Add libraries for decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import NMF


# Import utility functions from the data validation module
# Get the correct path to utility module
current_dir = Path(__file__).parent.absolute()
modules_dir = current_dir.parent  # Go up to /Code/Modules/
utility_dir = modules_dir / "Utility-module"

if str(utility_dir) not in sys.path:
    sys.path.insert(0, str(utility_dir))

# Import utility module
spec = importlib.util.spec_from_file_location(
    "utility_module",
    utility_dir / "20251009_Utility-module_HH.py"
)
utility_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility_module)

# Import the functions we need
is_anndata_raw = utility_module.is_anndata_raw
is_anndata_raw_layer = utility_module.is_anndata_raw_layer
X_is_raw = utility_module.X_is_raw

##############################################################################
# MODULE EXPORTS
##############################################################################

# Explicitly define what functions are exported when using "from module import *"
__all__ = [
    # Neighborhood preservation
    'compute_neighborhood_preservation',
    'evaluate_neighborhood_preservation',
    # Clustering quality
    'compute_clustering_similarity',
    'evaluate_clustering_quality',
    # Celltype identification
    'split_train_test_sets',
    'uniform_samples',
    'evaluate_celltype_identification',
]

##############################################################################
# MODULE CONFIGURATION
##############################################################################

# Global variables that will be set by the importing script
output_dir = None
DIMENSIONALITY_REDUCTION = "pca"  # Default, can be overridden

# ===================================================================
# LOGGING SETUP
# ===================================================================

def setup_logging(log_file):
    """Configure logging to both file and console"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def log_memory_usage(stage=""):
    """Log current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    logging.info(f"Memory usage {stage}: {mem_gb:.2f} GB")


##############################################################################
# NEIGHBORHOOD PRESERVATION METRICS
##############################################################################

def compute_neighborhood_preservation(ref_data, reduced_data, k_values=None, dimensionality_reduction="pca"):
    """
    Compute how well the k-nearest neighbors are preserved between 
    full transcriptome and reduced gene set across multiple k values.
    
    Parameters:
    -----------
    ref_data : AnnData
        Reference AnnData object (full transcriptome)
    reduced_data : AnnData
        Reduced gene set AnnData object
    k_values : list, optional
        List of k values to evaluate. If None, uses [5, 10, 15, 20, 30, 50]
    dimensionality_reduction : str, optional
        Which dimensionality reduction was used: "pca", "nmf", or "both" (default: "pca")
        
    Returns:
    --------
    dict
        Dictionary with preservation scores for each k value and the optimal k.
        If dimensionality_reduction="both", returns nested dict with 'pca' and 'nmf' keys,
        each containing their own results.
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 30, 50]
    
    # Match cell barcodes between datasets
    common_cells = list(set(ref_data.obs_names).intersection(set(reduced_data.obs_names)))
    
    if len(common_cells) == 0:
        logging.info("No common cells found between reference and test data")
        return {'optimal_k': None, 'scores': {}, 'best_score': np.nan}
    
    # Get reference and test data subsets for common cells
    ref_subset = ref_data[common_cells]
    reduced_subset = reduced_data[common_cells]
    n_cells = len(common_cells)
    
    # Run garbage collection to ensure we have memory available for this operation
    gc.collect()
    
    # Determine which representations to evaluate
    if dimensionality_reduction == "both":
        representations = ['pca', 'nmf']
        logging.info("Computing neighborhood preservation for both PCA and NMF representations")
    elif dimensionality_reduction == "nmf":
        representations = ['nmf']
    else:  # pca only (default)
        representations = ['pca']
    
    # Process each representation
    all_results = {}
    
    for rep in representations:
        if len(representations) > 1:
            logging.info(f"\n--- Evaluating {rep.upper()} representation ---")
        
        knn_overlap_scores = {}
        
        # For each k value, compute the preservation score
        for k in k_values:
            logging.info(f"Computing neighborhood preservation for k={k} ({rep.upper() if len(representations) > 1 else ''})...")
            
            # Construct the correct keys based on representation
            # For single-mode runs (pca or nmf only), the keys don't have the rep suffix
            # For "both" mode runs, they do have the suffix
            if rep == "nmf":
                # Try with suffix first (from "both" mode), then without (from "nmf" only mode)
                possible_keys = [
                    (f'neighbors_nmf_k{k}_connectivities' if k != 15 else 'neighbors_nmf_connectivities'),
                    (f'neighbors_k{k}_connectivities' if k != 15 else 'connectivities')
                ]
            else:  # pca
                # Try with suffix first (from "both" mode), then without (from "pca" only mode)
                possible_keys = [
                    (f'neighbors_pca_k{k}_connectivities' if k != 15 else 'neighbors_pca_connectivities'),
                    (f'neighbors_k{k}_connectivities' if k != 15 else 'connectivities')
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
                logging.info(f"Warning: Connectivities for k={k} not found in one or both datasets")
                logging.info(f"  Looking for one of: {possible_keys}")
                logging.info(f"  Available in ref: {list(ref_subset.obsp.keys())}")
                logging.info(f"  Available in reduced: {list(reduced_subset.obsp.keys())}")
                knn_overlap_scores[k] = np.nan
                continue
            
            ref_conn = ref_subset.obsp[ref_conn_key]
            reduced_conn = reduced_subset.obsp[reduced_conn_key]
            
            # Calculate neighborhood overlap for each cell
            knn_overlap_sum = 0
            
            for i in range(n_cells):
                # Get indices of k nearest neighbors in reference
                ref_neighbors = ref_conn[i].indices[:k] if ref_conn[i].indices.size >= k else ref_conn[i].indices
                
                # Get indices of k nearest neighbors in reduced
                reduced_neighbors = reduced_conn[i].indices[:k] if reduced_conn[i].indices.size >= k else reduced_conn[i].indices
                
                # Compute overlap (Jaccard index)
                intersection = len(set(ref_neighbors).intersection(set(reduced_neighbors)))
                union = len(set(ref_neighbors).union(set(reduced_neighbors)))
                overlap = intersection / union if union > 0 else 0
                knn_overlap_sum += overlap
            
            # Store average overlap for this k
            knn_overlap_scores[k] = knn_overlap_sum / n_cells
            logging.info(f"Average neighborhood preservation for k={k}: {knn_overlap_scores[k]:.4f}")
        
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
            'optimal_k': optimal_k,
            'scores': knn_overlap_scores,
            'best_score': best_score
        }
    
    # Return nested dict if multiple representations, flat dict if single representation
    if len(representations) > 1:
        return all_results
    else:
        return all_results[representations[0]]

def evaluate_neighborhood_preservation(sets, reference_key='full_transcriptome'):
    """
    Evaluate neighborhood preservation across all genesets compared to reference data.
    
    Parameters:
    -----------
    sets : dict
        Dictionary of AnnData objects for different genesets
    reference_key : str
        Key for the reference dataset in the sets dictionary
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with neighborhood preservation metrics for all genesets
    """

    # Initialize results storage
    results = []
    
    # Reference data
    reference_data = sets[reference_key]
    
    # Evaluate each geneset
    for dataset_name, dataset in sets.items():
        logging.info(f"Processing dataset: {dataset_name}")
        
        # For h5ad files, use the data directly with process_data
        if isinstance(dataset, sc.AnnData):
            logging.info(f"Dataset {dataset_name} is already an AnnData object")
            
            # Check if we have a counts layer
            if 'counts' in dataset.layers:
                logging.info(f"Using 'counts' layer for {dataset_name}")
                layer = "counts"
            else:
                logging.info(f"No 'counts' layer found, using X for {dataset_name}")
                # Store raw data in counts layer if not present
                dataset.layers["counts"] = dataset.X.copy()
                layer = "counts"

            # Skip processing if the dataset has already been processed
            # Check for either PCA or NMF embeddings AND any neighbor graph
            has_embeddings = 'X_pca' in dataset.obsm or 'X_nmf' in dataset.obsm
            # Check for neighbor graphs - can be 'neighbors_pca_k5', 'neighbors_nmf_k10', etc.
            has_neighbors = any('neighbors' in k and '_k' in k for k in dataset.uns.keys())
            
            if has_embeddings and has_neighbors:
                logging.info(f"Dataset {dataset_name} is preprocessed and ready for evaluation")
            else:
                # Dataset needs processing - this should have been done in preprocessing step
                logging.info(f"WARNING: Dataset {dataset_name} may not be fully preprocessed.")
                logging.info(f"         Expected preprocessing to include embeddings and neighbor graphs.")
                logging.info(f"         Has embeddings: {has_embeddings}, Has neighbors: {has_neighbors}")
                logging.info(f"         Attempting to proceed with evaluation anyway...")

        else:
            logging.info(f"Unknown dataset type for {dataset_name}, skipping")
        
        # Run garbage collection after processing each dataset to free memory
        gc.collect()
        
        if dataset_name == reference_key:
            continue
        
        logging.info(f"\nEvaluating neighborhood preservation for: {dataset_name}")
        
        # Compute neighborhood preservation across different k values
        # Pass the dimensionality reduction method
        preservation_results = compute_neighborhood_preservation(
            reference_data, 
            dataset,
            dimensionality_reduction=DIMENSIONALITY_REDUCTION
        )
        
        # Handle both flat (single representation) and nested (both representations) results
        if DIMENSIONALITY_REDUCTION == "both":
            # Nested results with 'pca' and 'nmf' keys
            for rep in ['pca', 'nmf']:
                if rep in preservation_results:
                    # Store results for each k value
                    for k, score in preservation_results[rep]['scores'].items():
                        results.append({
                            'dataset': dataset_name,
                            'representation': rep,
                            'k': k,
                            'preservation_score': score,
                            'is_optimal': k == preservation_results[rep]['optimal_k']
                        })
                    
                    # Add a summary row
                    results.append({
                        'dataset': dataset_name,
                        'representation': rep,
                        'k': 'optimal',
                        'preservation_score': preservation_results[rep]['best_score'],
                        'optimal_k': preservation_results[rep]['optimal_k']
                    })
        else:
            # Flat results (single representation)
            # Store results for each k value
            for k, score in preservation_results['scores'].items():
                results.append({
                    'dataset': dataset_name,
                    'representation': DIMENSIONALITY_REDUCTION,
                    'k': k,
                    'preservation_score': score,
                    'is_optimal': k == preservation_results['optimal_k']
                })
            
            # Add a summary row
            results.append({
                'dataset': dataset_name,
                'representation': DIMENSIONALITY_REDUCTION,
                'k': 'optimal',
                'preservation_score': preservation_results['best_score'],
                'optimal_k': preservation_results['optimal_k']
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Return results to calling function for saving
    # Note: The calling script (Run-Evaluation.py) handles saving and plotting
    
    return results_df

##############################################################################
# CLUSTERING QUALITY METRICS
##############################################################################

def compute_clustering_similarity(reference_data, test_data):
    """
    Compute the similarity between clustering results of reference and test datasets.
    Uses Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) to compare clusters.
    
    Parameters:
    -----------
    reference_data : AnnData
        Reference AnnData object with Leiden clustering results
    test_data : AnnData
        Test AnnData object with Leiden clustering results
    
    Returns:
    --------
    dict
        Dictionary with ARI and NMI scores for different cluster resolutions.
        Structure: {'ari': {...}, 'nmi': {...}}
        If both PCA and NMF were used, nested dict with 'pca' and 'nmf' keys within each metric.
    """
    # Match cell barcodes between datasets
    common_cells = list(set(reference_data.obs_names).intersection(set(test_data.obs_names)))
    
    if len(common_cells) == 0:
        logging.info("No common cells found between reference and test data")
        return {}
    
    # Get reference and test data subsets for common cells
    ref_subset = reference_data[common_cells]
    test_subset = test_data[common_cells]
    
    # Check if we have both PCA and NMF results (indicated by _pca and _nmf suffixes)
    test_leiden_cols = [col for col in test_subset.obs.columns if col.startswith('leiden_') and '_clusters' in col]
    has_pca_results = any('_pca' in col for col in test_leiden_cols)
    has_nmf_results = any('_nmf' in col for col in test_leiden_cols)
    
    # Initialize results dictionary for both metrics
    ari_scores = {}
    nmi_scores = {}
    
    # Determine representations to evaluate
    if has_pca_results and has_nmf_results:
        representations = ['pca', 'nmf']
        logging.info("Detected both PCA and NMF clustering results - evaluating both")
    elif has_nmf_results:
        representations = ['nmf']
    elif has_pca_results:
        representations = ['pca']
    else:
        representations = [None]  # Old format without suffix
    
    for rep in representations:
        if rep is not None:
            logging.info(f"\n--- Evaluating {rep.upper()} clustering ---")
            rep_results_ari = {}
            rep_results_nmi = {}
        else:
            rep_results_ari = ari_scores
            rep_results_nmi = nmi_scores
        
        # Get leiden cluster columns for this representation
        if rep is not None:
            ref_leiden_cols = [col for col in ref_subset.obs.columns 
                             if col.startswith('leiden_') and '_clusters' in col and f'_{rep}' in col]
            test_leiden_cols_rep = [col for col in test_leiden_cols if f'_{rep}' in col]
        else:
            ref_leiden_cols = [col for col in ref_subset.obs.columns 
                             if col.startswith('leiden_') and '_clusters' in col 
                             and '_pca' not in col and '_nmf' not in col]
            test_leiden_cols_rep = [col for col in test_leiden_cols 
                                  if '_pca' not in col and '_nmf' not in col]
        
        # For each reference cluster resolution
        for ref_col in ref_leiden_cols:
            # Extract cluster count
            try:
                parts = ref_col.split('_')
                # Handle both formats: leiden_15_clusters_pca and leiden_15_clusters
                if rep is not None:
                    ref_n_clusters = int(parts[1])
                else:
                    ref_n_clusters = int(parts[1])
            except (IndexError, ValueError):
                continue
            
            # Find matching test column with same number of clusters
            if rep is not None:
                pattern = f'leiden_{ref_n_clusters}_clusters_{rep}'
            else:
                pattern = f'leiden_{ref_n_clusters}_clusters'
            
            matching_col = [col for col in test_leiden_cols_rep if pattern in col]
            
            if not matching_col:
                continue
            
            test_col = matching_col[0]
            
            # Calculate Adjusted Rand Index and Normalized Mutual Information
            ari = adjusted_rand_score(
                ref_subset.obs[ref_col].astype(str),
                test_subset.obs[test_col].astype(str)
            )
            nmi = normalized_mutual_info_score(
                ref_subset.obs[ref_col].astype(str),
                test_subset.obs[test_col].astype(str)
            )
            
            rep_results_ari[ref_n_clusters] = ari
            rep_results_nmi[ref_n_clusters] = nmi
            logging.info(f"ARI for {ref_n_clusters} clusters: {ari:.4f}")
            logging.info(f"NMI for {ref_n_clusters} clusters: {nmi:.4f}")
        
        # Store results for this representation
        if rep is not None:
            ari_scores[rep] = rep_results_ari
            nmi_scores[rep] = rep_results_nmi
    
    # Also calculate ARI and NMI for default leiden clustering
    if 'leiden' in ref_subset.obs and 'leiden' in test_subset.obs:
        default_ari = adjusted_rand_score(
            ref_subset.obs['leiden'].astype(str),
            test_subset.obs['leiden'].astype(str)
        )
        default_nmi = normalized_mutual_info_score(
            ref_subset.obs['leiden'].astype(str),
            test_subset.obs['leiden'].astype(str)
        )
        if has_pca_results and has_nmf_results:
            # In both mode, store default separately
            ari_scores['default'] = default_ari
            nmi_scores['default'] = default_nmi
        else:
            ari_scores['default'] = default_ari
            nmi_scores['default'] = default_nmi
        logging.info(f"Default leiden ARI: {default_ari:.4f}")
        logging.info(f"Default leiden NMI: {default_nmi:.4f}")
    
    return {'ari': ari_scores, 'nmi': nmi_scores}

def evaluate_clustering_quality(sets, reference_key='full_transcriptome'):
    """
    Evaluate clustering quality across all genesets compared to reference data.
    
    Parameters:
    -----------
    sets : dict
        Dictionary of AnnData objects for different genesets
    reference_key : str
        Key for the reference dataset in the sets dictionary
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with clustering similarity metrics for all genesets
    """
    # Initialize results storage
    results = []
    
    # Reference data
    reference_data = sets[reference_key]
    
    # Evaluate each geneset
    for dataset_name, dataset in sets.items():
        if dataset_name == reference_key:
            continue
        
        logging.info(f"\nEvaluating clustering quality for: {dataset_name}")
        
        # Compute clustering similarity (returns dict with 'ari' and 'nmi' keys)
        clustering_scores = compute_clustering_similarity(reference_data, dataset)
        ari_scores = clustering_scores['ari']
        nmi_scores = clustering_scores['nmi']
        
        # Determine which representation(s) were used
        # Check if we have both PCA and NMF embeddings to know if we used "both" mode
        has_pca = 'X_pca' in dataset.obsm
        has_nmf = 'X_nmf' in dataset.obsm
        
        # Check if ari_scores is nested (both mode) or flat (single mode)
        if isinstance(ari_scores, dict) and any(k in ari_scores for k in ['pca', 'nmf']):
            # Both mode - results are nested by representation
            for rep in ['pca', 'nmf']:
                if rep in ari_scores:
                    for n_clusters in ari_scores[rep].keys():
                        ari = ari_scores[rep][n_clusters]
                        nmi = nmi_scores[rep][n_clusters]
                        results.append({
                            'dataset': dataset_name,
                            'n_clusters': n_clusters,
                            'ARI': ari,
                            'NMI': nmi,
                            'representation': rep
                        })
            # Handle default if present
            if 'default' in ari_scores:
                results.append({
                    'dataset': dataset_name,
                    'n_clusters': 'default',
                    'ARI': ari_scores['default'],
                    'NMI': nmi_scores['default'],
                    'representation': 'both'
                })
        else:
            # Single mode - results are flat
            if has_nmf:
                representation = 'nmf'
            elif has_pca:
                representation = 'pca'
            else:
                representation = 'unknown'
            
            # Store results
            for n_clusters in ari_scores.keys():
                ari = ari_scores[n_clusters]
                nmi = nmi_scores[n_clusters]
                results.append({
                    'dataset': dataset_name,
                    'n_clusters': n_clusters,
                    'ARI': ari,
                    'NMI': nmi,
                    'representation': representation
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Return results to calling function for saving
    # Note: The calling script (Run-Evaluation.py) handles saving and plotting
    
    return results_df
    #   plot_clustering_quality_nmi_heatmap(results_df, cluster_output_dir, PNG_DPI)
    
    return results_df

##############################################################################
# CELLTYPE IDENTIFICATION METRICS
##############################################################################

def split_train_test_sets(
    adata: sc.AnnData, split: int = 4, seed: int = 2020, verbose: bool = True, obs_key: str = None
) -> None:
    """Split data to train and test set.

    This function was copied from the Spapros package (Kuemmerle, Nature Methods (2024))
    
    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        split:
            Number of splits (train:test ratio will be split:1).
        seed:
            Random number seed.
        verbose:
            Verbosity level > 1.
        obs_key:
            Provide a column name of adata.obs. If an obs_key is provided each group is split with the defined ratio.
    """
    if not obs_key:
        n_train = (adata.n_obs // (split + 1)) * split
        np.random.seed(seed=seed)
        train_obs = np.random.choice(adata.n_obs, n_train, replace=False)
        test_obs = np.array([True for i in range(adata.n_obs)])
        test_obs[train_obs] = False
        train_obs = np.invert(test_obs)
        if verbose:
            logging.info(f"Split data to ratios {split}:1 (train:test)")
            logging.info(f"datapoints: {adata.n_obs}")
            logging.info(f"train data: {np.sum(train_obs)}")
            logging.info(f"test data: {np.sum(test_obs)}")
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
                logging.info(f"Split data for group {group}")
                logging.info(f"to ratios {split}:1 (train:test)")
                logging.info(f"datapoints: {n_obs}")
                logging.info(f"train data: {np.sum(train_obs)}")
                logging.info(f"test data: {np.sum(test_obs)}")
            adata.obs.loc[df.index, "train_set"] = train_obs
            adata.obs.loc[df.index, "test_set"] = test_obs


def uniform_samples(
    adata: sc.AnnData,
    ct_key: str,
    set_key: str = "train_set",
    subsample: int = 500,
    seed: int = 2020,
    celltypes: Union[list, str] = "all",
) -> tuple:
    """Subsample `subsample` cells per celltype.
    This function was copied from the Spapros package (Kuemmerle, Nature Methods (2024))
    If the number of cells of a celltype is lower we're oversampling that celltype.
    
    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        ct_key:
            Column of `adata.obs` with cell type annotation.
        set_key:
            Column of `adata.obs` with indicating the train set.
        subsample:
            Number of random choices.
        seed:
            Random number seed.
        celltypes:
            List of celltypes to consider or `all`.
            
    Returns:
        tuple: (X, y, cts) where:
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


def evaluate_celltype_identification(sets, reference_key='full_transcriptome', celltype_col='new_annot'):
    """
    Evaluate celltype identification accuracy using a decision tree classifier.
    Trains on the complete dataset and predicts celltype labels for each geneset.
    
    Parameters:
    -----------
    sets : dict
        Dictionary of AnnData objects for different genesets
    reference_key : str
        Key for the reference dataset in the sets dictionary
    celltype_col : str
        Name of the column in adata.obs containing celltype labels
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with classification accuracy metrics for all genesets
    """
    # Initialize results storage
    results = []
    feature_importances_list = []  # Store feature importances for each dataset
    
    # Get reference data
    reference_data = sets[reference_key]
    
    # Check if celltype column exists in reference data
    if celltype_col not in reference_data.obs.columns:
        logging.error(f"Error: celltype column '{celltype_col}' not found in reference data")
        logging.info(f"Available columns: {list(reference_data.obs.columns)}")
        return pd.DataFrame()
    
    # Iterate through all genesets
    for dataset_name, dataset in sets.items():
        if dataset_name == reference_key:
            continue
        
        logging.info(f"\nEvaluating celltype identification for: {dataset_name}")
        
        # Find common cells between reference and test dataset
        common_cells = list(set(reference_data.obs_names).intersection(set(dataset.obs_names)))
        
        if len(common_cells) == 0:
            logging.warning(f"No common cells found between reference and {dataset_name}")
            continue
        
        # Get reference and test data subsets for common cells
        ref_subset = reference_data[common_cells].copy()
        test_subset = dataset[common_cells].copy() # It is called test_subset but it is actually the original anndata of the evaluated panel subsetted to common cells with the reference data

        # Check if adata.X is raw
        is_raw_X = is_anndata_raw(test_subset)

        if not is_raw_X:
            logging.info(f"Using log normalized data for testing cell type classification accuracy")
        else:
            logging.info(f"No log normalized data available, using raw data instead")
        
        # Get celltype labels from reference data and add to test_subset
        try:
            test_subset.obs[celltype_col] = ref_subset.obs[celltype_col].values
        except KeyError as e:
            logging.error(f"KeyError accessing celltype column: {e}")
            continue
        
        # Apply train/test split (like the reference method)
        # Use 4:1 split ratio (80% train, 20% test) stratified by celltype
        split_train_test_sets(test_subset, split=4, seed=42, verbose=False, obs_key=celltype_col)
        
        # Check if we have both train and test cells for all celltypes
        celltypes = test_subset.obs[celltype_col].unique()
        celltypes_with_train = test_subset.obs.loc[test_subset.obs["train_set"], celltype_col].unique()
        celltypes_with_test = test_subset.obs.loc[test_subset.obs["test_set"], celltype_col].unique()
        
        valid_celltypes = [ct for ct in celltypes 
                          if ct in celltypes_with_train and ct in celltypes_with_test]
        
        if len(valid_celltypes) == 0:
            logging.warning(f"No celltypes with both train and test samples in {dataset_name}")
            continue
                
        try:
            # Use uniform sampling to get balanced train and test sets
            subsample_train = 500  # Number of cells per celltype for training
            subsample_test = 500   # Number of cells per celltype for testing
            
            # Get uniformly sampled training data
            # Note: uniform_samples returns (X, y_dict, cts) where:
            #   - X: expression matrix
            #   - y_dict: binary labels per celltype (for one-vs-rest, not used here)
            #   - cts: actual celltype labels (used for multi-class classification)
            X_train, _, y_train = uniform_samples(
                test_subset, # It is called test_subset but it is actually the original anndata of the evaluated panel subsetted to common cells with the reference data
                ct_key=celltype_col,
                set_key="train_set",
                subsample=subsample_train,
                seed=42,
                celltypes=valid_celltypes
            )
            
            # Get uniformly sampled test data
            X_test, _, y_test = uniform_samples(
                test_subset,
                ct_key=celltype_col,
                set_key="test_set",
                subsample=subsample_test,
                seed=42,
                celltypes=valid_celltypes
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
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                
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
            feature_importance_df = pd.DataFrame({
                'gene': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Store features for this dataset
            for idx, row in feature_importance_df.iterrows():
                feature_importances_list.append({
                    'dataset': dataset_name,
                    'gene': row['gene'],
                    'importance': row['importance'],
                    'rank': idx + 1
                })
            
            # Note: Feature importance saving is handled by calling script
            
            logging.info(f"Top 5 important genes for {dataset_name}: {', '.join(feature_importance_df['gene'].head(5).tolist())}")
            
            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Extract macro F1-score for display
            macro_f1 = class_report['macro avg']['f1-score']
            
            # Note: Confusion matrix and decision tree plotting have been moved to plotting module
            # To create these visualizations, use:
            #   from plotting_module import plot_confusion_matrix, plot_decision_tree_visualization
            #   plot_confusion_matrix(y_test, y_pred, dt_classifier.classes_, dataset_name, 
            #                        accuracy, macro_f1, dt_output_dir, PNG_DPI)
            #   plot_decision_tree_visualization(dt_classifier, list(test_subset.var_names), 
            #                                   list(dt_classifier.classes_), dataset_name, 
            #                                   dt_output_dir, PNG_DPI)
            
            # Store results - include both macro and weighted F1 scores
            results.append({
                'dataset': dataset_name,
                'accuracy': accuracy,
                'best_max_depth': best_depth,
                'n_genes': len(test_subset.var_names),
                'macro_f1': class_report['macro avg']['f1-score'],
                'weighted_f1': class_report['weighted avg']['f1-score'],
                'weighted_precision': class_report['weighted avg']['precision'],
                'weighted_recall': class_report['weighted avg']['recall']
            })
            
            # Also store per-class metrics
            for celltype in dt_classifier.classes_:
                if celltype in class_report:
                    results.append({
                        'dataset': dataset_name,  # Keep dataset name separate (don't append celltype)
                        'celltype': celltype,
                        'accuracy': accuracy,  # Overall accuracy
                        'precision': class_report[celltype]['precision'],
                        'recall': class_report[celltype]['recall'],
                        'f1-score': class_report[celltype]['f1-score'],
                        'support': class_report[celltype]['support']
                    })
            
            logging.info(f"Celltype identification accuracy for {dataset_name}: {accuracy:.4f}")
            logging.info(f"Best max_depth parameter: {best_depth}")
            logging.info(f"Macro F1 score: {class_report['macro avg']['f1-score']:.4f}")
            logging.info(f"Weighted F1 score: {class_report['weighted avg']['f1-score']:.4f}")
            
        except Exception as e:
            logging.error(f"Error in model training for {dataset_name}: {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average feature importance per dataset (but don't save separately)
    if feature_importances_list:
        feature_importances_df = pd.DataFrame(feature_importances_list)
        avg_importance_per_dataset = feature_importances_df.groupby('dataset')['importance'].mean().to_dict()
    else:
        avg_importance_per_dataset = {}
    
    # Return results to calling function for saving
    # Note: The calling script (Run-Evaluation.py) handles saving and plotting
    if not results_df.empty:
        # Check if we have celltype-specific results (which contain celltype column)
        if 'celltype' in results_df.columns:
            # Filter for only the dataset-level results (not per-celltype)
            dataset_results = results_df[pd.isna(results_df.get('celltype'))].copy()
            # Filter for per-celltype results (for heatmap)
            celltype_results = results_df[results_df['dataset'].str.contains('_')].copy()
        else:
            # If no celltype column, assume all rows are dataset-level
            dataset_results = results_df.copy()
        
        # Add average feature importance to dataset_results
        if avg_importance_per_dataset:
            dataset_results['avg_feature_importance'] = dataset_results['dataset'].map(avg_importance_per_dataset)
        else:
            dataset_results['avg_feature_importance'] = np.nan
        
        # Sort dataset results by macro F1-score (descending) for ranking
        dataset_results = dataset_results.sort_values('macro_f1', ascending=False)
        
        # Add ranking columns
        dataset_results['accuracy_rank'] = dataset_results['accuracy'].rank(ascending=False, method='min').astype(int)
        dataset_results['macro_f1_rank'] = dataset_results['macro_f1'].rank(ascending=False, method='min').astype(int)
        dataset_results['weighted_f1_rank'] = dataset_results['weighted_f1'].rank(ascending=False, method='min').astype(int)
        dataset_results['avg_feature_importance_rank'] = dataset_results['avg_feature_importance'].rank(ascending=False, method='min').astype(int)
        
        # Calculate average rank across all metrics (including feature importance)
        dataset_results['average_rank'] = dataset_results[['accuracy_rank', 'macro_f1_rank', 'weighted_f1_rank', 'avg_feature_importance_rank']].mean(axis=1)
        dataset_results = dataset_results.sort_values('average_rank')
        
        # Print ranking summary (but don't save - calling script handles saving)
        logging.info("\n=== PERFORMANCE RANKING SUMMARY ===")
        logging.info("Ranked by average rank across all metrics (lower is better):")
        for idx, row in dataset_results.iterrows():
            logging.info(f"  {row['dataset']}: Avg Rank={row['average_rank']:.1f}, "
                       f"Accuracy={row['accuracy']:.4f} (rank {row['accuracy_rank']}), "
                       f"Macro F1={row['macro_f1']:.4f} (rank {row['macro_f1_rank']}), "
                       f"Weighted F1={row['weighted_f1']:.4f} (rank {row['weighted_f1_rank']}), "
                       f"Avg Feature Importance={row['avg_feature_importance']:.4f} (rank {row['avg_feature_importance_rank']})")
        
        # Note: The calling script (Run-Evaluation.py) handles saving and plotting
        
    return results_df
