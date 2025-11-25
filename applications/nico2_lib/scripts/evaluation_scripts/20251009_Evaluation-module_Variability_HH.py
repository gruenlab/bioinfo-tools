#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NMF Evaluation Module for Variability Estimation
# ==============================================================================
# This is a streamlined function library for NMF-based probeset evaluation
#
# - Script written by: Helene Hemmer
# - Date: 09.10.2025
# - Based on: 20250918_NMF_Evaluation-metrics_DG-method-Saturation-curve-v2_HH.py
# 
# - Python environment: SpaprosProbeDesign
# - Python container: python.sif
#
# ==============================================================================
# INCLUDED FUNCTIONS:
# ==============================================================================
#
# 1. METRIC CALCULATION
#    - calculate_mse(X_original, X_reconstructed)
#      → Computes Mean Squared Error between original and reconstructed data
#    - calculate_explained_variance(X_original, X_reconstructed)
#      → Computes explained variance as 1 - (residual variance / total variance)
#    - calculate_macro_mse(celltype_results)
#      → Computes macro-averaged MSE across cell types
#    - calculate_macro_explained_variance(celltype_results)
#      → Computes macro-averaged explained variance across cell types
#
# 2. MECHANISTIC REPRESENTATION EVALUATION
#    - evaluate_mechanistic_representation(adata, probeset_genes, A_train, A_test, ...)
#      → Evaluates how well probe-derived biological factors can explain full transcriptome
#    - evaluate_mechanistic_representation_by_celltype(adata, probeset_genes, ...)
#      → Same as above but performed separately for each cell type
#
# 3. MAPPING PERFORMANCE EVALUATION
#    - evaluate_mapping_performance(adata, probeset_genes, A_train, A_test, ...)
#      → Evaluates how well probe patterns can map to full transcriptome
#    - evaluate_mapping_performance_by_celltype(adata, probeset_genes, ...)
#      → Same as above but performed separately for each cell type
# ==============================================================================

# Import required packages
import numpy as np
import pandas as pd
import scipy
import sys
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
# When calculating the explained variance score with sklearn
from sklearn.metrics import explained_variance_score
from tqdm import tqdm
import logging
import psutil
import os
import gc

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
# MODULE EXPORTS
##############################################################################

__all__ = [
    # Metric calculation
    'calculate_mse',
    'calculate_explained_variance',
    'calculate_macro_mse',
    'calculate_macro_explained_variance',
    # Mechanistic representation
    'evaluate_mechanistic_representation',
    'evaluate_mechanistic_representation_by_celltype',
    # Mapping performance
    'evaluate_mapping_performance',
    'evaluate_mapping_performance_by_celltype',
]

##############################################################################
# METRIC CALCULATION FUNCTIONS
##############################################################################

def calculate_mse(X_original, X_reconstructed):
    """
    Calculate Mean Squared Error between original and reconstructed data matrices.
    
    Parameters:
    -----------
    X_original : array-like
        Original data matrix
    X_reconstructed : array-like
        Reconstructed data matrix from NMF
        
    Returns:
    --------
    float
        Mean squared error value
    """
    # Convert sparse matrices to dense if needed
    if scipy.sparse.issparse(X_original):
        X_original = X_original.toarray()
    if scipy.sparse.issparse(X_reconstructed):
        X_reconstructed = X_reconstructed.toarray()
    
    # Calculate MSE
    mse = np.mean((X_original - X_reconstructed) ** 2)
    return mse


'''
##############################################################
# R-squared function for explained variability calculation
##############################################################

def calculate_explained_variance(X_original, X_reconstructed):
    """
    Calculate explained variance (R²) between original and reconstructed data matrices.
    
    Formula: R² = 1 - (MSE / Variance)
    Where:
        - MSE = (1/n) * Σ(X_original - X_reconstructed)²
        - Variance = (1/n) * Σ(X_original - mean(X_original))²
    
    This measures the proportion of variance in the original data that is explained 
    by the reconstruction. Values closer to 1 indicate better reconstruction.
    
    Parameters:
    -----------
    X_original : array-like
        Original data matrix
    X_reconstructed : array-like
        Reconstructed data matrix from NMF
        
    Returns:
    --------
    float
        Explained variance value (between 0 and 1, can be negative for poor fits)
    """
    # Convert sparse matrices to dense if needed
    if scipy.sparse.issparse(X_original):
        X_original = X_original.toarray()
    if scipy.sparse.issparse(X_reconstructed):
        X_reconstructed = X_reconstructed.toarray()
    
    # Calculate MSE (mean squared error): (1/n) * Σ(X - X_recon)²
    mse = np.mean((X_original - X_reconstructed) ** 2)
    
    # Calculate total variance: (1/n) * Σ(X - mean(X))²
    total_variance = np.var(X_original)
    
    if total_variance == 0:
        return 0.0
    
    # R² = 1 - (MSE / Variance)
    explained_var = 1 - (mse / total_variance)
    
    return explained_var
'''

def calculate_explained_variance(X_original, X_reconstructed):
    """
    Calculate explained variance using sklearn's metric.
    This ignores constant offsets in predictions.
    """

    if scipy.sparse.issparse(X_original):
        X_original = X_original.toarray()
    if scipy.sparse.issparse(X_reconstructed):
        X_reconstructed = X_reconstructed.toarray()
    
    # Flatten to treat as one long vector
    return explained_variance_score(
        X_original.ravel(), 
        X_reconstructed.ravel()
    )


def calculate_macro_explained_variance(celltype_results):
    """
    Calculate macro-averaged explained variance across cell types.
    
    Macro average gives equal weight to each cell type, regardless of cell count.
    This is useful for understanding performance across cell types without bias
    toward more abundant cell types.
    
    Parameters:
    -----------
    celltype_results : dict
        Dictionary containing cell type-specific results with 'expvar_test_probe' values
        
    Returns:
    --------
    float
        Macro-averaged explained variance
    """
    valid_expvar = []
    
    for celltype, metrics in celltype_results.items():
        if not metrics.get('skipped', False):
            expvar = metrics.get('expvar_test_probe', np.nan)
            if not np.isnan(expvar):
                valid_expvar.append(expvar)
    
    if not valid_expvar:
        return np.nan
    
    # Simple average across cell types (equal weight per cell type)
    macro_expvar = np.mean(valid_expvar)
    
    return macro_expvar


def calculate_macro_mse(celltype_results):
    """
    Calculate macro-averaged MSE across cell types.
    
    Macro average gives equal weight to each cell type, regardless of cell count.
    
    Parameters:
    -----------
    celltype_results : dict
        Dictionary containing cell type-specific results with 'mse_test_probe' values
        
    Returns:
    --------
    float
        Macro-averaged MSE
    """
    valid_mse = []
    
    for celltype, metrics in celltype_results.items():
        if not metrics.get('skipped', False):
            mse = metrics.get('mse_test_probe', np.nan)
            if not np.isnan(mse):
                valid_mse.append(mse)
    
    if not valid_mse:
        return np.nan
    
    # Simple average across cell types (equal weight per cell type)
    macro_mse = np.mean(valid_mse)
    
    return macro_mse

def calculate_weighted_mse(celltype_results):
    """
    Calculate weighted MSE across cell types (weighted by number of cells).
    
    Parameters:
    -----------
    celltype_results : dict
        Dictionary containing cell type-specific results with 'mse_test_probe' and 'n_cells' values
        
    Returns:
    --------
    float
        Weighted MSE
    """
    valid_results = {ct: res for ct, res in celltype_results.items() 
                    if not res.get('skipped', False) and 'mse_test_probe' in res and 'n_cells' in res}
    
    if not valid_results:
        return np.nan
    
    total_cells = sum(res['n_cells'] for res in valid_results.values())
    if total_cells == 0:
        return np.nan
    
    weighted_mse = sum(res['mse_test_probe'] * res['n_cells'] for res in valid_results.values()) / total_cells
    return weighted_mse


def calculate_weighted_explained_variance(celltype_results):
    """
    Calculate weighted explained variance across cell types (weighted by number of cells).
    
    Parameters:
    -----------
    celltype_results : dict
        Dictionary containing cell type-specific results with 'expvar_test_probe' and 'n_cells' values
        
    Returns:
    --------
    float
        Weighted explained variance
    """
    valid_results = {ct: res for ct, res in celltype_results.items() 
                    if not res.get('skipped', False) and 'expvar_test_probe' in res and 'n_cells' in res}
    
    if not valid_results:
        return np.nan
    
    total_cells = sum(res['n_cells'] for res in valid_results.values())
    if total_cells == 0:
        return np.nan
    
    weighted_expvar = sum(res['expvar_test_probe'] * res['n_cells'] for res in valid_results.values()) / total_cells
    return weighted_expvar


def calculate_weighted_mse_baseline(celltype_results):
    """
    Calculate weighted MSE for baseline across cell types (weighted by number of cells).
    
    Parameters:
    -----------
    celltype_results : dict
        Dictionary containing cell type-specific results with 'mse_test_baseline' and 'n_cells' values
        
    Returns:
    --------
    float
        Weighted baseline MSE
    """
    valid_results = {ct: res for ct, res in celltype_results.items() 
                    if not res.get('skipped', False) and 'mse_test_baseline' in res and 'n_cells' in res}
    
    if not valid_results:
        return np.nan
    
    total_cells = sum(res['n_cells'] for res in valid_results.values())
    if total_cells == 0:
        return np.nan
    
    weighted_mse = sum(res['mse_test_baseline'] * res['n_cells'] for res in valid_results.values()) / total_cells
    return weighted_mse


def calculate_weighted_explained_variance_baseline(celltype_results):
    """
    Calculate weighted explained variance for baseline across cell types (weighted by number of cells).
    
    Parameters:
    -----------
    celltype_results : dict
        Dictionary containing cell type-specific results with 'expvar_test_baseline' and 'n_cells' values
        
    Returns:
    --------
    float
        Weighted baseline explained variance
    """
    valid_results = {ct: res for ct, res in celltype_results.items() 
                    if not res.get('skipped', False) and 'expvar_test_baseline' in res and 'n_cells' in res}
    
    if not valid_results:
        return np.nan
    
    total_cells = sum(res['n_cells'] for res in valid_results.values())
    if total_cells == 0:
        return np.nan
    
    weighted_expvar = sum(res['expvar_test_baseline'] * res['n_cells'] for res in valid_results.values()) / total_cells
    return weighted_expvar



##############################################################################
# MECHANISTIC REPRESENTATION EVALUATION
##############################################################################

def evaluate_mechanistic_representation(adata, probeset_genes, A_train, A_test, n_components=5, 
                                      max_iter=1000, random_state=42, cached_full_nmf=None):
    """
    Evaluate how well a probe gene subset can represent the full transcriptome through mechanistic representation.
    
    This method evaluates if the patterns/factors learned from just the probe genes (a small subset)
    can be used to reconstruct the full transcriptome when combined with those genes' activities
    in new samples.
    
    Following scikit-learn convention: 
    - A (samples/cells × features/genes): Data matrix
    - W (samples/cells × components): Sample factor matrix
    - H (components × features/genes): Feature factor matrix
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with full transcriptome
    probeset_genes : list
        List of genes in the probeset to evaluate
    A_train : array-like
        Pre-split training data (cells × genes)
    A_test : array-like
        Pre-split test data (cells × genes)
    n_components : int, optional (default: 5)
        Number of NMF components to use
    max_iter : int, optional (default: 1000)
        Maximum number of iterations for NMF
    random_state : int, optional (default: 42)
        Random state for reproducibility
    cached_full_nmf : dict, optional
        Pre-computed full NMF results to avoid recomputation
        Should contain NMF computed values for reuse
        
    Returns:
    --------
    dict: Contains MSE and explained variance metrics
    """
    logging.info(f"Evaluating mechanistic representation with {len(probeset_genes)} genes")
    
    # Get indices of probe genes
    probeset_mask = np.array([gene in probeset_genes for gene in adata.var_names])
    if sum(probeset_mask) == 0:
        logging.error(f"No probeset genes found in the dataset")
        return None
    
    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logging.info(f"Found {len(probeset_genes_found)} out of {len(probeset_genes)} probeset genes in the dataset")
    
    # Use pre-split data and ensure float64 dtype for NMF compatibility
    A_train = A_train.astype(np.float64)
    A_test = A_test.astype(np.float64)
    A_P_train = A_train[:, probe_indices]  # (cells × probe_genes)
    A_P_test = A_test[:, probe_indices]    # (cells × probe_genes)
    
    logging.info(f"Training: {A_train.shape[0]} cells × {A_train.shape[1]} genes")
    logging.info(f"Testing: {A_test.shape[0]} cells × {A_test.shape[1]} genes")
    logging.info(f"Probe subset: {len(probe_indices)} genes")
    
    # Validate dimensions
    if A_P_train.shape[1] != len(probe_indices):
        logging.error(f"Dimension mismatch: A_P_train columns ({A_P_train.shape[1]}) != probe_indices ({len(probe_indices)})")
    if A_P_test.shape[1] != len(probe_indices):
        logging.error(f"Dimension mismatch: A_P_test columns ({A_P_test.shape[1]}) != probe_indices ({len(probe_indices)})")
    
    # ================================================================
    # TRAINING PHASE
    # ================================================================
    logging.info("--- Training Phase ---")
    
    # Step 1: Full NMF on training data
    if cached_full_nmf is not None and 'training' in cached_full_nmf:
        # Use cached values if available
        logging.info("Using cached full NMF results for training data")
        
        training_cache = cached_full_nmf['training']
        # Check if the training split size matches
        if training_cache['A_train'].shape == A_train.shape:
            # Verify random seed is the same by checking if A_train matrices are identical
            if np.allclose(training_cache['A_train'], A_train):
                # Ensure cached NMF results are float64 to match converted input data
                H_full_train = training_cache['H_full_train'].astype(np.float64)
                W_full_train = training_cache['W_full_train'].astype(np.float64)
                A_train_baseline = training_cache['A_train_baseline']
                mse_train_baseline = training_cache['mse_train_baseline']
                expvar_train_baseline = training_cache['expvar_train_baseline']
                logging.info("Successfully reused cached training NMF data")
            else:
                logging.warning("Cached training data doesn't match current split, recomputing...")
                cached_full_nmf = None
        else:
            logging.warning("Cached training data shape doesn't match, recomputing...")
            cached_full_nmf = None
    
    # If cache is not available or not valid, compute NMF
    if cached_full_nmf is None or 'training' not in cached_full_nmf:
        logging.info("Computing Full NMF on training data")
        # Solve: A_train ≈ W_full_train @ H_full_train
        nmf_full_train = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
        W_full_train = nmf_full_train.fit_transform(A_train)  # (test_samples x k)
        H_full_train = nmf_full_train.components_              # (k x genes)
        
        # Calculate baseline reconstruction and metrics for training data
        A_train_baseline = W_full_train @ H_full_train
        mse_train_baseline = calculate_mse(A_train, A_train_baseline)
        expvar_train_baseline = calculate_explained_variance(A_train, A_train_baseline)
    
    # Step 2: Extract gene subset patterns
    logging.info("Step 1: Extract probe gene patterns")
    H_P = H_full_train[:, probe_indices]  # (components × probe_genes)
    
    logging.info(f"H_full_train shape: {H_full_train.shape} (components × genes)")
    logging.info(f"H_P shape: {H_P.shape} (components × probe_genes)")
    logging.info(f"W_full_train shape: {W_full_train.shape} (cells × components)")
    
    # Validate pattern extraction dimensions
    if H_P.shape[1] != len(probe_indices):
        logging.error(f"Dimension mismatch: H_P columns ({H_P.shape[1]}) != probe_indices ({len(probe_indices)})")

    # Instead of W_train = H_P^† @ A_P_train (pseudo-inverse)
    # Solve iteratively: minimize ||A_P_train - W_train @ H_P||²
    logging.info("Step 2: Iterative solve for train sample factors")
    
    # NOTE: We're using a scikit-learn internal method (_fit_transform) which is not part of the public API.
    # A more robust approach would be to implement a custom NMF solver or use a library that supports
    # fixing one of the matrices during factorization.
    nmf_constrained = NMF(n_components=n_components, init='custom', max_iter=max_iter, random_state=random_state)
    # Solve: A_P_train ≈ W_train @ H_P with H_P fixed
    W_train = nmf_constrained._fit_transform(
        A_P_train, W=None, H=H_P, update_H=False
    )[0]
    logging.info(f"W_train shape: {W_train.shape} (cells × components)")

    # ================================================================
    # TESTING PHASE: Iterative Solution
    # ================================================================
    logging.info("--- Testing Phase ---")
    
    # Check if we have cached test data
    W_full_test = None
    A_test_baseline = None
    mse_test_baseline = None
    expvar_test_baseline = None
    
    # Step 1: Full NMF on testing data
    if cached_full_nmf is not None and 'testing' in cached_full_nmf:
        # Use cached values if available
        logging.info("Using cached full NMF results for testing data")
        
        testing_cache = cached_full_nmf['testing']
        # Check if the testing split size matches
        if testing_cache['A_test'].shape == A_test.shape:
            # Verify random seed is the same by checking if A_test matrices are identical
            if np.allclose(testing_cache['A_test'], A_test):
                # Ensure cached NMF results are float64 to match converted input data
                H_full_test = testing_cache['H_full_test'].astype(np.float64)
                W_full_test = testing_cache['W_full_test'].astype(np.float64)
                A_test_baseline = testing_cache['A_test_baseline']
                mse_test_baseline = testing_cache['mse_test_baseline']
                expvar_test_baseline = testing_cache['expvar_test_baseline']
                logging.info("Successfully reused cached testing NMF data")
            else:
                logging.warning("Cached testing data doesn't match current split, recomputing...")
                cached_full_nmf['testing'] = None
        else:
            logging.warning("Cached testing data shape doesn't match, recomputing...")
            cached_full_nmf['testing'] = None
    
    # If cache is not available or not valid, compute NMF
    if cached_full_nmf is None or 'testing' not in cached_full_nmf:
        logging.info("Computing Full NMF on testing data")
        # Solve: A_test ≈ W_full_test @ H_full_test
        nmf_full_test = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
        W_full_test = nmf_full_test.fit_transform(A_test)  # (test_samples x k)
        H_full_test = nmf_full_test.components_              # (k x genes)
        
        # Calculate baseline reconstruction and metrics for testing data
        A_test_baseline = W_full_test @ H_full_test
        mse_test_baseline = calculate_mse(A_test, A_test_baseline)
        expvar_test_baseline = calculate_explained_variance(A_test, A_test_baseline)
    
    # Instead of W_test = H_P^† @ A_P_test (pseudo-inverse)
    # Solve iteratively: minimize ||A_P_test - W_test @ H_P||²
    logging.info("Step 2: Iterative solve for test sample factors")
    
    # NOTE: We're using a scikit-learn internal method (_fit_transform) which is not part of the public API.
    # A more robust approach would be to implement a custom NMF solver or use a library that supports
    # fixing one of the matrices during factorization.
    nmf_constrained = NMF(n_components=n_components, init='custom', max_iter=max_iter, random_state=random_state)
    # Solve: A_P_test ≈ W_test @ H_P with H_P fixed
    W_test = nmf_constrained._fit_transform(
        A_P_test, W=None, H=H_P, update_H=False
    )[0]
    logging.info(f"W_test shape: {W_test.shape} (cells × components)")
    
    # ================================================================
    # RECONSTRUCTION AND EVALUATION
    # ================================================================
    logging.info("--- Reconstruction and Evaluation ---")
        
    # Probe method reconstructions for both train and test
    A_train_recon = W_train @ H_full_train
    A_test_recon = W_test @ H_full_train

    # Calculate probe metrics for training data
    mse_train_probe = calculate_mse(A_train, A_train_recon)
    expvar_train_probe = calculate_explained_variance(A_train, A_train_recon)
    # Performance ratios
    mse_ratio_train = mse_train_probe / mse_train_baseline if mse_train_baseline > 0 else float('inf')
    expvar_ratio_train = expvar_train_probe / expvar_train_baseline if expvar_train_baseline > 0 else 0

    # Calculate probe metrics for testing data
    mse_test_probe = calculate_mse(A_test, A_test_recon)
    expvar_test_probe = calculate_explained_variance(A_test, A_test_recon)
    # Performance ratios
    mse_ratio = mse_test_probe / mse_test_baseline if mse_test_baseline > 0 else float('inf')
    expvar_ratio = expvar_test_probe / expvar_test_baseline if expvar_test_baseline > 0 else 0
    
    logging.info(f"Training MSE (baseline): {mse_train_baseline:.6f}")
    logging.info(f"Training MSE (probe): {mse_train_probe:.6f}")
    logging.info(f"Test MSE (baseline): {mse_test_baseline:.6f}")
    logging.info(f"Test MSE (probe): {mse_test_probe:.6f}")
    logging.info(f"MSE ratio train (probe/baseline): {mse_ratio_train:.3f}")
    logging.info(f"MSE ratio test (probe/baseline): {mse_ratio:.3f}")
    logging.info(f"ExpVar ratio train (probe/baseline): {expvar_ratio_train:.3f}")
    logging.info(f"ExpVar ratio test (probe/baseline): {expvar_ratio:.3f}")
    
    # Prepare result dict
    result = {
        'mse_train_baseline': mse_train_baseline,
        'mse_test_baseline': mse_test_baseline,
        'mse_test_probe': mse_test_probe,
        'expvar_train_baseline': expvar_train_baseline,
        'expvar_test_baseline': expvar_test_baseline,
        'expvar_test_probe': expvar_test_probe,
        'mse_ratio': mse_ratio,
        'expvar_ratio': expvar_ratio,
        'probeset_size': len(probeset_genes),
        'probeset_genes_found': len(probeset_genes_found)
    }
    
    # Add cache data if we computed it (only if not already cached)
    if cached_full_nmf is None:
        # Create new cache object
        computed_full_nmf = {
            'training': {
                'A_train': A_train,
                'W_full_train': W_full_train,
                'H_full_train': H_full_train,
                'A_train_baseline': A_train_baseline,
                'mse_train_baseline': mse_train_baseline,
                'expvar_train_baseline': expvar_train_baseline
            },
            'testing': {
                'A_test': A_test,
                'W_full_test': W_full_test,
                'H_full_test': H_full_test,
                'A_test_baseline': A_test_baseline,
                'mse_test_baseline': mse_test_baseline,
                'expvar_test_baseline': expvar_test_baseline
            }
        }
        result['computed_full_nmf'] = computed_full_nmf
    
    # Clean up temporary NMF objects to prevent memory issues
    try:
        if 'nmf_constrained' in locals():
            del nmf_constrained
        gc.collect()
        logging.debug("Cleaned up temporary NMF objects from evaluate_mechanistic_representation")
    except:
        pass
    
    return result


def evaluate_mechanistic_representation_by_celltype(adata, probeset_genes, celltype_column='celltypes_v2', 
                                                   n_components=5, max_iter=1000, random_state=42, 
                                                   cached_full_nmf_by_celltype=None, test_size=0.3):
    """
    Evaluate mechanistic representation for each celltype separately using train-test split approach.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with full transcriptome
    probeset_genes : list
        List of genes in the probeset to evaluate
    celltype_column : str
        Column name in adata.obs that contains celltype information
    n_components : int
        Number of NMF components to use
    max_iter : int
        Maximum number of iterations for NMF
    random_state : int
        Random state for reproducibility
    cached_full_nmf_by_celltype : dict, optional
        Pre-computed full NMF results by celltype to avoid recomputation
    test_size : float
        Proportion of samples to use for testing
        
    Returns:
    --------
    dict: Contains MSE and explained variance metrics by celltype
    """
    logging.info(f"Evaluating mechanistic representation by celltype with {len(probeset_genes)} genes")
    
    # Ensure we have raw counts
    if "counts" not in adata.layers:
        logging.warning("No 'counts' layer found, using .X")
        adata.layers['counts'] = adata.X.copy()
    
    # Check if celltype column exists
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Celltype column '{celltype_column}' not found in adata.obs")
    
    # Get unique celltypes
    celltypes = adata.obs[celltype_column].unique()
    logging.info(f"Found {len(celltypes)} celltypes: {list(celltypes)}")
    
    # Subset to only probeset genes
    probeset_mask = np.array([gene in probeset_genes for gene in adata.var_names])
    if sum(probeset_mask) == 0:
        raise ValueError(f"None of the probeset genes were found in the dataset")
    
    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logging.info(f"Found {len(probeset_genes_found)} out of {len(probeset_genes)} probeset genes in the dataset")
    
    # Initialize results dictionary
    celltype_results = {}
    
    # Initialize cache if not provided
    if cached_full_nmf_by_celltype is None:
        cached_full_nmf_by_celltype = {}
    
    # Process each celltype
    for celltype in tqdm(celltypes, desc="Processing celltypes"):
        logging.info(f"Processing celltype: {celltype}")
        
        # Subset data to this celltype
        celltype_mask = adata.obs[celltype_column] == celltype
        adata_celltype = adata[celltype_mask].copy()
        
        logging.info(f"Celltype {celltype}: {adata_celltype.shape[0]:,} cells × {adata_celltype.shape[1]:,} genes")
        
        # Skip celltypes with too few cells for NMF or train-test split
        min_cells_needed = max(n_components * 2, int(1 / test_size))  # Need at least double n_components and enough for test split
        if adata_celltype.shape[0] < min_cells_needed:
            logging.warning(f"Skipping celltype {celltype}: only {adata_celltype.shape[0]} cells (need at least {min_cells_needed})")
            celltype_results[celltype] = {
                'mse_train_baseline': np.nan,
                'mse_test_baseline': np.nan,
                'mse_train_probe': np.nan,
                'mse_test_probe': np.nan,
                'expvar_train_baseline': np.nan,
                'expvar_test_baseline': np.nan,
                'expvar_train_probe': np.nan,
                'expvar_test_probe': np.nan,
                'mse_ratio_train': np.nan,
                'mse_ratio_test': np.nan,
                'expvar_ratio_train': np.nan,
                'expvar_ratio_test': np.nan,
                'generalization_gap': np.nan,
                'probeset_size': len(probeset_genes),
                'probeset_genes_found': len(probeset_genes_found),
                'n_cells': adata_celltype.shape[0],
                'skipped': True,
                'skip_reason': 'insufficient_cells'
            }
            continue
        
        try:
            # Step 1: Check if celltype exists in cache
            if celltype in cached_full_nmf_by_celltype:
                logging.info(f"Found cached data for celltype {celltype}")
                cached_ct = cached_full_nmf_by_celltype[celltype]
                
                # Step 2: Check if training and testing data exists in cache
                if ('training' in cached_ct and 'A_train' in cached_ct['training'] and 
                    'testing' in cached_ct and 'A_test' in cached_ct['testing']):
                    logging.info(f"Using cached train-test split for celltype {celltype}")
                    
                    # Use cached train-test splits and ensure float64 dtype for NMF compatibility
                    A_train_ct = cached_ct['training']['A_train'].astype(np.float64)
                    A_test_ct = cached_ct['testing']['A_test'].astype(np.float64)
                    
                    # Get probe gene subsets from cached splits
                    A_P_train_ct = A_train_ct[:, probe_indices]
                    A_P_test_ct = A_test_ct[:, probe_indices]
                else:
                    # Step 2 fallback: Recompute train-test split but warn about inconsistency
                    logging.warning(f"Cached train-test split not found for celltype {celltype}, recomputing (may be inconsistent with other evaluation functions)")
                    
                    # Get data matrix for this celltype
                    if "counts" in adata_celltype.layers:
                        A_celltype = adata_celltype.layers['counts'].toarray() if scipy.sparse.issparse(adata_celltype.layers['counts']) else adata_celltype.layers['counts']
                    else:
                        A_celltype = adata_celltype.X.toarray() if scipy.sparse.issparse(adata_celltype.X) else adata_celltype.X
                    # Ensure float64 dtype for NMF compatibility
                    A_celltype = A_celltype.astype(np.float64)
                    
                    # Create new train-test split
                    A_train_ct, A_test_ct = train_test_split(A_celltype, test_size=test_size, random_state=42)
                    
                    # Get probe gene subsets from new splits
                    A_P_train_ct = A_train_ct[:, probe_indices]
                    A_P_test_ct = A_test_ct[:, probe_indices]
                    
                    logging.info(f"Training: {A_train_ct.shape[0]} cells × {A_train_ct.shape[1]} genes")
                    logging.info(f"Testing: {A_test_ct.shape[0]} cells × {A_test_ct.shape[1]} genes")
                    logging.info(f"Probe subset: {len(probe_indices)} genes")
                    
                    # Update cache with new train-test split data
                    if celltype not in cached_full_nmf_by_celltype:
                        cached_full_nmf_by_celltype[celltype] = {}
                    if 'training' not in cached_full_nmf_by_celltype[celltype]:
                        cached_full_nmf_by_celltype[celltype]['training'] = {}
                    if 'testing' not in cached_full_nmf_by_celltype[celltype]:
                        cached_full_nmf_by_celltype[celltype]['testing'] = {}
                    
                    cached_full_nmf_by_celltype[celltype]['training']['A_train'] = A_train_ct
                    cached_full_nmf_by_celltype[celltype]['testing']['A_test'] = A_test_ct
                
                # Step 3: Check if NMF W and H matrices exist in cache (refresh cache reference)
                cached_ct = cached_full_nmf_by_celltype[celltype]  # Refresh reference after potential updates
                if ('training' in cached_ct and all(k in cached_ct['training'] for k in ['W_full_train', 'H_full_train', 'mse_train_baseline', 'expvar_train_baseline']) and
                    'testing' in cached_ct and all(k in cached_ct['testing'] for k in ['W_full_test', 'H_full_test', 'mse_test_baseline', 'expvar_test_baseline'])):
                    logging.info(f"Using cached NMF results for celltype {celltype}")
                    
                    # Use cached NMF results
                    W_full_train = cached_ct['training']['W_full_train'].astype(np.float64)
                    H_full_train = cached_ct['training']['H_full_train'].astype(np.float64)
                    mse_train_baseline = cached_ct['training']['mse_train_baseline']
                    expvar_train_baseline = cached_ct['training']['expvar_train_baseline']
                    
                    W_full_test = cached_ct['testing']['W_full_test'].astype(np.float64)
                    H_full_test = cached_ct['testing']['H_full_test'].astype(np.float64)
                    mse_test_baseline = cached_ct['testing']['mse_test_baseline']
                    expvar_test_baseline = cached_ct['testing']['expvar_test_baseline']
                else:
                    # Step 3 fallback: Recompute NMF using train-test data from step 2
                    logging.info(f"Cached NMF results not found for celltype {celltype}, recomputing NMF")
                    
                    # Compute full NMF for train and test
                    logging.info("Computing Full NMF on training data")
                    nmf_full_train = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                    W_full_train = nmf_full_train.fit_transform(A_train_ct)
                    H_full_train = nmf_full_train.components_
                    A_train_baseline = W_full_train @ H_full_train
                    mse_train_baseline = calculate_mse(A_train_ct, A_train_baseline)
                    expvar_train_baseline = calculate_explained_variance(A_train_ct, A_train_baseline)
                    
                    logging.info("Computing Full NMF on testing data")  
                    nmf_full_test = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                    W_full_test = nmf_full_test.fit_transform(A_test_ct)
                    H_full_test = nmf_full_test.components_
                    A_test_baseline = W_full_test @ H_full_test
                    mse_test_baseline = calculate_mse(A_test_ct, A_test_baseline)
                    expvar_test_baseline = calculate_explained_variance(A_test_ct, A_test_baseline)
                    
                    # Update cache with new NMF results
                    if celltype not in cached_full_nmf_by_celltype:
                        cached_full_nmf_by_celltype[celltype] = {}
                    
                    cached_full_nmf_by_celltype[celltype]['training'] = {
                        'A_train': A_train_ct,
                        'W_full_train': W_full_train,
                        'H_full_train': H_full_train,
                        'A_train_baseline': A_train_baseline,
                        'mse_train_baseline': mse_train_baseline,
                        'expvar_train_baseline': expvar_train_baseline
                    }
                    cached_full_nmf_by_celltype[celltype]['testing'] = {
                        'A_test': A_test_ct,
                        'W_full_test': W_full_test,
                        'H_full_test': H_full_test,
                        'A_test_baseline': A_test_baseline,
                        'mse_test_baseline': mse_test_baseline,
                        'expvar_test_baseline': expvar_test_baseline
                    }
            else:
                # Step 1 fallback: No cache data found for this celltype, compute everything from scratch
                logging.warning(f"No cached data found for celltype {celltype}, computing everything from scratch")
                
                # Get data matrix for this celltype
                if "counts" in adata_celltype.layers:
                    A_celltype = adata_celltype.layers['counts'].toarray() if scipy.sparse.issparse(adata_celltype.layers['counts']) else adata_celltype.layers['counts']
                else:
                    A_celltype = adata_celltype.X.toarray() if scipy.sparse.issparse(adata_celltype.X) else adata_celltype.X
                
                # Create train-test split for this celltype
                A_train_ct, A_test_ct = train_test_split(A_celltype, test_size=test_size, random_state=42)
                
                # Get probe gene subsets from train-test splits
                A_P_train_ct = A_train_ct[:, probe_indices]
                A_P_test_ct = A_test_ct[:, probe_indices]
                
                logging.info(f"Training: {A_train_ct.shape[0]} cells × {A_train_ct.shape[1]} genes")
                logging.info(f"Testing: {A_test_ct.shape[0]} cells × {A_test_ct.shape[1]} genes")
                logging.info(f"Probe subset: {len(probe_indices)} genes")
                
                # Compute full NMF for train and test
                logging.info("Computing Full NMF on training data")
                nmf_full_train = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                W_full_train = nmf_full_train.fit_transform(A_train_ct)
                H_full_train = nmf_full_train.components_
                A_train_baseline = W_full_train @ H_full_train
                mse_train_baseline = calculate_mse(A_train_ct, A_train_baseline)
                expvar_train_baseline = calculate_explained_variance(A_train_ct, A_train_baseline)
                
                logging.info("Computing Full NMF on testing data")  
                nmf_full_test = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                W_full_test = nmf_full_test.fit_transform(A_test_ct)
                H_full_test = nmf_full_test.components_
                A_test_baseline = W_full_test @ H_full_test
                mse_test_baseline = calculate_mse(A_test_ct, A_test_baseline)
                expvar_test_baseline = calculate_explained_variance(A_test_ct, A_test_baseline)
                
                # Cache results for potential reuse
                cached_full_nmf_by_celltype[celltype] = {
                    'training': {
                        'A_train': A_train_ct,
                        'W_full_train': W_full_train,
                        'H_full_train': H_full_train,
                        'A_train_baseline': A_train_baseline,
                        'mse_train_baseline': mse_train_baseline,
                        'expvar_train_baseline': expvar_train_baseline
                    },
                    'testing': {
                        'A_test': A_test_ct,
                        'W_full_test': W_full_test,
                        'H_full_test': H_full_test,
                        'A_test_baseline': A_test_baseline,
                        'mse_test_baseline': mse_test_baseline,
                        'expvar_test_baseline': expvar_test_baseline
                    }
                }
            

            
            # Apply mechanistic representation approach
            logging.info(f"Running mechanistic representation for celltype {celltype}")
            
            # Step 1: Extract gene subset patterns from full training NMF
            H_P = H_full_train[:, probe_indices]  # (components × probe_genes)
            
            # Step 2: Solve for W_train with H_P fixed
            nmf_constrained_train = NMF(n_components=n_components, init='custom', max_iter=max_iter, random_state=random_state)
            W_train = nmf_constrained_train._fit_transform(
                A_P_train_ct, W=None, H=H_P, update_H=False
            )[0]
            
            # Step 3: Solve for W_test with H_P fixed
            nmf_constrained_test = NMF(n_components=n_components, init='custom', max_iter=max_iter, random_state=random_state)
            W_test = nmf_constrained_test._fit_transform(
                A_P_test_ct, W=None, H=H_P, update_H=False
            )[0]
            
            # Step 4: Reconstruct full transcriptome
            A_train_recon = W_train @ H_full_train
            A_test_recon = W_test @ H_full_train
            
            # Calculate metrics
            mse_train_probe = calculate_mse(A_train_ct, A_train_recon)
            expvar_train_probe = calculate_explained_variance(A_train_ct, A_train_recon)
            
            mse_test_probe = calculate_mse(A_test_ct, A_test_recon)
            expvar_test_probe = calculate_explained_variance(A_test_ct, A_test_recon)
            
            # Performance ratios
            mse_ratio = mse_test_probe / mse_test_baseline if mse_test_baseline > 0 else float('inf')
            expvar_ratio = expvar_test_probe / expvar_test_baseline if expvar_test_baseline > 0 else 0
            
            logging.info(f"Celltype {celltype} - Test MSE (baseline): {mse_test_baseline:.6f}, (probe): {mse_test_probe:.6f}, ratio: {mse_ratio:.3f}")
            
            # Store results for this celltype
            mse_ratio_train = mse_train_probe / mse_train_baseline if mse_train_baseline > 0 else float('inf')
            mse_ratio_test = mse_ratio  # Already calculated above
            expvar_ratio_train = expvar_train_probe / expvar_train_baseline if expvar_train_baseline > 0 else 0
            expvar_ratio_test = expvar_ratio  # Already calculated above
            generalization_gap = mse_ratio_test - mse_ratio_train
            
            celltype_results[celltype] = {
                'mse_train_baseline': mse_train_baseline,
                'mse_test_baseline': mse_test_baseline,
                'mse_train_probe': mse_train_probe,
                'mse_test_probe': mse_test_probe,
                'expvar_train_baseline': expvar_train_baseline,
                'expvar_test_baseline': expvar_test_baseline,
                'expvar_train_probe': expvar_train_probe,
                'expvar_test_probe': expvar_test_probe,
                'mse_ratio_train': mse_ratio_train,
                'mse_ratio_test': mse_ratio_test,
                'expvar_ratio_train': expvar_ratio_train,
                'expvar_ratio_test': expvar_ratio_test,
                'generalization_gap': generalization_gap,
                'probeset_size': len(probeset_genes),
                'probeset_genes_found': len(probeset_genes_found),
                'n_cells': adata_celltype.shape[0],
                'skipped': False
            }
            
        except Exception as e:
            logging.error(f"Failed mechanistic evaluation for celltype {celltype}: {e}")
            celltype_results[celltype] = {
                'mse_train_baseline': np.nan,
                'mse_test_baseline': np.nan,
                'mse_train_probe': np.nan,
                'mse_test_probe': np.nan,
                'expvar_train_baseline': np.nan,
                'expvar_test_baseline': np.nan,
                'expvar_train_probe': np.nan,
                'expvar_test_probe': np.nan,
                'mse_ratio_train': np.nan,
                'mse_ratio_test': np.nan,
                'expvar_ratio_train': np.nan,
                'expvar_ratio_test': np.nan,
                'generalization_gap': np.nan,
                'probeset_size': len(probeset_genes),
                'probeset_genes_found': len(probeset_genes_found),
                'n_cells': adata_celltype.shape[0],
                'skipped': True,
                'skip_reason': 'evaluation_failed'
            }
    
    # Calculate summary statistics across celltypes
    valid_results = {ct: res for ct, res in celltype_results.items() if not res.get('skipped', False)}
    
    if valid_results:
        # Calculate WEIGHTED averages (weighted by number of cells)
        total_cells = sum(res['n_cells'] for res in valid_results.values())
        
        weighted_mse_test_probe = calculate_weighted_mse(valid_results)
        weighted_expvar_test_probe = calculate_weighted_explained_variance(valid_results)
        weighted_mse_test_baseline = calculate_weighted_mse_baseline(valid_results)
        weighted_expvar_test_baseline = calculate_weighted_explained_variance_baseline(valid_results)
        
        # Calculate MACRO averages (equal weight per cell type)
        macro_mse_test_probe = calculate_macro_mse(valid_results)
        macro_expvar_test_probe = calculate_macro_explained_variance(valid_results)
        macro_mse_test_baseline = np.mean([res['mse_test_baseline'] for res in valid_results.values()])
        macro_expvar_test_baseline = np.mean([res['expvar_test_baseline'] for res in valid_results.values()])
        
        summary_results = {
            'celltype_results': celltype_results,
            'summary': {
                # Weighted (by n_cells) metrics - absolute values
                'weighted_mse_test_probe': weighted_mse_test_probe,
                'weighted_mse_test_baseline': weighted_mse_test_baseline,
                'weighted_expvar_test_probe': weighted_expvar_test_probe,
                'weighted_expvar_test_baseline': weighted_expvar_test_baseline,
                # Macro (unweighted) metrics - absolute values
                'macro_mse_test_probe': macro_mse_test_probe,
                'macro_mse_test_baseline': macro_mse_test_baseline,
                'macro_expvar_test_probe': macro_expvar_test_probe,
                'macro_expvar_test_baseline': macro_expvar_test_baseline,
                # Metadata
                'total_cells': total_cells,
                'n_celltypes_processed': len(valid_results),
                'n_celltypes_skipped': len(celltype_results) - len(valid_results)
            }
        }
    else:
        summary_results = {
            'celltype_results': celltype_results,
            'summary': {
                'weighted_mse_test_probe': np.nan,
                'weighted_mse_test_baseline': np.nan,
                'weighted_expvar_test_probe': np.nan,
                'weighted_expvar_test_baseline': np.nan,
                'macro_mse_test_probe': np.nan,
                'macro_mse_test_baseline': np.nan,
                'macro_expvar_test_probe': np.nan,
                'macro_expvar_test_baseline': np.nan,
                'total_cells': 0,
                'n_celltypes_processed': 0,
                'n_celltypes_skipped': len(celltype_results)
            }
        }
    
    return summary_results

##############################################################################
# MAPPING PERFORMANCE EVALUATION
##############################################################################

def evaluate_mapping_performance(adata, probeset_genes, A_train, A_test, n_components=5, 
                               max_iter=1000, random_state=42, cached_full_nmf=None):
    """
    Evaluate how well a probe gene subset can map to the full transcriptome.
    
    Research question: Can probe-derived gene patterns work with complete 
    sample biology to reconstruct full transcriptome expression?
    """
    logging.info(f"Evaluating mapping performance with {len(probeset_genes)} genes")
    
    # Get indices of probe genes
    probeset_mask = np.array([gene in probeset_genes for gene in adata.var_names])
    if sum(probeset_mask) == 0:
        logging.error(f"No probeset genes found in the dataset")
        return None
    
    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logging.info(f"Found {len(probeset_genes_found)} out of {len(probeset_genes)} probeset genes in the dataset")
    
    # Use pre-split data and ensure float64 dtype for NMF compatibility
    A_train = A_train.astype(np.float64)
    A_test = A_test.astype(np.float64)
    A_P_train = A_train[:, probe_indices]  # (cells × probe_genes)
    A_P_test = A_test[:, probe_indices]    # (cells × probe_genes)

    logging.info(f"Training: {A_train.shape[0]} cells × {A_train.shape[1]} genes")
    logging.info(f"Testing: {A_test.shape[0]} cells × {A_test.shape[1]} genes")
    logging.info(f"Probe subset: {len(probe_indices)} genes")
    
    # Training phase
    logging.info("--- Training Phase ---")
    
    # Step 1: Probe NMF
    nmf_probe = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
    H_probe_train = nmf_probe.fit_transform(A_P_train.T).T # factors x genes
    W_probe_train = nmf_probe.components_.T               # cells x factors
    
    # Step 2: Full NMF - check if we can use cached data
    if cached_full_nmf is not None and 'training' in cached_full_nmf:
        training_cache = cached_full_nmf['training']
        if (training_cache['A_train'].shape == A_train.shape and 
            np.allclose(training_cache['A_train'], A_train)):
            # Ensure cached NMF results are float64 to match converted input data
            H_full_train = training_cache['H_full_train'].astype(np.float64)
            W_full_train = training_cache['W_full_train'].astype(np.float64)
            A_train_baseline = training_cache['A_train_baseline']
            mse_train_baseline = training_cache['mse_train_baseline']
            expvar_train_baseline = training_cache['expvar_train_baseline']
            logging.info("Using cached training NMF results")
        else:
            cached_full_nmf = None
    
    if cached_full_nmf is None or 'training' not in cached_full_nmf:
        logging.info("Computing Full NMF on training data")
        nmf_full_train = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
        H_full_train = nmf_full_train.fit_transform(A_train.T).T # factors x genes
        W_full_train = nmf_full_train.components_.T # cells x factors
        A_train_baseline = (W_full_train @ H_full_train).T
        mse_train_baseline = calculate_mse(A_train, A_train_baseline)
        expvar_train_baseline = calculate_explained_variance(A_train, A_train_baseline)
    
    # Step 3: Extension
    nmf_extend = NMF(n_components=n_components, init='custom', max_iter=max_iter, random_state=random_state)
    H_extended_train = nmf_extend._fit_transform(
        A_train.T, W=None, H=W_probe_train.T, update_H=False
    )[0] # factors × genes
    
    # Testing phase
    logging.info("--- Testing Phase ---")
    
    nmf_probe = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
    H_probe_test = nmf_probe.fit_transform(A_P_test.T).T # factors x genes
    W_probe_test = nmf_probe.components_.T               # cells x factors

    if cached_full_nmf is not None and 'testing' in cached_full_nmf:
        testing_cache = cached_full_nmf['testing']
        if (testing_cache['A_test'].shape == A_test.shape and 
            np.allclose(testing_cache['A_test'], A_test)):
            # Ensure cached NMF results are float64 to match converted input data
            H_full_test = testing_cache['H_full_test'].astype(np.float64)
            W_full_test = testing_cache['W_full_test'].astype(np.float64)
            A_test_baseline = testing_cache['A_test_baseline']
            mse_test_baseline = testing_cache['mse_test_baseline']
            expvar_test_baseline = testing_cache['expvar_test_baseline']
            logging.info("Using cached testing NMF results")
        else:
            cached_full_nmf['testing'] = None
    
    if cached_full_nmf is None or 'testing' not in cached_full_nmf:
        logging.info("Computing Full NMF on testing data")
        nmf_full_test = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
        H_full_test = nmf_full_test.fit_transform(A_test.T).T # factors x genes
        W_full_test = nmf_full_test.components_.T             # cells x factors
        A_test_baseline = (W_full_test @ H_full_test).T
        mse_test_baseline = calculate_mse(A_test, A_test_baseline)
        expvar_test_baseline = calculate_explained_variance(A_test, A_test_baseline)
    
    # Reconstruction and evaluation
    logging.info("--- Reconstruction and Evaluation ---")
    
    A_train_recon = W_probe_train @ H_extended_train.T
    A_test_recon = W_probe_test @ H_extended_train.T
    
    mse_train_probe = calculate_mse(A_train, A_train_recon)
    mse_test_probe = calculate_mse(A_test, A_test_recon)
    expvar_train_probe = calculate_explained_variance(A_train, A_train_recon)
    expvar_test_probe = calculate_explained_variance(A_test, A_test_recon)
    
    # Performance ratios
    mse_ratio_train = mse_train_probe / mse_train_baseline if mse_train_baseline > 0 else float('inf')
    mse_ratio_test = mse_test_probe / mse_test_baseline if mse_test_baseline > 0 else float('inf')
    expvar_ratio_train = expvar_train_probe / expvar_train_baseline if expvar_train_baseline > 0 else 0
    expvar_ratio_test = expvar_test_probe / expvar_test_baseline if expvar_test_baseline > 0 else 0
    
    generalization_gap = mse_ratio_test - mse_ratio_train
    
    logging.info(f"Training MSE (baseline): {mse_train_baseline:.6f}")
    logging.info(f"Training MSE (probe): {mse_train_probe:.6f}")
    logging.info(f"Test MSE (baseline): {mse_test_baseline:.6f}")
    logging.info(f"Test MSE (probe): {mse_test_probe:.6f}")
    logging.info(f"MSE ratio train (probe/baseline): {mse_ratio_train:.3f}")
    logging.info(f"MSE ratio test (probe/baseline): {mse_ratio_test:.3f}")
    logging.info(f"Generalization gap: {generalization_gap:.3f}")
    
    result = {
        'mse_train_baseline': mse_train_baseline,
        'mse_test_baseline': mse_test_baseline,
        'mse_train_probe': mse_train_probe, 
        'mse_test_probe': mse_test_probe,
        'expvar_train_baseline': expvar_train_baseline,
        'expvar_test_baseline': expvar_test_baseline,
        'expvar_train_probe': expvar_train_probe,
        'expvar_test_probe': expvar_test_probe,
        'mse_ratio_train': mse_ratio_train,
        'mse_ratio_test': mse_ratio_test,
        'expvar_ratio_train': expvar_ratio_train,
        'expvar_ratio_test': expvar_ratio_test,
        'generalization_gap': generalization_gap,
        'probeset_size': len(probeset_genes),
        'probeset_genes_found': len(probeset_genes_found)
    }
    
    # Add cache data if we computed it
    if cached_full_nmf is None:
        computed_full_nmf = {
            'training': {
                'A_train': A_train,
                'W_full_train': W_full_train,
                'H_full_train': H_full_train,
                'A_train_baseline': A_train_baseline,
                'mse_train_baseline': mse_train_baseline,
                'expvar_train_baseline': expvar_train_baseline
            },
            'testing': {
                'A_test': A_test,
                'W_full_test': W_full_test,
                'H_full_test': H_full_test,
                'A_test_baseline': A_test_baseline,
                'mse_test_baseline': mse_test_baseline,
                'expvar_test_baseline': expvar_test_baseline
            }
        }
        result['computed_full_nmf'] = computed_full_nmf
    
    return result


def evaluate_mapping_performance_by_celltype(adata, probeset_genes, celltype_column='celltypes_v2', 
                                            n_components=5, max_iter=1000, random_state=42, 
                                            cached_full_nmf_by_celltype=None, test_size=0.3):
    """
    Evaluate mapping performance for each celltype separately using train-test split approach.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with full transcriptome
    probeset_genes : list
        List of genes in the probeset to evaluate
    celltype_column : str
        Column name in adata.obs that contains celltype information
    n_components : int
        Number of NMF components to use
    max_iter : int
        Maximum number of iterations for NMF
    random_state : int
        Random state for reproducibility
    cached_full_nmf_by_celltype : dict, optional
        Pre-computed full NMF results by celltype to avoid recomputation
    test_size : float
        Proportion of samples to use for testing
        
    Returns:
    --------
    dict: Contains MSE and explained variance metrics by celltype
    """
    logging.info(f"Evaluating mapping performance by celltype with {len(probeset_genes)} genes")
    
    # Ensure we have raw counts
    if "counts" not in adata.layers:
        logging.warning("No 'counts' layer found, using .X")
        adata.layers['counts'] = adata.X.copy()
    
    # Check if celltype column exists
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Celltype column '{celltype_column}' not found in adata.obs")
    
    # Get unique celltypes
    celltypes = adata.obs[celltype_column].unique()
    logging.info(f"Found {len(celltypes)} celltypes: {list(celltypes)}")
    
    # Subset to only probeset genes
    probeset_mask = np.array([gene in probeset_genes for gene in adata.var_names])
    if sum(probeset_mask) == 0:
        raise ValueError(f"None of the probeset genes were found in the dataset")
    
    probe_indices = np.where(probeset_mask)[0]
    probeset_genes_found = adata.var_names[probeset_mask]
    logging.info(f"Found {len(probeset_genes_found)} out of {len(probeset_genes)} probeset genes in the dataset")
    
    # Initialize results dictionary
    celltype_results = {}
    
    # Initialize cache if not provided
    if cached_full_nmf_by_celltype is None:
        cached_full_nmf_by_celltype = {}
    
    # Process each celltype
    for celltype in tqdm(celltypes, desc="Processing celltypes"):
        logging.info(f"Processing celltype: {celltype}")
        
        # Subset data to this celltype
        celltype_mask = adata.obs[celltype_column] == celltype
        adata_celltype = adata[celltype_mask].copy()
        
        logging.info(f"Celltype {celltype}: {adata_celltype.shape[0]:,} cells × {adata_celltype.shape[1]:,} genes")
        
        # Skip celltypes with too few cells for NMF or train-test split
        min_cells_needed = max(n_components * 2, int(1 / test_size))
        if adata_celltype.shape[0] < min_cells_needed:
            logging.warning(f"Skipping celltype {celltype}: only {adata_celltype.shape[0]} cells (need at least {min_cells_needed})")
            celltype_results[celltype] = {
                'mse_train_baseline': np.nan,
                'mse_test_baseline': np.nan,
                'mse_train_probe': np.nan,
                'mse_test_probe': np.nan,
                'expvar_train_baseline': np.nan,
                'expvar_test_baseline': np.nan,
                'expvar_train_probe': np.nan,
                'expvar_test_probe': np.nan,
                'mse_ratio_train': np.nan,
                'mse_ratio_test': np.nan,
                'expvar_ratio_train': np.nan,
                'expvar_ratio_test': np.nan,
                'generalization_gap': np.nan,
                'probeset_size': len(probeset_genes),
                'probeset_genes_found': len(probeset_genes_found),
                'n_cells': adata_celltype.shape[0],
                'skipped': True,
                'skip_reason': 'insufficient_cells'
            }
            continue
        
        try:
            # Step 1: Check if celltype exists in cache
            if celltype in cached_full_nmf_by_celltype:
                logging.info(f"Found cached data for celltype {celltype}")
                cached_ct = cached_full_nmf_by_celltype[celltype]
                
                # Step 2: Check if training and testing data exists in cache
                if ('training' in cached_ct and 'A_train' in cached_ct['training'] and 
                    'testing' in cached_ct and 'A_test' in cached_ct['testing']):
                    logging.info(f"Using cached train-test split for celltype {celltype}")
                    
                    # Use cached train-test splits and ensure float64 dtype for NMF compatibility
                    A_train_ct = cached_ct['training']['A_train'].astype(np.float64)
                    A_test_ct = cached_ct['testing']['A_test'].astype(np.float64)
                    
                    # Get probe gene subsets from cached splits
                    A_P_train_ct = A_train_ct[:, probe_indices]
                    A_P_test_ct = A_test_ct[:, probe_indices]
                else:
                    # Step 2 fallback: Recompute train-test split but warn about inconsistency
                    logging.warning(f"Cached train-test split not found for celltype {celltype}, recomputing (may be inconsistent with other evaluation functions)")
                    
                    # Get data matrix for this celltype
                    if "counts" in adata_celltype.layers:
                        A_celltype = adata_celltype.layers['counts'].toarray() if scipy.sparse.issparse(adata_celltype.layers['counts']) else adata_celltype.layers['counts']
                    else:
                        A_celltype = adata_celltype.X.toarray() if scipy.sparse.issparse(adata_celltype.X) else adata_celltype.X
                    # Ensure float64 dtype for NMF compatibility
                    A_celltype = A_celltype.astype(np.float64)
                    
                    # Create new train-test split
                    A_train_ct, A_test_ct = train_test_split(A_celltype, test_size=test_size, random_state=42)
                    
                    # Get probe gene subsets from new splits
                    A_P_train_ct = A_train_ct[:, probe_indices]
                    A_P_test_ct = A_test_ct[:, probe_indices]
                    
                    logging.info(f"Training: {A_train_ct.shape[0]} cells × {A_train_ct.shape[1]} genes")
                    logging.info(f"Testing: {A_test_ct.shape[0]} cells × {A_test_ct.shape[1]} genes")
                    logging.info(f"Probe subset: {len(probe_indices)} genes")
                    
                    # Update cache with new train-test split data
                    if celltype not in cached_full_nmf_by_celltype:
                        cached_full_nmf_by_celltype[celltype] = {}
                    if 'training' not in cached_full_nmf_by_celltype[celltype]:
                        cached_full_nmf_by_celltype[celltype]['training'] = {}
                    if 'testing' not in cached_full_nmf_by_celltype[celltype]:
                        cached_full_nmf_by_celltype[celltype]['testing'] = {}
                    
                    cached_full_nmf_by_celltype[celltype]['training']['A_train'] = A_train_ct
                    cached_full_nmf_by_celltype[celltype]['testing']['A_test'] = A_test_ct
                
                # Step 3: Check if NMF W and H matrices exist in cache (refresh cache reference)
                cached_ct = cached_full_nmf_by_celltype[celltype]  # Refresh reference after potential updates
                if ('training' in cached_ct and all(k in cached_ct['training'] for k in ['W_full_train', 'H_full_train', 'mse_train_baseline', 'expvar_train_baseline']) and
                    'testing' in cached_ct and all(k in cached_ct['testing'] for k in ['W_full_test', 'H_full_test', 'mse_test_baseline', 'expvar_test_baseline'])):
                    logging.info(f"Using cached NMF results for celltype {celltype}")
                    
                    # Use cached NMF results
                    W_full_train = cached_ct['training']['W_full_train'].astype(np.float64)
                    H_full_train = cached_ct['training']['H_full_train'].astype(np.float64)
                    mse_train_baseline = cached_ct['training']['mse_train_baseline']
                    expvar_train_baseline = cached_ct['training']['expvar_train_baseline']
                    
                    W_full_test = cached_ct['testing']['W_full_test'].astype(np.float64)
                    H_full_test = cached_ct['testing']['H_full_test'].astype(np.float64)
                    mse_test_baseline = cached_ct['testing']['mse_test_baseline']
                    expvar_test_baseline = cached_ct['testing']['expvar_test_baseline']
                else:
                    # Step 3 fallback: Recompute NMF using train-test data from step 2
                    logging.info(f"Cached NMF results not found for celltype {celltype}, recomputing NMF")
                    
                    # Compute full NMF for train and test (using swapped convention for mapping)
                    logging.info("Computing Full NMF on training data (swapped convention)")
                    nmf_full_train_sw = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                    H_full_train_sw = nmf_full_train_sw.fit_transform(A_train_ct.T).T  # (components × genes)
                    W_full_train_sw = nmf_full_train_sw.components_.T                  # (cells × components)
                    A_train_baseline_sw = W_full_train_sw @ H_full_train_sw
                    mse_train_baseline_sw = calculate_mse(A_train_ct, A_train_baseline_sw)
                    expvar_train_baseline_sw = calculate_explained_variance(A_train_ct, A_train_baseline_sw)
                    
                    logging.info("Computing Full NMF on testing data (swapped convention)")
                    nmf_full_test_sw = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                    H_full_test_sw = nmf_full_test_sw.fit_transform(A_test_ct.T).T   # (components × genes)
                    W_full_test_sw = nmf_full_test_sw.components_.T                  # (cells × components)
                    A_test_baseline_sw = W_full_test_sw @ H_full_test_sw
                    mse_test_baseline_sw = calculate_mse(A_test_ct, A_test_baseline_sw)
                    expvar_test_baseline_sw = calculate_explained_variance(A_test_ct, A_test_baseline_sw)
                    
                    # Update cache with new NMF results
                    if celltype not in cached_full_nmf_by_celltype:
                        cached_full_nmf_by_celltype[celltype] = {}
                    
                    cached_full_nmf_by_celltype[celltype]['training'] = {
                        'A_train': A_train_ct,
                        'W_full_train': W_full_train_sw,
                        'H_full_train': H_full_train_sw,
                        'A_train_baseline': A_train_baseline_sw,
                        'mse_train_baseline': mse_train_baseline_sw,
                        'expvar_train_baseline': expvar_train_baseline_sw
                    }
                    cached_full_nmf_by_celltype[celltype]['testing'] = {
                        'A_test': A_test_ct,
                        'W_full_test': W_full_test_sw,
                        'H_full_test': H_full_test_sw,
                        'A_test_baseline': A_test_baseline_sw,
                        'mse_test_baseline': mse_test_baseline_sw,
                        'expvar_test_baseline': expvar_test_baseline_sw
                    }
            else:
                # Step 1 fallback: No cache data found for this celltype, compute everything from scratch
                logging.warning(f"No cached data found for celltype {celltype}, computing everything from scratch")
                
                # Get data matrix for this celltype
                if "counts" in adata_celltype.layers:
                    A_celltype = adata_celltype.layers['counts'].toarray() if scipy.sparse.issparse(adata_celltype.layers['counts']) else adata_celltype.layers['counts']
                else:
                    A_celltype = adata_celltype.X.toarray() if scipy.sparse.issparse(adata_celltype.X) else adata_celltype.X
                
                # Create train-test split for this celltype
                A_train_ct, A_test_ct = train_test_split(A_celltype, test_size=test_size, random_state=42)
                
                # Get probe gene subsets from train-test splits
                A_P_train_ct = A_train_ct[:, probe_indices]
                A_P_test_ct = A_test_ct[:, probe_indices]
                
                logging.info(f"Training: {A_train_ct.shape[0]} cells × {A_train_ct.shape[1]} genes")
                logging.info(f"Testing: {A_test_ct.shape[0]} cells × {A_test_ct.shape[1]} genes")
                logging.info(f"Probe subset: {len(probe_indices)} genes")
                
                # Compute full NMF for train and test (using swapped convention for mapping)
                logging.info("Computing Full NMF on training data (swapped convention)")
                nmf_full_train_sw = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                H_full_train_sw = nmf_full_train_sw.fit_transform(A_train_ct.T).T  # (components × genes)
                W_full_train_sw = nmf_full_train_sw.components_.T                  # (cells × components)
                A_train_baseline_sw = W_full_train_sw @ H_full_train_sw
                mse_train_baseline_sw = calculate_mse(A_train_ct, A_train_baseline_sw)
                expvar_train_baseline_sw = calculate_explained_variance(A_train_ct, A_train_baseline_sw)
                
                logging.info("Computing Full NMF on testing data (swapped convention)")
                nmf_full_test_sw = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
                H_full_test_sw = nmf_full_test_sw.fit_transform(A_test_ct.T).T   # (components × genes)
                W_full_test_sw = nmf_full_test_sw.components_.T                  # (cells × components)
                A_test_baseline_sw = W_full_test_sw @ H_full_test_sw
                mse_test_baseline_sw = calculate_mse(A_test_ct, A_test_baseline_sw)
                expvar_test_baseline_sw = calculate_explained_variance(A_test_ct, A_test_baseline_sw)
                
                # Cache results for potential reuse
                cached_full_nmf_by_celltype[celltype] = {
                    'training': {
                        'A_train': A_train_ct,
                        'W_full_train': W_full_train_sw,
                        'H_full_train': H_full_train_sw,
                        'A_train_baseline': A_train_baseline_sw,
                        'mse_train_baseline': mse_train_baseline_sw,
                        'expvar_train_baseline': expvar_train_baseline_sw
                    },
                    'testing': {
                        'A_test': A_test_ct,
                        'W_full_test': W_full_test_sw,
                        'H_full_test': H_full_test_sw,
                        'A_test_baseline': A_test_baseline_sw,
                        'mse_test_baseline': mse_test_baseline_sw,
                        'expvar_test_baseline': expvar_test_baseline_sw
                    }
                }
            

            
            # Apply mapping performance approach
            logging.info(f"Running mapping performance for celltype {celltype}")
            
            # Step 1: Probe NMF on training data
            nmf_probe_train = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
            H_probe_train = nmf_probe_train.fit_transform(A_P_train_ct.T).T  # (components × probe_genes)
            W_probe_train = nmf_probe_train.components_.T                    # (cells × components)
            
            # Step 2: Probe NMF on testing data
            nmf_probe_test = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
            H_probe_test = nmf_probe_test.fit_transform(A_P_test_ct.T).T   # (components × probe_genes)
            W_probe_test = nmf_probe_test.components_.T                    # (cells × components)
            
            # Step 3: Extension - solve for H_extended using probe factors
            # Training
            nmf_extend_train = NMF(n_components=n_components, init='custom', max_iter=max_iter, random_state=random_state)
            H_extended_train = nmf_extend_train._fit_transform(
                A_train_ct.T, W=None, H=W_probe_train.T, update_H=False
            )[0]  # (components × genes)
            
            # Step 4: Reconstruct full transcriptome
            A_train_recon = W_probe_train @ H_extended_train.T
            A_test_recon = W_probe_test @ H_extended_train.T
            
            # Calculate metrics
            mse_train_probe = calculate_mse(A_train_ct, A_train_recon)
            expvar_train_probe = calculate_explained_variance(A_train_ct, A_train_recon)
            
            mse_test_probe = calculate_mse(A_test_ct, A_test_recon)
            expvar_test_probe = calculate_explained_variance(A_test_ct, A_test_recon)
            
            # Performance ratios
            mse_ratio_train = mse_train_probe / mse_train_baseline if mse_train_baseline > 0 else float('inf')
            mse_ratio_test = mse_test_probe / mse_test_baseline if mse_test_baseline > 0 else float('inf')
            expvar_ratio_train = expvar_train_probe / expvar_train_baseline if expvar_train_baseline > 0 else 0
            expvar_ratio_test = expvar_test_probe / expvar_test_baseline if expvar_test_baseline > 0 else 0
            
            generalization_gap = mse_ratio_test - mse_ratio_train
            
            logging.info(f"Celltype {celltype} - Test MSE ratio: {mse_ratio_test:.3f}, Generalization gap: {generalization_gap:.3f}")
            
            # Store results for this celltype
            celltype_results[celltype] = {
                'mse_train_baseline': mse_train_baseline,
                'mse_test_baseline': mse_test_baseline,
                'mse_train_probe': mse_train_probe,
                'mse_test_probe': mse_test_probe,
                'expvar_train_baseline': expvar_train_baseline,
                'expvar_test_baseline': expvar_test_baseline,
                'expvar_train_probe': expvar_train_probe,
                'expvar_test_probe': expvar_test_probe,
                'mse_ratio_train': mse_ratio_train,
                'mse_ratio_test': mse_ratio_test,
                'expvar_ratio_train': expvar_ratio_train,
                'expvar_ratio_test': expvar_ratio_test,
                'generalization_gap': generalization_gap,
                'probeset_size': len(probeset_genes),
                'probeset_genes_found': len(probeset_genes_found),
                'n_cells': adata_celltype.shape[0],
                'skipped': False
            }
            
        except Exception as e:
            logging.error(f"Failed mapping evaluation for celltype {celltype}: {e}")
            celltype_results[celltype] = {
                'mse_train_baseline': np.nan,
                'mse_test_baseline': np.nan,
                'mse_train_probe': np.nan,
                'mse_test_probe': np.nan,
                'expvar_train_baseline': np.nan,
                'expvar_test_baseline': np.nan,
                'expvar_train_probe': np.nan,
                'expvar_test_probe': np.nan,
                'mse_ratio_train': np.nan,
                'mse_ratio_test': np.nan,
                'expvar_ratio_train': np.nan,
                'expvar_ratio_test': np.nan,
                'generalization_gap': np.nan,
                'probeset_size': len(probeset_genes),
                'probeset_genes_found': len(probeset_genes_found),
                'n_cells': adata_celltype.shape[0],
                'skipped': True,
                'skip_reason': 'evaluation_failed'
            }
    
    # Calculate summary statistics across celltypes
    valid_results = {ct: res for ct, res in celltype_results.items() if not res.get('skipped', False)}
    
    if valid_results:
        # Calculate WEIGHTED averages (weighted by number of cells)
        total_cells = sum(res['n_cells'] for res in valid_results.values())
        
        weighted_mse_test_probe = calculate_weighted_mse(valid_results)
        weighted_expvar_test_probe = calculate_weighted_explained_variance(valid_results)
        weighted_mse_test_baseline = calculate_weighted_mse_baseline(valid_results)
        weighted_expvar_test_baseline = calculate_weighted_explained_variance_baseline(valid_results)
        
        # Calculate MACRO averages (equal weight per cell type)
        macro_mse_test_probe = calculate_macro_mse(valid_results)
        macro_expvar_test_probe = calculate_macro_explained_variance(valid_results)
        macro_mse_test_baseline = np.mean([res['mse_test_baseline'] for res in valid_results.values()])
        macro_expvar_test_baseline = np.mean([res['expvar_test_baseline'] for res in valid_results.values()])
        
        summary_results = {
            'celltype_results': celltype_results,
            'summary': {
                # Weighted (by n_cells) metrics - absolute values
                'weighted_mse_test_probe': weighted_mse_test_probe,
                'weighted_mse_test_baseline': weighted_mse_test_baseline,
                'weighted_expvar_test_probe': weighted_expvar_test_probe,
                'weighted_expvar_test_baseline': weighted_expvar_test_baseline,
                # Macro (unweighted) metrics - absolute values
                'macro_mse_test_probe': macro_mse_test_probe,
                'macro_mse_test_baseline': macro_mse_test_baseline,
                'macro_expvar_test_probe': macro_expvar_test_probe,
                'macro_expvar_test_baseline': macro_expvar_test_baseline,
                # Metadata
                'total_cells': total_cells,
                'n_celltypes_processed': len(valid_results),
                'n_celltypes_skipped': len(celltype_results) - len(valid_results)
            }
        }
    else:
        summary_results = {
            'celltype_results': celltype_results,
            'summary': {
                'weighted_mse_test_probe': np.nan,
                'weighted_mse_test_baseline': np.nan,
                'weighted_expvar_test_probe': np.nan,
                'weighted_expvar_test_baseline': np.nan,
                'macro_mse_test_probe': np.nan,
                'macro_mse_test_baseline': np.nan,
                'macro_expvar_test_probe': np.nan,
                'macro_expvar_test_baseline': np.nan,
                'total_cells': 0,
                'n_celltypes_processed': 0,
                'n_celltypes_skipped': len(celltype_results)
            }
        }
    
    return summary_results
