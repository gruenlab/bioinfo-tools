#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NMF-INDEPENDENT RECONSTRUCTION EVALUATION MODULE
================================================

This module evaluates probe panel reconstruction quality using multiple methods:
1. Simple Neural Network (supervised reconstruction)
2. VAE variants (Variational Autoencoders):
   - vae_simple: Simple VAE via nn_utils.vae_predictor
   - LVAE: Linear encoder + Linear decoder
   - LEVAE: Linear encoder + Nonlinear decoder
   - LDVAE: Nonlinear encoder + Linear decoder
   - VAE: Nonlinear encoder + Nonlinear decoder
3. Tangram (spatial mapping)

All methods are trained/evaluated independently of NMF-based approaches to provide
complementary evaluation perspectives.

KEY FEATURES:
-------------
- Two evaluation modes:
  * GLOBAL: Train one model on all cells (mixed cell types)
  * PER-CELL-TYPE: Train separate models for each cell type
- Train/test split evaluation for Neural Network and VAE (generalization assessment)
- Full dataset evaluation for Tangram (method limitation)
- Per-cell-type metrics (macro and weighted averaging)
- Comprehensive metrics: MSE, explained variance, RMSE, MAE, R², Pearson correlation
- Compatible with existing evaluation infrastructure
- Multiple VAE architectures from colleague's nn_deep_CE.py

WORKFLOW:
---------
1. Load preprocessed h5ad (full transcriptome)
2. Load selected gene panel (CSV)
3. Choose evaluation mode (global/per_celltype/both)
4. GLOBAL MODE:
   - Split data into train/test sets (70/30, random_state=42)
   - Train reconstruction methods on ALL cells
   - Evaluate globally and per-cell-type
5. PER-CELL-TYPE MODE:
   - For each cell type separately:
     * Filter to cells of that type
     * Train/test split within cell type
     * Train separate model on that cell type only
     * Evaluate on held-out test set of same cell type
   - Aggregate results across cell types

OUTPUT:
-------
GLOBAL MODE:
  results/Global_Reconstruction/<Method>/
    - metrics_global.csv         # Overall metrics across all cells
    - reconstructed_expression.h5ad

PER-CELL-TYPE MODE:
  results/PerCellType_Reconstruction/<CellType>/<Method>/
    - metrics.csv                # Metrics for this cell type
  results/PerCellType_Reconstruction/summary/
    - aggregated_metrics.csv     # Macro/weighted across all cell types

Script written by: Helene Hemmer (with GitHub Copilot assistance)
Date: 10.11.2025
Python environment: evaluation-probes
"""

##############################################################################
# IMPORTS
##############################################################################

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import tangram as tg
except ImportError:
    print("Warning: tangram not installed. Install with: pip install tangram-sc")
    tg = None

try:
    import lightning as L
except ImportError:
    print("Warning: lightning not installed. Install with: pip install lightning")
    L = None
    
import logging
import sys
import os
import gc
import psutil
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import importlib.util

# Import existing metric functions from Variability module
variability_module_path = '/home/gruengroup/helene/SpatialProbeDesign_tmp/Code/Modules/Evaluation-module/20251009_Evaluation-module_Variability_HH.py'
spec = importlib.util.spec_from_file_location("evaluation_variability", variability_module_path)
evaluation_variability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation_variability)

# Import metric functions
calculate_mse = evaluation_variability.calculate_mse
calculate_explained_variance = evaluation_variability.calculate_explained_variance
calculate_macro_mse = evaluation_variability.calculate_macro_mse
calculate_macro_explained_variance = evaluation_variability.calculate_macro_explained_variance
calculate_weighted_mse = evaluation_variability.calculate_weighted_mse
calculate_weighted_explained_variance = evaluation_variability.calculate_weighted_explained_variance

# Import colleague NN / VAE utilities
nn_utils_path = '/home/gruengroup/helene/SpatialProbeDesign_tmp/nn_utils_CE.py'
spec_nn = importlib.util.spec_from_file_location("nn_utils_CE", nn_utils_path)
nn_utils = importlib.util.module_from_spec(spec_nn)
spec_nn.loader.exec_module(nn_utils)

nn_deep_path = '/home/gruengroup/helene/SpatialProbeDesign_tmp/nn_deep_CE.py'
spec_deep = importlib.util.spec_from_file_location("nn_deep_CE", nn_deep_path)
nn_deep = importlib.util.module_from_spec(spec_deep)
spec_deep.loader.exec_module(nn_deep)

##############################################################################
# LOGGING SETUP
##############################################################################

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
# EVALUATION UTILITY FUNCTIONS
##############################################################################

def calculate_global_metrics(X_full_ref, X_pred):
    """
    Calculate global metrics across all cells (no per-cell-type breakdown).
    
    Parameters:
    -----------
    X_full_ref : np.ndarray
        Original full transcriptome (n_cells × n_genes)
    X_pred : np.ndarray
        Reconstructed full transcriptome (n_cells × n_genes)
        
    Returns:
    --------
    global_metrics : dict
        Dictionary with global metrics
    """
    mse = calculate_mse(X_full_ref, X_pred)
    expvar = calculate_explained_variance(X_full_ref, X_pred)
    
    global_metrics = {
        'mse_global': float(mse),
        'expvar_global': float(expvar),
        'rmse_global': float(np.sqrt(mse)),
        'mae_global': float(np.mean(np.abs(X_full_ref - X_pred))),
        'r2_global': float(r2_score(X_full_ref.flatten(), X_pred.flatten())),
        'pearson_global': float(pearsonr(X_full_ref.flatten(), X_pred.flatten())[0]),
        'n_cells': int(X_full_ref.shape[0]),
        'n_genes': int(X_full_ref.shape[1])
    }
    
    return global_metrics


def calculate_summary_metrics(celltype_results):
    """
    Calculate macro and weighted metrics from per-cell-type results.
    
    Parameters:
    -----------
    celltype_results : dict
        Dictionary with per-cell-type metrics
        
    Returns:
    --------
    summary : dict
        Dictionary with macro and weighted metrics
    """
    # Get valid results
    valid_results = {ct: res for ct, res in celltype_results.items() 
                    if not res.get('skipped', False)}
    
    if not valid_results:
        return {
            'macro_mse_test': np.nan,
            'macro_expvar_test': np.nan,
            'weighted_mse_test': np.nan,
            'weighted_expvar_test': np.nan,
            'n_celltypes_evaluated': 0,
            'n_celltypes_skipped': len(celltype_results),
            'total_cells': 0
        }
    
    # Calculate macro metrics
    macro_metrics = {
        'macro_mse_test': calculate_macro_mse(celltype_results),
        'macro_expvar_test': calculate_macro_explained_variance(celltype_results),
        'macro_rmse_test': np.mean([res['rmse_test_probe'] for res in valid_results.values()]),
        'macro_mae_test': np.mean([res['mae_test_probe'] for res in valid_results.values()]),
        'macro_r2_test': np.mean([res['r2_test_probe'] for res in valid_results.values()]),
        'macro_pearson_test': np.mean([res['pearson_test_probe'] for res in valid_results.values()])
    }
    
    # Calculate weighted metrics
    total_cells = sum(res['n_cells'] for res in valid_results.values())
    weighted_metrics = {
        'weighted_mse_test': calculate_weighted_mse(celltype_results),
        'weighted_expvar_test': calculate_weighted_explained_variance(celltype_results),
        'weighted_rmse_test': sum(res['rmse_test_probe'] * res['n_cells'] for res in valid_results.values()) / total_cells,
        'weighted_mae_test': sum(res['mae_test_probe'] * res['n_cells'] for res in valid_results.values()) / total_cells,
        'weighted_r2_test': sum(res['r2_test_probe'] * res['n_cells'] for res in valid_results.values()) / total_cells,
        'weighted_pearson_test': sum(res['pearson_test_probe'] * res['n_cells'] for res in valid_results.values()) / total_cells
    }
    
    # Combine
    summary = {
        **macro_metrics,
        **weighted_metrics,
        'n_celltypes_evaluated': len(valid_results),
        'n_celltypes_skipped': len(celltype_results) - len(valid_results),
        'total_cells': total_cells
    }
    
    return summary


def save_unified_metrics(method_dir, dataset_name, global_metrics=None, per_celltype_results=None, 
                         n_probe_genes=None, gene_list_name=None):
    """
    Save unified metrics CSV with both global and per-cell-type results.
    
    Format matches variability evaluation output:
    - One row for global results (if computed)
    - One row per cell type for per-cell-type results
    
    Parameters:
    -----------
    method_dir : str
        Directory for this method
    dataset_name : str
        Name of the dataset/gene list
    global_metrics : dict, optional
        Global metrics dictionary
    per_celltype_results : dict, optional
        Per-cell-type results dictionary
    n_probe_genes : int, optional
        Number of probe genes
    gene_list_name : str, optional
        Name of gene list file
    """
    rows = []
    
    # Add global row if available
    if global_metrics is not None:
        global_row = {
            'mse_global': global_metrics.get('mse_global', np.nan),
            'expvar_global': global_metrics.get('expvar_global', np.nan),
            'rmse_global': global_metrics.get('rmse_global', np.nan),
            'mae_global': global_metrics.get('mae_global', np.nan),
            'r2_global': global_metrics.get('r2_global', np.nan),
            'pearson_global': global_metrics.get('pearson_global', np.nan),
            'n_cells': global_metrics.get('n_cells', np.nan),
            'n_genes': global_metrics.get('n_genes', np.nan),
            'analysis_type': 'global',
            'celltype': '',
            'skipped': False
        }
        if n_probe_genes is not None:
            global_row['n_probe_genes'] = n_probe_genes
        if gene_list_name is not None:
            global_row['gene_list'] = gene_list_name
        rows.append(global_row)
    
    # Add per-cell-type rows if available
    if per_celltype_results is not None:
        for celltype, metrics in per_celltype_results.items():
            ct_row = {
                'mse_test_probe': metrics.get('mse_test_probe', np.nan),
                'expvar_test_probe': metrics.get('expvar_test_probe', np.nan),
                'rmse_test_probe': metrics.get('rmse_test_probe', np.nan),
                'mae_test_probe': metrics.get('mae_test_probe', np.nan),
                'r2_test_probe': metrics.get('r2_test_probe', np.nan),
                'pearson_test_probe': metrics.get('pearson_test_probe', np.nan),
                'n_cells': metrics.get('n_cells', np.nan),
                'n_train': metrics.get('n_train', np.nan),
                'n_test': metrics.get('n_test', np.nan),
                'analysis_type': 'per_celltype',
                'celltype': celltype,
                'skipped': metrics.get('skipped', False)
            }
            if metrics.get('skipped'):
                ct_row['skip_reason'] = metrics.get('skip_reason', 'unknown')
            if n_probe_genes is not None:
                ct_row['n_probe_genes'] = n_probe_genes
            if gene_list_name is not None:
                ct_row['gene_list'] = gene_list_name
            rows.append(ct_row)
    
    # Save to CSV
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(method_dir, f'{dataset_name}.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved unified metrics to {csv_path}")
    else:
        logging.warning("No metrics to save!")


##############################################################################
# GLOBAL RECONSTRUCTION FUNCTIONS
##############################################################################

# Global reconstruction trains ONE model on ALL cells (mixed cell types)
# and evaluates generalization on held-out test set

# Note: Neural Network global reconstruction is handled inline in main function
# using nn_utils.nn_predictor (no separate function needed)
# nn_predictor uses HARDCODED parameters from nn_utils_CE.py:
#   - n_latent=6 (not configurable)
#   - lr=0.01
#   - max_epochs=10000
#   - loss_fn=HuberLoss()
#   - convergence_cutoff=-0.001
#   - batch_size=64 (default)
# The nn_epochs, nn_batch_size, nn_lr parameters in this module are for
# documentation purposes and potential future customization.

# Note: VAE global reconstruction is handled inline in main function
# using train_lightning_vae + predict_lightning_vae or nn_utils.vae_predictor
# 
# vae_simple uses HARDCODED parameters from nn_utils_CE.py:
#   - n_latent=6 (not configurable)
#   - lr=0.01
#   - max_epochs=10000
#   - loss_fn=HuberLoss()
#   - convergence_cutoff=-0.001
#   - batch_size=64 (default)
# 
# Lightning VAE models (LVAE, LEVAE, LDVAE, VAE) use configurable parameters:
#   - latent_features (vae_n_latent)
#   - lr (vae_lr)
#   - max_epochs (vae_max_epochs)
#   - batch_size (nn_batch_size)

# Note: Tangram global reconstruction uses reconstruct_with_tangram function
# (runs on full dataset, no train/test split)

##############################################################################
# VAE RECONSTRUCTION
##############################################################################

# Multiple VAE architectures available:
# - vae_simple: Uses nn_utils.vae_predictor (simple VAE)
# - LVAE: Linear encoder + Linear decoder
# - LEVAE: Linear encoder + Nonlinear decoder
# - LDVAE: Nonlinear encoder + Linear decoder
# - VAE: Nonlinear encoder + Nonlinear decoder

from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader


def train_lightning_vae(
    X_subset_train, X_full_train,
    vae_type='LVAE',
    latent_features=64,
    lr=1e-3,
    max_epochs=100,
    batch_size=128
):
    """
    Train one of the Lightning VAE models from nn_deep_CE.py.
    
    Parameters:
    -----------
    X_subset_train : np.ndarray
        Training probe genes (n_cells × n_probe_genes)
    X_full_train : np.ndarray
        Training full transcriptome (n_cells × n_full_genes)
    vae_type : str
        VAE architecture: 'LVAE', 'LEVAE', 'LDVAE', 'VAE'
    latent_features : int
        Latent dimension size
    lr : float
        Learning rate
    max_epochs : int
        Maximum training epochs
    batch_size : int
        Batch size
        
    Returns:
    --------
    vae_model : BaseVAE
        Trained VAE model
    """
    if L is None:
        raise ImportError("PyTorch Lightning is not installed. Install with: pip install lightning")
    
    logging.info(f"Training {vae_type} model...")
    
    # Prepare data (expects raw counts, will log1p internally)
    input_tensor = torch.FloatTensor(X_subset_train)
    target_tensor = torch.FloatTensor(X_full_train)
    
    dataset = TensorDataset(input_tensor, target_tensor)
    train_loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model based on type
    n_input = X_subset_train.shape[1]
    n_output = X_full_train.shape[1]
    
    if vae_type == 'LVAE':
        vae_model = nn_deep.LVAE(n_input, n_output, latent_features, lr=lr)
    elif vae_type == 'LEVAE':
        vae_model = nn_deep.LEVAE(n_input, n_output, latent_features, lr=lr)
    elif vae_type == 'LDVAE':
        vae_model = nn_deep.LDVAE(n_input, n_output, latent_features, lr=lr)
    elif vae_type == 'VAE':
        vae_model = nn_deep.VAE(n_input, n_output, latent_features, lr=lr)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    
    # Train with Lightning
    trainer = L.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        enable_model_summary=True,
        accelerator='auto',
        logger=False,
        enable_checkpointing=False
    )
    
    trainer.fit(vae_model, train_loader)
    
    logging.info(f"{vae_type} training complete.")
    
    return vae_model


def predict_lightning_vae(vae_model, X_subset):
    """
    Generate predictions using trained Lightning VAE.
    
    Parameters:
    -----------
    vae_model : BaseVAE
        Trained VAE model
    X_subset : np.ndarray
        Probe gene matrix (n_cells × n_probe_genes)
        
    Returns:
    --------
    X_reconstructed : np.ndarray
        Reconstructed full transcriptome (n_cells × n_full_genes)
    """
    vae_model.eval()
    
    input_tensor = torch.FloatTensor(X_subset)
    
    with torch.no_grad():
        # Forward pass (log_space=False means it will log1p internally)
        mu_x, _, _ = vae_model(input_tensor, log_space=False)
        # Transform back to count space
        reconstructed = torch.expm1(mu_x).clamp(min=0)
    
    return reconstructed.numpy()


##############################################################################
# TANGRAM RECONSTRUCTION
##############################################################################

def reconstruct_with_tangram(adata_full, adata_subset, num_epochs=500):
    """
    Use Tangram to reconstruct full transcriptome from probe genes.
    
    NOTE: Tangram uses FULL dataset (no train/test split).
    
    Parameters:
    -----------
    adata_full : AnnData
        Full transcriptome data (all cells)
    adata_subset : AnnData
        Probe panel data (all cells)
    num_epochs : int
        Number of Tangram training epochs
        
    Returns:
    --------
    ad_ge : AnnData
        Reconstructed gene expression
    """
    if tg is None:
        raise ImportError("Tangram is not installed. Install with: pip install tangram-sc")
    
    logging.info("Running Tangram reconstruction...")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Preprocess data: find common genes and prepare for mapping
    # This modifies the AnnData objects in place, so we need to make copies
    ad_sc_copy = adata_full.copy()
    ad_sp_copy = adata_subset.copy()
    
    logging.info("Preprocessing data with tg.pp_adatas()...")
    tg.pp_adatas(ad_sc_copy, ad_sp_copy, genes=None)
    logging.info("Preprocessing complete.")
    
    # Map cells
    ad_map = tg.map_cells_to_space(
        ad_sc_copy,
        ad_sp_copy,
        mode='cells',
        density_prior='rna_count_based',
        num_epochs=num_epochs,
        device=device
    )
    
    logging.info("Tangram mapping complete. Projecting genes...")
    
    # Project genes
    ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=ad_sc_copy)
    
    logging.info("Tangram reconstruction complete.")
    
    return ad_ge


##############################################################################
# PER-CELL-TYPE RECONSTRUCTION FUNCTIONS
##############################################################################

# Per-cell-type reconstruction trains SEPARATE models for EACH cell type
# and evaluates each model independently


def train_per_celltype_neural_network(
    adata_full, adata_subset, celltype_col, method_dir,
    test_size, random_state, min_cells=30, per_ct_splits=None
):
    """
    Train and evaluate Neural Network separately for each cell type.
    
    Parameters:
    -----------
    adata_full : AnnData
        Full transcriptome data
    adata_subset : AnnData
        Probe panel data
    celltype_col : str
        Column name for cell types
    method_dir : str
        Base directory for this method's results
    test_size : float
        Fraction for test set
    random_state : int
        Random seed
    min_cells : int
        Minimum cells required per cell type
    per_ct_splits : dict, optional
        Pre-computed train/test splits per cell type {celltype: (train_idx, test_idx)}
        If provided, these splits will be used instead of creating new ones
        
    Returns:
    --------
    all_results : dict
        Dictionary mapping cell type to metrics
    """
    all_results = {}
    celltypes = adata_full.obs[celltype_col].unique()
    
    X_full = adata_full.X.toarray() if scipy.sparse.issparse(adata_full.X) else adata_full.X
    X_subset = adata_subset.X.toarray() if scipy.sparse.issparse(adata_subset.X) else adata_subset.X
    
    for celltype in celltypes:
        logging.info(f"\n  Processing cell type: {celltype}")
        
        # Filter to this cell type
        ct_mask = adata_full.obs[celltype_col] == celltype
        n_cells_ct = ct_mask.sum()
        
        if n_cells_ct < min_cells:
            logging.warning(f"  Skipping {celltype}: only {n_cells_ct} cells (minimum: {min_cells})")
            all_results[celltype] = {
                'skipped': True,
                'skip_reason': 'insufficient_cells',
                'n_cells': int(n_cells_ct)
            }
            continue
        
        X_full_ct = X_full[ct_mask]
        X_subset_ct = X_subset[ct_mask]
        
        # Train/test split (use pre-computed if available)
        if per_ct_splits is not None and celltype in per_ct_splits:
            train_idx, test_idx = per_ct_splits[celltype]
            logging.info(f"  Using pre-computed train/test split")
        else:
            train_idx, test_idx = train_test_split(
                np.arange(n_cells_ct),
                test_size=test_size,
                random_state=random_state
            )
        
        X_subset_train = X_subset_ct[train_idx]
        X_subset_test = X_subset_ct[test_idx]
        X_full_train = X_full_ct[train_idx]
        X_full_test = X_full_ct[test_idx]
        
        logging.info(f"  Train: {len(train_idx)} cells, Test: {len(test_idx)} cells")
        
        # Train and predict
        reference_train = np.hstack([X_subset_train, X_full_train])
        X_recon_test = nn_utils.nn_predictor(X_subset_test, reference_train)
        
        # Calculate metrics
        mse = calculate_mse(X_full_test, X_recon_test)
        expvar = calculate_explained_variance(X_full_test, X_recon_test)
        
        all_results[celltype] = {
            'mse_test_probe': float(mse),
            'expvar_test_probe': float(expvar),
            'rmse_test_probe': float(np.sqrt(mse)),
            'mae_test_probe': float(np.mean(np.abs(X_full_test - X_recon_test))),
            'r2_test_probe': float(r2_score(X_full_test.flatten(), X_recon_test.flatten())),
            'pearson_test_probe': float(pearsonr(X_full_test.flatten(), X_recon_test.flatten())[0]),
            'n_cells': int(n_cells_ct),
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'skipped': False
        }
        
        logging.info(f"  {celltype} complete: MSE={mse:.4f}, ExpVar={expvar:.4f}")
    
    return all_results


def train_per_celltype_vae(
    adata_full, adata_subset, celltype_col, method_dir, vae_method,
    test_size, random_state, vae_n_latent, vae_lr, vae_max_epochs,
    nn_batch_size, min_cells=30, per_ct_splits=None
):
    """
    Train and evaluate VAE separately for each cell type.
    
    Parameters:
    -----------
    adata_full : AnnData
        Full transcriptome data
    adata_subset : AnnData
        Probe panel data
    celltype_col : str
        Column name for cell types
    method_dir : str
        Base directory for this method's results
    vae_method : str
        VAE type: 'vae_simple', 'LVAE', 'LEVAE', 'LDVAE', 'VAE'
    test_size : float
        Fraction for test set
    random_state : int
        Random seed
    vae_n_latent : int
        Latent dimensions
    vae_lr : float
        Learning rate
    vae_max_epochs : int
        Training epochs
    nn_batch_size : int
        Batch size
    min_cells : int
        Minimum cells required per cell type
    per_ct_splits : dict, optional
        Pre-computed train/test splits per cell type {celltype: (train_idx, test_idx)}
        If provided, these splits will be used instead of creating new ones
        
    Returns:
    --------
    all_results : dict
        Dictionary mapping cell type to metrics
    """
    all_results = {}
    celltypes = adata_full.obs[celltype_col].unique()
    
    X_full = adata_full.X.toarray() if scipy.sparse.issparse(adata_full.X) else adata_full.X
    X_subset = adata_subset.X.toarray() if scipy.sparse.issparse(adata_subset.X) else adata_subset.X
    
    for celltype in celltypes:
        logging.info(f"\n  Processing cell type: {celltype}")
        
        # Filter to this cell type
        ct_mask = adata_full.obs[celltype_col] == celltype
        n_cells_ct = ct_mask.sum()
        
        if n_cells_ct < min_cells:
            logging.warning(f"  Skipping {celltype}: only {n_cells_ct} cells (minimum: {min_cells})")
            all_results[celltype] = {
                'skipped': True,
                'skip_reason': 'insufficient_cells',
                'n_cells': int(n_cells_ct)
            }
            continue
        
        X_full_ct = X_full[ct_mask]
        X_subset_ct = X_subset[ct_mask]
        
        # Train/test split (use pre-computed if available)
        if per_ct_splits is not None and celltype in per_ct_splits:
            train_idx, test_idx = per_ct_splits[celltype]
            logging.info(f"  Using pre-computed train/test split")
        else:
            train_idx, test_idx = train_test_split(
                np.arange(n_cells_ct),
                test_size=test_size,
                random_state=random_state
            )
        
        X_subset_train = X_subset_ct[train_idx]
        X_subset_test = X_subset_ct[test_idx]
        X_full_train = X_full_ct[train_idx]
        X_full_test = X_full_ct[test_idx]
        
        logging.info(f"  Train: {len(train_idx)} cells, Test: {len(test_idx)} cells")
        
        # Train and predict
        if vae_method == 'vae_simple':
            reference_train = np.hstack([X_subset_train, X_full_train])
            X_recon_test = nn_utils.vae_predictor(X_subset_test, reference_train)
            X_recon_full_ct = nn_utils.vae_predictor(X_subset_ct, reference_train)
        else:
            vae_model = train_lightning_vae(
                X_subset_train, X_full_train,
                vae_type=vae_method,
                latent_features=vae_n_latent,
                lr=vae_lr,
                max_epochs=vae_max_epochs,
                batch_size=nn_batch_size
            )
            X_recon_test = predict_lightning_vae(vae_model, X_subset_test)
            X_recon_full_ct = predict_lightning_vae(vae_model, X_subset_ct)
            del vae_model
        
        # Calculate metrics
        mse = calculate_mse(X_full_test, X_recon_test)
        expvar = calculate_explained_variance(X_full_test, X_recon_test)
        
        all_results[celltype] = {
            'mse_test_probe': float(mse),
            'expvar_test_probe': float(expvar),
            'rmse_test_probe': float(np.sqrt(mse)),
            'mae_test_probe': float(np.mean(np.abs(X_full_test - X_recon_test))),
            'r2_test_probe': float(r2_score(X_full_test.flatten(), X_recon_test.flatten())),
            'pearson_test_probe': float(pearsonr(X_full_test.flatten(), X_recon_test.flatten())[0]),
            'n_cells': int(n_cells_ct),
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'skipped': False
        }
        
        logging.info(f"  {celltype} complete: MSE={mse:.4f}, ExpVar={expvar:.4f}")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_results


def train_per_celltype_tangram(
    adata_full, adata_subset, celltype_col, method_dir,
    tangram_num_epochs, min_cells=10
):
    """
    Run Tangram reconstruction separately for each cell type.
    
    NOTE: Tangram uses ALL cells of each cell type (no train/test split).
    Lower threshold (min_cells=10) since no split needed.
    
    Parameters:
    -----------
    adata_full : AnnData
        Full transcriptome data
    adata_subset : AnnData
        Probe panel data
    celltype_col : str
        Column name for cell types
    method_dir : str
        Base directory for this method's results
    tangram_num_epochs : int
        Tangram training epochs
    min_cells : int
        Minimum cells required per cell type (default: 10)
        
    Returns:
    --------
    all_results : dict
        Dictionary mapping cell type to metrics
    """
    if tg is None:
        raise ImportError("Tangram is not installed. Install with: pip install tangram-sc")
    
    all_results = {}
    celltypes = adata_full.obs[celltype_col].unique()
    
    for celltype in celltypes:
        logging.info(f"\n  Processing cell type: {celltype}")
        
        # Filter to this cell type
        ct_mask = adata_full.obs[celltype_col] == celltype
        n_cells_ct = ct_mask.sum()
        
        if n_cells_ct < min_cells:
            logging.warning(f"  Skipping {celltype}: only {n_cells_ct} cells (minimum: {min_cells})")
            all_results[celltype] = {
                'skipped': True,
                'skip_reason': 'insufficient_cells',
                'n_cells': int(n_cells_ct)
            }
            continue
        
        # Get cell type data
        adata_full_ct = adata_full[ct_mask].copy()
        adata_subset_ct = adata_subset[ct_mask].copy()
        
        logging.info(f"  Using all {n_cells_ct} cells (Tangram: no train/test split)")
        
        # Reconstruct with Tangram
        ad_ge = reconstruct_with_tangram(
            adata_full_ct, adata_subset_ct,
            num_epochs=tangram_num_epochs
        )
        
        # Tangram only reconstructs COMMON genes between full and subset
        # We need to align the full transcriptome to match
        # NOTE: Tangram's preprocessing may modify gene names (e.g., lowercase)
        # so we need case-insensitive matching
        common_genes = ad_ge.var_names
        
        # Create case-insensitive mapping: lowercase -> original gene name
        gene_mapping = {g.lower(): g for g in adata_full_ct.var_names}
        
        # Map Tangram's gene names to original gene names
        common_genes_mapped = [gene_mapping.get(g.lower(), g) for g in common_genes]
        
        # Filter to genes that exist in original data
        valid_genes = [g for g in common_genes_mapped if g in adata_full_ct.var_names]
        
        if len(valid_genes) < len(common_genes):
            logging.warning(f"  Only {len(valid_genes)}/{len(common_genes)} genes from Tangram found in original data")
        
        # Subset both to valid genes
        adata_full_ct_aligned = adata_full_ct[:, valid_genes].copy()
        ad_ge_aligned = ad_ge[:, [g for g in common_genes if gene_mapping.get(g.lower(), g) in valid_genes]].copy()
        
        # Calculate metrics (on common genes only)
        X_full_ct = adata_full_ct_aligned.X.toarray() if scipy.sparse.issparse(adata_full_ct_aligned.X) else adata_full_ct_aligned.X
        X_recon_ct = ad_ge_aligned.X.toarray() if scipy.sparse.issparse(ad_ge_aligned.X) else ad_ge_aligned.X
        
        logging.info(f"  Comparing on {len(valid_genes)} common genes (out of {adata_full_ct.n_vars} total)")
        
        mse = calculate_mse(X_full_ct, X_recon_ct)
        expvar = calculate_explained_variance(X_full_ct, X_recon_ct)
        
        all_results[celltype] = {
            'mse_test_probe': float(mse),
            'expvar_test_probe': float(expvar),
            'rmse_test_probe': float(np.sqrt(mse)),
            'mae_test_probe': float(np.mean(np.abs(X_full_ct - X_recon_ct))),
            'r2_test_probe': float(r2_score(X_full_ct.flatten(), X_recon_ct.flatten())),
            'pearson_test_probe': float(pearsonr(X_full_ct.flatten(), X_recon_ct.flatten())[0]),
            'n_cells': int(n_cells_ct),
            'n_train': int(n_cells_ct),
            'n_test': 0,
            'note': 'tangram_uses_full_celltype_dataset',
            'skipped': False
        }
        
        logging.info(f"  {celltype} complete: MSE={mse:.4f}, ExpVar={expvar:.4f}")
        
        # Cleanup
        del ad_ge, ad_ge_aligned
        gc.collect()
    
    return all_results


##############################################################################
# MAIN EVALUATION FUNCTION
##############################################################################

def run_reconstruction_evaluation(
    input_file,
    gene_list_file,
    output_dir,
    celltype_col='celltype',
    filter_name='Scanpy-Filter',
    gene_pool='All-Genes',
    methods=['neural_network', 'LVAE', 'tangram'],
    evaluation_mode='global',
    test_size=0.3,
    random_state=42,
    nn_epochs=10000,
    nn_batch_size=64,
    nn_lr=0.01,
    vae_n_latent=64,
    vae_max_epochs=100,
    vae_lr=0.001,
    tangram_num_epochs=500,
    device='cuda',
    exclude_probe_genes_from_eval=False
):
    """
    Run reconstruction evaluation with multiple methods.
    
    Parameters:
    -----------
    input_file : str
        Path to preprocessed h5ad file (full transcriptome)
    gene_list_file : str
        Path to CSV file with selected genes
    output_dir : str
        Output directory for results
    celltype_col : str
        Column name for cell types in .obs
    filter_name : str
        Filter method name (e.g., 'Scanpy-Filter', 'Xenium-Filter')
    gene_pool : str
        Gene pool name (e.g., 'All-Genes', 'HVG')
    methods : list
        Methods to use: ['neural_network', 'vae_simple', 'LVAE', 'LEVAE', 'LDVAE', 'VAE', 'tangram']
    evaluation_mode : str
        Evaluation mode: 'global', 'per_celltype', or 'both'
        - 'global': Train one model on all cells (mixed cell types)
        - 'per_celltype': Train separate models for each cell type
        - 'both': Run both global and per-cell-type evaluations
    test_size : float
        Fraction of data for test set (for NN and VAE)
    random_state : int
        Random seed for train/test split
    nn_epochs : int
        Neural Network training epochs (default: 10000 with convergence check)
    nn_batch_size : int
        Neural Network batch size (default: 64)
    nn_lr : float
        Neural Network learning rate (default: 0.01)
    vae_n_latent : int
        VAE latent dimensions
    vae_max_epochs : int
        VAE training epochs
    vae_lr : float
        VAE learning rate
    tangram_num_epochs : int
        Tangram training epochs
    device : str
        'cuda' or 'cpu'
    """
    
    # Create output directories (method-first structure like baseline evaluation)
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract strategy and size from gene list file
    gene_list_basename = os.path.splitext(os.path.basename(gene_list_file))[0]
    
    # Get directory path to extract strategy info
    gene_list_dir = os.path.dirname(gene_list_file)
    
    # Extract filter name from actual path (e.g., Scanpy-Filter, Xenium-Filter)
    # Path structure: .../Scanpy-Filter/All-Genes/strategy/100-genes/results/selected_genes.csv
    path_parts = gene_list_dir.split('/')
    
    # Find filter name (should be before All-Genes or HVG in the path)
    actual_filter_name = filter_name  # Default to parameter
    actual_gene_pool = gene_pool      # Default to parameter
    
    for i, part in enumerate(path_parts):
        if 'Filter' in part or part in ['Scanpy-Filter', 'Xenium-Filter', 'No-Filter']:
            actual_filter_name = part
        if part in ['All-Genes', 'HVG']:
            actual_gene_pool = part
    
    # Extract size from directory structure (e.g., "100-genes" or "100")
    size = None
    if '-genes' in gene_list_dir:
        size_match = gene_list_dir.split('/')[-2] if '/results/' in gene_list_dir else gene_list_dir.split('/')[-1]
        size = size_match.replace('-genes', '')
    else:
        # For external datasets like Spapros (e.g., .../Spapros/Scanpy-Filter_HVG/100/probeset.csv)
        parts = gene_list_dir.split('/')
        for part in reversed(parts):
            if part.isdigit():
                size = part
                break
    
    # Extract strategy from path
    # For strategy folders: .../Selected-panels/Xenium-Filter/All-Genes/dt_nmf_DT0.75.../100-genes/results/selected_genes.csv
    # For external: .../Spapros/Scanpy-Filter_HVG/100/probeset.csv
    if '/probeset.csv' in gene_list_file:
        # External dataset - extract from path
        parts = gene_list_dir.split('/')
        # Get dataset name (e.g., "Spapros")
        dataset_name_base = parts[-3] if len(parts) >= 3 else "external"
        strategy = dataset_name_base
    else:
        # Strategy folder - extract from directory structure
        # Find the strategy name between filter/gene_pool and size folder
        parts = gene_list_dir.split('/')
        strategy_start_idx = -1
        for i, part in enumerate(parts):
            if part in [actual_filter_name, actual_gene_pool]:
                strategy_start_idx = i + 1
                break
        
        if strategy_start_idx > 0 and strategy_start_idx < len(parts):
            # Get all parts between gene_pool and size folder
            strategy_parts = []
            for i in range(strategy_start_idx, len(parts)):
                if parts[i].endswith('-genes') or parts[i] == 'results':
                    break
                strategy_parts.append(parts[i])
            strategy = '_'.join(strategy_parts) if strategy_parts else gene_list_basename
        else:
            strategy = gene_list_basename
    
    # Construct dataset name matching baseline format: {filter}_{pool}_{strategy}_{size}
    dataset_name = f"{actual_filter_name}_{actual_gene_pool}_{strategy}_{size}"
    
    # Setup logging
    log_file = os.path.join(output_dir, 'logs', f'reconstruction_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info("NMF-INDEPENDENT RECONSTRUCTION EVALUATION")
    logging.info("="*80)
    logging.info(f"Dataset name: {dataset_name}")
    logging.info(f"Filter: {actual_filter_name}")
    logging.info(f"Gene pool: {actual_gene_pool}")
    logging.info(f"Strategy: {strategy}")
    logging.info(f"Size: {size}")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Gene list: {gene_list_file}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Cell type column: {celltype_col}")
    logging.info(f"Methods: {methods}")
    logging.info(f"Evaluation mode: {evaluation_mode}")
    logging.info(f"Test size: {test_size}")
    logging.info(f"Random state: {random_state}")
    logging.info(f"Device: {device}")
    
    # Load data
    logging.info("Loading data...")
    adata_full = sc.read_h5ad(input_file)
    selected_genes = pd.read_csv(gene_list_file).iloc[:, 0].values
    
    logging.info(f"Full transcriptome: {adata_full.shape[0]} cells x {adata_full.shape[1]} genes")
    logging.info(f"Selected genes: {len(selected_genes)}")
    
    # Data validation: Check if data is log-transformed
    sample_max = adata_full.X.max() if not scipy.sparse.issparse(adata_full.X) else adata_full.X.data.max()
    if sample_max < 20:
        logging.warning(
            "\n" + "="*80 + "\n"
            "WARNING: Input data appears to be log-transformed (max value < 20).\n"
            "Neural network and VAE methods expect RAW COUNT data.\n"
            "Using log-transformed data may lead to suboptimal reconstruction quality.\n"
            "Consider using raw count data or re-preprocessing without log transformation.\n"
            + "="*80
        )
    
    # Check cell type column
    if celltype_col not in adata_full.obs.columns:
        raise ValueError(f"Cell type column '{celltype_col}' not found in .obs")
    
    # Subset to selected genes
    selected_genes = [g for g in selected_genes if g in adata_full.var_names]
    logging.info(f"Genes found in data: {len(selected_genes)}")
    
    adata_subset = adata_full[:, selected_genes].copy()
    
    # Convert to dense arrays
    X_full = adata_full.X.toarray() if scipy.sparse.issparse(adata_full.X) else adata_full.X.copy()
    X_subset = adata_subset.X.toarray() if scipy.sparse.issparse(adata_subset.X) else adata_subset.X.copy()
    
    log_memory_usage("after loading data")
    
    # ========================================================================
    # DETERMINE EVALUATION MODES TO RUN
    # ========================================================================
    
    run_global = evaluation_mode in ['global', 'both']
    run_per_celltype = evaluation_mode in ['per_celltype', 'both']
    
    # ========================================================================
    # CREATE TRAIN/TEST SPLITS (ONCE, SHARED ACROSS ALL METHODS)
    # ========================================================================
    
    global_train_idx = None
    global_test_idx = None
    per_ct_splits = {}
    
    if run_global:
        # Create global split once (stratified by cell type)
        logging.info("\n" + "="*80)
        logging.info("CREATING GLOBAL TRAIN/TEST SPLIT")
        logging.info(f"Split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test (random_state={random_state})")
        logging.info("This split will be used by ALL methods for fair comparison")
        logging.info("="*80)
        
        global_train_idx, global_test_idx = train_test_split(
            np.arange(X_full.shape[0]), 
            test_size=test_size, 
            random_state=random_state,
            stratify=adata_full.obs[celltype_col].values
        )
        
        logging.info(f"Global split: {len(global_train_idx)} train cells, {len(global_test_idx)} test cells")
    
    if run_per_celltype:
        # Create per-cell-type splits once (will be shared across all methods)
        logging.info("\n" + "="*80)
        logging.info("CREATING PER-CELL-TYPE TRAIN/TEST SPLITS")
        logging.info(f"Split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test (random_state={random_state})")
        logging.info("These splits will be used by ALL methods for fair comparison")
        logging.info("="*80)
        
        celltypes = adata_full.obs[celltype_col].unique()
        for celltype in celltypes:
            ct_mask = adata_full.obs[celltype_col] == celltype
            n_cells_ct = ct_mask.sum()
            
            if n_cells_ct >= 30:  # Only split if sufficient cells
                train_idx, test_idx = train_test_split(
                    np.arange(n_cells_ct),
                    test_size=test_size,
                    random_state=random_state
                )
                per_ct_splits[celltype] = (train_idx, test_idx)
                logging.info(f"  {celltype}: {len(train_idx)} train, {len(test_idx)} test")
            else:
                logging.info(f"  {celltype}: {n_cells_ct} cells (insufficient for split)")
        
        logging.info(f"Created splits for {len(per_ct_splits)} cell types")
    
    # ========================================================================
    # GLOBAL RECONSTRUCTION MODE
    # ========================================================================
    
    if run_global:
        logging.info("\n" + "="*80)
        logging.info("GLOBAL RECONSTRUCTION MODE")
        logging.info("Training models on ALL cells (mixed cell types)")
        logging.info("Using SHARED train/test split across all methods")
        logging.info("="*80)
        
        global_results_dir = os.path.join(results_dir, 'Global_Reconstruction')
        os.makedirs(global_results_dir, exist_ok=True)
        
        # Prepare global train/test data (shared across all methods)
        X_subset_train_global = X_subset[global_train_idx]
        X_subset_test_global = X_subset[global_test_idx]
        X_full_train_global = X_full[global_train_idx]
        X_full_test_global = X_full[global_test_idx]
        
        # ====================================================================
        # GLOBAL MODE: NEURAL NETWORK
        # ====================================================================
        
        if 'neural_network' in methods:
            logging.info("\n" + "-"*80)
            logging.info("GLOBAL: SIMPLE NEURAL NETWORK")
            logging.info("-"*80)
            
            method_dir = os.path.join(global_results_dir, 'Neural_Network')
            os.makedirs(method_dir, exist_ok=True)
            
            logging.info(f"Using shared split: {len(global_train_idx)} train, {len(global_test_idx)} test")
            
            # Log evaluation mode
            if exclude_probe_genes_from_eval:
                logging.info("Evaluation mode: PROBE-EXCLUDED (only evaluating non-probe gene prediction)")
            else:
                logging.info("Evaluation mode: FULL TRANSCRIPTOME (evaluating all genes including probe genes)")
            
            # Train and predict (using shared split)
            reference_train = np.hstack([X_subset_train_global, X_full_train_global])
            X_recon_test = nn_utils.nn_predictor(X_subset_test_global, reference_train)
            X_recon_full = nn_utils.nn_predictor(X_subset, reference_train)
            
            # Calculate global metrics
            logging.info("Calculating global metrics...")
            global_metrics = calculate_global_metrics(X_full_test_global, X_recon_test)
            
            # Save unified metrics CSV
            save_unified_metrics(
                method_dir=method_dir,
                dataset_name=dataset_name,
                global_metrics=global_metrics,
                per_celltype_results=None,
                n_probe_genes=len(selected_genes),
                gene_list_name=dataset_name
            )
            
            # Save reconstructed expression
            adata_recon = sc.AnnData(X=X_recon_full, obs=adata_full.obs, var=adata_full.var)
            adata_recon.write_h5ad(os.path.join(method_dir, 'reconstructed_expression.h5ad'))
            
            logging.info(f"Results saved to {method_dir}")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # ====================================================================
        # GLOBAL MODE: VAE MODELS
        # ====================================================================
        
        vae_methods = [m for m in methods if m in ['vae_simple', 'LVAE', 'LEVAE', 'LDVAE', 'VAE']]
        
        for vae_method in vae_methods:
            logging.info("\n" + "-"*80)
            logging.info(f"GLOBAL: {vae_method.upper()}")
            logging.info("-"*80)
            
            method_dir = os.path.join(global_results_dir, vae_method)
            os.makedirs(method_dir, exist_ok=True)
            
            logging.info(f"Using shared split: {len(global_train_idx)} train, {len(global_test_idx)} test")
            
            # Train and predict (using shared split)
            if vae_method == 'vae_simple':
                reference_train = np.hstack([X_subset_train_global, X_full_train_global])
                X_recon_test = nn_utils.vae_predictor(X_subset_test_global, reference_train)
                X_recon_full = nn_utils.vae_predictor(X_subset, reference_train)
            else:
                vae_model = train_lightning_vae(
                    X_subset_train_global, X_full_train_global,
                    vae_type=vae_method,
                    latent_features=vae_n_latent,
                    lr=vae_lr,
                    max_epochs=vae_max_epochs,
                    batch_size=nn_batch_size
                )
                X_recon_test = predict_lightning_vae(vae_model, X_subset_test_global)
                X_recon_full = predict_lightning_vae(vae_model, X_subset)
                del vae_model
            
            # Calculate global metrics
            global_metrics = calculate_global_metrics(X_full_test_global, X_recon_test)
            
            # Save unified metrics CSV
            save_unified_metrics(
                method_dir=method_dir,
                dataset_name=dataset_name,
                global_metrics=global_metrics,
                per_celltype_results=None,
                n_probe_genes=len(selected_genes),
                gene_list_name=dataset_name
            )
            
            # Save reconstructed expression
            adata_recon = sc.AnnData(X=X_recon_full, obs=adata_full.obs, var=adata_full.var)
            adata_recon.write_h5ad(os.path.join(method_dir, 'reconstructed_expression.h5ad'))
            
            logging.info(f"Results saved to {method_dir}")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # ====================================================================
        # GLOBAL MODE: TANGRAM
        # ====================================================================
        
        if 'tangram' in methods:
            if tg is None:
                logging.warning("Tangram not installed. Skipping.")
            else:
                logging.info("\n" + "-"*80)
                logging.info("GLOBAL: TANGRAM")
                logging.info("-"*80)
                logging.info("NOTE: Tangram uses FULL dataset (no train/test split)")
                
                method_dir = os.path.join(global_results_dir, 'Tangram')
                os.makedirs(method_dir, exist_ok=True)
                
                # Reconstruct
                ad_ge = reconstruct_with_tangram(
                    adata_full, adata_subset,
                    num_epochs=tangram_num_epochs
                )
                
                # Tangram only reconstructs COMMON genes between full and subset
                # We need to align the full transcriptome to match
                # NOTE: Tangram's preprocessing may modify gene names (e.g., lowercase)
                # so we need case-insensitive matching
                common_genes = ad_ge.var_names
                
                # Create case-insensitive mapping: lowercase -> original gene name
                gene_mapping = {g.lower(): g for g in adata_full.var_names}
                
                # Map Tangram's gene names to original gene names
                common_genes_mapped = [gene_mapping.get(g.lower(), g) for g in common_genes]
                
                # Filter to genes that exist in original data
                valid_genes = [g for g in common_genes_mapped if g in adata_full.var_names]
                
                if len(valid_genes) < len(common_genes):
                    logging.warning(f"Only {len(valid_genes)}/{len(common_genes)} genes from Tangram found in original data")
                
                # Subset both to valid genes
                adata_full_aligned = adata_full[:, valid_genes].copy()
                ad_ge_aligned = ad_ge[:, [g for g in common_genes if gene_mapping.get(g.lower(), g) in valid_genes]].copy()
                
                X_full_aligned = adata_full_aligned.X.toarray() if scipy.sparse.issparse(adata_full_aligned.X) else adata_full_aligned.X
                
                logging.info(f"Comparing on {len(valid_genes)} common genes (out of {adata_full.n_vars} total)")
                
                # Calculate global metrics (on common genes only)
                X_recon = ad_ge_aligned.X.toarray() if scipy.sparse.issparse(ad_ge_aligned.X) else ad_ge_aligned.X
                global_metrics = calculate_global_metrics(X_full_aligned, X_recon)
                global_metrics['note'] = 'tangram_uses_full_dataset_no_split'
                global_metrics['n_common_genes'] = len(valid_genes)
                
                # Save unified metrics CSV
                save_unified_metrics(
                    method_dir=method_dir,
                    dataset_name=dataset_name,
                    global_metrics=global_metrics,
                    per_celltype_results=None,
                    n_probe_genes=len(selected_genes),
                    gene_list_name=dataset_name
                )
                
                # Save reconstructed expression (aligned version)
                ad_ge_aligned.write_h5ad(os.path.join(method_dir, 'reconstructed_expression.h5ad'))
                
                logging.info(f"Results saved to {method_dir}")
                del ad_ge, ad_ge_aligned
                gc.collect()
    
    # ========================================================================
    # PER-CELL-TYPE RECONSTRUCTION MODE
    # ========================================================================
    
    if run_per_celltype:
        logging.info("\n" + "="*80)
        logging.info("PER-CELL-TYPE RECONSTRUCTION MODE")
        logging.info("Training separate models for each cell type")
        logging.info("Using SHARED train/test splits across all methods")
        logging.info("="*80)
        
        per_ct_results_dir = os.path.join(results_dir, 'PerCellType_Reconstruction')
        os.makedirs(per_ct_results_dir, exist_ok=True)
        
        # ====================================================================
        # PER-CELL-TYPE: NEURAL NETWORK
        # ====================================================================
        
        if 'neural_network' in methods:
            logging.info("\n" + "-"*80)
            logging.info("PER-CELL-TYPE: NEURAL NETWORK")
            logging.info("-"*80)
            
            method_dir = os.path.join(per_ct_results_dir, 'neural_network')
            os.makedirs(method_dir, exist_ok=True)
            all_results = train_per_celltype_neural_network(
                adata_full, adata_subset, celltype_col, method_dir,
                test_size, random_state, min_cells=30, per_ct_splits=per_ct_splits
            )
            
            # Save unified metrics CSV
            save_unified_metrics(
                method_dir=method_dir,
                dataset_name=dataset_name,
                global_metrics=None,
                per_celltype_results=all_results,
                n_probe_genes=len(selected_genes),
                gene_list_name=dataset_name
            )
            
            logging.info(f"Neural Network per-cell-type evaluation complete")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # ====================================================================
        # PER-CELL-TYPE: VAE MODELS
        # ====================================================================
        
        vae_methods = [m for m in methods if m in ['vae_simple', 'LVAE', 'LEVAE', 'LDVAE', 'VAE']]
        
        for vae_method in vae_methods:
            logging.info("\n" + "-"*80)
            logging.info(f"PER-CELL-TYPE: {vae_method.upper()}")
            logging.info("-"*80)
            
            method_dir = os.path.join(per_ct_results_dir, vae_method)
            os.makedirs(method_dir, exist_ok=True)
            all_results = train_per_celltype_vae(
                adata_full, adata_subset, celltype_col, method_dir, vae_method,
                test_size, random_state, vae_n_latent, vae_lr, vae_max_epochs,
                nn_batch_size, min_cells=30, per_ct_splits=per_ct_splits
            )
            
            # Save unified metrics CSV
            save_unified_metrics(
                method_dir=method_dir,
                dataset_name=dataset_name,
                global_metrics=None,
                per_celltype_results=all_results,
                n_probe_genes=len(selected_genes),
                gene_list_name=dataset_name
            )
            
            logging.info(f"{vae_method} per-cell-type evaluation complete")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # ====================================================================
        # PER-CELL-TYPE: TANGRAM
        # ====================================================================
        
        if 'tangram' in methods:
            if tg is None:
                logging.warning("Tangram not installed. Skipping.")
            else:
                logging.info("\n" + "-"*80)
                logging.info("PER-CELL-TYPE: TANGRAM")
                logging.info("-"*80)
                logging.info("NOTE: Tangram uses all cells per cell type (no train/test split)")
                
                method_dir = os.path.join(per_ct_results_dir, 'tangram')
                os.makedirs(method_dir, exist_ok=True)
                all_results = train_per_celltype_tangram(
                    adata_full, adata_subset, celltype_col, method_dir,
                    tangram_num_epochs, min_cells=10
                )
                
                # Save unified metrics CSV
                save_unified_metrics(
                    method_dir=method_dir,
                    dataset_name=dataset_name,
                    global_metrics=None,
                    per_celltype_results=all_results,
                    n_probe_genes=len(selected_genes),
                    gene_list_name=dataset_name
                )
                
                logging.info(f"Tangram per-cell-type evaluation complete")
                gc.collect()
    
    logging.info("\n" + "="*80)
    logging.info("EVALUATION COMPLETE")
    logging.info("="*80)
    log_memory_usage("final")
    
    return results_dir


##############################################################################
# PLOTTING FUNCTIONS (from 20251119_Plot-NMF-independent-by-method_HH.py)
##############################################################################

def extract_gene_count(dataset_name):
    """
    Extract gene count (100 or 200) from dataset name.
    
    Parameters:
    -----------
    dataset_name : str
        Dataset name (e.g., 'Xenium-Filter_All-Genes_hvg_100')
        
    Returns:
    --------
    gene_count : int or None
        Gene count (100 or 200) or None if not found
    """
    if '_100' in dataset_name:
        return 100
    elif '_200' in dataset_name:
        return 200
    else:
        return None

def simplify_dataset_label(dataset_name):
    """
    Simplify dataset label by removing redundant prefix.
    
    Parameters:
    -----------
    dataset_name : str
        Full dataset name (e.g., 'Xenium-Filter_All-Genes_hvg_100')
        
    Returns:
    --------
    simplified_name : str
        Simplified name (e.g., 'hvg_100' or 'nmf_global_method_b_200')
    """
    # Remove common prefix 'Xenium-Filter_All-Genes_'
    if 'Xenium-Filter_All-Genes_' in dataset_name:
        simplified = dataset_name.replace('Xenium-Filter_All-Genes_', '')
    elif 'Spapros_' in dataset_name:
        # For Spapros datasets, keep the format
        simplified = dataset_name
    else:
        simplified = dataset_name
    
    return simplified

def calculate_macro_metrics_from_df(df):
    """
    Calculate macro-averaged metrics from per-cell-type results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with per-cell-type metrics
        
    Returns:
    --------
    metrics : dict
        Dictionary with macro metrics
    """
    # Filter out skipped cell types
    valid_df = df[~df['skipped']].copy()
    
    if len(valid_df) == 0:
        return {
            'macro_mse': np.nan,
            'macro_expvar': np.nan,
            'macro_rmse': np.nan
        }
    
    return {
        'macro_mse': valid_df['mse_test_probe'].mean(),
        'macro_expvar': valid_df['expvar_test_probe'].mean(),
        'macro_rmse': valid_df['rmse_test_probe'].mean()
    }

def calculate_weighted_metrics_from_df(df):
    """
    Calculate weighted-averaged metrics from per-cell-type results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with per-cell-type metrics
        
    Returns:
    --------
    metrics : dict
        Dictionary with weighted metrics
    """
    # Filter out skipped cell types
    valid_df = df[~df['skipped']].copy()
    
    if len(valid_df) == 0:
        return {
            'weighted_mse': np.nan,
            'weighted_expvar': np.nan,
            'weighted_rmse': np.nan
        }
    
    total_cells = valid_df['n_cells'].sum()
    
    return {
        'weighted_mse': (valid_df['mse_test_probe'] * valid_df['n_cells']).sum() / total_cells,
        'weighted_expvar': (valid_df['expvar_test_probe'] * valid_df['n_cells']).sum() / total_cells,
        'weighted_rmse': (valid_df['rmse_test_probe'] * valid_df['n_cells']).sum() / total_cells
    }

def collect_method_results(base_dir, method_name):
    """
    Collect all results for a specific method across all datasets.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing all dataset folders
    method_name : str
        Method name: 'neural_network', 'LVAE', 'tangram'
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with columns: dataset, macro_mse, macro_expvar, weighted_mse, weighted_expvar
    """
    results = []
    
    # Iterate through all dataset directories
    base_path = Path(base_dir)
    for dataset_dir in sorted(base_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        # Skip the comparison_plots directory itself
        if dataset_dir.name.startswith('comparison_plots'):
            continue
        
        # Look for method results
        method_results_dir = dataset_dir / 'results' / 'PerCellType_Reconstruction' / method_name
        
        if not method_results_dir.exists():
            logging.warning(f"No {method_name} results found for {dataset_dir.name}")
            continue
        
        # Find CSV file
        csv_files = list(method_results_dir.glob('*.csv'))
        if len(csv_files) == 0:
            logging.warning(f"No CSV files found in {method_results_dir}")
            continue
        
        csv_file = csv_files[0]  # Take first CSV file
        
        logging.info(f"Reading {method_name} results from {dataset_dir.name}: {csv_file.name}")
        
        try:
            # Read per-cell-type metrics
            df = pd.read_csv(csv_file)
            
            # Calculate macro and weighted metrics
            macro_metrics = calculate_macro_metrics_from_df(df)
            weighted_metrics = calculate_weighted_metrics_from_df(df)
            
            # Extract dataset name from directory
            dataset_name = dataset_dir.name
            gene_count = extract_gene_count(dataset_name)
            simplified_label = simplify_dataset_label(dataset_name)
            
            # Store results
            results.append({
                'dataset': dataset_name,
                'dataset_label': simplified_label,
                'gene_count': gene_count,
                'macro_mse': macro_metrics['macro_mse'],
                'macro_expvar': macro_metrics['macro_expvar'],
                'macro_rmse': macro_metrics['macro_rmse'],
                'weighted_mse': weighted_metrics['weighted_mse'],
                'weighted_expvar': weighted_metrics['weighted_expvar'],
                'weighted_rmse': weighted_metrics['weighted_rmse']
            })
            
        except Exception as e:
            logging.error(f"Error processing {csv_file}: {e}")
            continue
    
    if len(results) == 0:
        logging.warning(f"No valid results found for method: {method_name}")
        return None
    
    results_df = pd.DataFrame(results)
    
    logging.info(f"Collected {len(results_df)} datasets for {method_name}")
    
    return results_df

def plot_global_metrics(results_dir, output_dir, metrics=['mse', 'expvar'], dpi=300):
    """
    Create bar plots comparing global reconstruction metrics across methods.
    
    DEPRECATED: Use plot_method_mse_comparison and plot_method_expvar_comparison instead.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing Global_Reconstruction results
    output_dir : str
        Directory to save plots
    metrics : list
        Metrics to plot: 'mse', 'expvar', 'rmse', 'mae', 'r2', 'pearson'
    dpi : int
        Plot resolution (default: 300)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logging.info("\n" + "="*80)
    logging.info("PLOTTING GLOBAL METRICS")
    logging.info("="*80)
    
    # Find all method directories
    method_dirs = [d for d in os.listdir(results_dir) 
                   if os.path.isdir(os.path.join(results_dir, d))]
    
    if not method_dirs:
        logging.warning("No method directories found in results")
        return
    
    # Collect data from all methods
    plot_data = []
    for method_name in method_dirs:
        method_dir = os.path.join(results_dir, method_name)
        # Find CSV files in method directory (should be one per dataset)
        csv_files = [f for f in os.listdir(method_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            metrics_file = os.path.join(method_dir, csv_file)
            df = pd.read_csv(metrics_file)
            # Extract global row (analysis_type == 'global')
            global_rows = df[df['analysis_type'] == 'global']
            if len(global_rows) > 0:
                row = global_rows.iloc[0].to_dict()
                row['method'] = method_name
                plot_data.append(row)
                break  # Only process first CSV (should be only one per method)
        
        if not csv_files:
            logging.warning(f"No CSV files found for {method_name}")
    
    if not plot_data:
        logging.warning("No valid global metrics data found")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define method colors - updated with more vibrant, distinctive palette
    method_colors = {
        'Neural_Network': '#3498db',  # Bright blue
        'vae_simple': '#e74c3c',      # Vivid red
        'LVAE': '#2ecc71',            # Emerald green
        'LEVAE': '#f39c12',           # Orange
        'LDVAE': '#9b59b6',           # Purple
        'VAE': '#1abc9c',             # Turquoise
        'Tangram': '#e91e63'          # Pink
    }
    
    # Create plots for requested metrics
    os.makedirs(output_dir, exist_ok=True)
    
    # MSE plot
    if 'mse' in metrics and 'mse_global' in plot_df.columns:
        plt.figure(figsize=(10, 6))
        df_sorted = plot_df.sort_values('mse_global', ascending=True)
        colors = [method_colors.get(m, '#cccccc') for m in df_sorted['method']]
        
        sns.barplot(x='mse_global', y='method', data=df_sorted, palette=colors)
        plt.xlabel('MSE (lower is better)', fontsize=12)
        plt.ylabel('Method', fontsize=12)
        plt.title('Global Reconstruction: MSE Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'global_mse_comparison.png')
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {plot_path}")
    
    # Explained Variance plot
    if 'expvar' in metrics and 'expvar_global' in plot_df.columns:
        plt.figure(figsize=(10, 6))
        df_sorted = plot_df.sort_values('expvar_global', ascending=False)
        colors = [method_colors.get(m, '#cccccc') for m in df_sorted['method']]
        
        sns.barplot(x='expvar_global', y='method', data=df_sorted, palette=colors)
        plt.xlabel('Explained Variance (higher is better)', fontsize=12)
        plt.ylabel('Method', fontsize=12)
        plt.title('Global Reconstruction: Explained Variance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'global_expvar_comparison.png')
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {plot_path}")
    
    # RMSE plot
    if 'rmse' in metrics and 'rmse_global' in plot_df.columns:
        plt.figure(figsize=(10, 6))
        df_sorted = plot_df.sort_values('rmse_global', ascending=True)
        colors = [method_colors.get(m, '#cccccc') for m in df_sorted['method']]
        
        sns.barplot(x='rmse_global', y='method', data=df_sorted, palette=colors)
        plt.xlabel('RMSE (lower is better)', fontsize=12)
        plt.ylabel('Method', fontsize=12)
        plt.title('Global Reconstruction: RMSE Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'global_rmse_comparison.png')
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {plot_path}")
    
    # Pearson correlation plot
    if 'pearson' in metrics and 'pearson_global' in plot_df.columns:
        plt.figure(figsize=(10, 6))
        df_sorted = plot_df.sort_values('pearson_global', ascending=False)
        colors = [method_colors.get(m, '#cccccc') for m in df_sorted['method']]
        
        sns.barplot(x='pearson_global', y='method', data=df_sorted, palette=colors)
        plt.xlabel('Pearson Correlation (higher is better)', fontsize=12)
        plt.ylabel('Method', fontsize=12)
        plt.title('Global Reconstruction: Pearson Correlation Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'global_pearson_comparison.png')
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {plot_path}")


def plot_method_mse_comparison(results_df, method_name, gene_count, output_dir, dpi=300):
    """
    Create MSE comparison plot (macro + weighted) for a specific method and gene count.
    Uses horizontal bar plot style.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with aggregated metrics per dataset
    method_name : str
        Method name for plot title
    gene_count : int
        Gene count (100 or 200)
    output_dir : str
        Output directory for plots
    dpi : int
        Plot resolution
    """
    if results_df is None or len(results_df) == 0:
        logging.warning(f"No data to plot for {method_name}")
        return
    
    # Create figure with 2 subplots (macro and weighted)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(results_df) * 0.4)))
    
    # Define colors for datasets - use a vibrant color palette
    # Choose between different palettes based on number of datasets
    if len(results_df) <= 10:
        colors = sns.color_palette("Set2", len(results_df))
    else:
        colors = sns.color_palette("tab20", len(results_df))
    
    # Macro MSE
    ax = axes[0]
    df_sorted = results_df.sort_values('macro_mse', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['macro_mse'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('MSE (lower is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Macro Average MSE', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    max_val = df_sorted['macro_mse'].max()
    ax.set_xlim([0, max_val * 1.15])
    
    for i, (bar, value) in enumerate(zip(bars, df_sorted['macro_mse'])):
        if not np.isnan(value):
            if value > max_val * 0.15:
                ax.text(value - max_val * 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            else:
                ax.text(value + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='left', va='center', fontsize=8)
    
    # Weighted MSE
    ax = axes[1]
    df_sorted = results_df.sort_values('weighted_mse', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['weighted_mse'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('MSE (lower is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Weighted Average MSE', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    max_val = df_sorted['weighted_mse'].max()
    ax.set_xlim([0, max_val * 1.15])
    
    for i, (bar, value) in enumerate(zip(bars, df_sorted['weighted_mse'])):
        if not np.isnan(value):
            if value > max_val * 0.15:
                ax.text(value - max_val * 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            else:
                ax.text(value + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='left', va='center', fontsize=8)
    
    plt.suptitle(f'Per-Cell-Type Reconstruction: {method_name} MSE Comparison ({gene_count} genes)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{method_name}_{gene_count}genes_mse_comparison.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_file}")

def plot_method_expvar_comparison(results_df, method_name, gene_count, output_dir, dpi=300):
    """
    Create Explained Variance comparison plot (macro + weighted) for a specific method and gene count.
    Uses horizontal bar plot style.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with aggregated metrics per dataset
    method_name : str
        Method name for plot title
    gene_count : int
        Gene count (100 or 200)
    output_dir : str
        Output directory for plots
    dpi : int
        Plot resolution
    """
    if results_df is None or len(results_df) == 0:
        logging.warning(f"No data to plot for {method_name}")
        return
    
    # Create figure with 2 subplots (macro and weighted)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(results_df) * 0.4)))
    
    # Define colors for datasets - use a vibrant color palette
    # Choose between different palettes based on number of datasets
    if len(results_df) <= 10:
        colors = sns.color_palette("Set2", len(results_df))
    else:
        colors = sns.color_palette("tab20", len(results_df))
    
    # Macro Explained Variance
    ax = axes[0]
    df_sorted = results_df.sort_values('macro_expvar', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['macro_expvar'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('Explained Variance (higher is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Macro Average Explained Variance', fontsize=13, fontweight='bold')
    
    min_val = df_sorted['macro_expvar'].min()
    max_val = df_sorted['macro_expvar'].max()
    if min_val < 0:
        ax.set_xlim([min_val * 1.1, max(1.0, max_val * 1.15)])
    else:
        ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, df_sorted['macro_expvar'])):
        if not np.isnan(value):
            if value > 0.15:
                ax.text(value - 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            elif value >= 0:
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center', fontsize=8)
            else:
                ax.text(value - 0.05, bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}', ha='right', va='center', fontsize=8, color='red')
    
    # Weighted Explained Variance
    ax = axes[1]
    df_sorted = results_df.sort_values('weighted_expvar', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['weighted_expvar'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('Explained Variance (higher is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Weighted Average Explained Variance', fontsize=13, fontweight='bold')
    
    min_val = df_sorted['weighted_expvar'].min()
    max_val = df_sorted['weighted_expvar'].max()
    if min_val < 0:
        ax.set_xlim([min_val * 1.1, max(1.0, max_val * 1.15)])
    else:
        ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, df_sorted['weighted_expvar'])):
        if not np.isnan(value):
            if value > 0.15:
                ax.text(value - 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            elif value >= 0:
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center', fontsize=8)
            else:
                ax.text(value - 0.05, bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}', ha='right', va='center', fontsize=8, color='red')
    
    plt.suptitle(f'Per-Cell-Type Reconstruction: {method_name} Explained Variance Comparison ({gene_count} genes)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{method_name}_{gene_count}genes_expvar_comparison.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_file}")

def plot_aggregated_metrics(results_dir, output_dir, metrics=['mse', 'expvar'], dpi=300):
    """
    Create comparison plots for each method, split by gene count.
    
    OPTIMIZED VERSION: Creates separate plots per method and gene count.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing PerCellType_Reconstruction results
    output_dir : str
        Directory to save plots
    metrics : list
        Metrics to plot: 'mse', 'expvar' (default: both)
    dpi : int
        Plot resolution (default: 300)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logging.info("\n" + "="*80)
    logging.info("PLOTTING AGGREGATED PER-CELL-TYPE METRICS (BY METHOD)")
    logging.info("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parent directory (contains all dataset folders)
    parent_dir = os.path.dirname(results_dir.rstrip('/'))
    
    # Find all method directories
    method_dirs = [d for d in os.listdir(results_dir) 
                   if os.path.isdir(os.path.join(results_dir, d))]
    
    if not method_dirs:
        logging.warning("No method directories found")
        return
    
    # Process each method
    for method_name in method_dirs:
        logging.info(f"\nProcessing method: {method_name}")
        logging.info("-"*80)
        
        # Collect results for this method
        results_df = collect_method_results(parent_dir, method_name)
        
        if results_df is not None:
            # Split by gene count
            for gene_count in [100, 200]:
                df_subset = results_df[results_df['gene_count'] == gene_count].copy()
                
                if len(df_subset) == 0:
                    logging.warning(f"No {gene_count}-gene datasets found for {method_name}")
                    continue
                
                logging.info(f"  Processing {gene_count}-gene datasets ({len(df_subset)} datasets)")
                
                # Create plots based on requested metrics
                if 'mse' in metrics:
                    plot_method_mse_comparison(df_subset, method_name, gene_count, output_dir, dpi=dpi)
                
                if 'expvar' in metrics:
                    plot_method_expvar_comparison(df_subset, method_name, gene_count, output_dir, dpi=dpi)
                
                # Save metrics to CSV
                csv_file = os.path.join(output_dir, f'{method_name}_{gene_count}genes_metrics.csv')
                df_subset.to_csv(csv_file, index=False)
                logging.info(f"  Saved metrics: {csv_file}")
        else:
            logging.warning(f"Skipping {method_name} - no valid results found")


##############################################################################
# COMMAND LINE INTERFACE
##############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NMF-independent reconstruction evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to preprocessed h5ad file (full transcriptome)')
    parser.add_argument('--gene_list_file', type=str, required=True,
                       help='Path to CSV file with selected genes')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--celltype_col', type=str, default='celltype',
                       help='Column name for cell types in .obs (default: celltype)')
    parser.add_argument('--filter_name', type=str, default='Scanpy-Filter',
                       help='Filter method name for dataset naming (default: Scanpy-Filter). '
                            'Options: Scanpy-Filter, Xenium-Filter, No-Filter')
    parser.add_argument('--gene_pool', type=str, default='All-Genes',
                       help='Gene pool for dataset naming (default: All-Genes). '
                            'Options: All-Genes, HVG')
    parser.add_argument('--evaluation_mode', type=str, default='global',
                       choices=['global', 'per_celltype', 'both'],
                       help='Evaluation mode (default: global). Options: '
                            'global (train on all cells), '
                            'per_celltype (train separate model per cell type), '
                            'both (run both modes)')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['neural_network', 'LVAE', 'tangram'],
                       choices=['neural_network', 'vae_simple', 'LVAE', 'LEVAE', 'LDVAE', 'VAE', 'tangram'],
                       help='Methods to use (default: neural_network, LVAE, tangram). '
                            'VAE options: vae_simple (nn_utils), LVAE (linear), LEVAE (linear enc + nonlinear dec), '
                            'LDVAE (nonlinear enc + linear dec), VAE (nonlinear)')
    parser.add_argument('--test_size', type=float, default=0.3,
                       help='Test set fraction for NN/VAE (default: 0.3)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for train/test split (default: 42)')
    
    # Neural Network parameters (DOCUMENTATION ONLY - nn_utils.nn_predictor uses hardcoded values)
    # nn_utils.nn_predictor uses: n_latent=6, lr=0.01, max_epochs=10000, batch_size=64
    parser.add_argument('--nn_epochs', type=int, default=10000,
                       help='[NOT USED by neural_network method - uses hardcoded value in nn_utils.py] '
                            'Neural Network training epochs (default: 10000, with convergence check)')
    parser.add_argument('--nn_batch_size', type=int, default=64,
                       help='[SHARED] Batch size used by Lightning VAE models (default: 64). '
                            'neural_network method uses hardcoded value in nn_utils.py')
    parser.add_argument('--nn_lr', type=float, default=0.01,
                       help='[NOT USED by neural_network method - uses hardcoded value in nn_utils.py] '
                            'Neural Network learning rate (default: 0.01)')
    
    # VAE parameters (used by LVAE, LEVAE, LDVAE, VAE methods)
    # vae_simple method uses hardcoded values from nn_utils.vae_predictor
    parser.add_argument('--vae_n_latent', type=int, default=64,
                       help='[Lightning VAE only] VAE latent dimensions (default: 64). '
                            'Used by LVAE, LEVAE, LDVAE, VAE. vae_simple uses hardcoded n_latent=6')
    parser.add_argument('--vae_max_epochs', type=int, default=100,
                       help='[Lightning VAE only] VAE training epochs (default: 100). '
                            'Used by LVAE, LEVAE, LDVAE, VAE. vae_simple uses hardcoded max_epochs=10000')
    parser.add_argument('--vae_lr', type=float, default=0.001,
                       help='[Lightning VAE only] VAE learning rate (default: 0.001). '
                            'Used by LVAE, LEVAE, LDVAE, VAE. vae_simple uses hardcoded lr=0.01')
    
    # Tangram parameters
    parser.add_argument('--tangram_num_epochs', type=int, default=500,
                       help='Tangram training epochs (default: 500)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    # Evaluation mode
    parser.add_argument('--exclude_probe_genes_from_eval', action='store_true',
                       help='Exclude probe genes from reconstruction target (evaluate only non-probe gene prediction). '
                            'Default: False (evaluate full transcriptome reconstruction quality including probe genes)')
    
    # Plotting
    parser.add_argument('--create_plots', action='store_true',
                       help='Create comparison plots after evaluation')
    parser.add_argument('--plot_metrics', type=str, nargs='+', 
                       default=['mse', 'expvar'],
                       choices=['mse', 'expvar', 'rmse', 'mae', 'r2', 'pearson'],
                       help='Metrics to plot (default: mse, expvar)')
    parser.add_argument('--plot_dpi', type=int, default=300,
                       help='Plot resolution DPI (default: 300)')
    
    args = parser.parse_args()
    
    # Run evaluation
    results_dir = run_reconstruction_evaluation(
        input_file=args.input_file,
        gene_list_file=args.gene_list_file,
        output_dir=args.output_dir,
        celltype_col=args.celltype_col,
        filter_name=args.filter_name,
        gene_pool=args.gene_pool,
        methods=args.methods,
        evaluation_mode=args.evaluation_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        nn_epochs=args.nn_epochs,
        nn_batch_size=args.nn_batch_size,
        nn_lr=args.nn_lr,
        vae_n_latent=args.vae_n_latent,
        vae_max_epochs=args.vae_max_epochs,
        vae_lr=args.vae_lr,
        tangram_num_epochs=args.tangram_num_epochs,
        device=args.device,
        exclude_probe_genes_from_eval=args.exclude_probe_genes_from_eval
    )
    
    # Create plots if requested
    if args.create_plots and results_dir:
        plots_dir = os.path.join(args.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot global metrics if global mode was run
        if args.evaluation_mode in ['global', 'both']:
            global_results_dir = os.path.join(results_dir, 'Global_Reconstruction')
            if os.path.exists(global_results_dir):
                plot_global_metrics(
                    global_results_dir, 
                    plots_dir,
                    metrics=args.plot_metrics,
                    dpi=args.plot_dpi
                )
        
        # Plot aggregated metrics if per-cell-type mode was run
        if args.evaluation_mode in ['per_celltype', 'both']:
            per_ct_results_dir = os.path.join(results_dir, 'PerCellType_Reconstruction')
            if os.path.exists(per_ct_results_dir):
                plot_aggregated_metrics(
                    per_ct_results_dir,
                    plots_dir,
                    metrics=args.plot_metrics,
                    dpi=args.plot_dpi
                )
