"""Dimension reduction (NMF/PCA) gene selection with factor-aware resolution.

This module provides gene selection based on NMF or PCA loadings, supporting both
global (whole-dataset) and per-celltype analysis. Integrates factor-aware duplicate
resolution from _factor_aware.py.

Author: Refactored from _selection.py
Date: 2026-02-08
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from sklearn.decomposition import NMF, PCA

# Import utility functions for data validation
SCRIPT_DIR = Path(__file__).parent.absolute()
UTILITY_DIR = SCRIPT_DIR.parent / "Utility-module"
sys.path.insert(0, str(UTILITY_DIR))

try:
    from _validation import is_anndata_raw, is_anndata_raw_layer
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported utility functions from Utility-module")
    UTILITY_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import from Utility-module: {e}. Using fallback functions.")
    UTILITY_AVAILABLE = False
    
    # Fallback: Provide simple validation functions
    def is_anndata_raw(adata):
        """Check if data appears to be raw counts."""
        if hasattr(adata, 'raw') and adata.raw is not None:
            return True
        # Check if X contains integer-like values (raw counts)
        if adata.X is not None:
            sample = adata.X[:100, :100] if adata.X.shape[0] > 100 else adata.X
            if scipy.sparse.issparse(sample):
                sample = sample.toarray()
            return np.allclose(sample, np.round(sample), atol=1e-6)
        return False
    
    def is_anndata_raw_layer(adata, layer_name):
        """Check if layer appears to be raw counts."""
        if layer_name not in adata.layers:
            return False
        sample = adata.layers[layer_name][:100, :100] if adata.layers[layer_name].shape[0] > 100 else adata.layers[layer_name]
        if scipy.sparse.issparse(sample):
            sample = sample.toarray()
        return np.allclose(sample, np.round(sample), atol=1e-6)

# Import explained variance calculation from evaluation module
try:
    EVALUATION_DIR = SCRIPT_DIR.parent / "Evaluation-module"
    sys.path.insert(0, str(EVALUATION_DIR))
    from metrics import calculate_explained_variance, calculate_mse
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported explained variance functions from Evaluation-module")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import from Evaluation-module: {e}. Using local implementation.")
    # Fallback implementation
    def calculate_explained_variance(X_original, X_reconstructed):
        """Fallback: Calculate R² = 1 - (MSE / Variance)"""
        import scipy.sparse
        if scipy.sparse.issparse(X_original):
            X_original = X_original.toarray()
        if scipy.sparse.issparse(X_reconstructed):
            X_reconstructed = X_reconstructed.toarray()
        mse = np.mean((X_original - X_reconstructed) ** 2)
        total_variance = np.var(X_original)
        if total_variance == 0:
            return 0.0
        return 1 - (mse / total_variance)

# Use absolute imports (for script execution)
from _constants import (
    COL_CELLTYPE,
    COL_COMPONENT,
    COL_GENE,
    COL_RANK,
    COL_SELECTION_SCORE,
    DEFAULT_DIMRED_GENES_PER_COMPONENT,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
    DEFAULT_NMF_COMPONENTS,
    DEFAULT_N_COMPONENTS_PCA,
    DEFAULT_RANDOM_STATE,
    DEFAULT_NMF_INIT,
    DEFAULT_NMF_BETA_LOSS,
    DEFAULT_NMF_SOLVER,
    DEFAULT_NMF_MAX_ITER,
    DEFAULT_MIN_XENIUM_EXPRESSION,
    DEFAULT_MAX_XENIUM_EXPRESSION,
    MIN_FACTOR_CONTRIBUTION,
)
from _factor_aware import (
    resolve_duplicates_factor_aware_global,
    resolve_duplicates_factor_aware_per_celltype,
    fill_gap_factor_aware_global,
    fill_gap_factor_aware_per_celltype,
)
from _gene_list_builder import GeneListBuilder

logger = logging.getLogger(__name__)


# ============================================================================
# NMF/PCA COMPUTATION FUNCTIONS
# ============================================================================
# Note: calculate_explained_variance is imported from Evaluation-module/_variability.py

def compute_nmf_global(
    adata: AnnData,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    init: str = DEFAULT_NMF_INIT,
    beta_loss: str = DEFAULT_NMF_BETA_LOSS,
    solver: str = DEFAULT_NMF_SOLVER,
    max_iter: int = DEFAULT_NMF_MAX_ITER,
    cache_dir: Optional[str] = None,
    # cNMF options
    use_consensus_nmf: bool = False,
    k_min: int = 3,
    k_max: int = 15,
    k_step: int = 2,
    cnmf_n_iter: int = 20,
    k_selection_method: str = "silhouette",
    use_consensus_H: bool = False,
    cnmf_plot_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Compute global NMF on raw counts.

    Args:
        adata: Annotated data matrix (must have raw counts)
        n_components: Number of NMF factors
        random_state: Random seed
        cache_dir: Directory to save/load cached models (optional)

    Returns:
        Tuple of (loadings_df, model_data):
            - loadings_df: DataFrame with gene loadings (genes × factors)
            - model_data: Dict with W, H, model object, reconstruction error

    Raises:
        ValueError: If counts not available or not raw
    """
    logger.info(f"Computing global NMF with {n_components} components...")
    
    # Check for cached model
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'nmf_models', 'global_nmf.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached NMF model from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info("✓ Successfully loaded cached NMF model")
                return cached_data['loadings_df'], cached_data['model_data']
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}. Recomputing...")

    # Get raw counts
    if hasattr(adata, 'raw') and adata.raw is not None:
        counts = adata.raw.X
        gene_names = adata.raw.var_names
        logger.info("Using adata.raw.X for NMF (should contain raw counts)")
    elif 'counts' in adata.layers:
        counts = adata.layers['counts']
        gene_names = adata.var_names
        # Verify layer contains raw counts
        if not is_anndata_raw_layer(adata, 'counts'):
            logger.warning("Layer 'counts' does not appear to contain raw integer counts!")
        logger.info("Using adata.layers['counts'] for NMF")
    else:
        raise ValueError("No raw counts found (need adata.raw or adata.layers['counts'])")

    # Convert to dense if sparse
    if scipy.sparse.issparse(counts):
        counts_dense = counts.toarray()
    else:
        counts_dense = np.asarray(counts)

    # Filter genes by standard deviation
    std = np.std(counts_dense, axis=0)

    # Filter genes with zero standard deviation
    ind = np.where((std > 0))
    index = ind[0].astype(int)
    n2 = len(index)
    counts_dense = counts_dense[:, index]
    
    # Skip if too few features
    if n2 < n_components:
        logging.warning(f"Data has too few features with non-zero standard deviation ({n2}), skipping NMF...")
        return []

    # ── Optional cNMF pre-step: determine optimal K automatically ────────────
    if use_consensus_nmf:
        try:
            # Import consensus NMF from Utility-module
            import sys
            from pathlib import Path
            utility_module_path = Path(__file__).parent.parent / 'Utility-module'
            if str(utility_module_path) not in sys.path:
                sys.path.insert(0, str(utility_module_path))

            from _consensus_nmf import run_consensus_nmf_global, select_optimal_k
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            k_values = list(range(k_min, k_max + 1, k_step))
            logger.info(f"Running cNMF (global) to select optimal K from {k_values} ...")

            # Build a temporary AnnData with only the std-filtered genes so that
            # _consensus_nmf uses the same gene set we already prepared.
            import anndata as _ad
            import scipy.sparse as _sp
            _counts_sub = _sp.csr_matrix(counts_dense)
            _adata_sub = _ad.AnnData(X=_counts_sub)
            _adata_sub.var_names = gene_names[index]
            _adata_sub.obs_names = adata.obs_names

            cnmf_result = run_consensus_nmf_global(
                _adata_sub,
                k_values=k_values,
                n_iter=cnmf_n_iter,
                random_state=random_state,
            )
            cnmf_obj = cnmf_result["cnmf_object"]
            consensus_by_k = cnmf_result["consensus_by_k"]
            optimal_k = select_optimal_k(cnmf_obj, method=k_selection_method)
            logger.info(f"✓ cNMF selected optimal K={optimal_k} (method: {k_selection_method})")

            # Save k-selection plot
            if cnmf_plot_dir:
                Path(cnmf_plot_dir).mkdir(parents=True, exist_ok=True)
                _fig = cnmf_obj.plot_k_selection()
                _fig.savefig(
                    Path(cnmf_plot_dir) / "k_selection_global.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(_fig)
                logger.info(f"  K-selection plot saved to {cnmf_plot_dir}/k_selection_global.png")

            if use_consensus_H and optimal_k in consensus_by_k:
                # Use the consensus H matrix directly (factors × genes)
                H = consensus_by_k[optimal_k]["consensus_H"]
                W = consensus_by_k[optimal_k]["consensus_W"]
                n_components = optimal_k
                model_meta = {"source": "consensus_H", "optimal_k": optimal_k,
                              "k_selection_method": k_selection_method}
                logger.info(f"  Using consensus H matrix (shape: {H.shape})")
            else:
                # Re-run standard NMF with the optimal K
                n_components = optimal_k
                _nmf_opt = NMF(
                    n_components=n_components,
                    init=init,
                    random_state=random_state,
                    beta_loss=beta_loss,
                    solver=solver,
                    max_iter=max_iter,
                    alpha_W=0.0,
                    alpha_H=0.0,
                    l1_ratio=0,
                )
                W = _nmf_opt.fit_transform(counts_dense)
                H = _nmf_opt.components_
                model_meta = {"source": "nmf_with_optimal_k", "optimal_k": optimal_k,
                              "k_selection_method": k_selection_method,
                              "reconstruction_err": _nmf_opt.reconstruction_err_,
                              "nmf_model": _nmf_opt}
                logger.info(f"  Re-ran standard NMF with K={optimal_k}")

        except Exception as _exc:
            logger.warning(
                f"cNMF failed ({_exc}). Falling back to standard NMF with "
                f"n_components={n_components}."
            )
            use_consensus_nmf = False  # fall through to standard path below
            model_meta = {}

    if not use_consensus_nmf:
        # Standard NMF
        model_meta = {}

    if not use_consensus_nmf:
        # Run NMF
        nmf = NMF(n_components=n_components,
                init=init,
                random_state=random_state,
                beta_loss=beta_loss,
                solver=solver,
                max_iter=max_iter,
                alpha_W=0.0,
                alpha_H=0.0,
                l1_ratio=0)

        W = nmf.fit_transform(counts_dense)  # Cell × factors
        H = nmf.components_  # Factors × genes
        model_meta.update({
            'reconstruction_err': nmf.reconstruction_err_,
            'n_iter': nmf.n_iter_,
            'model': nmf,
        })

    # Create loadings DataFrame (genes × factors)
    loadings_df = pd.DataFrame(
        H.T,
        index=gene_names[index],
        columns=[f"NMF_{i+1}" for i in range(n_components)]
    )

    # Store model data
    model_data = {
        'W': W,
        'H': H,
        'gene_names': gene_names[index].tolist(),
        'n_components': n_components,
        'filtered_gene_indices': index,
        **model_meta,
    }
    
    # Cache model if directory provided
    if cache_dir:
        model_dir = os.path.join(cache_dir, 'nmf_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, 'global_nmf.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'loadings_df': loadings_df, 'model_data': model_data}, f)
            logger.info(f"✓ Saved NMF model to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")

    logger.info(f"✓ Global NMF complete: {loadings_df.shape}")
    return loadings_df, model_data


def compute_pca_global(
    adata: AnnData,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Compute global PCA on normalized data.

    Args:
        adata: Annotated data matrix
        n_components: Number of PCs
        random_state: Random seed
        cache_dir: Directory to save/load cached models (optional)

    Returns:
        Tuple of (loadings_df, model_data):
            - loadings_df: DataFrame with gene loadings (genes × PCs)
            - model_data: Dict with components, explained variance, model object

    Raises:
        ValueError: If data not normalized
    """
    logger.info(f"Computing global PCA with {n_components} components...")
    
    # Check for cached model
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'pca_models', 'global_pca.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached PCA model from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info("✓ Successfully loaded cached PCA model")
                return cached_data['loadings_df'], cached_data['model_data']
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}. Recomputing...")

    # Use normalized data (adata.X)
    if adata.X is None:
        raise ValueError("adata.X is None - need normalized data for PCA")

    # Convert to dense if sparse
    if scipy.sparse.issparse(adata.X):
        X_dense = adata.X.toarray()
    else:
        X_dense = np.asarray(adata.X)

    # Run PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_dense)

    # Create loadings DataFrame (genes × PCs)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=adata.var_names,
        columns=[f"PC_{i+1}" for i in range(n_components)]
    )
    
    # Store model data
    model_data = {
        'components': pca.components_,
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_,
        'model': pca,
        'gene_names': adata.var_names.tolist(),
        'n_components': n_components,
    }
    
    # Cache model if directory provided
    if cache_dir:
        model_dir = os.path.join(cache_dir, 'pca_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, 'global_pca.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'loadings_df': loadings_df, 'model_data': model_data}, f)
            logger.info(f"✓ Saved PCA model to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")

    logger.info(f"✓ Global PCA complete: {loadings_df.shape}")
    logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return loadings_df, model_data


def compute_nmf_per_celltype(
    adata: AnnData,
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_cells: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    init: str = DEFAULT_NMF_INIT,
    beta_loss: str = DEFAULT_NMF_BETA_LOSS,
    solver: str = DEFAULT_NMF_SOLVER,
    max_iter: int = DEFAULT_NMF_MAX_ITER,
    n_jobs: int = 1,
    parallel_backend: str = 'process',
    cache_dir: Optional[str] = None,
    # cNMF options
    use_consensus_nmf: bool = False,
    k_min: int = 3,
    k_max: int = 15,
    k_step: int = 2,
    cnmf_n_iter: int = 20,
    k_selection_method: str = "silhouette",
    use_consensus_H: bool = False,
    cnmf_plot_dir: Optional[str] = None,
) -> Tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Compute per-celltype NMF on raw counts.

    Args:
        adata: Annotated data matrix
        celltype_column: Column with cell type labels
        n_components: Number of NMF factors per celltype
        random_state: Random seed
        min_cells: Minimum cells per celltype
        n_jobs: Number of workers for per-celltype fits. 1 keeps sequential behavior.
        parallel_backend: Parallel backend when n_jobs > 1 ('process' or 'thread').

    Returns:
        Tuple of:
            - Dict mapping celltype → loadings DataFrame (genes × factors)
            - Dict mapping celltype → pd.Series of explained variance per factor
              Format: {celltype: pd.Series({factor_name: r2, ...})}

    Raises:
        ValueError: If counts not available or celltype column missing
    """
    logger.info(f"Computing per-celltype NMF (n_components={n_components})...")
    
    # Check for cached model
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'nmf_models', 'per_celltype_nmf.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached per-celltype NMF models from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # Validate cache
                if cached_data['n_components'] == n_components:
                    logger.info(f"✓ Successfully loaded cached per-celltype NMF: {len(cached_data['loadings'])} celltypes")
                    return cached_data['loadings'], cached_data['explained_variance']
                else:
                    logger.warning(
                        f"Cache n_components mismatch: cached={cached_data['n_components']}, "
                        f"requested={n_components}. Recomputing..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}. Recomputing...")

    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not in adata.obs")

    # ROBUST RAW COUNTS DETECTION
    # Always compare to raw counts for metrics
    # Test if adata.X is raw, else use counts layer
    is_log = not is_anndata_raw(adata)
    
    if is_log:
        # Data is normalized/log-transformed, need to find raw counts
        logger.info("adata.X appears to be normalized/log-transformed, searching for raw counts...")
        
        if 'counts' in adata.layers:
            counts_is_raw = is_anndata_raw_layer(adata, 'counts')
            if not counts_is_raw:
                logger.error("Counts layer is not raw data! Cannot compute metrics correctly.")
                raise ValueError("Counts layer does not contain raw data")
            
            counts_full = adata.layers['counts']
            gene_names = adata.var_names
            logger.info("Using adata.layers['counts'] for per-celltype NMF (verified as raw counts)")
        else:
            logger.error("No 'counts' layer found! Cannot compute metrics correctly.")
            raise ValueError("No raw counts layer found - cannot compute metrics")
    else:
        # If not log, check that adata.X is actually raw, else error
        if is_anndata_raw(adata):
            counts_full = adata.X
            gene_names = adata.var_names
            logger.info("Using adata.X for per-celltype NMF (detected as raw counts)")
        else:
            logger.error("Data is not raw and no counts layer is available! Cannot compute metrics.")
            raise ValueError("No raw data available for metrics computation.")

    celltype_loadings = {}
    celltype_explained_variance = {}

    nmf_params = {
        'init': init,
        'beta_loss': beta_loss,
        'solver': solver,
        'max_iter': max_iter,
    }

    # ── Optional cNMF pre-step: determine optimal K per celltype ──────────────
    # Maps celltype name → optimal K (int). Empty when cNMF is not requested.
    _optimal_k_per_ct: dict = {}
    _cnmf_ct_consensus_H: dict = {}   # celltype → consensus_H ndarray (for use_consensus_H)

    if use_consensus_nmf:
        try:
            # Import consensus NMF from Utility-module
            import sys
            from pathlib import Path
            utility_module_path = Path(__file__).parent.parent / 'Utility-module'
            if str(utility_module_path) not in sys.path:
                sys.path.insert(0, str(utility_module_path))

            from _consensus_nmf import run_consensus_nmf_per_celltype, select_optimal_k as _select_k
            import anndata as _ad_ct
            import scipy.sparse as _sp_ct

            logger.info(
                f"Running cNMF per-celltype (k={k_min}..{k_max} step {k_step}, "
                f"n_iter={cnmf_n_iter}, method={k_selection_method})..."
            )

            # Build a temporary AnnData with counts layer required by cNMF
            if _sp_ct.issparse(counts_full):
                _X_ct = counts_full.toarray().astype(np.float32)
            else:
                _X_ct = np.asarray(counts_full, dtype=np.float32)

            _obs_cnmf = adata.obs[[celltype_column]].copy()
            _var_cnmf = pd.DataFrame(index=pd.Index(gene_names))
            _adata_cnmf = _ad_ct.AnnData(X=_X_ct, obs=_obs_cnmf, var=_var_cnmf)
            _adata_cnmf.layers['counts'] = _X_ct.copy()

            k_values_list = list(range(k_min, k_max + 1, k_step))
            _cnmf_ct_results = run_consensus_nmf_per_celltype(
                _adata_cnmf,
                groupby=celltype_column,
                k_values=k_values_list,
                n_iter=cnmf_n_iter,
                random_state=random_state,
                results_dir=cnmf_plot_dir,
            )

            for _ct, _ct_res in _cnmf_ct_results.items():
                _cnmf_obj = _ct_res.get("cnmf_object")
                if _cnmf_obj is None:
                    continue
                try:
                    _opt_k = _select_k(_cnmf_obj, method=k_selection_method)
                    _optimal_k_per_ct[_ct] = _opt_k
                    logger.info(f"  cNMF {_ct}: optimal K = {_opt_k}")
                except Exception as _ke:
                    logger.warning(
                        f"  cNMF K-selection failed for {_ct} ({_ke}); "
                        f"using default n_components={n_components}"
                    )
                    _optimal_k_per_ct[_ct] = n_components

                # Store consensus H if requested (factors×genes → genes×factors after transpose)
                if use_consensus_H:
                    _opt_k2 = _optimal_k_per_ct.get(_ct, n_components)
                    _cby_k = _ct_res.get("consensus_by_k", {})
                    if _opt_k2 in _cby_k and _cby_k[_opt_k2] is not None:
                        # consensus_H shape: (factors, genes) — transpose to (genes, factors)
                        _H_raw = _cby_k[_opt_k2].get("consensus_H")
                        if _H_raw is not None:
                            _cnmf_ct_consensus_H[_ct] = np.asarray(_H_raw).T  # genes×factors

        except Exception as _exc:
            logger.warning(
                f"cNMF per-celltype failed ({_exc}). "
                f"Falling back to standard NMF with n_components={n_components}.",
                exc_info=True,
            )
            _optimal_k_per_ct = {}
            _cnmf_ct_consensus_H = {}
            use_consensus_nmf = False

    # ── If use_consensus_H: build loadings directly from consensus H matrices ──
    if use_consensus_nmf and use_consensus_H and _cnmf_ct_consensus_H:
        logger.info("Building per-celltype loadings from consensus H matrices...")
        for _ct, _H_genes_factors in _cnmf_ct_consensus_H.items():
            # _H_genes_factors: (n_genes_std, n_factors)
            # Map back to full gene_names index
            _opt_k3 = _H_genes_factors.shape[1]
            _factor_names = [f"Factor_{i+1}" for i in range(_opt_k3)]
            # Compute explained variance using simple R² proxy (same as _compute_single_celltype_nmf)
            _mask_ct = (adata.obs[celltype_column] == _ct).values
            _counts_ct_raw = counts_full[_mask_ct, :]
            if scipy.sparse.issparse(_counts_ct_raw):
                _counts_ct_dense = _counts_ct_raw.toarray().astype(np.float64)
            else:
                _counts_ct_dense = np.asarray(_counts_ct_raw, dtype=np.float64)
            _std_ct = np.std(_counts_ct_dense, axis=0)
            _nz = np.where(_std_ct > 0)[0]
            if len(_nz) < _opt_k3:
                logger.warning(f"  {_ct}: too few std-filtered genes for consensus H; skipping")
                continue
            _counts_filt = _counts_ct_dense[:, _nz]
            # Use only the std-filtered slice of H (rows = std-filtered genes)
            _H_filt = _H_genes_factors[:len(_nz), :]  # best-effort alignment
            # Reconstruct and compute per-factor R²
            _ev_vals = {}
            for _fi, _fn in enumerate(_factor_names):
                _h_vec = _H_filt[:, _fi].reshape(1, -1)  # (1, genes)
                _w_vec = _counts_filt @ _h_vec.T          # (cells, 1) — projection
                _recon = _w_vec @ _h_vec                  # (cells, genes)
                _ss_res = float(np.sum((_counts_filt - _recon) ** 2))
                _ss_tot = float(np.sum((_counts_filt - _counts_filt.mean(axis=0)) ** 2))
                _ev_vals[_fn] = float(1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else 0.0

            # Build loadings DataFrame (genes × factors) for std-filtered genes
            _gene_names_filt = gene_names[_nz]
            _loadings_df_ct = pd.DataFrame(
                _H_filt,
                index=pd.Index(_gene_names_filt),
                columns=pd.Index(_factor_names),
            )
            celltype_loadings[_ct] = _loadings_df_ct
            celltype_explained_variance[_ct] = pd.Series(_ev_vals)
            logger.info(f"  ✓ {_ct}: consensus H loadings {_loadings_df_ct.shape}")

        logger.info(f"✓ Per-celltype cNMF (consensus H) complete: {len(celltype_loadings)} celltypes")
        return celltype_loadings, celltype_explained_variance

    # Build task list — use per-celltype optimal K when cNMF determined it
    nmf_tasks = []
    for celltype in adata.obs[celltype_column].unique():
        mask = (adata.obs[celltype_column] == celltype).values
        n_cells_ct = int(mask.sum())

        if n_cells_ct < min_cells:
            logger.warning(f"Skipping {celltype}: only {n_cells_ct} cells (need >={min_cells})")
            continue

        _n_comp_ct = _optimal_k_per_ct.get(celltype, n_components)
        logger.info(f"  {celltype}: {n_cells_ct} cells, n_components={_n_comp_ct}")
        counts_ct = counts_full[mask, :]
        nmf_tasks.append((celltype, n_cells_ct, counts_ct, gene_names, _n_comp_ct, random_state, nmf_params))

    if n_jobs < 1:
        logger.warning(f"Invalid n_jobs={n_jobs}; falling back to sequential execution")
        n_jobs = 1

    if n_jobs == 1:
        for task in nmf_tasks:
            celltype, n_cells_ct, nmf_result, error_msg = _compute_single_celltype_nmf_task(task)
            if error_msg is not None:
                logger.warning(f"Skipping {celltype}: NMF task failed with error: {error_msg}")
                continue
            if nmf_result is None:
                continue

            loadings_df, explained_variance_series = nmf_result
            celltype_loadings[celltype] = loadings_df
            celltype_explained_variance[celltype] = explained_variance_series
            mean_r2 = explained_variance_series.mean()
            factor_r2 = explained_variance_series.to_dict()
            logger.info(
                f"  ✓ {celltype}: {loadings_df.shape}, "
                f"mean R² per factor={mean_r2:.4f}, range=[{min(factor_r2.values()):.4f}, {max(factor_r2.values()):.4f}]"
            )
    else:
        backend = parallel_backend.lower()
        if backend not in {'process', 'thread'}:
            logger.warning(
                f"Unsupported parallel_backend='{parallel_backend}', defaulting to 'process'"
            )
            backend = 'process'

        logger.info(
            f"Running per-celltype NMF in parallel with n_jobs={n_jobs}, backend={backend}"
        )

        executor_cls = ProcessPoolExecutor if backend == 'process' else ThreadPoolExecutor
        with executor_cls(max_workers=n_jobs) as executor:
            # executor.map preserves input order for deterministic logs/output.
            for celltype, n_cells_ct, nmf_result, error_msg in executor.map(
                _compute_single_celltype_nmf_task,
                nmf_tasks,
            ):
                if error_msg is not None:
                    logger.warning(f"Skipping {celltype}: NMF task failed with error: {error_msg}")
                    continue
                if nmf_result is None:
                    continue

                loadings_df, explained_variance_series = nmf_result
                celltype_loadings[celltype] = loadings_df
                celltype_explained_variance[celltype] = explained_variance_series
                mean_r2 = explained_variance_series.mean()
                factor_r2 = explained_variance_series.to_dict()
                logger.info(
                    f"  ✓ {celltype}: {loadings_df.shape}, "
                    f"mean R² per factor={mean_r2:.4f}, range=[{min(factor_r2.values()):.4f}, {max(factor_r2.values()):.4f}]"
                )

    logger.info(f"✓ Per-celltype NMF complete: {len(celltype_loadings)} celltypes")
    
    # Cache models if directory provided
    if cache_dir:
        model_dir = os.path.join(cache_dir, 'nmf_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, 'per_celltype_nmf.pkl')
        try:
            cache_data = {
                'loadings': celltype_loadings,
                'explained_variance': celltype_explained_variance,
                'n_components': n_components,
                'celltypes': list(celltype_loadings.keys()),
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✓ Saved per-celltype NMF models to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")
    
    return celltype_loadings, celltype_explained_variance


def _compute_single_celltype_nmf(
    celltype: str,
    counts_ct,
    gene_names: pd.Index,
    n_components: int,
    random_state: int,
    nmf_params: dict[str, object],
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Run NMF for a single celltype and return loadings + per-factor explained variance."""
    if scipy.sparse.issparse(counts_ct):
        counts_dense = counts_ct.toarray()
    else:
        counts_dense = np.asarray(counts_ct)

    std = np.std(counts_dense, axis=0)
    nonzero_index = np.where(std > 0)[0].astype(int)
    n_features = len(nonzero_index)

    if n_features < n_components:
        logger.warning(
            f"Skipping {celltype}: only {n_features} features with non-zero std "
            f"(need >= {n_components})"
        )
        return None

    counts_dense = counts_dense[:, nonzero_index]

    nmf = NMF(
        n_components=n_components,
        init=nmf_params['init'],
        random_state=random_state,
        beta_loss=nmf_params['beta_loss'],
        solver=nmf_params['solver'],
        max_iter=nmf_params['max_iter'],
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0,
    )
    W = nmf.fit_transform(counts_dense)
    H = nmf.components_

    factor_names = [f"{celltype}_NMF_{i+1}" for i in range(n_components)]
    factor_r2 = {}

    for factor_idx, factor_name in enumerate(factor_names):
        W_factor = W[:, factor_idx:factor_idx + 1]
        H_factor = H[factor_idx:factor_idx + 1, :]
        X_recon_factor = W_factor @ H_factor
        factor_r2[factor_name] = calculate_explained_variance(counts_dense, X_recon_factor)

    explained_variance_series = pd.Series(factor_r2)
    loadings_df = pd.DataFrame(
        H.T,
        index=gene_names[nonzero_index],
        columns=factor_names,
    )
    return loadings_df, explained_variance_series


def _compute_single_celltype_nmf_task(
    task: tuple[str, int, object, pd.Index, int, int, dict[str, object]],
) -> tuple[str, int, Optional[Tuple[pd.DataFrame, pd.Series]], Optional[str]]:
    """Task wrapper for optional parallel execution of per-celltype NMF."""
    celltype, n_cells_ct, counts_ct, gene_names, n_components, random_state, nmf_params = task
    try:
        nmf_result = _compute_single_celltype_nmf(
            celltype=celltype,
            counts_ct=counts_ct,
            gene_names=gene_names,
            n_components=n_components,
            random_state=random_state,
            nmf_params=nmf_params,
        )
        return celltype, n_cells_ct, nmf_result, None
    except Exception as exc:
        return celltype, n_cells_ct, None, str(exc)


def compute_pca_per_celltype(
    adata: AnnData,
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_cells: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    n_jobs: int = 1,
    parallel_backend: str = 'process',
    cache_dir: Optional[str] = None,
) -> Tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Compute per-celltype PCA on normalized data.

    Args:
        adata: Annotated data matrix
        celltype_column: Column with cell type labels
        n_components: Number of PCs per celltype
        random_state: Random seed
        min_cells: Minimum cells per celltype
        n_jobs: Number of workers for per-celltype fits. 1 keeps sequential behavior.
        parallel_backend: Parallel backend when n_jobs > 1 ('process' or 'thread').
        cache_dir: Directory to save/load cached models (optional)

    Returns:
        Tuple of:
            - Dict mapping celltype → loadings DataFrame (genes × PCs)
            - Dict mapping celltype → pd.Series of explained variance per PC
              Format: {celltype: pd.Series({pc_name: r2, ...})}
    """
    logger.info(f"Computing per-celltype PCA (n_components={n_components})...")
    
    # Check for cached model
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'pca_models', 'per_celltype_pca.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached per-celltype PCA models from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"✓ Successfully loaded cached per-celltype PCA: {len(cached_data['loadings'])} celltypes")
                return cached_data['loadings'], cached_data['explained_variance']
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}. Recomputing...")

    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not in adata.obs")

    if adata.X is None:
        raise ValueError("adata.X is None - need normalized data for PCA")

    celltype_loadings = {}
    celltype_explained_variance = {}

    pca_tasks = []
    for celltype in adata.obs[celltype_column].unique():
        mask = (adata.obs[celltype_column] == celltype).values
        n_cells_ct = int(mask.sum())

        if n_cells_ct < min_cells:
            logger.warning(f"Skipping {celltype}: only {n_cells_ct} cells (need >={min_cells})")
            continue

        logger.info(f"  {celltype}: {n_cells_ct} cells")
        X_ct = adata.X[mask, :]
        pca_tasks.append((celltype, n_cells_ct, X_ct, adata.var_names, n_components, random_state))

    if n_jobs < 1:
        logger.warning(f"Invalid n_jobs={n_jobs}; falling back to sequential execution")
        n_jobs = 1

    if n_jobs == 1:
        for task in pca_tasks:
            celltype, n_cells_ct, pca_result, error_msg = _compute_single_celltype_pca_task(task)
            if error_msg is not None:
                logger.warning(f"Skipping {celltype}: PCA task failed with error: {error_msg}")
                continue
            if pca_result is None:
                continue

            loadings_df, explained_variance_series, total_marginal_r2 = pca_result
            celltype_loadings[celltype] = loadings_df
            celltype_explained_variance[celltype] = explained_variance_series
            mean_r2 = explained_variance_series.mean()
            logger.info(
                f"  ✓ {celltype}: {loadings_df.shape}, "
                f"mean R² per PC={mean_r2:.4f}, "
                f"marginal (sklearn)={total_marginal_r2:.4f}"
            )
    else:
        backend = parallel_backend.lower()
        if backend not in {'process', 'thread'}:
            logger.warning(
                f"Unsupported parallel_backend='{parallel_backend}', defaulting to 'process'"
            )
            backend = 'process'

        logger.info(
            f"Running per-celltype PCA in parallel with n_jobs={n_jobs}, backend={backend}"
        )

        executor_cls = ProcessPoolExecutor if backend == 'process' else ThreadPoolExecutor
        with executor_cls(max_workers=n_jobs) as executor:
            for celltype, n_cells_ct, pca_result, error_msg in executor.map(
                _compute_single_celltype_pca_task,
                pca_tasks,
            ):
                if error_msg is not None:
                    logger.warning(f"Skipping {celltype}: PCA task failed with error: {error_msg}")
                    continue
                if pca_result is None:
                    continue

                loadings_df, explained_variance_series, total_marginal_r2 = pca_result
                celltype_loadings[celltype] = loadings_df
                celltype_explained_variance[celltype] = explained_variance_series
                mean_r2 = explained_variance_series.mean()
                logger.info(
                    f"  ✓ {celltype}: {loadings_df.shape}, "
                    f"mean R² per PC={mean_r2:.4f}, "
                    f"marginal (sklearn)={total_marginal_r2:.4f}"
                )

    logger.info(f"✓ Per-celltype PCA complete: {len(celltype_loadings)} celltypes")
    
    # Cache models if directory provided
    if cache_dir:
        model_dir = os.path.join(cache_dir, 'pca_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, 'per_celltype_pca.pkl')
        try:
            cache_data = {
                'loadings': celltype_loadings,
                'explained_variance': celltype_explained_variance,
                'n_components': n_components,
                'celltypes': list(celltype_loadings.keys()),
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✓ Saved per-celltype PCA models to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")
    
    return celltype_loadings, celltype_explained_variance


def _compute_single_celltype_pca(
    celltype: str,
    X_ct,
    gene_names: pd.Index,
    n_components: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, float]:
    """Run PCA for a single celltype and return loadings + explained variance metrics."""
    if scipy.sparse.issparse(X_ct):
        X = X_ct.toarray()
    else:
        X = np.asarray(X_ct)

    pca = PCA(n_components=n_components, random_state=random_state)
    W = pca.fit_transform(X)
    H = pca.components_

    component_names = [f"{celltype}_PC_{i+1}" for i in range(n_components)]
    factor_r2 = {}

    for component_idx, component_name in enumerate(component_names):
        W_component = W[:, component_idx:component_idx + 1]
        H_component = H[component_idx:component_idx + 1, :]
        X_recon_component = W_component @ H_component + pca.mean_
        factor_r2[component_name] = calculate_explained_variance(X, X_recon_component)

    explained_variance_series = pd.Series(factor_r2)
    total_marginal_r2 = float(pca.explained_variance_ratio_.sum())
    loadings_df = pd.DataFrame(
        H.T,
        index=gene_names,
        columns=component_names,
    )
    return loadings_df, explained_variance_series, total_marginal_r2


def _compute_single_celltype_pca_task(
    task: tuple[str, int, object, pd.Index, int, int],
) -> tuple[str, int, Optional[Tuple[pd.DataFrame, pd.Series, float]], Optional[str]]:
    """Task wrapper for optional parallel execution of per-celltype PCA."""
    celltype, n_cells_ct, X_ct, gene_names, n_components, random_state = task
    try:
        pca_result = _compute_single_celltype_pca(
            celltype=celltype,
            X_ct=X_ct,
            gene_names=gene_names,
            n_components=n_components,
            random_state=random_state,
        )
        return celltype, n_cells_ct, pca_result, None
    except Exception as exc:
        return celltype, n_cells_ct, None, str(exc)


# ============================================================================
# GENE SELECTION FUNCTIONS
# ============================================================================


def select_genes_from_nmf(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    analysis_type: str = "global",
    method: str = "method_a",
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    pool_size_per_celltype: int = 200,
    pool_size_per_factor: int = 200,
    genes_per_component: int = DEFAULT_DIMRED_GENES_PER_COMPONENT,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    random_state: int = DEFAULT_RANDOM_STATE,
    nmf_n_jobs: int = 1,
    nmf_parallel_backend: str = 'process',
    nmf_loadings_global: Optional[pd.DataFrame] = None,
    nmf_loadings_per_celltype: Optional[dict[str, pd.DataFrame]] = None,
    results_dir: Optional[str] = None,
    nmf_model_cache_dir: Optional[str] = None,
    mean_expr_per_ct: Optional[dict] = None,
    # cNMF options
    use_consensus_nmf: bool = False,
    k_min: int = 3,
    k_max: int = 15,
    k_step: int = 2,
    cnmf_n_iter: int = 20,
    k_selection_method: str = "silhouette",
    use_consensus_H: bool = False,
) -> GeneListBuilder:
    """Select genes using NMF loadings with factor-aware duplicate resolution.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes
        analysis_type: 'global' or 'per_celltype'
        method: Gene selection method (only 'method_a' — top genes by absolute factor loading)
        celltype_column: Cell type column (for per_celltype analysis)
        n_components: Number of NMF factors to use
        nmf_n_jobs: Workers for per-celltype NMF fitting when loadings are computed internally
        nmf_parallel_backend: Backend for per-celltype NMF parallelism ('process' or 'thread')
        genes_per_component: Genes per factor
        min_cells_per_celltype: Minimum cells per cell type
        random_state: Random seed for reproducibility
        nmf_loadings_global: Pre-computed global NMF loadings (genes × factors)
        nmf_loadings_per_celltype: Pre-computed per-celltype loadings {celltype: DataFrame}
        results_dir: Directory to save selection results
        nmf_model_cache_dir: Shared directory for NMF model pkl files (reused across
            different probeset sizes / strategies). Takes priority over results_dir
            for model caching. Leave None to use results_dir (old behaviour).

    Returns:
        GeneListBuilder with factor assignments and provenance tracking

    Examples:
        >>> # Global NMF
        >>> builder = select_genes_from_nmf(
        ...     adata,
        ...     probeset_size=500,
        ...     analysis_type='global',
        ...     nmf_loadings_global=loadings_df
        ... )
        
        >>> # Per-celltype NMF
        >>> builder = select_genes_from_nmf(
        ...     adata,
        ...     probeset_size=500,
        ...     analysis_type='per_celltype',
        ...     nmf_loadings_per_celltype=celltype_loadings
        ... )
    """
    # Resolve model cache directory: explicit nmf_model_cache_dir takes priority,
    # falling back to results_dir so existing behaviour is preserved.
    _nmf_cache_dir = nmf_model_cache_dir if nmf_model_cache_dir else results_dir

    # cNMF plot dir — save inside results_dir/cNMF/ if available
    _cnmf_plot_dir = str(Path(results_dir) / "cNMF") if results_dir else None

    # Compute loadings if not provided
    if analysis_type == "global" and nmf_loadings_global is None:
        logger.info("Computing global NMF loadings...")
        if _nmf_cache_dir:
            logger.info(f"NMF model cache dir: {_nmf_cache_dir}")
        nmf_loadings_global, _ = compute_nmf_global(
            adata=adata,
            n_components=n_components,
            random_state=random_state,
            cache_dir=_nmf_cache_dir,
            use_consensus_nmf=use_consensus_nmf,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            cnmf_n_iter=cnmf_n_iter,
            k_selection_method=k_selection_method,
            use_consensus_H=use_consensus_H,
            cnmf_plot_dir=_cnmf_plot_dir,
        )
    elif analysis_type == "per_celltype" and nmf_loadings_per_celltype is None:
        logger.info("Computing per-celltype NMF loadings...")
        if _nmf_cache_dir:
            logger.info(f"NMF model cache dir: {_nmf_cache_dir}")
        nmf_loadings_per_celltype, nmf_explained_variance = compute_nmf_per_celltype(
            adata=adata,
            celltype_column=celltype_column,
            n_components=n_components,
            random_state=random_state,
            min_cells=min_cells_per_celltype,
            n_jobs=nmf_n_jobs,
            parallel_backend=nmf_parallel_backend,
            cache_dir=_nmf_cache_dir,
            use_consensus_nmf=use_consensus_nmf,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            cnmf_n_iter=cnmf_n_iter,
            k_selection_method=k_selection_method,
            use_consensus_H=use_consensus_H,
            cnmf_plot_dir=_cnmf_plot_dir,
        )
    else:
        # Loadings were pre-computed - no explained variance available
        nmf_explained_variance = None

    return _select_genes_from_dimred(
        adata=adata,
        probeset_size=probeset_size,
        reduction_type="nmf",
        analysis_type=analysis_type,
        method=method,
        n_components=n_components,
        pool_size_per_celltype=pool_size_per_celltype,
        pool_size_per_factor=pool_size_per_factor,
        genes_per_component=genes_per_component,
        dimred_loadings_global=nmf_loadings_global,
        dimred_loadings_per_celltype=nmf_loadings_per_celltype,
        dimred_explained_variance=nmf_explained_variance if analysis_type == 'per_celltype' else None,
        results_dir=results_dir,
        mean_expr_per_ct=mean_expr_per_ct,
    )


def select_genes_from_pca(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    analysis_type: str = "global",
    method: str = "method_a",
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_N_COMPONENTS_PCA,
    pool_size_per_celltype: int = 200,
    pool_size_per_factor: int = 200,
    top_n_pcs: int = 5,
    genes_per_component: int = DEFAULT_DIMRED_GENES_PER_COMPONENT,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    random_state: int = DEFAULT_RANDOM_STATE,
    pca_n_jobs: int = 1,
    pca_parallel_backend: str = 'process',
    pca_loadings_global: Optional[pd.DataFrame] = None,
    pca_loadings_per_celltype: Optional[dict[str, pd.DataFrame]] = None,
    results_dir: Optional[str] = None,
    nmf_model_cache_dir: Optional[str] = None,
    mean_expr_per_ct: Optional[dict] = None,
) -> GeneListBuilder:
    """Select genes using PCA loadings with factor-aware duplicate resolution.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes
        analysis_type: 'global' or 'per_celltype'
        method: Gene selection method (only 'method_a' — top genes by absolute PC loading)
        celltype_column: Cell type column (for per_celltype analysis)
        n_components: Total number of PCs computed
        top_n_pcs: Number of top PCs to use for gene selection
        genes_per_component: Genes per PC
        min_cells_per_celltype: Minimum cells per cell type
        random_state: Random seed for reproducibility
        pca_n_jobs: Number of workers for per-celltype PCA when analysis_type='per_celltype'
        pca_parallel_backend: Parallel backend for per-celltype PCA ('process' or 'thread')
        pca_loadings_global: Pre-computed global PCA loadings (genes × PCs)
        pca_loadings_per_celltype: Pre-computed per-celltype loadings {celltype: DataFrame}
        results_dir: Directory to save selection results
        nmf_model_cache_dir: Shared directory for PCA model pkl files (reused across
            different probeset sizes / strategies). Takes priority over results_dir
            for model caching. Leave None to use results_dir (old behaviour).

    Returns:
        GeneListBuilder with factor assignments and provenance tracking

    Examples:
        >>> # Global PCA (top 5 PCs)
        >>> builder = select_genes_from_pca(
        ...     adata,
        ...     probeset_size=500,
        ...     top_n_pcs=5,
        ...     pca_loadings_global=loadings_df
        ... )
    """
    # Resolve model cache directory: explicit nmf_model_cache_dir takes priority,
    # falling back to results_dir so existing behaviour is preserved.
    _pca_cache_dir = nmf_model_cache_dir if nmf_model_cache_dir else results_dir

    # Compute loadings if not provided
    if analysis_type == "global" and pca_loadings_global is None:
        logger.info("Computing global PCA loadings...")
        if _pca_cache_dir:
            logger.info(f"PCA model cache dir: {_pca_cache_dir}")
        pca_loadings_global, _ = compute_pca_global(
            adata=adata,
            n_components=n_components,
            random_state=random_state,
            cache_dir=_pca_cache_dir,
        )
    elif analysis_type == "per_celltype" and pca_loadings_per_celltype is None:
        logger.info("Computing per-celltype PCA loadings...")
        if _pca_cache_dir:
            logger.info(f"PCA model cache dir: {_pca_cache_dir}")
        pca_loadings_per_celltype, pca_explained_variance = compute_pca_per_celltype(
            adata=adata,
            celltype_column=celltype_column,
            n_components=n_components,
            random_state=random_state,
            min_cells=min_cells_per_celltype,
            n_jobs=pca_n_jobs,
            parallel_backend=pca_parallel_backend,
            cache_dir=_pca_cache_dir,
        )
    else:
        # Loadings were pre-computed - no explained variance available
        pca_explained_variance = None

    # For PCA, limit to top N PCs
    n_components_to_use = min(top_n_pcs, n_components)
    logger.info(f"PCA: Using top {n_components_to_use} PCs (out of {n_components})")

    return _select_genes_from_dimred(
        adata=adata,
        probeset_size=probeset_size,
        reduction_type="pca",
        analysis_type=analysis_type,
        method=method,
        n_components=n_components_to_use,
        pool_size_per_celltype=pool_size_per_celltype,
        pool_size_per_factor=pool_size_per_factor,
        genes_per_component=genes_per_component,
        dimred_loadings_global=pca_loadings_global,
        dimred_loadings_per_celltype=pca_loadings_per_celltype,
        dimred_explained_variance=pca_explained_variance if analysis_type == 'per_celltype' else None,
        results_dir=results_dir,
        mean_expr_per_ct=mean_expr_per_ct,
    )


def _select_genes_from_dimred(
    adata: AnnData,
    probeset_size: int,
    reduction_type: str,
    analysis_type: str,
    method: str,
    n_components: int,
    pool_size_per_celltype: int,
    pool_size_per_factor: int,
    genes_per_component: int,
    dimred_loadings_global: Optional[pd.DataFrame],
    dimred_loadings_per_celltype: Optional[dict[str, pd.DataFrame]],
    dimred_explained_variance: Optional[dict[str, float]],
    results_dir: Optional[str],
    mean_expr_per_ct: Optional[dict] = None,
) -> GeneListBuilder:
    """Core dimension reduction gene selection with factor-aware resolution.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes
        reduction_type: 'nmf' or 'pca'
        analysis_type: 'global' or 'per_celltype'
        method: Gene selection method (only 'method_a')
        n_components: Number of components to use
        genes_per_component: Genes per component
        dimred_loadings_global: Global loadings (if analysis_type='global')
        dimred_loadings_per_celltype: Per-celltype loadings (if analysis_type='per_celltype')
        dimred_explained_variance: Explained variance per celltype (if analysis_type='per_celltype')
        results_dir: Results directory

    Returns:
        GeneListBuilder with selected genes and factor assignments
    """
    logger.info(f"=== {reduction_type.upper()} Gene Selection ===")
    logger.info(f"Analysis: {analysis_type}, Method: {method}")
    logger.info(f"Target: {probeset_size} genes, Components: {n_components}")

    # Initialize GeneListBuilder
    strategy_name = f"dimred_only_{reduction_type}_{analysis_type}_{method}"
    builder = GeneListBuilder(
        strategy_name=strategy_name,
        analysis_type=analysis_type,
    )

    if analysis_type == "global":
        if dimred_loadings_global is None:
            raise ValueError(f"Global {reduction_type} loadings required but not provided")

        _select_genes_global(
            builder=builder,
            loadings_df=dimred_loadings_global,
            method=method,
            n_components=n_components,
            pool_size_per_factor=pool_size_per_factor,
            genes_per_component=genes_per_component,
            probeset_size=probeset_size,
            results_dir=results_dir,
            mean_expr_per_ct=mean_expr_per_ct,
        )

    elif analysis_type == "per_celltype":
        if dimred_loadings_per_celltype is None:
            raise ValueError(f"Per-celltype {reduction_type} loadings required but not provided")

        _select_genes_per_celltype(
            builder=builder,
            celltype_loadings=dimred_loadings_per_celltype,
            celltype_explained_variance=dimred_explained_variance,
            method=method,
            n_components=n_components,
            pool_size_per_celltype=pool_size_per_celltype,
            genes_per_component=genes_per_component,
            probeset_size=probeset_size,
            results_dir=results_dir,
            mean_expr_per_ct=mean_expr_per_ct,
        )

    else:
        raise ValueError(f"Invalid analysis_type: {analysis_type}")

    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        builder.to_csv(os.path.join(results_dir, "selected_genes.csv"))

    logger.info(f"{reduction_type.upper()} selection complete: {len(builder.get_selected_genes('initial'))} genes")
    return builder


def _select_genes_global(
    builder: GeneListBuilder,
    loadings_df: pd.DataFrame,
    method: str,
    n_components: int,
    pool_size_per_factor: int,
    genes_per_component: int,
    probeset_size: int,
    results_dir: Optional[str] = None,
    mean_expr_per_ct: Optional[dict] = None,
) -> None:
    """Select genes from global dimension reduction with 3-phase pool-based architecture.
    
    **NEW POOL-BASED ARCHITECTURE:**
    - Phase 1: Create large pool (pool_size_per_factor genes/factor, e.g., 200)
    - Phase 2: Resolve cross-factor duplicates using absolute loading
    - Phase 3: Select final probeset_size genes from duplicate-free pool
    
    This provides consistency with per-celltype mode and enables larger pools for replacement.

    Args:
        builder: GeneListBuilder to populate
        loadings_df: DataFrame with genes as rows, components as columns
        method: Gene selection method (only 'method_a')
        n_components: Number of components to use
        pool_size_per_factor: Genes per factor in Phase 1 pool (e.g., 200)
        genes_per_component: Genes per component
        probeset_size: Target final size (e.g., 100)
        results_dir: Results directory for pool caching
        mean_expr_per_ct: Per-celltype mean expression dict for Xenium filter.
            A global mean is derived by averaging across all celltypes.
    """
    logger.info(f"=" * 80)
    logger.info(f"POOL-BASED GENE SELECTION: Global")
    logger.info(f"=" * 80)
    logger.info(f"Genes: {loadings_df.shape[0]}, Components: {n_components}")
    logger.info(f"Pool size per factor (Phase 1): {pool_size_per_factor} genes")
    logger.info(f"Final probeset size (Phase 3): {probeset_size} genes")

    # ============================================================================
    # PHASE 1: POOL CREATION (Large pool without size constraints)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 1: Creating large gene pool")
    logger.info("─" * 80)

    # Select components to use
    component_cols = loadings_df.columns[:n_components]
    loadings_subset = loadings_df[component_cols]

    # Collect genes per factor (POOL, not final selection)
    genes_per_factor = {}
    pool_stats = {"total_selections": 0}

    for comp_idx, comp_name in enumerate(component_cols):
        comp_loadings = loadings_subset[comp_name].abs()

        top_genes = comp_loadings.nlargest(pool_size_per_factor)
        selected_genes = top_genes.index.tolist()

        genes_per_factor[comp_name] = selected_genes
        pool_stats["total_selections"] += len(selected_genes)
        logger.info(f"  {comp_name}: {len(selected_genes)} genes in pool")

    logger.info(f"✓ Phase 1 complete: {pool_stats['total_selections']} genes in raw pool (with duplicates)")

    # ──────────────────────────────────────────────────────────────────────────
    # XENIUM FILTER: Remove genes outside expression range from Phase 1 pool
    # For global mode a gene-level mean is derived by averaging across all celltypes.
    # ──────────────────────────────────────────────────────────────────────────
    if mean_expr_per_ct is not None:
        # Compute global mean per gene by averaging across celltypes
        import numpy as np
        all_genes_in_pool = set(g for genes in genes_per_factor.values() for g in genes)
        global_mean_expr: dict[str, float] = {}
        for gene in all_genes_in_pool:
            ct_values = [
                ct_expr[gene]
                for ct_expr in mean_expr_per_ct.values()
                if gene in ct_expr
            ]
            global_mean_expr[gene] = float(np.mean(ct_values)) if ct_values else 0.0

        n_before = pool_stats['total_selections']
        n_removed = 0
        for factor in list(genes_per_factor.keys()):
            before = len(genes_per_factor[factor])
            genes_per_factor[factor] = [
                g for g in genes_per_factor[factor]
                if DEFAULT_MIN_XENIUM_EXPRESSION
                <= global_mean_expr.get(g, 0.0)
                <= DEFAULT_MAX_XENIUM_EXPRESSION
            ]
            n_removed += before - len(genes_per_factor[factor])

        pool_stats['total_selections'] = sum(len(g) for g in genes_per_factor.values())
        logger.info(
            f"✓ Xenium filter (global mean): {n_before} → {pool_stats['total_selections']} pool genes "
            f"({n_removed} removed, expression range "
            f"[{DEFAULT_MIN_XENIUM_EXPRESSION}, {DEFAULT_MAX_XENIUM_EXPRESSION}])"
        )
    else:
        logger.info("Xenium filter skipped (no mean expression data provided)")

    # Cache pool to disk if results_dir provided
    if results_dir:
        import pickle
        from pathlib import Path
        
        pool_cache_dir = Path(results_dir) / "nmf_pools"
        pool_cache_dir.mkdir(parents=True, exist_ok=True)
        
        pool_cache_file = pool_cache_dir / "global_pool.pkl"
        with open(pool_cache_file, "wb") as f:
            pickle.dump(genes_per_factor, f)
        logger.info(f"✓ Pool cached to: {pool_cache_file}")

    # ============================================================================
    # PHASE 2: DUPLICATE RESOLUTION (Create duplicate-free pool)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 2: Resolving cross-factor duplicates")
    logger.info("─" * 80)

    abs_loadings_df = loadings_df.abs()

    # Build gene_weights format from genes_per_factor
    gene_weights = {}
    for factor, genes in genes_per_factor.items():
        for gene in genes:
            if gene not in gene_weights:
                gene_weights[gene] = {}
            gene_weights[gene][factor] = float(abs_loadings_df.at[gene, factor])
    
    # Resolve duplicates (assign each gene to best factor by loading)
    logger.info("Resolving duplicates using absolute loading...")
    
    # For global mode, use a simpler approach: assign each gene to its highest-loading factor
    pool_genes = []
    pool_factor_assignments = {}
    
    for gene, factor_loadings in gene_weights.items():
        best_factor = max(factor_loadings.items(), key=lambda x: x[1])[0]
        pool_factor_assignments[gene] = best_factor
        pool_genes.append(gene)
    
    duplicates_resolved = pool_stats["total_selections"] - len(pool_genes)
    
    logger.info(
        f"✓ Phase 2 complete: {len(pool_genes)} unique genes "
        f"({duplicates_resolved} duplicates resolved)"
    )

    # Cache duplicate-free pool to disk
    if results_dir:
        import pandas as pd
        from pathlib import Path
        
        pool_cache_dir = Path(results_dir) / "nmf_pools"
        
        # Save as CSV for easy inspection
        pool_df = pd.DataFrame({
            "gene": pool_genes,
            "factor": [pool_factor_assignments.get(g) for g in pool_genes],
        })
        resolved_pool_file = pool_cache_dir / "global_resolved_pool.csv"
        pool_df.to_csv(resolved_pool_file, index=False)
        logger.info(f"✓ Resolved pool saved to: {resolved_pool_file}")
        
        # Save duplicate resolution statistics
        import json
        stats_file = pool_cache_dir / "global_resolution_stats.json"
        with open(stats_file, "w") as f:
            json.dump({
                "pool_size_per_factor": pool_size_per_factor,
                "raw_pool_size": pool_stats["total_selections"],
                "resolved_pool_size": len(pool_genes),
                "duplicates_resolved": duplicates_resolved,
                "n_components": n_components,
            }, f, indent=2)
        logger.info(f"✓ Resolution stats saved to: {stats_file}")

    # ============================================================================
    # PHASE 3: FINAL SELECTION (Factor-aware selection from duplicate-free pool)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 3: Selecting final genes from duplicate-free pool")
    logger.info("─" * 80)

    # Calculate genes per factor for FINAL selection (round UP)
    import math
    genes_per_factor_final = probeset_size / n_components
    genes_per_factor_final_rounded = math.ceil(genes_per_factor_final)
    
    logger.info(
        f"Final allocation: {genes_per_factor_final:.2f} genes/factor "
        f"(rounded UP to {genes_per_factor_final_rounded})"
    )

    # Select genes from pool per factor
    final_selected_genes = []
    final_factor_assignments = {}

    for factor in component_cols:
        # Get pool genes for this factor
        factor_pool_genes = [
            g for g in pool_genes
            if pool_factor_assignments.get(g) == factor
        ]

        if len(factor_pool_genes) == 0:
            phase1_count = genes_per_factor.get(factor, 0)
            logger.warning(
                f"No pool genes for {factor} - empty after Phase 2 duplicate resolution "
                f"(Phase 1: {phase1_count} genes → {duplicates_resolved} total duplicates resolved; "
                f"need {genes_per_factor_final_rounded}/factor for Phase 3)"
            )
            continue

        # Rank by absolute loading
        gene_loadings = {
            gene: float(abs_loadings_df.at[gene, factor])
            for gene in factor_pool_genes
        }

        # Select top N genes
        sorted_genes = sorted(gene_loadings.items(), key=lambda x: x[1], reverse=True)
        n_to_select = min(genes_per_factor_final_rounded, len(sorted_genes))
        selected_genes_factor = [g for g, _ in sorted_genes[:n_to_select]]

        # Add to final selection
        for gene in selected_genes_factor:
            if gene not in final_selected_genes:
                final_selected_genes.append(gene)
                final_factor_assignments[gene] = factor

    logger.info(f"✓ Phase 3 complete: {len(final_selected_genes)} genes selected from pool")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL TRIMMING: Reduce to exactly probeset_size (factor-balanced removal)
    # ──────────────────────────────────────────────────────────────────────────
    if len(final_selected_genes) > probeset_size:
        from collections import Counter
        target_per_factor = probeset_size / n_components  # float
        overage_threshold = target_per_factor * 1.1
        n_to_remove = len(final_selected_genes) - probeset_size
        logger.info(
            f"Trimming {len(final_selected_genes)} → {probeset_size} genes "
            f"({n_to_remove} to remove, factor-balanced; "
            f"target/factor={target_per_factor:.2f}, overage threshold={overage_threshold:.2f})"
        )
        fallback_used = 0
        while len(final_selected_genes) > probeset_size:
            factor_counts = Counter(
                final_factor_assignments.get(g, '__unknown__')
                for g in final_selected_genes
            )
            eligible_factors = {
                f for f, cnt in factor_counts.items()
                if cnt > overage_threshold
            }
            candidates: list[tuple[float, str]] = []
            for g in final_selected_genes:
                fac = final_factor_assignments.get(g, '__unknown__')
                if fac not in eligible_factors:
                    continue
                # Respect MIN_FACTOR_CONTRIBUTION floor
                if factor_counts[fac] <= MIN_FACTOR_CONTRIBUTION:
                    continue
                loading = float(abs_loadings_df.at[g, fac]) if g in abs_loadings_df.index else 0.0
                candidates.append((loading, g))
            if candidates:
                worst_gene = min(candidates, key=lambda x: x[0])[1]
            else:
                fallback_used += 1
                worst_gene = min(
                    final_selected_genes,
                    key=lambda g: float(abs_loadings_df.at[g, final_factor_assignments.get(g, component_cols[0])])
                    if g in abs_loadings_df.index else 0.0
                )
            final_selected_genes.remove(worst_gene)
            del final_factor_assignments[worst_gene]
        if fallback_used:
            logger.warning(
                f"  {fallback_used} fallback removals (no over-represented factor "
                f"with removable gene found)"
            )
        logger.info(f"✓ Trimmed to exactly {len(final_selected_genes)} genes")
    elif len(final_selected_genes) < probeset_size:
        logger.warning(
            f"Selected {len(final_selected_genes)} genes (target: {probeset_size}). "
            f"Pool may need larger size."
        )

    # Log final factor distribution
    final_genes_per_factor = {}
    for factor in component_cols:
        factor_genes = [g for g in final_selected_genes if final_factor_assignments.get(g) == factor]
        final_genes_per_factor[factor] = len(factor_genes)

    logger.info(f"Final factor distribution (target: {probeset_size} genes):")
    for factor, count in sorted(final_genes_per_factor.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {factor}: {count} genes")

    # ============================================================================
    # ADD GENES TO BUILDER
    # Add ALL Phase 2 pool genes so ODT has a large replacement pool.
    # Only Phase 3 final genes are marked as selected_initial.
    # ============================================================================
    logger.info("")
    selected_set = set(final_selected_genes)
    logger.info(
        f"Adding Phase 2 pool to builder: {len(pool_genes)} genes "
        f"({len(final_selected_genes)} selected + "
        f"{len(pool_genes) - len(selected_set)} ODT replacement candidates)"
    )

    for gene in pool_genes:
        factor = pool_factor_assignments.get(gene, str(component_cols[0]))
        selection_score = float(abs_loadings_df.at[gene, factor]) if gene in abs_loadings_df.index else 0.0
        builder.add_gene(
            gene=gene,
            selection_score=selection_score,
            rank=None,
            component=str(factor),
            metadata={
                "loading": float(selection_score),
                "pool_size_per_factor": pool_size_per_factor,
                "phase": "pool_based",
            },
        )

    for gene in final_selected_genes:
        builder.mark_selected(gene)

    logger.info(
        f"✓ Added {len(pool_genes)} pool genes; "
        f"{len(final_selected_genes)} marked selected, "
        f"{len(pool_genes) - len(selected_set)} available as ODT replacement candidates"
    )
    logger.info(f"=" * 80)


def _select_genes_per_celltype(
    builder: GeneListBuilder,
    celltype_loadings: dict[str, pd.DataFrame],
    celltype_explained_variance: Optional[dict[str, float]],
    method: str,
    n_components: int,
    pool_size_per_celltype: int,
    genes_per_component: int,
    probeset_size: int,
    results_dir: Optional[str] = None,
    mean_expr_per_ct: Optional[dict] = None,
) -> None:
    """Select genes from per-celltype dimension reduction with 3-phase pool-based architecture.
    
    **NEW POOL-BASED ARCHITECTURE:**
    - Phase 1: Create large pool (pool_size_per_celltype genes/celltype, e.g., 200)
    - Phase 2: Resolve cross-celltype duplicates using R²×loading contribution
    - Phase 3: Select final probeset_size genes from duplicate-free pool
    
    This eliminates the gap-filling problem where cross-celltype duplicates were lost.

    Args:
        builder: GeneListBuilder to populate
        celltype_loadings: Dict mapping celltype -> loadings DataFrame
        celltype_explained_variance: Explained variance (R²) per celltype
        method: Gene selection method (only 'method_a')
        n_components: Number of components per celltype
        pool_size_per_celltype: Genes per celltype in Phase 1 pool (e.g., 200)
        genes_per_component: Genes per component
        probeset_size: Target final size (e.g., 100)
        results_dir: Results directory for pool caching
    """
    logger.info(f"=" * 80)
    logger.info(f"POOL-BASED GENE SELECTION: Per-Celltype")
    logger.info(f"=" * 80)
    logger.info(f"Cell types: {len(celltype_loadings)}")
    logger.info(f"Components per celltype: {n_components}")
    logger.info(f"Pool size per celltype (Phase 1): {pool_size_per_celltype} genes")
    logger.info(f"Final probeset size (Phase 3): {probeset_size} genes")

    # ============================================================================
    # PHASE 1: POOL CREATION (Large pool without size constraints)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 1: Creating large gene pool")
    logger.info("─" * 80)

    # Calculate genes per celltype-factor combo for pool
    # pool_size_per_celltype = TOTAL genes per celltype distributed across factors
    # Example: 200 genes/celltype ÷ 5 factors = 40 genes per celltype-factor combo
    pool_genes_per_comp = max(1, pool_size_per_celltype // n_components)
    logger.info(f"Pool allocation: {pool_genes_per_comp} genes per celltype-factor combination")

    # Collect genes per celltype and factor (POOL, not final selection)
    celltype_genes_per_factor = {}
    pool_stats = {"total_selections": 0, "celltypes": {}}

    for celltype, loadings_df in celltype_loadings.items():
        component_cols = loadings_df.columns[:n_components]
        genes_per_factor = {}

        for comp_name in component_cols:
            comp_loadings = loadings_df[comp_name].abs()

            top_genes = comp_loadings.nlargest(pool_genes_per_comp)
            selected_genes = top_genes.index.tolist()

            genes_per_factor[comp_name] = selected_genes

        # Store pool data
        celltype_genes_per_factor[celltype] = {
            "genes_per_factor": genes_per_factor,
            "loadings_df": loadings_df,
            "component_cols": component_cols.tolist(),
        }

        ct_total = sum(len(g) for g in genes_per_factor.values())
        pool_stats["total_selections"] += ct_total
        pool_stats["celltypes"][celltype] = ct_total

        logger.info(f"  {celltype}: {ct_total} genes in pool")

    logger.info(f"✓ Phase 1 complete: {pool_stats['total_selections']} genes in raw pool (with duplicates)")

    # ──────────────────────────────────────────────────────────────────────────
    # XENIUM FILTER: Remove genes outside expression range from Phase 1 pool
    # ──────────────────────────────────────────────────────────────────────────
    if mean_expr_per_ct is not None:
        n_before = pool_stats['total_selections']
        n_removed = 0
        for celltype in list(celltype_genes_per_factor.keys()):
            if celltype not in mean_expr_per_ct:
                logger.warning(
                    f"No mean expression data for '{celltype}', "
                    f"skipping Xenium filter for this celltype"
                )
                continue
            ct_mean_expr = mean_expr_per_ct[celltype]
            for factor in list(celltype_genes_per_factor[celltype]['genes_per_factor'].keys()):
                before = len(celltype_genes_per_factor[celltype]['genes_per_factor'][factor])
                celltype_genes_per_factor[celltype]['genes_per_factor'][factor] = [
                    g for g in celltype_genes_per_factor[celltype]['genes_per_factor'][factor]
                    if DEFAULT_MIN_XENIUM_EXPRESSION
                    <= ct_mean_expr.get(g, 0.0)
                    <= DEFAULT_MAX_XENIUM_EXPRESSION
                ]
                after = len(celltype_genes_per_factor[celltype]['genes_per_factor'][factor])
                n_removed += before - after

        # Recalculate pool_stats totals after filtering
        pool_stats['total_selections'] = sum(
            len(g)
            for ct_data in celltype_genes_per_factor.values()
            for g in ct_data['genes_per_factor'].values()
        )
        logger.info(
            f"✓ Xenium filter: {n_before} → {pool_stats['total_selections']} pool genes "
            f"({n_removed} removed, expression range "
            f"[{DEFAULT_MIN_XENIUM_EXPRESSION}, {DEFAULT_MAX_XENIUM_EXPRESSION}])"
        )
    else:
        logger.info("Xenium filter skipped (no mean expression data provided)")

    # Cache pool to disk if results_dir provided
    if results_dir:
        import pickle
        from pathlib import Path
        
        pool_cache_dir = Path(results_dir) / "nmf_pools"
        pool_cache_dir.mkdir(parents=True, exist_ok=True)
        
        pool_cache_file = pool_cache_dir / "per_celltype_pool.pkl"
        with open(pool_cache_file, "wb") as f:
            pickle.dump(celltype_genes_per_factor, f)
        logger.info(f"✓ Pool cached to: {pool_cache_file}")

    # ============================================================================
    # PHASE 2: DUPLICATE RESOLUTION (Create duplicate-free pool)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 2: Resolving cross-celltype duplicates")
    logger.info("─" * 80)

    # Validate explained variance availability
    if celltype_explained_variance is None:
        logger.error(
            "No explained variance provided - duplicate resolution requires R² values!\n"
            "Provide celltype_explained_variance from compute_nmf_per_celltype or compute_pca_per_celltype."
        )
        raise ValueError("celltype_explained_variance required for pool-based selection")

    celltype_factor_explained_variance = celltype_explained_variance
    
    # Log variance statistics
    for celltype, factor_r2_series in celltype_factor_explained_variance.items():
        logger.debug(
            f"  {celltype}: {len(factor_r2_series)} factors, "
            f"mean R²={factor_r2_series.mean():.4f}"
        )

    # Apply factor-aware duplicate resolution
    logger.info("Resolving duplicates using R² × loading contribution...")
    resolved = resolve_duplicates_factor_aware_per_celltype(
        celltype_genes_per_factor=celltype_genes_per_factor,
        celltype_factor_explained_variance=celltype_factor_explained_variance,
    )

    pool_genes = resolved["selected_genes"]  # Duplicate-free pool
    pool_factor_assignments = resolved["factor_assignments"]
    pool_celltype_assignments = resolved["celltype_assignments"]

    logger.info(
        f"✓ Phase 2 complete: {len(pool_genes)} unique genes "
        f"({resolved['duplicates_resolved']} duplicates resolved)"
    )

    # Cache duplicate-free pool to disk
    if results_dir:
        import pandas as pd
        from pathlib import Path
        
        pool_cache_dir = Path(results_dir) / "nmf_pools"
        
        # Save as CSV for easy inspection
        pool_df = pd.DataFrame({
            "gene": pool_genes,
            "celltype": [pool_celltype_assignments.get(g) for g in pool_genes],
            "factor": [pool_factor_assignments.get(g) for g in pool_genes],
        })
        resolved_pool_file = pool_cache_dir / "resolved_pool.csv"
        pool_df.to_csv(resolved_pool_file, index=False)
        logger.info(f"✓ Resolved pool saved to: {resolved_pool_file}")
        
        # Save duplicate resolution statistics
        import json
        stats_file = pool_cache_dir / "resolution_stats.json"
        with open(stats_file, "w") as f:
            json.dump({
                "pool_size_per_celltype": pool_size_per_celltype,
                "raw_pool_size": pool_stats["total_selections"],
                "resolved_pool_size": len(pool_genes),
                "duplicates_resolved": resolved["duplicates_resolved"],
                "celltypes": list(celltype_loadings.keys()),
                "n_components": n_components,
            }, f, indent=2)
        logger.info(f"✓ Resolution stats saved to: {stats_file}")

    # ============================================================================
    # PHASE 3: FINAL SELECTION (Factor-aware selection from duplicate-free pool)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 3: Selecting final genes from duplicate-free pool")
    logger.info("─" * 80)

    # Calculate genes per celltype-factor combo for FINAL selection (round UP)
    # Example: 100 genes ÷ 16 celltypes ÷ 5 factors = 1.25 → 2 genes per combo
    import math
    genes_per_celltype_final = probeset_size / len(celltype_loadings)
    genes_per_combo_final = genes_per_celltype_final / n_components
    genes_per_combo_final_rounded = math.ceil(genes_per_combo_final)  # ROUND UP
    
    logger.info(
        f"Final allocation: {genes_per_celltype_final:.2f} genes/celltype, "
        f"{genes_per_combo_final:.2f} genes/combo (rounded UP to {genes_per_combo_final_rounded})"
    )

    # Select genes from pool per celltype-factor combo
    final_selected_genes = []
    final_factor_assignments = {}
    final_celltype_assignments = {}
    abs_loadings_per_celltype = {
        celltype: loadings_df.abs()
        for celltype, loadings_df in celltype_loadings.items()
    }

    for celltype in celltype_loadings.keys():
        abs_loadings_df = abs_loadings_per_celltype[celltype]
        component_cols = celltype_loadings[celltype].columns[:n_components]
        factor_r2_series = celltype_explained_variance[celltype]

        for factor in component_cols:
            # Get pool genes for this celltype-factor combo
            combo_pool_genes = [
                g for g in pool_genes
                if pool_celltype_assignments.get(g) == celltype
                and pool_factor_assignments.get(g) == factor
            ]

            if len(combo_pool_genes) == 0:
                phase1_count = celltype_genes_per_factor[celltype].get(factor, 0)
                logger.warning(
                    f"No pool genes for {celltype}/{factor} - empty after Phase 2 duplicate resolution "
                    f"(Phase 1: {phase1_count} genes → {resolved['duplicates_resolved']} total duplicates resolved; "
                    f"need {genes_per_combo_final_rounded}/combo for Phase 3)"
                )
                continue

            # Calculate variance-weighted contributions for ranking
            gene_contributions = {}
            for gene in combo_pool_genes:
                if gene in abs_loadings_df.index:
                    loading = float(abs_loadings_df.at[gene, factor])
                    factor_r2 = factor_r2_series.get(factor, 0.0)
                    gene_contributions[gene] = loading * factor_r2
                else:
                    gene_contributions[gene] = 0.0

            # Select top N genes by contribution
            sorted_genes = sorted(gene_contributions.items(), key=lambda x: x[1], reverse=True)
            n_to_select = min(genes_per_combo_final_rounded, len(sorted_genes))
            selected_genes_combo = [g for g, _ in sorted_genes[:n_to_select]]

            # Add to final selection
            for gene in selected_genes_combo:
                if gene not in final_selected_genes:
                    final_selected_genes.append(gene)
                    final_factor_assignments[gene] = factor
                    final_celltype_assignments[gene] = celltype

    logger.info(f"✓ Phase 3 complete: {len(final_selected_genes)} genes selected from pool")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL TRIMMING: Reduce to exactly probeset_size (factor-balanced removal)
    #
    # Rules:
    #  - Only remove from celltypes that have >10% more genes than the target
    #    (target = probeset_size / n_celltypes).  Celltypes at or below
    #    target * 1.1 are protected from removal.
    #  - Within an eligible celltype, a gene is only removable when its factor
    #    would still have >= MIN_FACTOR_CONTRIBUTION genes after removal.
    #  - Per iteration, among ALL eligible genes across ALL over-represented
    #    celltypes, remove the one with the lowest selection score.
    #  - If no eligible gene exists (all celltypes at/below threshold, or
    #    all remaining genes are factor-contribution floors), fall back to
    #    global lowest-score removal without floor protection (with a warning).
    # ──────────────────────────────────────────────────────────────────────────
    if len(final_selected_genes) > probeset_size:
        from collections import Counter
        n_to_remove = len(final_selected_genes) - probeset_size
        n_celltypes_with_genes = len(celltype_loadings)
        target_per_ct = probeset_size / n_celltypes_with_genes  # float, e.g. 6.25
        overage_threshold = target_per_ct * 1.1
        logger.info(
            f"Trimming {len(final_selected_genes)} → {probeset_size} genes "
            f"({n_to_remove} to remove, factor-balanced; "
            f"target/CT={target_per_ct:.2f}, overage threshold={overage_threshold:.2f})"
        )

        fallback_used = 0
        while len(final_selected_genes) > probeset_size:
            # Current per-celltype gene counts
            ct_counts = Counter(
                final_celltype_assignments.get(g, '__unknown__')
                for g in final_selected_genes
            )
            # Current per-(celltype, factor) gene counts
            ct_factor_counts: dict[tuple, int] = Counter(
                (final_celltype_assignments.get(g, '__unknown__'),
                 final_factor_assignments.get(g, '__unknown__'))
                for g in final_selected_genes
            )

            # Celltypes eligible for removal (>10% over target)
            eligible_cts = {
                ct for ct, cnt in ct_counts.items()
                if cnt > overage_threshold
            }

            # Build candidate list: (score, gene) for all removable genes
            candidates: list[tuple[float, str]] = []
            for g in final_selected_genes:
                ct = final_celltype_assignments.get(g, '__unknown__')
                if ct not in eligible_cts:
                    continue
                fac = final_factor_assignments.get(g, '__unknown__')
                # Guard: after removal this factor must still meet MIN_FACTOR_CONTRIBUTION
                if ct_factor_counts[(ct, fac)] <= MIN_FACTOR_CONTRIBUTION:
                    continue
                ct_abs_loadings_df = abs_loadings_per_celltype[ct]
                factor_r2_series = celltype_explained_variance[ct]
                if fac and g in ct_abs_loadings_df.index:
                    loading = float(ct_abs_loadings_df.at[g, fac])
                    r2 = factor_r2_series.get(fac, 0.0)
                    score = loading * r2
                else:
                    score = 0.0
                candidates.append((score, g))

            if candidates:
                worst_gene = min(candidates, key=lambda x: x[0])[1]
            else:
                # Fallback: no cell type is sufficiently over threshold or all
                # remaining genes are at the factor floor — remove global worst
                fallback_used += 1
                fallback_candidates: list[tuple[float, str]] = []
                for g in final_selected_genes:
                    ct = final_celltype_assignments.get(g, '__unknown__')
                    fac = final_factor_assignments.get(g, '__unknown__')
                    ct_abs_loadings_df = abs_loadings_per_celltype.get(ct)
                    if ct_abs_loadings_df is not None and fac and g in ct_abs_loadings_df.index:
                        loading = float(ct_abs_loadings_df.at[g, fac])
                        r2 = celltype_explained_variance[ct].get(fac, 0.0)
                        score = loading * r2
                    else:
                        score = 0.0
                    fallback_candidates.append((score, g))
                worst_gene = min(fallback_candidates, key=lambda x: x[0])[1]

            final_selected_genes.remove(worst_gene)
            del final_factor_assignments[worst_gene]
            del final_celltype_assignments[worst_gene]

        if fallback_used:
            logger.warning(
                f"  {fallback_used} fallback removals were needed (no over-represented "
                f"celltype with removable gene found — factor floor or threshold exhausted)"
            )
        logger.info(f"✓ Trimmed to exactly {len(final_selected_genes)} genes")
    elif len(final_selected_genes) < probeset_size:
        logger.warning(
            f"Selected {len(final_selected_genes)} genes (target: {probeset_size}). "
            f"Pool may need larger size or more balanced allocation."
        )

    # Log final celltype distribution
    final_genes_per_ct = {}
    for ct in celltype_loadings.keys():
        ct_genes = [g for g in final_selected_genes if final_celltype_assignments.get(g) == ct]
        final_genes_per_ct[ct] = len(ct_genes)

    logger.info(f"Final celltype distribution (target: {probeset_size} genes):")
    for ct, count in sorted(final_genes_per_ct.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {ct}: {count} genes")

    # ============================================================================
    # ADD GENES TO BUILDER
    # Add ALL Phase 2 pool genes so ODT has a large replacement pool.
    # Only Phase 3 final genes are marked as selected_initial.
    # Non-selected pool genes automatically become ODT replacement candidates.
    # ============================================================================
    logger.info("")
    logger.info(
        f"Adding Phase 2 pool to builder: {len(pool_genes)} genes "
        f"({len(final_selected_genes)} selected + "
        f"{len(pool_genes) - len(final_selected_genes)} ODT replacement candidates)"
    )

    selected_set = set(final_selected_genes)

    for gene in pool_genes:
        celltype = pool_celltype_assignments.get(gene)
        factor = pool_factor_assignments.get(gene)

        if not celltype or not factor:
            logger.warning(f"Gene {gene} missing celltype/factor assignment - skipping")
            continue

        abs_loadings_df = abs_loadings_per_celltype[celltype]

        if gene not in abs_loadings_df.index:
            logger.warning(f"Gene {gene} not found in {celltype} loadings - skipping")
            continue

        loading = float(abs_loadings_df.at[gene, factor])
        factor_r2 = celltype_explained_variance[celltype].get(factor, 0.0)
        selection_score = loading * factor_r2

        builder.add_gene(
            gene=gene,
            selection_score=selection_score,
            rank=None,
            celltype=celltype,
            component=str(factor),
            metadata={
                "loading": float(loading),
                "factor_r2": float(factor_r2),
                "pool_size_per_celltype": pool_size_per_celltype,
                "phase": "pool_based",
            },
        )

    # Mark only Phase 3 final genes as initially selected
    for gene in final_selected_genes:
        builder.mark_selected(gene)

    logger.info(
        f"✓ Added {len(pool_genes)} pool genes; "
        f"{len(final_selected_genes)} marked selected, "
        f"{len(pool_genes) - len(selected_set)} available as ODT replacement candidates"
    )
    logger.info(f"=" * 80)


def calculate_genes_per_celltype(probeset_size: int, n_celltypes: int) -> int:
    """Calculate target genes per cell type for equal distribution.

    Args:
        probeset_size: Total target size
        n_celltypes: Number of cell types

    Returns:
        Genes per cell type

    Examples:
        >>> calculate_genes_per_celltype(500, 10)
        50
        >>> calculate_genes_per_celltype(500, 7)
        72  # Rounded up to ensure coverage
    """
    return math.ceil(probeset_size / n_celltypes)
