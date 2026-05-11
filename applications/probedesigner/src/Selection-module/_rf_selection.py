"""Random Forest gene selection.

This module provides classifier-based gene selection using Random Forest to rank
genes by feature importance. Uses cross-validation with F1-weighted scoring.

Author: Refactored from _selection.py
Date: 2026-02-08
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Import utility functions for data validation
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
    UTILITY_DIR = SCRIPT_DIR.parent.parent / "Utility-module"
    sys.path.insert(0, str(UTILITY_DIR))
    from _validation import is_anndata_raw
except ImportError:
    # Fallback implementation if utility module not available
    def is_anndata_raw(adata):
        """Fallback: check if data appears to be raw counts"""
        if issparse(adata.X):
            sample_data = adata.X.data[:10000] if len(adata.X.data) > 10000 else adata.X.data
        else:
            flat_data = adata.X.flatten()
            sample_data = flat_data[:10000] if len(flat_data) > 10000 else flat_data
        non_zero = sample_data[sample_data != 0]
        return len(non_zero) > 0 and np.allclose(non_zero, np.round(non_zero))

# Use absolute imports (for script execution)
from _constants import (
    COL_CELLTYPE,
    COL_GENE,
    COL_RANK,
    COL_SELECTION_SCORE,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_RF_N_FOLDS,
    DEFAULT_RF_RANDOM_SEEDS,
)
from _deg_selection import filter_celltypes_by_min_cells
from _gene_list_builder import GeneListBuilder

logger = logging.getLogger(__name__)


def select_genes_with_rf(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    celltype_column: str = COL_CELLTYPE,
    max_depth: int = DEFAULT_RF_MAX_DEPTH,
    n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
    n_folds: int = DEFAULT_RF_N_FOLDS,
    random_seeds: Optional[list[int]] = None,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    use_deg_prefilter: bool = False,
    n_deg_per_group: Optional[int] = None,
    deg_results: Optional[GeneListBuilder] = None,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Select genes using Random Forest classifier with cross-validation.

    Trains a Random Forest classifier to predict cell types from gene expression
    and selects genes with highest feature importance across folds. Optionally
    pre-filters to DEG genes before classification.

    Args:
        adata: Processed AnnData object (log-normalized)
        probeset_size: Target number of genes to select
        celltype_column: Column in adata.obs for classification labels
        max_depth: Maximum depth of each tree in forest
        n_estimators: Number of trees in Random Forest
        n_folds: Number of cross-validation folds
        random_seeds: List of random seeds for multiple CV runs
        min_cells_per_celltype: Minimum cells per cell type for inclusion
        use_deg_prefilter: If True, pre-filter to DEG genes before classification
        n_deg_per_group: Number of DEG genes per group to use (if use_deg_prefilter=True)
        deg_results: Pre-computed DEG results (optional, for caching)
        rf_simple_results: Pre-computed simple RF results (optional, for rf_deg strategy)
        results_dir: Directory to save results (optional)

    Returns:
        GeneListBuilder with:
            - selected_genes: Top genes by F1-weighted importance
            - selection_score: F1-weighted feature importance
            - rank: Rank by weighted importance
            - metadata: Average importance, max importance, average F1

    Raises:
        ValueError: If not enough valid cell types after filtering

    Examples:
        >>> # Simple RF selection
        >>> builder = select_genes_with_rf(adata, probeset_size=100)
        
        >>> # RF with DEG pre-filtering
        >>> builder = select_genes_with_rf(
        ...     adata,
        ...     probeset_size=100,
        ...     use_deg_prefilter=True,
        ...     n_deg_per_group=200
        ... )
        
        >>> # RF with cached DEG results
        >>> deg_builder = select_DEGs(adata, ...)
        >>> rf_builder = select_genes_with_rf(
        ...     adata,
        ...     use_deg_prefilter=True,
        ...     deg_results=deg_builder
        ... )
    """
    logger.info("=== Random Forest Gene Selection ===")
    logger.info(f"Target probeset size: {probeset_size}")
    logger.info(f"Use DEG pre-filter: {use_deg_prefilter}")

    # Initialize GeneListBuilder
    strategy_name = "rf_deg" if use_deg_prefilter else "rf_simple"
    builder = GeneListBuilder(
        strategy_name=strategy_name,
        analysis_type="global",
    )

    # Default random seeds if not provided
    if random_seeds is None:
        random_seeds = DEFAULT_RF_RANDOM_SEEDS

    # Validate celltype column
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not found in adata.obs")

    # Filter cell types by minimum cell count
    if min_cells_per_celltype > 0:
        valid_celltypes, excluded_celltypes, celltype_counts = filter_celltypes_by_min_cells(
            adata, celltype_column, min_cells_per_celltype
        )

        if len(excluded_celltypes) > 0:
            logger.info(f"Filtering to {len(valid_celltypes)} valid cell types")
            adata = adata[adata.obs[celltype_column].isin(valid_celltypes)].copy()
            adata.obs[celltype_column] = adata.obs[celltype_column].cat.remove_unused_categories()

    if adata.shape[0] == 0:
        raise ValueError("No cells remaining after filtering")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(adata.obs[celltype_column].values)
    class_names = list(label_encoder.classes_)
    logger.info(f"Classes: {len(class_names)} ({', '.join(class_names[:5])}...)")

    # Optional DEG pre-filtering
    candidate_genes = adata.var_names.tolist()
    if use_deg_prefilter:
        logger.info("Applying DEG pre-filter...")
        
        # Try to use cached DEG results first
        if deg_results is not None:
            logger.info("Using cached DEG results")
            candidate_genes = deg_results.get_selected_genes()
        else:
            # Fallback: compute DEGs
            logger.info("No cached DEG results - computing new DEGs")
            candidate_genes = _get_deg_genes_for_classification(
                adata,
                celltype_column,
                n_deg_per_group,
                probeset_size,
            )
        
        logger.info(f"DEG pre-filter: {len(candidate_genes)} candidate genes")

    # Ensure candidate genes are present and aligned with the feature matrix.
    candidate_genes = [g for g in candidate_genes if g in adata.var_names]
    if not candidate_genes:
        raise ValueError("No candidate genes available after filtering to adata.var_names")

    # Use expression data for RF classification on the candidate gene subset.
    # This avoids training on unused columns and keeps feature_importances_ aligned.
    X_source = adata[:, candidate_genes].X
    if issparse(X_source):
        X = X_source.toarray()
    else:
        X = np.asarray(X_source)
    
    # Verify data is normalized (RF works better with log-normalized data)
    if is_anndata_raw(adata):
        logger.warning(
            "Data appears to contain raw counts (integer values). "
            "Random Forest typically works better with log-normalized data. "
            "Consider using log1p-transformed data in adata.X for better performance."
        )
    
    # Check for cached RF results
    if results_dir:
        cache_file = os.path.join(results_dir, 'rf_models', 'rf_gene_scores.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached RF results from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                # Verify cache is compatible
                if (cached_results['n_genes'] == len(candidate_genes) and
                    cached_results['use_deg_prefilter'] == use_deg_prefilter):
                    
                    logger.info("✓ Using cached RF gene scores")
                    gene_summary = cached_results['gene_summary']
                    
                    # Rebuild GeneListBuilder from cached results in bulk
                    builder.add_genes(
                        genes=gene_summary["gene"].tolist(),
                        ranks=list(range(1, len(gene_summary) + 1)),
                        scores=gene_summary["final_score"].tolist(),
                        avg_importance=gene_summary["avg_importance"].tolist(),
                        max_importance=gene_summary["max_importance"].tolist(),
                        avg_f1_score=gene_summary["avg_f1_score"].tolist(),
                    )
                    
                    # Mark top N as selected
                    selected_genes = gene_summary.head(probeset_size)["gene"].tolist()
                    for gene in selected_genes:
                        builder.mark_selected(gene)
                    
                    logger.info(f"Selected {len(selected_genes)} genes from cache")
                    return builder
                else:
                    logger.warning("Cached results incompatible (different gene set). Recomputing...")
            except Exception as e:
                logger.warning(f"Failed to load cached RF results: {e}. Recomputing...")

    # Cross-validation with multiple seeds (not cached)
    all_gene_scores = []

    for seed_idx, seed in enumerate(random_seeds):
        logger.info(f"Running CV with seed {seed} ({seed_idx + 1}/{len(random_seeds)})")

        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y_encoded)):
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Train Random Forest
            sample_weight = compute_sample_weight("balanced", y_train)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=seed + fold_idx,  # Ensure different random state per fold,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)

            # Evaluate
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="macro")
            logger.info(f"    F1 score: {f1:.4f}")

            # Record gene scores
            for gene, importance in zip(candidate_genes, model.feature_importances_):
                all_gene_scores.append(
                    {
                        "gene": gene,
                        "fold": f"seed{seed}_fold{fold_idx}",
                        "importance": importance,
                        "f1_score": f1,
                    }
                )

    # Aggregate scores across folds
    gene_scoring_df = pd.DataFrame(all_gene_scores)

    # F1-weighted importance (genes important in good models ranked higher)
    # Each gene appears in multiple folds (n_folds × n_random_seeds)
    # Weight each appearance by model performance (F1), then average for consensus
    gene_scoring_df["weighted_importance"] = (
        gene_scoring_df["importance"] * gene_scoring_df["f1_score"]
    )

    # Aggregate across all fold appearances to get consensus gene ranking
    # - avg_importance: Mean feature importance across folds
    # - max_importance: Peak importance achieved in any fold
    # - avg_f1_score: Mean model performance where gene appeared
    # - weighted_importance: Mean of (importance × f1), prioritizes genes important in good models
    gene_summary = gene_scoring_df.groupby("gene").agg(
        avg_importance=("importance", "mean"),
        max_importance=("importance", "max"),
        avg_f1_score=("f1_score", "mean"),
        weighted_importance=("weighted_importance", "mean"),
    ).reset_index()

    gene_summary["final_score"] = gene_summary["weighted_importance"]
    gene_summary = gene_summary.sort_values("final_score", ascending=False)

    # Add all genes to builder in bulk
    builder.add_genes(
        genes=gene_summary["gene"].tolist(),
        ranks=list(range(1, len(gene_summary) + 1)),
        scores=gene_summary["final_score"].tolist(),
        avg_importance=gene_summary["avg_importance"].tolist(),
        max_importance=gene_summary["max_importance"].tolist(),
        avg_f1_score=gene_summary["avg_f1_score"].tolist(),
    )

    # Mark top N as selected
    selected_genes = gene_summary.head(probeset_size)["gene"].tolist()
    for gene in selected_genes:
        builder.mark_selected(gene)

    logger.info(f"Selected {len(selected_genes)} genes by Random Forest importance")

    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

        builder.to_csv(os.path.join(results_dir, "selected_genes.csv"))

        # Save diagnostic output: full gene rankings with importance metrics
        # NOTE: This is RF-specific diagnostic output (not present in other strategies)
        gene_summary_file = os.path.join(results_dir, "consensus_genes.csv")
        gene_summary.to_csv(gene_summary_file, index=False)
        logger.info(f"Saved gene summary (diagnostic) to {gene_summary_file}")
        
        # Cache RF results for reuse by combination strategies
        model_dir = os.path.join(results_dir, 'rf_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, 'rf_gene_scores.pkl')
        try:
            cache_data = {
                'gene_summary': gene_summary,
                'all_gene_scores': gene_scoring_df,
                'candidate_genes': candidate_genes,
                'n_genes': len(candidate_genes),
                'use_deg_prefilter': use_deg_prefilter,
                'n_folds': n_folds,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_seeds': random_seeds,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✓ Saved RF results to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save RF cache: {e}")

    return builder


def _get_deg_genes_for_classification(
    adata: AnnData,
    celltype_column: str,
    n_deg_per_group: Optional[int],
    probeset_size: int,
) -> list[str]:
    """Get DEG genes for pre-filtering before classification.
    
    Ranks genes by Scanpy DEG scores (Z-scores for Wilcoxon), ensuring
    genes are ordered by their differential expression significance.

    Args:
        adata: AnnData object
        celltype_column: Grouping column
        n_deg_per_group: Genes per group (if None, computed from probeset_size)
        probeset_size: Target size for final panel

    Returns:
        List of DEG gene names ranked by Scanpy scores (descending)
    """
    # Ensure categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[celltype_column]):
        adata.obs[celltype_column] = adata.obs[celltype_column].astype("category")

    groups = adata.obs[celltype_column].cat.categories.tolist()

    # Calculate genes per group if not specified
    if n_deg_per_group is None:
        n_deg_per_group = max(50, probeset_size // len(groups) * 3)  # 3x oversampling

    logger.info(f"Running DEG analysis: {n_deg_per_group} genes per group")

    # Run DEG analysis
    sc.tl.rank_genes_groups(
        adata,
        groupby=celltype_column,
        method="wilcoxon",
        use_raw=False,
        n_genes=None,  # Get all genes to extract scores
        key_added="rank_genes_groups",
    )

    # Extract genes with scores per group
    deg_records = []
    for group in groups:
        try:
            # Use scanpy's get_rank_genes_groups_df to get proper scores
            group_df = sc.get.rank_genes_groups_df(adata, group=group, key='rank_genes_groups')
            # Keep top N genes per group
            group_df = group_df.head(n_deg_per_group)
            group_df['group'] = group
            deg_records.append(group_df)
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not extract genes for group {group}: {e}")
            continue

    # Combine all groups
    if not deg_records:
        logger.warning("No DEG results extracted - returning empty list")
        return []
    
    deg_df = pd.concat(deg_records, ignore_index=True)
    
    # Remove duplicates, keeping highest score
    deg_df_dedup = deg_df.sort_values('scores', ascending=False).drop_duplicates(
        subset='names', keep='first'
    )
    
    # Return genes ranked by score
    deg_genes = deg_df_dedup['names'].tolist()
    
    logger.info(f"Extracted {len(deg_genes)} unique DEG genes ranked by Scanpy scores")

    return deg_genes
