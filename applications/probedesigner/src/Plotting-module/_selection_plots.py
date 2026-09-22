"""Gene selection result visualization utilities.

This module contains plotting functions for visualizing gene selection
results, including feature importance, confusion matrices, and gene
source distributions.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _clustering_plots import extract_display_name_from_dataset

from _constants import (
    COL_FEATURE_IMPORTANCE,
    COL_GENE,
    DEFAULT_PNG_DPI,
)

logger = logging.getLogger(__name__)

__all__ = [
    "plot_confusion_matrix",
    "plot_feature_importances",
    "plot_f1_distribution",
    "plot_final_gene_dotplot",
    "plot_gene_source_distribution",
    "plot_genes_per_celltype",
]

def plot_confusion_matrix(conf_matrix, class_names, seed, fold_idx, results_dir):
    """Plot and save one CV fold's confusion matrix as a heatmap.

    Args:
        conf_matrix: Square confusion-matrix array (true x predicted).
        class_names: Ordered class labels for both axes.
        seed: CV seed (used in the title and filename).
        fold_idx: 0-based fold index (shown/saved as ``fold_idx + 1``).
        results_dir: Directory the PNG is written to.
    """
    plt.figure(figsize=(10, 8))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    sns.heatmap(conf_matrix_df, annot=True, cmap='Blues', fmt='g')
    plt.title(f'Confusion Matrix - Seed {seed} - Fold {fold_idx+1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_seed{seed}_fold{fold_idx+1}.png'), dpi=DEFAULT_PNG_DPI)
    plt.close()

def plot_feature_importances(feature_importances, seed, fold_idx, results_dir):
    """Plot and save one CV fold's top-30 gene feature importances.

    Args:
        feature_importances: DataFrame with ``gene`` and ``feature_importance``
            columns, sorted best-first.
        seed: CV seed (used in the title and filename).
        fold_idx: 0-based fold index (shown/saved as ``fold_idx + 1``).
        results_dir: Directory the PNG is written to.
    """
    top_n = min(30, len(feature_importances))
    plt.figure(figsize=(12, 8))
    sns.barplot(x=COL_FEATURE_IMPORTANCE, y=COL_GENE, data=feature_importances.head(top_n))
    plt.title(f'Top {top_n} Gene Importances - Seed {seed} - Fold {fold_idx+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'feature_imp_seed{seed}_fold{fold_idx+1}.png'), dpi=DEFAULT_PNG_DPI)
    plt.close()

def plot_f1_distribution(f1_scores, results_dir):
    """Plot and save the macro-F1 distribution across all CV folds.

    Args:
        f1_scores: Iterable of per-fold macro-F1 values.
        results_dir: Directory the PNG is written to.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(f1_scores, kde=True)
    plt.axvline(np.mean(f1_scores), color='red', linestyle='--', 
                label=f'Mean F1 score: {np.mean(f1_scores):.4f}')
    plt.title('Macro F1 Score Distribution Across All Folds')
    plt.xlabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'f1_score_distribution.png'), dpi=DEFAULT_PNG_DPI)
    plt.close()

def _order_genes_by_celltype_contribution(adata, genes, groupby, target_celltype):
    """Order ``genes`` via hierarchical clustering, oriented so genes most specific
    to ``target_celltype`` lead.

    Computes each gene's mean expression per ``groupby`` category, z-scores each
    gene's row (so clustering reflects relative expression *pattern* across
    categories, not absolute magnitude -- a highly-expressed gene shouldn't dominate
    the tree just because of scale), and hierarchically clusters the genes
    (average-linkage, correlation distance) on those z-scored profiles. The
    resulting tree is walked from the root, and at every merge the two child
    branches are ordered by their mean z-scored value in ``target_celltype`` (higher
    first) -- a gene's z-scored value in that column doubles as its "specificity"
    for ``target_celltype`` (how much higher its expression there is relative to its
    own expression elsewhere, not just its raw level). The result is a leaf order
    where similar-pattern genes stay adjacent (the clustering) and the overall
    sequence trends from most- to least-specific for ``target_celltype`` (the
    ordering).

    Falls back to the input order (deduplicated, restricted to genes present in
    ``adata``) if there are fewer than 3 genes, ``target_celltype`` isn't a category
    of ``adata.obs[groupby]``, or clustering fails for any reason -- never raises,
    this is a display nicety.
    """
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    if len(genes) < 3 or target_celltype not in set(adata.obs[groupby].astype(str)):
        return genes

    try:
        import scipy.sparse as sp
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import pdist

        X = adata[:, genes].X
        if sp.issparse(X):
            X = X.toarray()
        expr_df = pd.DataFrame(np.asarray(X), columns=genes, index=adata.obs_names)
        expr_df[groupby] = adata.obs[groupby].astype(str).values
        mean_expr = expr_df.groupby(groupby, observed=True)[genes].mean().T  # genes x categories

        row_std = mean_expr.std(axis=1, ddof=0).replace(0, 1)
        z = mean_expr.sub(mean_expr.mean(axis=1), axis=0).div(row_std, axis=0)
        if target_celltype not in z.columns:
            return genes
        contribution = z[target_celltype]

        distance = pdist(z.values, metric='correlation')
        tree = sch.to_tree(sch.linkage(distance, method='average'))

        order: list[int] = []

        def _visit(node) -> None:
            if node.is_leaf():
                order.append(node.id)
                return
            left, right = node.get_left(), node.get_right()
            left_score = contribution.iloc[left.pre_order()].mean()
            right_score = contribution.iloc[right.pre_order()].mean()
            first, second = (left, right) if left_score >= right_score else (right, left)
            _visit(first)
            _visit(second)

        _visit(tree)
        return [genes[i] for i in order]
    except Exception as exc:
        logging.warning(f"Gene ordering by celltype contribution failed (non-fatal): {exc}")
        return genes


def plot_final_gene_dotplot(adata, probeset_genes, deg_genes, nmf_genes, gene_scores=None, groupby='original',
                           output_path=None, figsize=(28, 16), cmap='viridis', grid=True, filename_prefix=None,
                           title=None):
    """
    Create an enhanced dotplot of the probeset genes with genes ordered by their combined scores,
    with clear visual indicators for gene sources (NMF-only, DEG-only, or both).

    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.
    probeset_genes : list
        List of final selected probeset genes to display
    deg_genes : list or set
        List of genes selected from the DEG-based decision trees
    nmf_genes : list or set
        List of genes selected from the NMF analysis
    gene_scores : dict, optional
        Dictionary mapping genes to their combined scores (default: None).
    groupby : str, optional
        Column name in adata.obs for grouping cells (default: 'original').
    output_path : str, optional
        Path to save the figure (default: None, which uses results_dir).
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str, optional
        Colormap for dot expression values.
    grid : bool, optional
        Whether to show grid lines.
    filename_prefix : str, optional
        Prefix for the output filename (``{filename_prefix}_final_gene_dotplot.png``);
        used only when ``output_path`` is not given.
    title : str, optional
        Figure title, set via ``plt.suptitle``. When a gene list is split into
        multiple chunks (see below), chunks after the first get `` (part i/N)``
        appended.

    Panels larger than 100 genes are split into multiple figures of at most 100 genes
    each (``..._chunk1.png``, ``..._chunk2.png``, ...) so each dotplot stays legible;
    panels of 100 genes or fewer keep the single unsuffixed filename.
    """
    # Convert inputs to sets for easier operations
    probeset_set = set(probeset_genes)
    deg_set = set(deg_genes)
    nmf_set = set(nmf_genes)
    
    # Create sets for each category
    deg_only_set = probeset_set.intersection(deg_set) - nmf_set
    nmf_only_set = probeset_set.intersection(nmf_set) - deg_set
    both_set = probeset_set.intersection(deg_set).intersection(nmf_set)
    
    logging.info(f"Creating dotplot with {len(probeset_set)} probeset genes")
    logging.info(f"- {len(deg_only_set)} from DEG-based decision trees only")
    logging.info(f"- {len(nmf_only_set)} from NMF only")
    logging.info(f"- {len(both_set)} from both DEG and NMF")
    
    # Keep only genes present in the dataset. Order-preserving (not routed through a
    # plain set, whose iteration order does not reliably match probeset_genes' order)
    # so a caller-supplied order -- e.g. hierarchical-clustering output -- survives
    # when gene_scores is None (see _order_genes_by_celltype_contribution below).
    available_genes = set(adata.var_names)
    available_probeset_genes = [
        g for g in dict.fromkeys(probeset_genes) if g in available_genes
    ]
    
    if not available_probeset_genes:
        logging.error("No probeset genes found in the dataset. Cannot create dotplot.")
        return
    
    # Process gene scores if provided
    if gene_scores is not None:
        # Log gene_scores information for debugging
        if isinstance(gene_scores, pd.DataFrame):
            logging.info(f"Gene scores provided as DataFrame with columns: {gene_scores.columns.tolist()}")
            logging.info(f"Gene scores DataFrame shape: {gene_scores.shape}")
            if len(gene_scores) > 0:
                logging.info(f"First row sample: {gene_scores.iloc[0].to_dict()}")
        else:
            logging.info(f"Gene scores provided as dictionary with {len(gene_scores)} entries")
        # Check if gene_scores is a DataFrame (from the error message)
        if isinstance(gene_scores, pd.DataFrame):
            # Create a dictionary from the DataFrame's gene and score columns
            matching_gene_scores = {}
            for _, row in gene_scores.iterrows():
                if row[COL_GENE] in available_probeset_genes:
                    # Try different possible column names for the score
                    if 'selection_score' in row and pd.notna(row['selection_score']):
                        matching_gene_scores[row[COL_GENE]] = row['selection_score']
                    elif 'combined_score' in row and pd.notna(row['combined_score']):
                        matching_gene_scores[row[COL_GENE]] = row['combined_score']
                    elif 'score' in row and pd.notna(row['score']):
                        matching_gene_scores[row[COL_GENE]] = row['score']
                    elif 'metric_value' in row and pd.notna(row['metric_value']):
                        matching_gene_scores[row[COL_GENE]] = row['metric_value']
        else:
            # Original dictionary handling
            matching_gene_scores = {gene: score for gene, score in gene_scores.items()
                                if gene in available_probeset_genes}

        if matching_gene_scores:
            # Sort genes by their combined scores in descending order
            plot_genes = sorted(available_probeset_genes,
                            key=lambda g: matching_gene_scores.get(g, float('-inf')),
                            reverse=True)
            logging.info(f"Ordered {len(plot_genes)} genes by their combined scores")

            # Log the top 5 genes and their scores for verification
            top_genes = plot_genes[:min(5, len(plot_genes))]
            top_scores = [matching_gene_scores.get(g, 'N/A') for g in top_genes]
            logging.info(f"Top 5 genes by score: {list(zip(top_genes, top_scores))}")
        else:
            plot_genes = available_probeset_genes
            logging.warning("No matching genes found between provided scores and available probeset genes")
            logging.warning(f"Available genes sample: {available_probeset_genes[:5]}")
            if isinstance(gene_scores, pd.DataFrame) and not gene_scores.empty:
                sample_genes = gene_scores[COL_GENE].tolist()[:5]
                logging.warning(f"Gene scores DataFrame sample genes: {sample_genes}")
    else:
        plot_genes = available_probeset_genes
        logging.info("No gene scores provided. Using original gene order.")

    # Generate dotplot using scanpy
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})

    # Large panels are hard to read as one dotplot -- split into chunks of at most
    # 100 genes each, writing one figure per chunk (backward compatible: a single
    # chunk keeps the unsuffixed filename).
    DOTPLOT_CHUNK_SIZE = 100
    gene_chunks = [
        plot_genes[i:i + DOTPLOT_CHUNK_SIZE]
        for i in range(0, len(plot_genes), DOTPLOT_CHUNK_SIZE)
    ] or [plot_genes]

    for chunk_idx, chunk_genes in enumerate(gene_chunks, start=1):
        sc.pl.dotplot(
            adata,
            var_names=chunk_genes,
            var_group_positions=[],
            groupby=groupby,
            standard_scale='var',
            cmap=cmap,
            show=False,
            var_group_rotation=90,
        )

        if title:
            chunk_title = (
                f'{title} (part {chunk_idx}/{len(gene_chunks)})'
                if len(gene_chunks) > 1 and chunk_idx > 1 else title
            )
            plt.suptitle(chunk_title, y=1.02, fontsize=18)

        chunk_suffix = f'_chunk{chunk_idx}' if len(gene_chunks) > 1 else ''
        final_filename = (
            f'{filename_prefix}_final_gene_dotplot{chunk_suffix}.png'
            if filename_prefix else f'final_gene_dotplot{chunk_suffix}.png'
        )
        logging.info(f"Saving dotplot to: {os.path.join(output_path, final_filename)}")
        plt.savefig(os.path.join(output_path, final_filename), dpi=DEFAULT_PNG_DPI, bbox_inches='tight')
        plt.close()


def plot_panel_expression_distribution(
    adata, panel_genes, output_dir, raw_layer='counts', bins=60,
    PNG_DPI=DEFAULT_PNG_DPI, filename_prefix=None,
):
    """Histogram of panel-gene expression values, pooled across every (gene, cell)
    pair, comparing raw counts against log-normalized values side by side.

    Unlike a per-gene summary (mean/median), this pools every individual expression
    value across all panel genes and all cells into one distribution per space, to
    show the overall SHAPE of expression (e.g. how zero-inflated raw counts are vs.
    the lognorm spread) rather than per-gene detail.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw counts in ``adata.layers[raw_layer]`` and
        log-normalized values in ``adata.X``.
    panel_genes : list
        Panel gene symbols to include (only those present in ``adata.var_names`` are
        used).
    output_dir : str
        Directory to save the figure into.
    raw_layer : str, optional
        Layer holding raw counts (default: ``'counts'``).
    bins : int, optional
        Number of histogram bins per panel (default: 60).
    PNG_DPI : int, optional
        Output resolution.
    filename_prefix : str, optional
        Prefix for the output filename
        (``{filename_prefix}_panel_gene_expression_distribution.png``).

    Returns
    -------
    str or None
        Path to the saved PNG, or ``None`` on failure / empty input.
    """
    import scipy.sparse as sp

    available_genes = [g for g in panel_genes if g in set(adata.var_names)]
    if not available_genes:
        logging.error("No panel genes found in the dataset. Cannot create expression distribution plot.")
        return None
    if raw_layer not in adata.layers:
        logging.error(f"Layer '{raw_layer}' not found in adata.layers; cannot plot raw distribution.")
        return None

    sub = adata[:, available_genes]

    def _flatten(m):
        if sp.issparse(m):
            return np.asarray(m.todense()).ravel()
        return np.asarray(m).ravel()

    raw_values = _flatten(sub.layers[raw_layer])
    lognorm_values = _flatten(sub.X)
    n_total = raw_values.size
    pct_zero_raw = 100.0 * float(np.mean(raw_values == 0)) if n_total else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(raw_values, bins=bins, color='#4C72B0', edgecolor='black', linewidth=0.3)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Raw count')
    axes[0].set_ylabel('Number of (gene, cell) pairs')
    axes[0].set_title(f'Raw counts\n({n_total:,} points, {pct_zero_raw:.1f}% zero)')

    axes[1].hist(lognorm_values, bins=bins, color='#55A868', edgecolor='black', linewidth=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Log-normalized expression')
    axes[1].set_ylabel('Number of (gene, cell) pairs')
    axes[1].set_title(f'Log-normalized\n({n_total:,} points)')

    fig.suptitle(
        f'Panel gene expression distribution ({len(available_genes)} genes × {sub.n_obs:,} cells)',
        fontsize=16,
    )
    plt.tight_layout()

    filename = (
        f'{filename_prefix}_panel_gene_expression_distribution.png'
        if filename_prefix else 'panel_gene_expression_distribution.png'
    )
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=PNG_DPI, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved panel gene expression distribution plot to: {out_path}")
    return out_path


def _convert_provenance_to_gene_source(provenance_df, reduction_type='NMF'):
    """
    Convert provenance report format to gene_source format for plotting.
    
    Parameters:
    -----------
    provenance_df : pd.DataFrame
        Provenance report with initial_source, source_details columns
    reduction_type : str
        'NMF' or 'PCA' for dimensionality reduction type
        
    Returns:
    --------
    pd.DataFrame with gene_source column added
    """
    df = provenance_df.copy()
    
    def map_to_gene_source(row):
        """Map provenance columns to gene_source string"""
        initial_source = row['initial_source']
        source_details = row.get('source_details', '')
        
        # Force-include / user-defined markers
        if initial_source == 'force_include':
            return 'user-defined-marker'
        
        # Replacement genes
        if initial_source == 'blacklist_replacement':
            replaced_gene = row.get('blacklist_replacement_for', 'unknown')
            return f'replacement for {replaced_gene} (blacklist)'
        
        if initial_source == 'xenium_replacement':
            replaced_gene = row.get('xenium_replacement_for', 'unknown')
            return f'replacement for {replaced_gene} (xenium)'
        
        # Initial selection genes
        if initial_source == 'initial_selection':
            # Check source_details for specific source
            if source_details in ['DT', 'dt']:
                return 'DT'
            elif source_details.upper() in ['NMF', 'PCA']:
                return source_details.upper()
            elif 'both' in source_details.lower():
                return f'both (DT & {reduction_type})'
            elif 'dimred' in source_details.lower():
                return reduction_type
            elif 'gap-filling' in source_details.lower() or 'cell-type' in source_details.lower():
                if 'cell-type' in source_details.lower() or 'celltype' in source_details.lower():
                    return 'cell-type-specific gap-filling'
                elif 'global' in source_details.lower():
                    return 'global gap-filling'
                elif 'deg' in source_details.lower():
                    return 'DEG-based gap-filling'
                else:
                    return 'gap-filling'
            else:
                return source_details if source_details else 'initial_selection'
        
        return initial_source
    
    df['gene_source'] = df.apply(map_to_gene_source, axis=1)
    return df


def plot_gene_source_distribution(summary_df=None, summary_csv_path=None, results_dir=None, reduction_type='NMF', strategy_name=None, figsize=(12, 8)):
    """
    Create a bar plot showing the distribution of selected genes by source
    (DT-unique, dimred-unique, both, gap-filling, blacklist-replacement, xenium-replacement).
    
    This function visualizes how many genes came from each source based on the
    gene_source column in the gene_selection_replacement_summary or provenance report.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame, optional
        DataFrame with gene selection summary (if already loaded)
    summary_csv_path : str, optional
        Path to the gene_selection_replacement_summary.csv file (if not loaded)
    results_dir : str
        Directory to save the output plot
    reduction_type : str, optional
        Type of dimensionality reduction used ('PCA' or 'NMF'), for plot labels
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 8))
        
    Returns:
    --------
    str : Path to the saved plot
    """
    try:
        # Load data if not provided
        if summary_df is None:
            # Prefer provenance report over old summary file
            provenance_path = os.path.join(results_dir, 'final_panel_with_provenance.csv') if results_dir else None
            
            if provenance_path and os.path.exists(provenance_path):
                logging.info(f"Loading provenance report: {provenance_path}")
                summary_df = pd.read_csv(provenance_path)
                # Map provenance columns to gene_source format for plotting
                summary_df = _convert_provenance_to_gene_source(summary_df, reduction_type)
            elif summary_csv_path and os.path.exists(summary_csv_path):
                logging.info(f"Loading gene selection summary: {summary_csv_path}")
                summary_df = pd.read_csv(summary_csv_path)
            else:
                logging.warning(f"No summary data found. Tried: {provenance_path}, {summary_csv_path}")
                return None
        
        logging.info(f"Loaded gene selection summary: {len(summary_df)} total genes")
        
        # Filter to only genes in final panel (exclude blacklisted genes)
        # Genes with 'replaced as blacklisted' or 'blacklisted in component' are NOT in final panel
        selected_df = summary_df[
            ~summary_df['gene_source'].str.contains('blacklisted', case=False, na=False)
        ].copy()
        logging.info(f"Final panel genes for plot: {len(selected_df)} genes (excluded blacklisted)")
        
        if len(selected_df) == 0:
            logging.warning("No selected genes found in summary file")
            return None
        
        # Categorize genes by their source
        def categorize_source(gene_source):
            """Categorize gene source into plot-friendly categories"""
            gene_source_lower = gene_source.lower()
            
            # Check for user-defined markers
            if 'user-defined' in gene_source_lower or 'user-specified' in gene_source_lower:
                return 'User-defined Markers'
            
            # Check for replacement types
            if 'replacement for' in gene_source_lower:
                if 'blacklist' in gene_source_lower:
                    return 'Blacklist Replacement'
                elif 'xenium' in gene_source_lower:
                    return 'Xenium Replacement'
                else:
                    return 'Other Replacement'
            
            # Check for gap-filling. Accept the RecoVar_panel_information.csv display
            # strings ('gap-fill (NMF)', 'gap-fill (random forest)'), the internal
            # identifiers ('gap_fill_celltype', 'gap_fill_deg', 'gap_fill_global'), and
            # the legacy hyphenated 'gap-filling: cell-type' labels.
            if ('gap-filling' in gene_source_lower or 'gap-fill' in gene_source_lower
                    or 'gap_fill' in gene_source_lower):
                if ('cell-type' in gene_source_lower or 'celltype' in gene_source_lower
                        or 'nmf' in gene_source_lower or 'pca' in gene_source_lower):
                    return 'Gap-filling: Cell-type'
                elif 'global' in gene_source_lower:
                    return 'Gap-filling: Global'
                elif ('deg' in gene_source_lower or 'random forest' in gene_source_lower
                        or 'rf' in gene_source_lower):
                    return 'Gap-filling: DEG'
                else:
                    return 'Gap-filling: Other'

            # Check for both DT/RF and dimred. Legacy 'both …' labels plus the
            # current 'overlap' CSV display string (internally 'overlap→rf_deg',
            # RF ∩ dimred, RF wins).
            if 'both' in gene_source_lower:
                return f'Both (DT & {reduction_type})'
            if gene_source_lower == 'overlap' or (
                    'overlap' in gene_source_lower and ('rf' in gene_source_lower or 'dt' in gene_source_lower)):
                return f'Both (RF & {reduction_type})'

            # Check for DT/RF-only. Legacy 'DT' plus the internal 'rf_deg' /
            # 'rf_simple' identifiers and the current 'random forest' CSV display string.
            if gene_source == 'DT' or gene_source_lower == 'dt':
                return 'DT-unique'
            if gene_source_lower in ('rf_deg', 'rf_simple', 'rf', 'random forest'):
                return 'RF-unique'

            # Check for dimred-only. Legacy 'NMF'/'PCA' plus the internal 'dimred'
            # identifier (the current 'NMF'/'PCA' CSV display string is already
            # caught by the .upper() check below).
            if gene_source.upper() in ['NMF', 'PCA'] or gene_source_lower == 'dimred':
                return f'{reduction_type}-unique'

            # Force-included / user-supplied markers (current Selection-module vocab)
            if gene_source_lower in ('force_include', 'force-include'):
                return 'User-defined Markers'

            # Default: unknown
            return 'Other'
        
        selected_df['source_category'] = selected_df['gene_source'].apply(categorize_source)
        
        # Count genes by category
        category_counts = selected_df['source_category'].value_counts()
        
        # Define category order for consistent plotting
        preferred_order = [
            'User-defined Markers',
            'DT-unique',
            'RF-unique',
            f'{reduction_type}-unique',
            f'Both (DT & {reduction_type})',
            f'Both (RF & {reduction_type})',
            'Gap-filling: Cell-type',
            'Gap-filling: Global',
            'Gap-filling: DEG',
            'Gap-filling: Other',
            'Blacklist Replacement',
            'Xenium Replacement',
            'Other Replacement',
            'Other'
        ]
        
        # Reorder categories
        ordered_categories = [cat for cat in preferred_order if cat in category_counts.index]
        category_counts = category_counts.reindex(ordered_categories)
        
        # Create color palette
        colors = {
            'User-defined Markers': '#E67E22',  # Orange
            'DT-unique': '#E74C3C',  # Red
            'RF-unique': '#875223',  # Brown (matches _constants rf_deg colour)
            f'{reduction_type}-unique': '#3498DB',  # Blue
            f'Both (DT & {reduction_type})': '#9B59B6',  # Purple
            f'Both (RF & {reduction_type})': '#9B59B6',  # Purple
            'Gap-filling: Cell-type': '#2ECC71',  # Green
            'Gap-filling: Global': '#27AE60',  # Dark green
            'Gap-filling: DEG': '#16A085',  # Teal
            'Gap-filling: Other': '#1ABC9C',  # Light teal
            'Blacklist Replacement': '#F39C12',  # Gold
            'Xenium Replacement': '#E8B04B',  # Light gold
            'Other Replacement': '#F8E27D',  # Pale yellow
            'Other': '#95A5A6'  # Gray
        }
        
        bar_colors = [colors.get(cat, '#95A5A6') for cat in category_counts.index]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(range(len(category_counts)), category_counts.values, color=bar_colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.85)
        
        # Add value labels on top of bars
        for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/len(selected_df)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize plot
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Number of Genes', fontsize=13, fontweight='bold')
        ax.set_xlabel('Gene Source', fontsize=13, fontweight='bold')
        
        # Include strategy name in title if provided
        title_suffix = f' ({extract_display_name_from_dataset(strategy_name)})' if strategy_name else ''
        ax.set_title(f'Distribution of Selected Genes by Source{title_suffix}\n(Total: {len(selected_df)} genes)', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Adjust layout
        plt.tight_layout()
        
        # Settings/strategy info stays in the output folder path, not the filename.
        plot_path = os.path.join(results_dir, 'gene_source_distribution.png')
        plt.savefig(plot_path, dpi=DEFAULT_PNG_DPI, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved gene source distribution plot to {plot_path}")
        logging.info(f"Category breakdown:\n{category_counts}")
        
        return plot_path
        
    except Exception as e:
        logging.error(f"Failed to create gene source distribution plot: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None


# =============================================================================
# Genes-per-cell-type coverage diagnostic
# =============================================================================

_GPC_UNATTRIBUTED = "(unattributed)"
_GPC_BAR_COLOR = "#4c72b0"
_GPC_HIGHLIGHT_COLOR = "#c0392b"
_GPC_UNATTRIBUTED_COLOR = "#999999"


def _gpc_is_real_celltype(value) -> bool:
    """True when ``value`` is a usable cell-type label (not NaN/empty/'global')."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return bool(s) and s.lower() != "global"


def _gpc_split_celltype_list(value, known_celltypes) -> list[str]:
    """Split a ``contributing_celltypes`` string into cell-type names.

    The string is ``", ".join(sorted(names))`` and individual cell-type names may
    themselves contain ``", "`` (e.g. "CD16-negative, CD56-bright natural killer
    cell, human"). Fragments are split on ``", "`` and then, when a
    ``known_celltypes`` vocabulary is available, adjacent fragments are greedily
    re-joined into the longest run that forms a known name. Fragments that are not
    part of any known multi-fragment name pass through unchanged, so an incomplete
    vocabulary degrades gracefully (at worst it over-splits a comma-containing
    name into two bars, never collapses the whole list into one).
    """
    if not _gpc_is_real_celltype(value):
        return []
    frags = [f.strip() for f in str(value).strip().split(", ") if f.strip()]
    if not frags:
        return []

    known = set(known_celltypes or ())
    if not known or all(f in known for f in frags):
        return frags

    out: list[str] = []
    i = 0
    while i < len(frags):
        matched = None
        for j in range(len(frags), i, -1):
            candidate = ", ".join(frags[i:j])
            if candidate in known:
                matched = (candidate, j)
                break
        if matched:
            out.append(matched[0])
            i = matched[1]
        else:
            out.append(frags[i])
            i += 1
    return out


def _gpc_source_bucket(gene_source: str) -> str:
    """Map a ``gene_source`` / ``selection_strategy`` string to a coverage bucket."""
    s = str(gene_source).strip().lower()
    if "gap_fill_celltype" in s or ("gap" in s and "celltype" in s):
        return "gap_fill_celltype"
    if "gap_fill_deg" in s or ("gap" in s and "deg" in s):
        return "gap_fill_deg"
    if "force_include" in s or "force-include" in s:
        return "force_include"
    if "rf" in s or s == "dt" or "→dt" in s:
        return "rf"
    if "dimred" in s or "nmf" in s or "pca" in s:
        return "dimred"
    if "deg" in s:
        return "deg"
    return "other"


def plot_genes_per_celltype(
    summary_df=None,
    ranked_list_csv=None,
    results_dir=None,
    *,
    strategy_name=None,
    celltype_vocabulary=None,
    highlight_celltypes=None,
    figsize=None,
    png_dpi=DEFAULT_PNG_DPI,
):
    """Bar chart of how many selected genes are informative for each cell type.

    One bar per cell type = the number of distinct panel genes attributed to it.
    The attribution comes from the ``informative_celltypes`` column that the
    selection module writes into ``ranked_gene_list.csv`` (RF genes via the
    per-class Gini decomposition, NMF/dimred + cell-type gap-fill via
    ``contributing_celltypes``, DEG via ``celltype``); if that column is absent
    (older CSVs) the per-source columns are merged as a fallback. A gene
    informative for *k* cell types adds one to each of those ``k`` bars, so the
    bar heights sum to at least the panel size. Genes with no usable attribution
    fall into a trailing ``(unattributed)`` bar.

    Works on both the combination ``ranked_gene_list.csv`` (``RecoVar`` /
    ``RecoVar_PCA``) and the single-strategy one (``dimred_only`` / ``rf_deg`` /
    ``rf_simple`` / ``deg_only`` / ...). Writes no CSV — the per-cell-type counts
    are ``df['informative_celltypes'].str.split('|').explode().value_counts()``.

    Parameters
    ----------
    summary_df : pandas.DataFrame, optional
        Already-loaded ranked gene list. Takes precedence over the path args.
    ranked_list_csv : str, optional
        Path to a ``ranked_gene_list.csv``.
    results_dir : str, optional
        Directory to read ``ranked_gene_list.csv`` from (if the two args above are
        not given) and to write the plot to.
    strategy_name : str, optional
        Label for the title and output filename.
    celltype_vocabulary : list of str, optional
        Known cell-type names, used by the fallback path to split comma-containing
        ``contributing_celltypes`` strings. Defaults to the cell types seen in the
        panel's ``celltype`` / ``rf_celltype`` columns.
    highlight_celltypes : list of str, optional
        Cell types to draw in a contrasting colour (with a small legend). Nothing
        is highlighted by default.
    figsize : tuple, optional
        Overridden automatically from the cell-type count when not given.
    png_dpi : int, optional
        Output resolution (default: module ``DEFAULT_PNG_DPI``).

    Returns
    -------
    str or None
        Path to the saved PNG, or ``None`` on failure / empty input.
    """
    try:
        # --- load ---------------------------------------------------------------
        df = summary_df
        if df is None:
            path = ranked_list_csv
            if path is None and results_dir:
                for cand in ("RecoVar_panel_information.csv",
                             os.path.join("selection", "RecoVar_panel_information.csv"),
                             "ranked_gene_list.csv",
                             os.path.join("selection", "ranked_gene_list.csv"),
                             "final_panel_with_provenance.csv"):
                    p = os.path.join(results_dir, cand)
                    if os.path.exists(p):
                        path = p
                        break
                if path is None:
                    # Single-strategy output (deg_only/hvg/random/rf_simple/rf_deg/nmf/pca) --
                    # filename is strategy-specific, e.g. rf_deg_panel_information.csv.
                    for base in (results_dir, os.path.join(results_dir, "selection")):
                        if os.path.isdir(base):
                            matches = sorted(
                                f for f in os.listdir(base) if f.endswith("_panel_information.csv")
                            )
                            if matches:
                                path = os.path.join(base, matches[0])
                                break
            if not path or not os.path.exists(path):
                logging.warning("plot_genes_per_celltype: no ranked gene list found")
                return None
            df = pd.read_csv(path)
        df = df.copy()

        if df.empty:
            logging.warning("plot_genes_per_celltype: empty gene list")
            return None

        cols = set(df.columns)
        if "gene" not in cols and "names" in cols:
            df = df.rename(columns={"names": "gene"})
            cols = set(df.columns)
        if "gene_source" not in cols and "source" in cols:
            df = df.rename(columns={"source": "gene_source"})
            cols = set(df.columns)

        # --- restrict to final-panel genes -----------------------------------
        is_single_strategy = ("in_panel" in cols) or ("selected_initial" in cols) \
            or ("final_selection" in cols)
        if is_single_strategy:
            mask = None
            for flag_col in ("in_panel", "final_selection"):
                if flag_col in cols:
                    raw = df[flag_col]
                    mask = raw.map(
                        lambda v: str(v).strip().lower() in ("true", "1", "1.0")
                    ) if raw.dtype == object else raw.fillna(False).astype(bool)
                    break
            if mask is not None:
                df = df[mask]
        if df.empty:
            logging.warning("plot_genes_per_celltype: no panel genes after filtering")
            return None

        if "gene_source" not in df.columns:
            df["gene_source"] = df.get("selection_strategy", "other")

        for opt_col in ("celltype", "contributing_celltypes",
                        "rf_celltype", "rf_contributing_celltypes"):
            if opt_col not in df.columns:
                df[opt_col] = ""

        # --- cell-type vocabulary (for comma-safe splitting) ----------------
        if celltype_vocabulary:
            known = set(map(str, celltype_vocabulary))
        else:
            known = set()
            for c in ("celltype", "rf_celltype"):
                known.update(
                    str(v).strip() for v in df[c].tolist() if _gpc_is_real_celltype(v)
                )

        # --- per gene -> cell types it is informative for -------------------
        # Preferred path: the unified ``informative_celltypes`` column written into
        # ranked_gene_list.csv by the selection module. Fallback: merge the
        # per-source attribution columns (older CSVs without that column).
        genes_by_ct: dict[str, set] = {}

        def _register_many(cts, gene) -> None:
            for ct in dict.fromkeys(cts) or [_GPC_UNATTRIBUTED]:
                genes_by_ct.setdefault(ct, set()).add(gene)

        if "informative_celltypes" in df.columns:
            for _, row in df.iterrows():
                raw = row["informative_celltypes"]
                cts = [] if not _gpc_is_real_celltype(raw) else [
                    t.strip() for t in str(raw).split("|") if t.strip()
                ]
                if not cts and _gpc_source_bucket(row["gene_source"]) == "force_include":
                    continue  # force-include with no celltype -> not a real bar
                _register_many(cts, row["gene"])
        else:
            for _, row in df.iterrows():
                gene = row["gene"]
                bucket = _gpc_source_bucket(row["gene_source"])
                if bucket in ("dimred", "gap_fill_celltype"):
                    cts = _gpc_split_celltype_list(row["contributing_celltypes"], known) \
                        or ([str(row["celltype"]).strip()] if _gpc_is_real_celltype(row["celltype"]) else [])
                elif bucket in ("rf", "gap_fill_deg"):
                    cts = [t for t in str(row["rf_contributing_celltypes"]).split("|") if t.strip()] \
                        or ([str(row["rf_celltype"]).strip()] if _gpc_is_real_celltype(row["rf_celltype"]) else [])
                elif bucket == "force_include":
                    if _gpc_is_real_celltype(row["celltype"]):
                        _register_many([str(row["celltype"]).strip()], gene)
                    elif _gpc_is_real_celltype(row["rf_celltype"]):
                        _register_many([str(row["rf_celltype"]).strip()], gene)
                    continue
                else:  # deg / other
                    if _gpc_is_real_celltype(row["celltype"]):
                        cts = [str(row["celltype"]).strip()]
                    elif _gpc_is_real_celltype(row["rf_celltype"]):
                        cts = [str(row["rf_celltype"]).strip()]
                    else:
                        cts = []
                _register_many(cts, gene)

        if not genes_by_ct:
            logging.warning("plot_genes_per_celltype: nothing to plot")
            return None

        # --- aggregate ------------------------------------------------------
        cov = pd.DataFrame(
            [{"celltype": ct, "n_genes": len(genes)} for ct, genes in genes_by_ct.items()]
        )
        # Sort by count desc, keep the unattributed bar last.
        cov["_last"] = (cov["celltype"] == _GPC_UNATTRIBUTED).astype(int)
        cov = cov.sort_values(["_last", "n_genes"], ascending=[True, False]).drop(columns="_last")
        cov = cov.reset_index(drop=True)

        out_dir = results_dir or "."
        os.makedirs(out_dir, exist_ok=True)

        # --- plot ---------------------------------------------------------------
        n_ct = len(cov)
        if figsize is None:
            figsize = (max(8.0, 0.5 * n_ct + 3.0), 6.8)
        highlight = set(map(str, highlight_celltypes or []))

        bar_colors = []
        for ct in cov["celltype"]:
            if ct == _GPC_UNATTRIBUTED:
                bar_colors.append(_GPC_UNATTRIBUTED_COLOR)
            elif ct in highlight:
                bar_colors.append(_GPC_HIGHLIGHT_COLOR)
            else:
                bar_colors.append(_GPC_BAR_COLOR)

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(n_ct)
        ax.bar(x, cov["n_genes"].values, color=bar_colors)
        top = max(cov["n_genes"].max(), 1)
        for i, v in enumerate(cov["n_genes"].values):
            ax.text(i, v + top * 0.012, str(int(v)), ha="center", va="bottom", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(cov["celltype"], rotation=45, ha="right", fontsize=12)
        ax.set_ylabel("number of panel genes", fontsize=14)
        ax.margins(y=0.15)
        title = "Genes informative for each cell type"
        if strategy_name:
            title += f"  -  {extract_display_name_from_dataset(strategy_name)}"
        ax.set_title(title, fontsize=16)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        if highlight:
            from matplotlib.patches import Patch
            ax.legend(
                handles=[Patch(fc=_GPC_HIGHLIGHT_COLOR, label="highlighted cell type"),
                         Patch(fc=_GPC_BAR_COLOR, label="other cell type")],
                fontsize=12, loc="upper right",
            )
        fig.tight_layout()

        # Settings/strategy info stays in the output folder path, not the filename.
        plot_path = os.path.join(out_dir, "genes_per_celltype.png")
        fig.savefig(plot_path, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved genes-per-cell-type plot to {plot_path}")
        return plot_path

    except Exception as e:
        logging.error(f"Failed to create genes-per-cell-type plot: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None


