"""Differential expression gene (DEG) selection.

This module provides functions for selecting genes based on differential expression
analysis across cell types or clusters. Uses GeneListBuilder for unified output format.

Author: Refactored from _selection.py
Date: 2026-02-08
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# Use absolute imports (for script execution)
from _constants import (
    COL_CELLTYPE,
    COL_GENE,
    COL_RANK,
    COL_SELECTION_SCORE,
    DEFAULT_DEG_MAX_PVAL,
    DEFAULT_DEG_MIN_FOLD_CHANGE,
    DEFAULT_DEG_METHOD,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
)
from _gene_list_builder import GeneListBuilder

logger = logging.getLogger(__name__)


def filter_celltypes_by_min_cells(
    adata: AnnData,
    celltype_column: str = "celltype",
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Filter cell types based on minimum number of cells.

    Identifies cell types with sufficient cells for robust gene selection.

    Args:
        adata: Annotated data matrix
        celltype_column: Column name in adata.obs containing cell type labels
        min_cells_per_celltype: Minimum number of cells required per cell type

    Returns:
        Tuple of (valid_celltypes, excluded_celltypes, celltype_counts):
            - valid_celltypes: List of cell types with >= min cells
            - excluded_celltypes: List of cell types with < min cells
            - celltype_counts: Dict mapping cell type -> cell count

    Examples:
        >>> valid, excluded, counts = filter_celltypes_by_min_cells(adata, min_cells_per_celltype=10)
        >>> logger.info(f"Valid: {len(valid)}, Excluded: {len(excluded)}")
    """
    # Count cells per cell type
    celltype_counts = adata.obs[celltype_column].value_counts().to_dict()

    valid_celltypes = []
    excluded_celltypes = []

    for celltype, count in celltype_counts.items():
        if count >= min_cells_per_celltype:
            valid_celltypes.append(celltype)
        else:
            excluded_celltypes.append(celltype)
            logger.warning(
                f"Excluding cell type '{celltype}': only {count} cells "
                f"(min required: {min_cells_per_celltype})"
            )

    logger.info(
        f"Cell type filtering: {len(valid_celltypes)} valid, "
        f"{len(excluded_celltypes)} excluded (min cells: {min_cells_per_celltype})"
    )

    return valid_celltypes, excluded_celltypes, celltype_counts


def select_DEGs(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    celltype_column: str = COL_CELLTYPE,
    method: str = DEFAULT_DEG_METHOD,
    n_genes_per_group: Optional[int] = None,
    max_pval: float = DEFAULT_DEG_MAX_PVAL,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Select genes based on differential expression analysis.

    Performs differential gene expression (DEG) analysis for each category in the
    specified column (e.g., cell types or clusters). Ranks ALL genes by DEG score
    (log2FC * -log10(adj_pval)), excludes only genes with pval_adj > max_pval,
    and marks top N genes as selected.

    Note: min_fold_change is NOT used for exclusion, only for ranking. This ensures
    enough genes remain for downstream ODT/Xenium filtering.

    Args:
        adata: Processed AnnData object (log-normalized)
        probeset_size: Target number of genes in final panel
        celltype_column: Column name in adata.obs for grouping (fallback to 'leiden')
        method: Statistical test for DEG analysis ('wilcoxon', 't-test', etc.)
        n_genes_per_group: Genes per group (if None, calculated as probeset_size / n_groups)
        max_pval: Maximum adjusted p-value threshold for EXCLUSION ONLY
        min_cells_per_celltype: Minimum cells per cell type for analysis
        results_dir: Directory to save DEG results (optional)

    Returns:
        GeneListBuilder with:
            - selected_genes: DEG markers
            - selection_score: DEG score (log2fc * -log10(adj_pval))
            - rank: Per-group rank
            - celltype: Source cell type
            - filter_reason: Why genes were excluded

    Raises:
        ValueError: If no valid cell types remain after filtering

    Examples:
        >>> builder = select_DEGs(adata, probeset_size=500, celltype_column='celltype')
        >>> df = builder.to_dataframe()
        >>> logger.info(f"Selected {len(builder.get_selected_genes())} DEGs")
    """
    logger.info("=== Starting DEG-based gene selection ===")
    logger.info(f"Target probeset size: {probeset_size}")

    # Initialize GeneListBuilder
    builder = GeneListBuilder(
        strategy_name="deg_only",
        analysis_type="per_celltype"
    )

    # Validate and determine grouping column
    groupby = _determine_groupby_column(adata, celltype_column)

    # Ensure categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        logger.info(f"Converting {groupby} to categorical")
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    # Filter cell types by minimum cell count
    if min_cells_per_celltype > 0:
        valid_celltypes, excluded_celltypes, celltype_counts = filter_celltypes_by_min_cells(
            adata, groupby, min_cells_per_celltype
        )

        if len(excluded_celltypes) > 0:
            logger.info(f"Filtering dataset to include only {len(valid_celltypes)} valid cell types")
            adata = adata[adata.obs[groupby].isin(valid_celltypes)].copy()
            adata.obs[groupby] = adata.obs[groupby].cat.remove_unused_categories()
            logger.info(f"Dataset filtered: {adata.shape[0]} cells remaining")

    groups = list(adata.obs[groupby].cat.categories)
    n_groups = len(groups)
    logger.info(f"Found {n_groups} groups: {', '.join(map(str, groups))}")

    if n_groups == 0:
        raise ValueError("No groups found for DEG analysis after filtering")

    # Calculate genes per group
    if n_genes_per_group is None:
        n_genes_per_group = max(1, probeset_size // n_groups)
    logger.info(f"Will select {n_genes_per_group} genes per group")

    # Run differential expression analysis
    logger.info(f"Running {method} test for differential expression...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        use_raw=False,
        n_genes=None,
        key_added="rank_genes_groups",
    )
    logger.info("DEG analysis complete")

    # Extract DEG results
    deg_df = _extract_deg_results(adata, groups)
    logger.info(f"Extracted {len(deg_df)} DEG results across all groups")

    # Use scanpy's 'scores' directly for ranking (for Wilcoxon: Z-score, for t-test: t-statistic)
    # No need for redundant calculation - scanpy already provides proper ranking metric
    logger.info(f"Using scanpy '{method}' scores for ranking (higher = more significant)")

    # Filter DEGs by p-value ONLY (exclude genes with high p-value)
    logger.info(f"Excluding genes with adj. p-value > {max_pval}")
    deg_df_filtered = deg_df[deg_df["pvals_adj"] <= max_pval].copy()
    logger.info(f"After p-value filter: {len(deg_df_filtered)} genes remain (ranked by scanpy scores)")
    logger.info(f"Excluded: {len(deg_df) - len(deg_df_filtered)} genes with pval_adj > {max_pval}")

    # Add ALL genes passing pval filter to builder (for comprehensive tracking and ODT replacement)
    for _, row in deg_df_filtered.iterrows():
        builder.add_gene(
            gene=row["gene"],
            selection_score=row["scores"],  # Use scanpy's score directly
            rank=None,  # Will be set per-group
            celltype=row["group"],
            metadata={
                "logfoldchanges": row["logfoldchanges"],
                "pvals": row["pvals"],
                "pvals_adj": row["pvals_adj"],
            },
        )

    # Select top genes per group
    genes_per_group_actual = {}
    selected_count = 0

    for group in groups:
        group_degs = deg_df_filtered[deg_df_filtered["group"] == group].copy()
        group_degs = group_degs.sort_values("scores", ascending=False)

        # Mark top genes as selected
        top_genes = []
        for idx, (_, row) in enumerate(group_degs.iterrows()):
            gene = row["gene"]
            if gene not in builder.get_selected_genes() and len(top_genes) < n_genes_per_group:
                builder.mark_selected(gene, rank=idx + 1)
                top_genes.append(gene)
                selected_count += 1

        genes_per_group_actual[group] = len(top_genes)
        logger.info(f"Group '{group}': selected {len(top_genes)} genes")

    logger.info(f"Total unique genes selected: {selected_count}")

    # Handle gap-filling if needed
    if selected_count < probeset_size:
        gap = probeset_size - selected_count
        logger.warning(f"Only found {selected_count} DEGs meeting criteria (target: {probeset_size})")
        logger.info(f"Attempting to fill gap of {gap} genes by relaxing criteria...")

        selected_count = _fill_deg_gap(
            builder,
            deg_df_filtered,
            deg_df,
            groups,
            probeset_size,
            selected_count,
        )

    # Save results if directory provided
    if results_dir:
        _save_deg_results(
            results_dir,
            builder,
            deg_df,
            deg_df_filtered,
            groupby,
            method,
            n_groups,
            n_genes_per_group,
            probeset_size,
            max_pval,
            genes_per_group_actual,
        )

    logger.info(f"DEG selection complete: {len(builder.get_selected_genes())} genes selected from {n_groups} groups")
    return builder


def _determine_groupby_column(adata: AnnData, celltype_column: str) -> str:
    """Determine the grouping column to use for DEG analysis.

    Args:
        adata: Annotated data matrix
        celltype_column: Preferred column name

    Returns:
        Column name to use for grouping

    Raises:
        ValueError: If no suitable grouping column found
    """
    fallback_columns = ["leiden", "louvain", "clusters", "cluster"]

    if celltype_column in adata.obs.columns:
        logger.info(f"Using specified column: {celltype_column}")
        return celltype_column

    logger.warning(f"Column '{celltype_column}' not found in adata.obs")
    for col in fallback_columns:
        if col in adata.obs.columns:
            logger.info(f"Using fallback column: {col}")
            return col

    raise ValueError(
        f"No suitable grouping column found. Available columns: {', '.join(adata.obs.columns)}"
    )


def _extract_deg_results(adata: AnnData, groups: list[str]) -> pd.DataFrame:
    """Extract DEG results from scanpy's rank_genes_groups output.

    Args:
        adata: AnnData with rank_genes_groups results
        groups: List of group names

    Returns:
        DataFrame with columns: group, gene, logfoldchanges, pvals, pvals_adj, scores
    """
    all_results = []
    
    for group in groups:
        try:
            # Use scanpy's built-in function
            df = sc.get.rank_genes_groups_df(adata, group=group, key='rank_genes_groups')
            df['group'] = group  # Add group column
            all_results.append(df)
        except Exception as e:
            logger.error(f"Error extracting DEG results for group {group}: {e}")
            continue
    
    # Combine all groups
    deg_df = pd.concat(all_results, ignore_index=True)
    
    # Rename columns to match your original format
    deg_df = deg_df.rename(columns={'names': 'gene'})
    
    return deg_df[['group', 'gene', 'logfoldchanges', 'pvals', 'pvals_adj', 'scores']]


def _fill_deg_gap(
    builder: GeneListBuilder,
    deg_df_filtered: pd.DataFrame,
    deg_df: pd.DataFrame,
    groups: list[str],
    probeset_size: int,
    current_count: int,
) -> int:
    """Fill remaining slots by relaxing filtering criteria.

    Args:
        builder: GeneListBuilder to update
        deg_df_filtered: Filtered DEG results
        deg_df: Unfiltered DEG results
        groups: List of groups
        probeset_size: Target size
        current_count: Current number of selected genes

    Returns:
        Updated count of selected genes
    """
    # Try filtered results first (next best genes)
    for group in groups:
        if current_count >= probeset_size:
            break

        group_degs = deg_df_filtered[deg_df_filtered["group"] == group].copy()
        group_degs = group_degs.sort_values("scores", ascending=False)

        for _, row in group_degs.iterrows():
            gene = row["gene"]
            if gene not in builder.get_selected_genes() and current_count < probeset_size:
                builder.mark_selected(gene)
                current_count += 1

    # If still not enough, use unfiltered results
    if current_count < probeset_size:
        gap = probeset_size - current_count
        logger.warning(f"Still need {gap} more genes. Using unfiltered DEG results...")

        for group in groups:
            if current_count >= probeset_size:
                break

            group_degs = deg_df[deg_df["group"] == group].copy()
            group_degs = group_degs.sort_values("scores", ascending=False)

            for _, row in group_degs.iterrows():
                gene = row["gene"]
                if gene not in builder.get_selected_genes() and current_count < probeset_size:
                    # Add gene if not already tracked
                    if gene not in builder.get_all_genes():
                        builder.add_gene(
                            gene=gene,
                            selection_score=row["deg_score"],
                            celltype=row["group"],
                        )
                    builder.mark_selected(gene)
                    current_count += 1

    return current_count


def _save_deg_results(
    results_dir: str,
    builder: GeneListBuilder,
    deg_df: pd.DataFrame,
    deg_df_filtered: pd.DataFrame,
    groupby: str,
    method: str,
    n_groups: int,
    n_genes_per_group: int,
    probeset_size: int,
    max_pval: float,
    genes_per_group_actual: dict[str, int],
) -> None:
    """Save DEG selection results to disk.

    Args:
        results_dir: Directory to save results
        builder: GeneListBuilder with results
        deg_df: Full DEG results
        deg_df_filtered: Filtered DEG results (pval filter only)
        groupby: Grouping column used
        method: DEG method used
        n_groups: Number of groups analyzed
        n_genes_per_group: Target genes per group
        probeset_size: Target probeset size
        max_pval: Max p-value threshold
        genes_per_group_actual: Actual genes selected per group
    """
    import os

    os.makedirs(results_dir, exist_ok=True)

    # Save full DEG results
    deg_file = os.path.join(results_dir, "deg_results_all.csv")
    deg_df.to_csv(deg_file, index=False)
    logger.info(f"Saved all DEG results to: {deg_file}")

    # Save filtered DEG results
    deg_filtered_file = os.path.join(results_dir, "deg_results_filtered.csv")
    deg_df_filtered.to_csv(deg_filtered_file, index=False)
    logger.info(f"Saved filtered DEG results to: {deg_filtered_file}")

    # Save selected genes
    builder.to_csv(os.path.join(results_dir, "selected_genes.csv"))

    # Save summary
    summary_file = os.path.join(results_dir, "deg_selection_summary.txt")
    with open(summary_file, "w") as f:
        f.write("DEG-based Gene Selection Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Grouping column: {groupby}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Number of groups: {n_groups}\n")
        f.write(f"Target probeset size: {probeset_size}\n")
        f.write(f"Actual probeset size: {len(builder.get_selected_genes())}\n")
        f.write(f"Target genes per group: {n_genes_per_group}\n\n")
        f.write(f"Filter criteria:\n")
        f.write(f"  - Max adj. p-value: {max_pval} (EXCLUSION only)\n")
        f.write(f"  - Note: Genes ranked by log2FC, not excluded by fold change\n\n")
        f.write(f"  - Max adj. p-value: {max_pval}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Total DEGs: {len(deg_df)}\n")
        f.write(f"  - Filtered DEGs: {len(deg_df_filtered)}\n\n")
        f.write("Genes per group:\n")
        for group, count in genes_per_group_actual.items():
            f.write(f"  - {group}: {count} genes\n")

    logger.info(f"Saved summary to: {summary_file}")
