"""Highly Variable Genes (HVG) and Random gene selection.

This module provides simple baseline selection strategies using highly variable genes
or random selection. Uses GeneListBuilder for unified output format.

Author: Refactored from _selection.py
Date: 2026-02-08
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# Use absolute imports (for script execution)
from _constants import COL_GENE, COL_RANK, COL_SELECTION_SCORE, DEFAULT_PROBESET_SIZE
from _gene_list_builder import GeneListBuilder

logger = logging.getLogger(__name__)


def select_highly_variable_genes(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Select highly variable genes (HVGs) based on target panel size.

    Identifies genes with high variability across cells using scanpy's default HVG method.

    Args:
        adata: Annotated data matrix (log-normalized recommended)
        probeset_size: Target number of genes to select
        results_dir: Directory to save results (optional)

    Returns:
        GeneListBuilder with:
            - selected_genes: Top HVGs
            - selection_score: Normalized dispersion/variance
            - rank: Rank by variability
            - metadata: HVG statistics (mean, dispersion, etc.)

    Examples:
        >>> builder = select_highly_variable_genes(adata, probeset_size=500)
        >>> df = builder.to_dataframe()
        >>> logger.info(f"Selected {len(builder.get_selected_genes())} HVGs")
    """
    logger.info("=== Selecting Highly Variable Genes (HVGs) ===")
    logger.info(f"Target probeset size: {probeset_size}")

    # Initialize GeneListBuilder
    builder = GeneListBuilder(
        strategy_name="hvg",
        analysis_type="global",
    )

    # Make a copy to avoid modifying original
    adata_hvg = adata.copy()

    # Check if HVGs already computed
    hvgs_already_computed = False
    if "highly_variable" in adata.var.columns:
        n_existing_hvgs = adata.var["highly_variable"].sum()
        if n_existing_hvgs >= probeset_size:
            hvgs_already_computed = True
            logger.info(f"Reusing {n_existing_hvgs} pre-computed HVGs")

    # Calculate HVGs if needed
    if not hvgs_already_computed:
        logger.info("Computing highly variable genes...")
        # Guard against infinity values that can arise from expm1 overflow on
        # very large log-normalised counts — replace with the finite max so
        # scanpy's binning step doesn't crash.
        import scipy.sparse, numpy as np
        if scipy.sparse.issparse(adata_hvg.X):
            adata_hvg.X = adata_hvg.X.toarray()
        inf_mask = ~np.isfinite(adata_hvg.X)
        if inf_mask.any():
            logger.warning(
                f"Found {inf_mask.sum()} non-finite values in expression matrix "
                "before HVG computation — replacing with finite max."
            )
            finite_max = adata_hvg.X[np.isfinite(adata_hvg.X)].max() if np.isfinite(adata_hvg.X).any() else 0.0
            adata_hvg.X[inf_mask] = finite_max
        sc.pp.highly_variable_genes(adata_hvg, n_top_genes=probeset_size)

    # Extract HVG information
    hvg_info = adata_hvg.var[adata_hvg.var["highly_variable"]].copy()

    # Determine score column for ranking
    if "dispersions_norm" in hvg_info.columns:
        score_col = "dispersions_norm"
    elif "variances_norm" in hvg_info.columns:
        score_col = "variances_norm"
    elif "variances" in hvg_info.columns:
        score_col = "variances"
    else:
        logger.warning("No standard variability score found, using uniform scores")
        score_col = None

    # Sort by score and take top genes
    if score_col is not None:
        hvg_info = hvg_info.sort_values(score_col, ascending=False)

    selected_genes = hvg_info.index[:probeset_size].tolist()
    logger.info(f"Selected {len(selected_genes)} HVGs")

    # Add all genes to builder (for comprehensive tracking)
    all_genes_df = adata_hvg.var.copy()
    if score_col is not None and score_col in all_genes_df.columns:
        all_genes_df = all_genes_df.sort_values(score_col, ascending=False)

        for rank, (gene, row) in enumerate(all_genes_df.iterrows(), start=1):
            metadata = {
                "means": row.get("means", np.nan),
                "dispersions": row.get("dispersions", np.nan),
                "dispersions_norm": row.get("dispersions_norm", np.nan),
                "highly_variable": row.get("highly_variable", False),
            }

            builder.add_gene(
                gene=str(gene),
                selection_score=row.get(score_col, 0.0),
                rank=rank,
                metadata=metadata,
            )

            # Mark as selected if in top N
            if gene in selected_genes:
                builder.mark_selected(gene)
    else:
        # No score column - add genes with uniform scores
        for rank, gene in enumerate(selected_genes, start=1):
            builder.add_gene(gene=gene, selection_score=1.0, rank=rank)
            builder.mark_selected(gene)

    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

        # Save selected genes
        builder.to_csv(os.path.join(results_dir, "selected_genes.csv"))

        # Save HVG statistics
        hvg_stats_file = os.path.join(results_dir, f"hvg_statistics_{probeset_size}genes.csv")
        hvg_info.to_csv(hvg_stats_file)
        logger.info(f"Saved HVG statistics to {hvg_stats_file}")

    logger.info(f"HVG selection complete: {len(selected_genes)} genes")
    return builder


def select_random_genes(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    random_state: int = 42,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Select random genes as a baseline control.

    Randomly selects genes to serve as a consistent baseline for benchmarking.
    Uses a fixed random seed for reproducibility.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes to select
        random_state: Random seed for reproducibility
        results_dir: Directory to save results (optional)

    Returns:
        GeneListBuilder with:
            - selected_genes: Randomly selected genes
            - selection_score: Uniform score (all equal for random)
            - rank: Random rank

    Raises:
        ValueError: If not enough genes available

    Examples:
        >>> builder = select_random_genes(adata, probeset_size=100, random_state=42)
        >>> genes = builder.get_selected_genes()
        >>> # Same genes every time with same random_state
    """
    logger.info("=== Selecting Random Genes ===")
    logger.info(f"Target probeset size: {probeset_size}")
    logger.info(f"Random seed: {random_state}")

    # Initialize GeneListBuilder
    builder = GeneListBuilder(
        strategy_name="random",
        analysis_type="global",
    )

    all_genes = adata.var_names.tolist()
    logger.info(f"Total genes in dataset: {len(all_genes)}")

    # Check if we have enough genes
    if len(all_genes) < probeset_size:
        raise ValueError(
            f"Not enough genes available ({len(all_genes)}) for target size ({probeset_size})"
        )

    # Set random seed
    np.random.seed(random_state)

    # Randomly sample genes
    selected_genes = np.random.choice(
        all_genes,
        size=probeset_size,
        replace=False,
    ).tolist()

    logger.info(f"Selected {len(selected_genes)} random genes")

    # Create shuffled list of all genes (for ranking)
    all_genes_shuffled = all_genes.copy()
    np.random.shuffle(all_genes_shuffled)

    # Add all genes to builder with random ranking
    for rank, gene in enumerate(all_genes_shuffled, start=1):
        # For random selection, all genes have equal "score"
        builder.add_gene(
            gene=gene,
            selection_score=1.0,  # Uniform score
            rank=rank,
            metadata={"random_state": random_state},
        )

        # Mark as selected if in chosen set
        if gene in selected_genes:
            builder.mark_selected(gene)

    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        builder.to_csv(os.path.join(results_dir, "selected_genes.csv"))

    logger.info(f"Random gene selection complete: {len(selected_genes)} genes")
    return builder


def select_random_genes_bootstrap(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    n_bootstrap: int = 10,
    random_state: int = 42,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Select random genes with bootstrap resampling for robustness analysis.

    Generates multiple random gene sets to assess variability. Genes are ranked
    by selection frequency across bootstrap iterations. This method is useful for
    showing that random selection is unstable compared to informed methods.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes per set
        n_bootstrap: Number of random gene sets to generate
        random_state: Base random seed (uses random_state + i for each bootstrap)
        results_dir: Directory to save results (optional)

    Returns:
        GeneListBuilder with:
            - selected_genes: Top N genes by bootstrap frequency
            - selection_score: Bootstrap selection frequency (0.0-1.0)
            - rank: Rank by frequency (ties broken randomly)
            - metadata: {'selection_count': int, 'n_bootstrap': int}

    Examples:
        >>> builder = select_random_genes_bootstrap(adata, n_bootstrap=100)
        >>> genes = builder.get_selected_genes()
        >>> df = builder.to_dataframe()
        >>> # Check bootstrap frequency
        >>> print(df[df['selected']]['selection_score'].describe())
    """
    logger.info("=== Random Gene Selection with Bootstrap ===")
    logger.info(f"Target probeset size: {probeset_size}")
    logger.info(f"Bootstrap iterations: {n_bootstrap}")

    # Initialize GeneListBuilder
    builder = GeneListBuilder(
        strategy_name="random_bootstrap",
        analysis_type="global",
    )

    all_genes = adata.var_names.tolist()

    if len(all_genes) < probeset_size:
        raise ValueError(f"Not enough genes ({len(all_genes)}) for size {probeset_size}")

    bootstrap_gene_sets = []

    for bootstrap_idx in range(n_bootstrap):
        seed = random_state + bootstrap_idx
        np.random.seed(seed)

        selected_genes = np.random.choice(
            all_genes,
            size=probeset_size,
            replace=False,
        ).tolist()

        bootstrap_gene_sets.append(selected_genes)
        logger.info(f"Bootstrap {bootstrap_idx + 1}/{n_bootstrap}: {len(selected_genes)} genes (seed={seed})")

    # Calculate gene selection frequency
    gene_counts = {}
    for gene_set in bootstrap_gene_sets:
        for gene in gene_set:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

    gene_freq_df = pd.DataFrame(
        [
            {
                "gene": gene,
                "selection_count": count,
                "selection_frequency": count / n_bootstrap,
            }
            for gene, count in gene_counts.items()
        ]
    ).sort_values("selection_count", ascending=False)

    # Log statistics
    logger.info(f"Total unique genes across bootstraps: {len(gene_freq_df)}")
    logger.info(f"Genes selected once: {len(gene_freq_df[gene_freq_df['selection_count'] == 1])}")
    logger.info(f"Max selection count: {gene_freq_df['selection_count'].max()}")

    # Add all genes to builder, ranked by frequency
    for rank, (_, row) in enumerate(gene_freq_df.iterrows(), start=1):
        gene = row["gene"]
        selection_score = row["selection_frequency"]
        selection_count = int(row["selection_count"])

        builder.add_gene(
            gene=gene,
            selection_score=selection_score,
            rank=rank,
            metadata={
                "selection_count": selection_count,
                "n_bootstrap": n_bootstrap,
                "random_state": random_state,
            },
        )

        # Mark top N genes as selected
        if rank <= probeset_size:
            builder.mark_selected(gene)

    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

        # Save selected genes
        builder.to_csv(os.path.join(results_dir, "selected_genes.csv"))

        # Also save detailed frequency info
        freq_file = os.path.join(
            results_dir,
            f"gene_selection_frequency_{probeset_size}genes_{n_bootstrap}bootstraps.csv",
        )
        gene_freq_df.to_csv(freq_file, index=False)
        logger.info(f"Saved gene selection frequency to {freq_file}")

    logger.info(f"Bootstrap selection complete: {len(builder.get_selected_genes())} genes")
    return builder
