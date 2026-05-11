"""
Factor-aware duplicate resolution and gap filling for dimensionality reduction.

This module implements factor-aware strategies for maintaining balanced
representation across components (factors) in NMF/PCA-based gene selection.
Key features:
- Resolve duplicates by keeping genes in highest-weight factor
- Fill gaps while maintaining factor balance
- Support both global and per-celltype analysis
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Use absolute imports (for script execution)
from _constants import (
    DEFAULT_DIMRED_GENES_PER_COMPONENT,
    MIN_FACTOR_CONTRIBUTION,
    FACTOR_BALANCE_TOLERANCE,
    COL_COMPONENT,
    COL_RANK,
    COL_GENE,
)

logger = logging.getLogger(__name__)


def resolve_duplicates_factor_aware_global(
    gene_weights: Dict[str, Dict[str, float]],
    genes_per_component: int,
    loadings_df: pd.DataFrame,
    component_cols: List[str],
    context: str = 'final'
) -> Dict[str, any]:
    """
    Resolve duplicate genes by keeping them in factor with highest weight.
    
    When a gene appears in multiple factors, this function assigns it to the
    factor where it has the highest loading/weight, and selects replacement
    genes from the other factors to maintain balanced representation.
    
    Args:
        gene_weights: Mapping from gene to component weights.
            Format: {gene: {component: weight}}
        genes_per_component: Target number of genes per component.
        loadings_df: Full loadings dataframe (genes × components).
            Must have columns for each component in component_cols.
        component_cols: List of component column names (e.g., ['NMF1', 'NMF2']).
        
    Returns:
        Dictionary containing:
            - 'selected_genes': List of final selected gene names
            - 'factor_assignments': Dict mapping gene to assigned component
            - 'duplicates_resolved': Number of duplicate genes resolved
            - 'replacements_made': Number of replacement genes added
            - 'factor_to_genes': Dict mapping component to list of genes
            
    Raises:
        ValueError: If genes_per_component <= 0.
        ValueError: If component_cols is empty.
        ValueError: If loadings_df is empty.
        KeyError: If component_cols not in loadings_df columns.
        
    Examples:
        >>> gene_weights = {
        ...     'GeneA': {'NMF1': 8.5, 'NMF2': 3.2},
        ...     'GeneB': {'NMF1': 7.1},
        ...     'GeneC': {'NMF2': 6.8}
        ... }
        >>> result = resolve_duplicates_factor_aware_global(
        ...     gene_weights, 10, loadings_df, ['NMF1', 'NMF2']
        ... )
        >>> 'GeneA' in result['factor_to_genes']['NMF1']
        True
    """
    # Validate inputs
    if genes_per_component <= 0:
        raise ValueError(
            f"genes_per_component must be positive, got {genes_per_component}"
        )
    
    if not component_cols:
        raise ValueError(
            "component_cols cannot be empty. "
            "At least one component required for factor-aware resolution."
        )
    
    if loadings_df.empty:
        raise ValueError(
            "loadings_df is empty. Cannot perform factor-aware resolution "
            "without gene loadings data."
        )
    
    # Check component columns exist
    missing_cols = set(component_cols) - set(loadings_df.columns)
    if missing_cols:
        raise KeyError(
            f"Component columns {missing_cols} not found in loadings_df. "
            f"Available columns: {list(loadings_df.columns)}"
        )
    
    logger.debug(
        f"Processing {len(gene_weights)} genes across "
        f"{len(component_cols)} factors"
    )
    
    # Step 1: Identify duplicates (genes in multiple factors)
    duplicates = _find_duplicate_genes(gene_weights)
    logger.info(f"Found {len(duplicates)} duplicate genes across factors")
    
    # Step 2: Assign each duplicate to highest-weight factor
    gene_to_primary_factor = _assign_duplicates_to_primary_factor(
        duplicates, gene_weights
    )
    
    # Step 3: Build initial factor assignments
    factor_to_genes = _build_initial_factor_assignments(
        gene_weights, gene_to_primary_factor, component_cols
    )
    
    # Step 4: Identify which factors need replacement genes
    factors_needing_replacement = _identify_factors_needing_replacement(
        factor_to_genes, genes_per_component
    )
    
    logger.info(
        f"Factors needing replacement: "
        f"{list(factors_needing_replacement.keys())}"
    )
    
    # Step 5: Fill gaps with replacement genes from loadings
    replacements_made = 0
    for factor, gap in factors_needing_replacement.items():
        replacements = _get_replacement_genes_for_factor(
            factor=factor,
            gap_size=gap,
            loadings_df=loadings_df,
            already_selected=set(
                gene for genes in factor_to_genes.values() for gene in genes
            )
        )
        factor_to_genes[factor].extend(replacements)
        replacements_made += len(replacements)
        
        logger.debug(
            f"Added {len(replacements)} replacement genes to {factor}"
        )
    
    # Step 6: Validate factor balance (only warns for 'final' context; pool context suppresses deviation warnings)
    _validate_factor_balance(factor_to_genes, genes_per_component, context=context)
    
    # Compile results
    selected_genes = [
        gene for genes in factor_to_genes.values() for gene in genes
    ]
    
    factor_assignments = {
        gene: factor
        for factor, genes in factor_to_genes.items()
        for gene in genes
    }
    
    result = {
        'selected_genes': selected_genes,
        'factor_assignments': factor_assignments,
        'duplicates_resolved': len(duplicates),
        'replacements_made': replacements_made,
        'factor_to_genes': factor_to_genes,
    }
    
    # Log factor distribution
    factor_dist_str = ', '.join(f'{f}: {len(factor_to_genes[f])}' for f in component_cols)
    logger.info(
        f"✓ Resolution complete: {len(selected_genes)} genes, "
        f"{len(duplicates)} duplicates resolved, "
        f"{replacements_made} replacements made"
    )
    logger.debug(f"  Factor distribution: [{factor_dist_str}]")
    
    return result


def resolve_duplicates_factor_aware_per_celltype(
    celltype_genes_per_factor: Dict[str, Dict],
    celltype_factor_explained_variance: Optional[Dict[str, pd.Series]] = None
) -> Dict[str, any]:
    """
    Resolve duplicate genes with celltype-factor awareness.
    
    This function accepts the celltype_genes_per_factor dict directly from
    _select_genes_per_celltype and handles both within-celltype and cross-
    celltype duplicates.
    
    Args:
        celltype_genes_per_factor: Dict with structure:
            {celltype: {
                "genes_per_factor": {factor: [gene1, gene2, ...]},
                "loadings_df": pd.DataFrame (genes × factors),
                "component_cols": [factor1, factor2, ...]
            }}
        celltype_factor_explained_variance: Optional R² per factor per celltype.
            Format: {celltype: pd.Series([r2_factor1, r2_factor2, ...])}
            If provided, uses R²-weighted assignment for cross-celltype duplicates.
            If None, falls back to normalized loading comparison.
        
    Returns:
        Dictionary containing:
            - 'selected_genes': List of final selected gene names
            - 'factor_assignments': Dict mapping gene to component
            - 'celltype_assignments': Dict mapping gene to celltype
            - 'duplicates_resolved': Total duplicates resolved
            - 'replacements_made': Number of replacement genes added
            
    Examples:
        >>> celltype_data = {
        ...     'T_cells': {
        ...         'genes_per_factor': {'NMF1': ['GeneA', 'GeneB']},
        ...         'loadings_df': loadings_t,
        ...         'component_cols': ['NMF1', 'NMF2']
        ...     }
        ... }
        >>> result = resolve_duplicates_factor_aware_per_celltype(
        ...     celltype_data, 100
        ... )
    """
    # Extract loadings and component cols
    loadings_per_celltype = {
        ct: data["loadings_df"] for ct, data in celltype_genes_per_factor.items()
    }
    celltype_component_cols = {
        ct: data["component_cols"] for ct, data in celltype_genes_per_factor.items()
    }
    
    logger.info("Starting factor-aware duplicate resolution (per-celltype)")
    logger.debug(
        f"Processing {len(celltype_genes_per_factor)} celltypes"
    )
    
    celltype_factor_to_genes = {}
    within_ct_duplicates = 0
    cross_ct_duplicates = 0
    total_replacements = 0
    
    # Step 1: Resolve within-celltype duplicates for each celltype
    for celltype in celltype_genes_per_factor.keys():
        logger.debug(f"Processing celltype: {celltype}")
        
        genes_per_factor = celltype_genes_per_factor[celltype]["genes_per_factor"]
        loadings_df = loadings_per_celltype[celltype]
        abs_loadings_df = loadings_df.abs()
        component_cols = celltype_component_cols[celltype]
        
        # Build gene_weights format for this celltype
        gene_weights = {}
        for factor, genes in genes_per_factor.items():
            for gene in genes:
                if gene not in gene_weights:
                    gene_weights[gene] = {}
                gene_weights[gene][factor] = float(abs_loadings_df.at[gene, factor])
        
        # Resolve duplicates within this celltype
        # Calculate target per factor from total pool genes (not last loop variable)
        total_pool_genes = sum(len(g_list) for g_list in genes_per_factor.values())
        genes_per_component = max(1, total_pool_genes // len(component_cols)) if genes_per_factor else 1
        ct_result = resolve_duplicates_factor_aware_global(
            gene_weights=gene_weights,
            genes_per_component=genes_per_component,
            loadings_df=loadings_df,
            component_cols=component_cols,
            context='pool'  # Suppress deviation warnings for pool creation
        )
        
        # Log per-celltype duplicate summary
        factor_dist_str = ', '.join(
            f'{f.split("_")[-1]}: {len(ct_result["factor_to_genes"][f])}'
            for f in component_cols
        )
        logger.info(
            f"  [{celltype}] {ct_result['duplicates_resolved']} within-celltype duplicates resolved, "
            f"pool: {len(ct_result['selected_genes'])} genes ({factor_dist_str})"
        )
        
        celltype_factor_to_genes[celltype] = ct_result['factor_to_genes']
        within_ct_duplicates += ct_result['duplicates_resolved']
        total_replacements += ct_result['replacements_made']
    
    logger.info(
        f"Within-celltype resolution: {within_ct_duplicates} duplicates, "
        f"{total_replacements} replacements"
    )
    
    # Step 2: Resolve cross-celltype duplicates with iterative replacement
    all_selected_genes = [
        gene
        for ct_factors in celltype_factor_to_genes.values()
        for genes in ct_factors.values()
        for gene in genes
    ]
    
    cross_ct_dups = _find_cross_celltype_duplicates(all_selected_genes)
    cross_ct_duplicates_initial = len(cross_ct_dups)
    
    if cross_ct_duplicates_initial > 0:
        logger.info(
            f"Found {cross_ct_duplicates_initial} cross-celltype duplicates "
            f"(genes selected by multiple celltypes)"
        )
        logger.info("Resolving cross-celltype duplicates with variance-weighted assignment...")
        
        # Resolve with iterative replacement
        celltype_factor_to_genes, cross_ct_replacements = _resolve_cross_celltype_duplicates(
            celltype_factor_to_genes=celltype_factor_to_genes,
            loadings_per_celltype=loadings_per_celltype,
            celltype_factor_explained_variance=celltype_factor_explained_variance
        )
        
        total_replacements += cross_ct_replacements
        cross_ct_duplicates = cross_ct_duplicates_initial  # Track for output
    else:
        logger.info("No cross-celltype duplicates found")
    
    # Step 3: Build simple assignments for caller
    selected_genes = []
    factor_assignments = {}
    celltype_assignments = {}
    
    for celltype, factor_to_genes in celltype_factor_to_genes.items():
        for factor, genes in factor_to_genes.items():
            for gene in genes:
                if gene not in selected_genes:
                    selected_genes.append(gene)
                factor_assignments[gene] = factor
                celltype_assignments[gene] = celltype
    
    result = {
        'selected_genes': selected_genes,
        'factor_assignments': factor_assignments,
        'celltype_assignments': celltype_assignments,
        'duplicates_resolved': within_ct_duplicates + cross_ct_duplicates,
        'replacements_made': total_replacements,
    }
    
    logger.info(
        f"✓ Per-celltype resolution complete: {len(selected_genes)} genes, "
        f"{within_ct_duplicates} within-celltype duplicates, "
        f"{cross_ct_duplicates} cross-celltype duplicates"
    )
    
    return result


def fill_gap_factor_aware_global(
    existing_genes: Set[str],
    gap_needed: int,
    dimred_df: pd.DataFrame,
    n_components: int
) -> List[str]:
    """
    Perform factor-aware gap filling for global selection.
    
    Distributes gap equally across factors and selects next-best candidates
    from each factor to maintain balanced factor representation.
    
    Args:
        existing_genes: Set of already selected genes.
        gap_needed: Number of genes needed to fill gap.
        dimred_df: Loadings dataframe (genes × components).
        n_components: Number of components in dimensionality reduction.
        
    Returns:
        List of candidate genes ranked by factor, ready for filtering.
        
    Raises:
        ValueError: If gap_needed is negative.
        KeyError: If factor_to_genes is missing expected components.
        
    Examples:
        >>> existing = {'Gene1', 'Gene2', 'Gene3'}
        >>> candidates = fill_gap_factor_aware_global(
        ...     existing, 10, loadings_df, factor_map, 5
        ... )
        >>> len(candidates)
        10
        >>> all(g not in existing for g in candidates)
        True
    """
    if gap_needed < 0:
        raise ValueError(f"gap_needed must be non-negative, got {gap_needed}")
    
    if gap_needed == 0:
        return []
    
    logger.info(f"Filling gap of {gap_needed} genes (factor-aware)")
    
    # Calculate genes per factor for gap
    genes_per_factor = gap_needed // n_components
    remainder = gap_needed % n_components
    
    logger.debug(
        f"Distributing gap: {genes_per_factor} per factor, "
        f"{remainder} remainder"
    )
    
    # Get component columns from dimred_df
    # Use first n_components columns (already filtered by caller)
    component_cols = dimred_df.columns[:n_components].tolist()
    
    gap_fill_genes = []
    
    for idx, component in enumerate(component_cols[:n_components]):
        # Calculate how many genes needed from this component
        n_needed = genes_per_factor
        if idx < remainder:
            n_needed += 1  # Distribute remainder to first factors
        
        if n_needed == 0:
            continue
        
        # Get next-best genes from this component
        component_genes = _get_next_best_genes_for_component(
            component=component,
            n_genes=n_needed,
            loadings_df=dimred_df,
            exclude_genes=existing_genes.union(set(gap_fill_genes))
        )
        
        gap_fill_genes.extend(component_genes)
        
        logger.debug(
            f"Factor {component}: added {len(component_genes)}/{n_needed} genes"
        )
    
    logger.info(f"✓ Gap filled: {len(gap_fill_genes)} genes selected")
    
    return gap_fill_genes


def fill_gap_factor_aware_per_celltype(
    existing_genes_per_ct: Dict[str, Set[str]],
    gap_needed_per_ct: Dict[str, int],
    loadings_per_celltype: Dict[str, pd.DataFrame],
    factor_to_genes_per_ct: Dict[str, Dict[str, List[str]]],
    n_components: int
) -> Dict[str, List[str]]:
    """
    Perform factor-aware gap filling for per-celltype selection.
    
    Extends gap filling to per-celltype analysis, maintaining factor balance
    within each celltype independently.
    
    Args:
        existing_genes_per_ct: Already selected genes per celltype.
        gap_needed_per_ct: Gap size for each celltype.
        loadings_per_celltype: Loadings dataframe for each celltype.
        factor_to_genes_per_ct: Factor assignments per celltype.
        n_components: Number of components in dimensionality reduction.
        
    Returns:
        Dict mapping celltype to list of gap-fill genes.
        
    Raises:
        ValueError: If any gap_needed value is negative.
        
    Examples:
        >>> existing = {'T_cells': {'Gene1', 'Gene2'}, 'B_cells': {'Gene3'}}
        >>> gaps = {'T_cells': 5, 'B_cells': 8}
        >>> result = fill_gap_factor_aware_per_celltype(
        ...     existing, gaps, loadings, factors, 5
        ... )
        >>> len(result['T_cells'])
        5
    """
    if any(gap < 0 for gap in gap_needed_per_ct.values()):
        raise ValueError("gap_needed values must be non-negative")
    
    logger.info("Filling gaps (per-celltype, factor-aware)")
    
    gap_fill_genes_per_ct = {}
    
    for celltype, gap in gap_needed_per_ct.items():
        if gap == 0:
            gap_fill_genes_per_ct[celltype] = []
            continue
        
        logger.debug(f"Filling {gap} genes for celltype: {celltype}")
        
        # Use global gap filling logic for this celltype
        gap_genes = fill_gap_factor_aware_global(
            existing_genes=existing_genes_per_ct.get(celltype, set()),
            gap_needed=gap,
            dimred_df=loadings_per_celltype[celltype],
            n_components=n_components
        )
        
        gap_fill_genes_per_ct[celltype] = gap_genes
    
    total_filled = sum(len(genes) for genes in gap_fill_genes_per_ct.values())
    logger.info(f"✓ Per-celltype gap filling complete: {total_filled} genes total")
    
    return gap_fill_genes_per_ct


def build_factor_replacement_pools(
    gene_list_df: pd.DataFrame,
    group_by: str = COL_COMPONENT
) -> Dict[any, List[str]]:
    """
    Build replacement pools organized by factor/component.
    
    Creates pools of candidate genes for each factor, sorted by rank,
    for use in factor-aware replacement during filtering.
    
    Args:
        gene_list_df: Gene list with component assignments.
            Must have 'component' and 'rank' columns.
        group_by: Column to group by ('component' or ('celltype', 'component')).
        
    Returns:
        Dict mapping component (or tuple) to list of genes sorted by rank.
        For component-only: {component: [genes]}
        For celltype-component: {(celltype, component): [genes]}
        
    Raises:
        KeyError: If required columns missing from gene_list_df.
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'gene': ['A', 'B', 'C'],
        ...     'component': ['NMF1', 'NMF1', 'NMF2'],
        ...     'rank': [1, 2, 1]
        ... })
        >>> pools = build_factor_replacement_pools(df)
        >>> pools['NMF1']
        ['A', 'B']
    """
    required_cols = [COL_GENE, COL_RANK, group_by]
    if isinstance(group_by, tuple):
        required_cols = [COL_GENE, COL_RANK] + list(group_by)
    
    missing = set(required_cols) - set(gene_list_df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available: {list(gene_list_df.columns)}"
        )
    
    logger.debug(f"Building replacement pools grouped by: {group_by}")
    
    pools = {}
    
    if isinstance(group_by, tuple):
        # Per-celltype-component pools
        for group_vals, group_df in gene_list_df.groupby(list(group_by)):
            sorted_genes = group_df.sort_values(COL_RANK)[COL_GENE].tolist()
            pools[group_vals] = sorted_genes
    else:
        # Per-component pools
        for component, group_df in gene_list_df.groupby(group_by):
            sorted_genes = group_df.sort_values(COL_RANK)[COL_GENE].tolist()
            pools[component] = sorted_genes
    
    logger.debug(f"Created {len(pools)} replacement pools")
    
    return pools


# =============================================================================
# Private Helper Functions
# =============================================================================


def _find_duplicate_genes(gene_weights: Dict[str, Dict[str, float]]) -> Set[str]:
    """Find genes that appear in multiple factors."""
    duplicates = set()
    for gene, factors in gene_weights.items():
        if len(factors) > 1:
            duplicates.add(gene)
    return duplicates


def _assign_duplicates_to_primary_factor(
    duplicates: Set[str],
    gene_weights: Dict[str, Dict[str, float]]
) -> Dict[str, str]:
    """Assign each duplicate to the factor with highest weight."""
    assignments = {}
    for gene in duplicates:
        factors = gene_weights[gene]
        primary_factor = max(factors.items(), key=lambda x: x[1])[0]
        assignments[gene] = primary_factor
        logger.debug(
            f"Gene {gene}: assigned to {primary_factor} "
            f"(weights: {factors})"
        )
    return assignments


def _build_initial_factor_assignments(
    gene_weights: Dict[str, Dict[str, float]],
    gene_to_primary_factor: Dict[str, str],
    component_cols: List[str]
) -> Dict[str, List[str]]:
    """Build initial factor-to-genes mapping respecting primary assignments."""
    factor_to_genes = {factor: [] for factor in component_cols}
    
    for gene, factors in gene_weights.items():
        if gene in gene_to_primary_factor:
            # Duplicate: use primary assignment
            factor = gene_to_primary_factor[gene]
        else:
            # Non-duplicate: use only factor
            factor = list(factors.keys())[0]
        
        factor_to_genes[factor].append(gene)
    
    return factor_to_genes


def _identify_factors_needing_replacement(
    factor_to_genes: Dict[str, List[str]],
    target_per_factor: int
) -> Dict[str, int]:
    """Identify which factors need replacement genes and how many."""
    factors_needing_replacement = {}
    
    for factor, genes in factor_to_genes.items():
        current_count = len(genes)
        if current_count < target_per_factor:
            gap = target_per_factor - current_count
            factors_needing_replacement[factor] = gap
            logger.debug(
                f"Factor {factor}: {current_count}/{target_per_factor} genes, "
                f"gap = {gap}"
            )
    
    return factors_needing_replacement


def _get_replacement_genes_for_factor(
    factor: str,
    gap_size: int,
    loadings_df: pd.DataFrame,
    already_selected: Set[str]
) -> List[str]:
    """Get replacement genes for a specific factor."""
    # Sort genes by loading in this factor
    factor_loadings = loadings_df[factor]
    factor_loadings = factor_loadings.sort_values(ascending=False)
    
    # Filter to genes not already selected
    available_genes = [
        gene for gene in factor_loadings.index
        if gene not in already_selected
    ]
    
    # Take top gap_size genes
    replacement_genes = available_genes[:gap_size]
    
    if len(replacement_genes) < gap_size:
        logger.warning(
            f"Could only find {len(replacement_genes)}/{gap_size} "
            f"replacement genes for {factor}"
        )
    
    # Log each replacement gene with its loading
    for gene in replacement_genes:
        loading = float(factor_loadings.at[gene]) if gene in factor_loadings.index else 0.0
        logger.info(f"  Gap-fill for {factor}: '{gene}' (loading={loading:.4f})")
    
    return replacement_genes


def _validate_factor_balance(
    factor_to_genes: Dict[str, List[str]],
    target_per_factor: int,
    context: str = 'final'
) -> None:
    """Validate that factor representation is balanced.
    
    Args:
        factor_to_genes: Mapping from factor to gene list.
        target_per_factor: Target genes per factor.
        context: 'pool' or 'final'. In 'pool' context, over-representation is
            expected and intentional (pool is larger than final selection).
            Only factors with too FEW genes are warned about.
    """
    for factor, genes in factor_to_genes.items():
        count = len(genes)
        
        if count < MIN_FACTOR_CONTRIBUTION:
            logger.warning(
                f"Factor {factor} has only {count} genes "
                f"(minimum: {MIN_FACTOR_CONTRIBUTION})"
            )
        
        if context == 'pool':
            # In pool context: over-representation is expected, only log under-representation
            if count < target_per_factor:
                deviation = (target_per_factor - count) / target_per_factor
                logger.debug(
                    f"Factor {factor} pool: {count}/{target_per_factor} genes "
                    f"(under by {deviation:.1%})"
                )
        else:
            deviation = abs(count - target_per_factor) / target_per_factor
            if deviation > FACTOR_BALANCE_TOLERANCE:
                logger.warning(
                    f"Factor {factor} has {count} genes, "
                    f"target is {target_per_factor} "
                    f"(deviation: {deviation:.1%} > {FACTOR_BALANCE_TOLERANCE:.1%})"
                )


def _resolve_cross_celltype_duplicates(
    celltype_factor_to_genes: Dict[str, Dict[str, List[str]]],
    loadings_per_celltype: Dict[str, pd.DataFrame],
    celltype_factor_explained_variance: Optional[Dict[str, pd.Series]]
) -> Tuple[Dict[str, Dict[str, List[str]]], int]:
    """
    Resolve cross-celltype duplicates with iterative replacement.
    
    For each duplicate gene, assigns to celltype/factor with highest contribution
    (loading × R² if available, else normalized loading). Losing celltypes pull
    next-best candidates from their candidate pools.
    
    Returns:
        Updated celltype_factor_to_genes dict and number of replacements made
    """
    # Build candidate pools (sorted by loading descending)
    candidate_pools = {}
    abs_loadings_per_celltype = {
        ct: loadings_df.abs() for ct, loadings_df in loadings_per_celltype.items()
    }
    for ct, factors in celltype_factor_to_genes.items():
        candidate_pools[ct] = {}
        abs_loadings_df = abs_loadings_per_celltype[ct]
        for factor, genes in factors.items():
            # Sort all genes by loading on this factor (vectorized — avoids
            # ~20k Python-level .at[] calls per factor per celltype)
            sorted_genes = abs_loadings_df[factor].sort_values(
                ascending=False
            ).index.tolist()
            # Remove already selected genes
            selected_set = set(genes)
            candidate_pools[ct][factor] = [
                g for g in sorted_genes if g not in selected_set
            ]
    
    # Track assignments
    gene_to_assignments = {}  # gene -> [(ct, factor)]
    for ct, factors in celltype_factor_to_genes.items():
        for factor, genes in factors.items():
            for gene in genes:
                if gene not in gene_to_assignments:
                    gene_to_assignments[gene] = []
                gene_to_assignments[gene].append((ct, factor))
    
    # Find duplicates
    duplicates = {g: cts for g, cts in gene_to_assignments.items() if len(cts) > 1}
    
    if not duplicates:
        return celltype_factor_to_genes, 0
    
    logger.info(f"Resolving {len(duplicates)} cross-celltype duplicates...")
    
    replacements_made = 0
    
    # Resolve each duplicate
    for gene, candidates in duplicates.items():
        # Find best celltype/factor for this gene
        best_score = 0
        best_ct = None
        best_factor = None
        
        for ct, factor in candidates:
            abs_loadings_df = abs_loadings_per_celltype[ct]
            
            if gene not in abs_loadings_df.index:
                continue
            
            loading = float(abs_loadings_df.at[gene, factor])
            
            if celltype_factor_explained_variance is None:
                raise ValueError(
                    "celltype_factor_explained_variance is required for cross-celltype duplicate resolution. "
                    "Ensure compute_nmf_per_celltype or compute_pca_per_celltype returns explained variance "
                    "and it is passed to resolve_duplicates_factor_aware_per_celltype."
                )
            
            # Use R²-weighted contribution
            factor_r2 = celltype_factor_explained_variance[ct][factor]
            contribution = loading * factor_r2
            
            if contribution > best_score:
                best_score = contribution
                best_ct = ct
                best_factor = factor
        
        # Assign to winner, replace in losers
        for ct, factor in candidates:
            if ct == best_ct and factor == best_factor:
                # Winner keeps the gene
                continue
            else:
                # Loser: remove gene and pull replacement
                celltype_factor_to_genes[ct][factor].remove(gene)
                
                # Pull next candidate
                if candidate_pools[ct][factor]:
                    replacement = candidate_pools[ct][factor].pop(0)
                    celltype_factor_to_genes[ct][factor].append(replacement)
                    replacements_made += 1
                    # Log replacement gene with loading and r2*loading contribution
                    repl_loading = float(abs_loadings_per_celltype[ct].at[replacement, factor]) \
                        if replacement in abs_loadings_per_celltype[ct].index else 0.0
                    repl_r2 = float(celltype_factor_explained_variance[ct].get(factor, 0.0)) \
                        if celltype_factor_explained_variance else 0.0
                    repl_contribution = repl_loading * repl_r2
                    # Also log removed gene metrics for comparison
                    try:
                        old_loading = float(abs_loadings_per_celltype[ct].at[gene, factor]) \
                            if gene in abs_loadings_per_celltype[ct].index else 0.0
                        old_r2 = repl_r2  # Same R² for same factor/celltype
                        old_contribution = old_loading * old_r2
                    except (KeyError, ValueError):
                        old_loading = 0.0
                        old_r2 = 0.0
                        old_contribution = 0.0
                    logger.info(
                        f"  Cross-CT gap-fill [{ct}/{factor}]: {gene}(loading={old_loading:.4f}, "
                        f"r2={old_r2:.4f}, contribution={old_contribution:.4f}) replaced by "
                        f"{replacement}(loading={repl_loading:.4f}, r2={repl_r2:.4f}, "
                        f"contribution={repl_contribution:.4f})"
                    )
                else:
                    logger.warning(
                        f"  No replacement available for {ct}/{factor} "
                        f"(gene {gene} removed)"
                    )
        
        logger.info(
            f"  {gene}: assigned to {best_ct}/{best_factor} "
            f"(contribution={best_score:.4f})"
        )
    
    logger.info(
        f"Cross-celltype resolution complete: {replacements_made} replacements made"
    )
    
    return celltype_factor_to_genes, replacements_made


def _find_cross_celltype_duplicates(all_genes: List[str]) -> Set[str]:
    """Find genes that appear multiple times (cross-celltype duplicates)."""
    gene_counts = Counter(all_genes)
    return {gene for gene, count in gene_counts.items() if count > 1}


def _get_next_best_genes_for_component(
    component: str,
    n_genes: int,
    loadings_df: pd.DataFrame,
    exclude_genes: Set[str]
) -> List[str]:
    """Get next-best genes from a component, excluding already selected."""
    component_loadings = loadings_df[component]
    component_loadings = component_loadings.sort_values(ascending=False)
    
    available = [
        gene for gene in component_loadings.index
        if gene not in exclude_genes
    ]
    
    selected = available[:n_genes]
    
    if len(selected) < n_genes:
        logger.warning(
            f"Component {component}: only found {len(selected)}/{n_genes} genes"
        )
    
    return selected
