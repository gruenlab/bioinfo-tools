"""
Factor-aware duplicate resolution for dimensionality reduction.

This module implements factor-aware duplicate resolution for maintaining balanced
representation across components (factors) in NMF/PCA-based gene selection:
resolve duplicates by keeping each gene in its highest-weight factor, backfilling
the vacated slot with the next-best candidate. Supports both global and
per-celltype analysis (`resolve_duplicates_factor_aware_global` /
`resolve_duplicates_factor_aware_per_celltype`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import pandas as pd

# Use absolute imports (for script execution)
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import (
    MIN_FACTOR_CONTRIBUTION,
    FACTOR_BALANCE_TOLERANCE,
)

logger = logging.getLogger(__name__)


def resolve_duplicates_factor_aware_global(
    gene_weights: Dict[str, Dict[str, float]],
    genes_per_component: int,
    loadings_df: pd.DataFrame,
    component_cols: List[str],
    context: str = 'final'
) -> Dict[str, Any]:
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
        context: Balance-check mode passed to ``_validate_factor_balance`` —
            ``'final'`` warns on any imbalance, ``'pool'`` suppresses
            over-representation warnings.

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
) -> Dict[str, any]:
    """
    Resolve duplicate genes with celltype-factor awareness.

    This function accepts the celltype_genes_per_factor dict directly from
    _select_genes_per_celltype and handles within-celltype duplicates.
    Cross-celltype shared genes are intentionally kept and tracked via
    gene_celltype_mapping so the caller can rank by n_celltypes.

    Args:
        celltype_genes_per_factor: Dict with structure:
            {celltype: {
                "genes_per_factor": {factor: [gene1, gene2, ...]},
                "loadings_df": pd.DataFrame (genes × factors),
                "component_cols": [factor1, factor2, ...]
            }}

    Returns:
        Dictionary containing:
            - 'selected_genes': List of final selected gene names
            - 'factor_assignments': Dict mapping gene to component
            - 'celltype_assignments': Dict mapping gene to celltype
            - 'duplicates_resolved': Total within-celltype duplicates resolved
            - 'replacements_made': Number of replacement genes added
            - 'gene_celltype_mapping': Dict mapping gene to all celltypes that selected it
            - 'n_celltypes_per_gene': Dict mapping gene to count of celltypes

    Examples:
        >>> celltype_data = {
        ...     'T_cells': {
        ...         'genes_per_factor': {'NMF1': ['GeneA', 'GeneB']},
        ...         'loadings_df': loadings_t,
        ...         'component_cols': ['NMF1', 'NMF2']
        ...     }
        ... }
        >>> result = resolve_duplicates_factor_aware_per_celltype(celltype_data)
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

    # Step 2: Build gene_celltype_mapping from original input (captures all
    # contributions before within-CT resolution trimmed genes).
    # Cross-celltype shared genes are tracked here, not removed — the caller
    # ranks genes by n_celltypes to prefer multi-celltype genes.
    gene_celltype_mapping: Dict[str, List[str]] = {}
    for celltype, data in celltype_genes_per_factor.items():
        for factor, genes in data["genes_per_factor"].items():
            for gene in genes:
                if gene not in gene_celltype_mapping:
                    gene_celltype_mapping[gene] = []
                if celltype not in gene_celltype_mapping[gene]:
                    gene_celltype_mapping[gene].append(celltype)

    n_celltypes_per_gene = {gene: len(cts) for gene, cts in gene_celltype_mapping.items()}

    # Step 3: Build simple assignments for caller (best celltype = first assignment)
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

    cross_ct_count = sum(1 for n in n_celltypes_per_gene.values() if n > 1)

    result = {
        'selected_genes': selected_genes,
        'factor_assignments': factor_assignments,
        'celltype_assignments': celltype_assignments,
        'duplicates_resolved': within_ct_duplicates,
        'replacements_made': total_replacements,
        'gene_celltype_mapping': gene_celltype_mapping,
        'n_celltypes_per_gene': n_celltypes_per_gene,
    }

    logger.info(
        f"✓ Per-celltype resolution complete: {len(selected_genes)} unique genes "
        f"({within_ct_duplicates} within-celltype duplicates resolved, "
        f"{cross_ct_count} genes shared across multiple celltypes)"
    )

    return result


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


