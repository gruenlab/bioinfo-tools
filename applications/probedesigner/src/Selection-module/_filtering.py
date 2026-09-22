"""
Separated filtering functions for gene selection pipeline.

This module provides modular, composable filtering functions:
- Xenium expression filter (celltype-aware or global)
- Blacklist filter for unwanted gene patterns

Each function operates independently and can be applied in sequence
or customized based on pipeline needs.
"""

from __future__ import annotations

import logging
import scipy.sparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use absolute imports (for script execution)
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import (
    COL_GENE,
    COL_CELLTYPE,
    DEFAULT_MIN_XENIUM_EXPRESSION,
    DEFAULT_MAX_XENIUM_EXPRESSION,
    DEFAULT_BLACKLIST_PATTERNS
)
from _gene_list_builder import GeneListBuilder

logger = logging.getLogger(__name__)


def apply_xenium_filter_to_genelist(
    gene_list_builder: GeneListBuilder,
    mean_expr_per_ct: Dict[str, Dict[str, float]],
    min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
    celltype_aware: bool = True
) -> GeneListBuilder:
    """
    Apply Xenium expression filter to entire ranked gene list.
    
    The Xenium expression filter removes genes that have expression levels
    outside acceptable bounds. In celltype-aware mode, per-celltype genes
    are checked against their assigned celltype. In global mode, genes must
    fail the filter in ALL celltypes to be rejected.
    
    Args:
        gene_list_builder: Gene list with rankings and celltype assignments.
        mean_expr_per_ct: Expression values per celltype.
            Format: {celltype: {gene: mean_expression}}
        min_expr: Minimum acceptable expression level.
        max_expr: Maximum acceptable expression level.
        celltype_aware: If True, check per-celltype genes against assigned celltype.
            If False, gene must fail in ALL celltypes to be rejected.
        
    Returns:
        Updated GeneListBuilder with xenium filter results recorded.
        
    Raises:
        ValueError: If min_expr >= max_expr.
        KeyError: If required celltype missing from mean_expr_per_ct.
        
    Examples:
        >>> mean_expr = {'T_cells': {'Gene1': 5.0, 'Gene2': 150.0}}
        >>> builder = apply_xenium_filter_to_genelist(
        ...     builder, mean_expr, min_expr=0.1, max_expr=100.0
        ... )
        >>> # Gene1 passes, Gene2 fails (too high)
    """
    if min_expr >= max_expr:
        raise ValueError(
            f"min_expr must be < max_expr. Got min={min_expr}, max={max_expr}"
        )
    
    logger.info(
        f"Applying Xenium expression filter "
        f"(min={min_expr}, max={max_expr}, celltype_aware={celltype_aware})"
    )
    
    df = gene_list_builder.to_dataframe()
    
    passed_count = 0
    failed_count = 0
    
    for _, row in df.iterrows():
        gene = row[COL_GENE]
        celltype = row.get(COL_CELLTYPE, 'global')
        
        passed, reason = _check_xenium_criteria(
            gene=gene,
            celltype=celltype,
            mean_expr_per_ct=mean_expr_per_ct,
            min_expr=min_expr,
            max_expr=max_expr,
            celltype_aware=celltype_aware
        )
        
        gene_list_builder.mark_filter_result(
            gene=gene,
            filter_name='xenium',
            passed=passed,
            failure_reason=reason
        )
        
        if passed:
            passed_count += 1
        else:
            failed_count += 1
    
    logger.info(
        f"✓ Xenium filter complete: {passed_count} passed, {failed_count} failed"
    )
    
    return gene_list_builder


def apply_blacklist_filter(
    gene_list: List[str],
    blacklist_patterns: Optional[List[str]] = None,
    use_default_blacklist: bool = True,
    force_include_genes: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Filter genes based on blacklist patterns (case-insensitive prefix matching).
    
    This filter removes genes that match blacklist patterns (e.g., mitochondrial
    genes starting with 'mt-', heat shock proteins starting with 'hsp').
    Force-included genes override the blacklist.
    
    Args:
        gene_list: List of gene names to filter.
        blacklist_patterns: Additional patterns to blacklist (e.g., ['Malat1', 'Xist']).
            Case-insensitive prefix matching.
        use_default_blacklist: If True, apply default blacklist patterns.
        force_include_genes: Genes to force-include, overriding blacklist.
            
    Returns:
        Tuple containing:
            - filtered_genes: Genes passing the filter
            - removed_genes: Genes removed by blacklist
            - force_included: Genes that were force-included despite blacklist
            
    Examples:
        >>> genes = ['Actb', 'mt-Atp6', 'Hsp90aa1', 'Cd4', 'Malat1']
        >>> filtered, removed, forced = apply_blacklist_filter(
        ...     genes, blacklist_patterns=['Malat1'], use_default_blacklist=True
        ... )
        >>> filtered
        ['Actb', 'Cd4']
        >>> removed
        ['mt-Atp6', 'Hsp90aa1', 'Malat1']
    """
    logger.info("Applying blacklist filter to gene list")
    
    # Build combined blacklist
    combined_blacklist = []
    if use_default_blacklist:
        combined_blacklist.extend(DEFAULT_BLACKLIST_PATTERNS)
    if blacklist_patterns:
        combined_blacklist.extend(blacklist_patterns)
    
    # Convert to lowercase for case-insensitive matching
    blacklist_lower = [pattern.lower() for pattern in combined_blacklist]
    
    # Process force-include genes
    force_include_set = set()
    if force_include_genes:
        force_include_set = {g.lower() for g in force_include_genes}
        logger.debug(f"Force-include genes: {force_include_genes}")
    
    logger.debug(f"Blacklist patterns: {combined_blacklist}")
    
    filtered_genes = []
    removed_genes = []
    force_included = []
    
    for gene in gene_list:
        gene_lower = gene.lower()
        
        # Check if force-included
        if gene_lower in force_include_set:
            filtered_genes.append(gene)
            force_included.append(gene)
            logger.debug(f"  ✓ {gene}: force-included (overrides blacklist)")
            continue
        
        # Check against blacklist patterns (prefix matching)
        is_blacklisted = False
        for pattern in blacklist_lower:
            if gene_lower.startswith(pattern):
                is_blacklisted = True
                removed_genes.append(gene)
                logger.debug(f"  ✗ {gene}: blacklisted (matches pattern '{pattern}')")
                break
        
        if not is_blacklisted:
            filtered_genes.append(gene)
    
    logger.info(
        f"✓ Blacklist filter complete: {len(filtered_genes)} passed, "
        f"{len(removed_genes)} removed, {len(force_included)} force-included"
    )
    
    return filtered_genes, removed_genes, force_included

def compute_global_mean_expression(adata: Any) -> Dict[str, float]:
    """
    Compute mean expression per gene across all cells (global, ignoring cell type).

    Args:
        adata: AnnData object. Uses adata.X (should be the expression matrix at
            the pipeline stage where this is called — typically log-normalized or
            raw counts).

    Returns:
        Dict mapping gene name to mean expression across all cells.

    Examples:
        >>> global_mean = compute_global_mean_expression(adata)
        >>> global_mean['Cd4']
        3.2
    """
    X = adata.X
    if scipy.sparse.issparse(X):
        gene_means = np.array(X.mean(axis=0)).flatten()
    else:
        gene_means = np.array(X).mean(axis=0).flatten()
    return dict(zip(adata.var_names, gene_means.tolist()))


def apply_xenium_filter_global_to_genelist(
    gene_list_builder: GeneListBuilder,
    global_mean_expr: Dict[str, float],
    min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
) -> GeneListBuilder:
    """
    Apply global Xenium expression filter using mean expression across all cells.

    Unlike apply_xenium_filter_to_genelist (which passes a gene if its expression
    is in range in *at least one* celltype), this function uses a single pooled
    mean across all cells. A gene fails only if its global mean falls outside
    [min_expr, max_expr].

    Args:
        gene_list_builder: Gene list with rankings and celltype assignments.
        global_mean_expr: Global mean expression per gene, from
            compute_global_mean_expression. Format: {gene: mean_expression}.
        min_expr: Minimum acceptable expression level.
        max_expr: Maximum acceptable expression level.

    Returns:
        Updated GeneListBuilder with xenium filter results recorded.

    Raises:
        ValueError: If min_expr >= max_expr.

    Examples:
        >>> global_mean = {'Gene1': 5.0, 'Gene2': 150.0}
        >>> builder = apply_xenium_filter_global_to_genelist(
        ...     builder, global_mean, min_expr=0.1, max_expr=100.0
        ... )
        >>> # Gene1 passes, Gene2 fails (too high global mean)
    """
    if min_expr >= max_expr:
        raise ValueError(
            f"min_expr must be < max_expr. Got min={min_expr}, max={max_expr}"
        )

    logger.info(
        f"Applying global Xenium expression filter (min={min_expr}, max={max_expr})"
    )

    df = gene_list_builder.to_dataframe()
    passed_count = 0
    failed_count = 0

    for _, row in df.iterrows():
        gene = row[COL_GENE]
        expr = global_mean_expr.get(gene, 0.0)

        if expr < min_expr:
            passed = False
            reason = "too_low_expression_global"
        elif expr > max_expr:
            passed = False
            reason = "too_high_expression_global"
        else:
            passed = True
            reason = None

        gene_list_builder.mark_filter_result(
            gene=gene,
            filter_name='xenium',
            passed=passed,
            failure_reason=reason,
        )

        if passed:
            passed_count += 1
        else:
            failed_count += 1

    logger.info(
        f"✓ Global Xenium filter complete: {passed_count} passed, {failed_count} failed"
    )

    return gene_list_builder


# ==============================================================================
# PRIVATE HELPER FUNCTIONS
# ==============================================================================


def _check_xenium_criteria(
    gene: str,
    celltype: str,
    mean_expr_per_ct: Dict[str, Dict[str, float]],
    min_expr: float,
    max_expr: float,
    celltype_aware: bool
) -> Tuple[bool, Optional[str]]:
    """
    Check if gene passes Xenium expression criteria.
    
    Returns:
        (passed, failure_reason) tuple.
    """
    if celltype_aware:
        # Check gene in assigned celltype only
        if celltype == 'global' or celltype not in mean_expr_per_ct:
            # Global genes or missing celltype: check all celltypes
            return _check_expression_all_celltypes(
                gene, mean_expr_per_ct, min_expr, max_expr
            )
        
        # Check specific celltype
        ct_expr = mean_expr_per_ct.get(celltype, {})
        expr = ct_expr.get(gene, 0.0)
        
        if expr < min_expr:
            return False, f"too_low_expression_{celltype}"
        elif expr > max_expr:
            return False, f"too_high_expression_{celltype}"
        else:
            return True, None
    else:
        # Global mode: must fail in ALL celltypes to be rejected
        return _check_expression_all_celltypes(
            gene, mean_expr_per_ct, min_expr, max_expr
        )


def _check_expression_all_celltypes(
    gene: str,
    mean_expr_per_ct: Dict[str, Dict[str, float]],
    min_expr: float,
    max_expr: float
) -> Tuple[bool, Optional[str]]:
    """Check if gene passes in at least one celltype."""
    passed_in_any = False
    
    for celltype, ct_expr in mean_expr_per_ct.items():
        expr = ct_expr.get(gene, 0.0)
        if min_expr <= expr <= max_expr:
            passed_in_any = True
            break
    
    if passed_in_any:
        return True, None
    else:
        return False, "failed_all_celltypes"