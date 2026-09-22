"""
Gene selection strategies for spatial transcriptomics probe design.

This module provides eight different gene selection strategies:
- deg_only: Differential expression analysis
- rf_simple: Random forest classifier on all genes
- rf_deg: Random forest on DEG-filtered genes
- RecoVar: Hybrid random forest + NMF with gap-filling
- RecoVar_PCA: Hybrid random forest + PCA with gap-filling
- dimred_only: Pure dimensionality reduction (NMF/PCA)
- hvg: Highly variable genes (baseline)
- random: Random selection (baseline)

The module uses a unified output format (GeneListBuilder) to track gene
provenance through selection, filtering, and replacement stages.

"""

from __future__ import annotations

from ._gene_list_builder import GeneListBuilder
from ._factor_aware import (
    resolve_duplicates_factor_aware_global,
    resolve_duplicates_factor_aware_per_celltype,
)
from ._filtering import (
    apply_xenium_filter_to_genelist,
    apply_blacklist_filter,
)
from ._deg_selection import (
    select_degs,
    filter_celltypes_by_min_cells,
)
from ._baseline_selection import (
    select_highly_variable_genes,
    select_random_genes,
)
from ._rf_selection import (
    select_genes_with_rf,
)
from ._dimred_selection import (
    select_genes_from_nmf,
    select_genes_from_pca,
)

__all__ = [
    # Core data structure
    'GeneListBuilder',
    # Factor-aware functions
    'resolve_duplicates_factor_aware_global',
    'resolve_duplicates_factor_aware_per_celltype',
    # Filtering functions
    'apply_xenium_filter_to_genelist',
    'apply_blacklist_filter',

    # DEG selection
    'select_degs',
    'filter_celltypes_by_min_cells',

    # Baseline selection (HVG, Random)
    'select_highly_variable_genes',
    'select_random_genes',

    # Random forest selection
    'select_genes_with_rf',

    # Dimension reduction selection (NMF/PCA)
    'select_genes_from_nmf',
    'select_genes_from_pca',
]

__version__ = '2.0.0'
