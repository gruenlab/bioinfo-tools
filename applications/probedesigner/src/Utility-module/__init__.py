"""
Utility functions for the Spatial Probe Design pipeline.

This module provides shared utilities for data validation, dimensionality
reduction, Consensus NMF, Ensembl ID conversion, and gene panel management.
"""

from __future__ import annotations

from ._validation import (
    is_anndata_raw,
    is_anndata_raw_layer,
    X_is_raw,
)
from ._utils import (
    convert_ensembl_to_gene_symbols,
    subset_data_to_gene_lists,
    perform_nmf_pca_per_celltype,
    perform_consensus_nmf_per_celltype,
)
from ._consensus_nmf import (
    ConsensusNmf,
    run_consensus_nmf_per_celltype,
    run_consensus_nmf_global,
    select_optimal_k,
)
from ._panel_utils import (
    save_experiment_parameters,
    setup_logging,
    log_memory_usage,
    get_preprocessing_name,
    get_panel_cache_prefix,
    load_preprocessed_data,
    get_component_panel_path,
    load_cached_panel,
    save_panel_to_cache,
    save_gene_details,
    GeneProvenanceTracker,
)

__all__ = [
    # Data validation
    'is_anndata_raw',
    'is_anndata_raw_layer',
    'X_is_raw',
    # General utilities
    'convert_ensembl_to_gene_symbols',
    'subset_data_to_gene_lists',
    'perform_nmf_pca_per_celltype',
    'perform_consensus_nmf_per_celltype',
    # Consensus NMF
    'ConsensusNmf',
    'run_consensus_nmf_per_celltype',
    'run_consensus_nmf_global',
    'select_optimal_k',
    # Panel utilities
    'save_experiment_parameters',
    'setup_logging',
    'log_memory_usage',
    'get_preprocessing_name',
    'get_panel_cache_prefix',
    'load_preprocessed_data',
    'get_component_panel_path',
    'load_cached_panel',
    'save_panel_to_cache',
    'save_gene_details',
    'GeneProvenanceTracker',
]

__version__ = '2.0.0'
