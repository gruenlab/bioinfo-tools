"""
Utility functions for the Spatial Probe Design pipeline.

This module provides shared utilities for AnnData validation and Ensembl ID
conversion.
"""

from __future__ import annotations

from ._validation import (
    is_anndata_raw,
    is_anndata_raw_layer,
)
from ._utils import (
    convert_ensembl_to_gene_symbols,
)

__all__ = [
    # Data validation
    'is_anndata_raw',
    'is_anndata_raw_layer',
    # General utilities
    'convert_ensembl_to_gene_symbols',
]

__version__ = '2.0.0'
