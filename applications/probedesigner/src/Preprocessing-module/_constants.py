"""Constants for preprocessing module.

This module defines all configuration constants used throughout the preprocessing
pipeline, including dimensionality reduction parameters, normalization targets,
filtering thresholds, and logging configuration.
"""

from __future__ import annotations

# =============================================================================
# Dimensionality Reduction
# =============================================================================

DEFAULT_N_COMPONENTS_PCA: int = 50
"""Number of PCA components to compute."""

DEFAULT_N_COMPONENTS_NMF: int = 5
"""Number of NMF components to compute."""

DEFAULT_RANDOM_STATE: int = 42
"""Random state seed for reproducibility."""

# =============================================================================
# Normalization
# =============================================================================

NORMALIZE_TARGET_SUM: float = 1e4
"""Target sum for normalization (CPM scaling)."""

# =============================================================================
# Gene Selection
# =============================================================================

DEFAULT_N_HVG: int = 8000
"""Default number of highly variable genes to select."""

DEFAULT_HVG_FLAVOR: str = "cell_ranger"
"""Scanpy HVG flavor for highly variable gene computation."""

# =============================================================================
# Filtering
# =============================================================================

DEFAULT_MIN_GENES_PER_CELL: int = 100
"""Minimum number of genes expressed per cell for filtering."""

DEFAULT_MIN_CELLS_PER_GENE: int = 3
"""Minimum number of cells expressing a gene for filtering."""

FILTER_METHODS: list[str] = ['scanpy', 'no_filter']
"""Available filtering methods."""

# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
"""Format string for logging output."""

LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
"""Date format for logging timestamps."""

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'DEFAULT_N_COMPONENTS_PCA',
    'DEFAULT_N_COMPONENTS_NMF',
    'DEFAULT_RANDOM_STATE',
    'NORMALIZE_TARGET_SUM',
    'DEFAULT_N_HVG',
    'DEFAULT_HVG_FLAVOR',
    'DEFAULT_MIN_GENES_PER_CELL',
    'DEFAULT_MIN_CELLS_PER_GENE',
    'FILTER_METHODS',
    'LOG_FORMAT',
    'LOG_DATE_FORMAT',
]
