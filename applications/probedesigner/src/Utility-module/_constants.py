"""
Constants for utility functions in the Spatial Probe Design pipeline.

This module defines configuration constants for data validation, dimensionality
reduction, consensus NMF, and related utility operations.
"""

from __future__ import annotations

# =============================================================================
# Data Validation Constants
# =============================================================================

# Sample size for raw data detection
SAMPLE_SIZE_FOR_RAW_CHECK: int = 10_000

# Minimum cells required for dimensionality reduction
MIN_CELLS_FOR_DIMRED: int = 10

# Normalization target sum
NORMALIZE_TARGET_SUM: float = 1e4

# =============================================================================
# NMF Parameters
# =============================================================================

# NMF regularization parameters
DEFAULT_NMF_ALPHA_W: float = 0.0
DEFAULT_NMF_ALPHA_H: float = 0.0
DEFAULT_NMF_L1_RATIO: float = 0.0

# =============================================================================
# Consensus NMF Parameters
# =============================================================================

# Density threshold for consensus filtering (lower = stricter)
DEFAULT_DENSITY_THRESHOLD: float = 0.5

# Local neighborhood size for density calculation (fraction of iterations)
DEFAULT_LOCAL_NEIGHBORHOOD_SIZE: float = 0.30

# K-means n_init parameter
DEFAULT_KMEANS_N_INIT: int = 10

# =============================================================================
# Logging Configuration
# =============================================================================

# Log format
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
