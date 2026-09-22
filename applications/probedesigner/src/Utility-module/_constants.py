"""
Constants for utility functions in the Spatial Probe Design pipeline.

This module defines configuration constants for data validation, dimensionality
reduction, and related utility operations.
"""

from __future__ import annotations

# =============================================================================
# Data Validation Constants
# =============================================================================

# Sample size for raw data detection
SAMPLE_SIZE_FOR_RAW_CHECK: int = 10_000

# Minimum cells required for dimensionality reduction
MIN_CELLS_FOR_DIMRED: int = 10

# Normalization: sc.pp.normalize_total() is called with no target_sum (scanpy
# default, per-cell median), matching Preprocessing-module and Evaluation-module.
# No constant needed here.

# =============================================================================
# NMF Parameters
# =============================================================================

# NMF regularization parameters
DEFAULT_NMF_ALPHA_W: float = 0.0
DEFAULT_NMF_ALPHA_H: float = 0.0
DEFAULT_NMF_L1_RATIO: float = 0.0

# =============================================================================
# Logging Configuration
# =============================================================================

# Log format
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
