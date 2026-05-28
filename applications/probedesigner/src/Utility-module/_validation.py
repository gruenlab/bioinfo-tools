"""
Data validation functions for AnnData objects.

This module provides utilities for checking whether AnnData objects contain
raw count data or processed data, based on data type inspection.
"""

from __future__ import annotations

import logging
import numpy as np
import scipy.sparse
from anndata import AnnData

try:
    from ._constants import SAMPLE_SIZE_FOR_RAW_CHECK   # package import
except ImportError:
    SAMPLE_SIZE_FOR_RAW_CHECK = 10_000                  # flat-module fallback

__all__ = [
    'is_anndata_raw',
    'is_anndata_raw_layer',
    'X_is_raw',
]

logger = logging.getLogger(__name__)


def is_anndata_raw(adata: AnnData) -> bool:
    """
    Check if an AnnData object is raw by examining data types in .X.

    Raw data should contain integer count values, while processed data
    typically contains float values after normalization.

    Args:
        adata: The AnnData object to check.

    Returns:
        True if .X contains integer data (raw), False if float data (processed).

    Raises:
        ValueError: If the AnnData object has no data in .X.
    """
    if adata.X is None:
        raise ValueError("AnnData object has no data in .X")

    # Integer dtype is a definitive indicator of raw counts.
    # sc.pp.normalize_total converts to float, so float dtype indicates processed data
    # (though we still verify with a value check below for pipelines that store
    # normalized data as float from the start).
    if np.issubdtype(adata.X.dtype, np.integer):
        return True

    # Value-based check: sample specifically from non-zero positions to avoid the
    # all-zeros false positive on sparse panels (random sampling from the full
    # matrix can land entirely on zero-valued positions, making non_zero_data empty
    # and causing np.allclose([], []) to vacuously return True).
    if scipy.sparse.issparse(adata.X):
        # .data contains only the stored (non-zero) values — no zeros to sample.
        non_zero_data = adata.X.data
        if len(non_zero_data) > SAMPLE_SIZE_FOR_RAW_CHECK:
            idx = np.random.choice(len(non_zero_data), SAMPLE_SIZE_FOR_RAW_CHECK, replace=False)
            non_zero_data = non_zero_data[idx]
    else:
        flat = adata.X.flatten()
        nz_idx = np.flatnonzero(flat)
        if len(nz_idx) == 0:
            return True  # All zeros — conservative: assume raw
        if len(nz_idx) > SAMPLE_SIZE_FOR_RAW_CHECK:
            nz_idx = np.random.choice(nz_idx, SAMPLE_SIZE_FOR_RAW_CHECK, replace=False)
        non_zero_data = flat[nz_idx]

    if len(non_zero_data) == 0:
        return True  # Sparse with no stored values — assume raw

    return bool(np.allclose(non_zero_data, np.round(non_zero_data)))


def is_anndata_raw_layer(adata: AnnData, layer_name: str) -> bool:
    """
    Check if a specific layer in an AnnData object contains raw data.

    Args:
        adata: The AnnData object to check.
        layer_name: Name of the layer to check.

    Returns:
        True if layer contains integer data (raw), False if float data (processed).

    Raises:
        ValueError: If the layer is not found in adata.layers.
    """
    if layer_name not in adata.layers:
        raise ValueError(f"Layer '{layer_name}' not found in adata.layers")

    layer_data = adata.layers[layer_name]

    if np.issubdtype(layer_data.dtype, np.integer):
        return True

    if scipy.sparse.issparse(layer_data):
        non_zero_data = layer_data.data
        if len(non_zero_data) > SAMPLE_SIZE_FOR_RAW_CHECK:
            idx = np.random.choice(len(non_zero_data), SAMPLE_SIZE_FOR_RAW_CHECK, replace=False)
            non_zero_data = non_zero_data[idx]
    else:
        flat = layer_data.flatten()
        nz_idx = np.flatnonzero(flat)
        if len(nz_idx) == 0:
            return True
        if len(nz_idx) > SAMPLE_SIZE_FOR_RAW_CHECK:
            nz_idx = np.random.choice(nz_idx, SAMPLE_SIZE_FOR_RAW_CHECK, replace=False)
        non_zero_data = flat[nz_idx]

    if len(non_zero_data) == 0:
        return True

    return bool(np.allclose(non_zero_data, np.round(non_zero_data)))


def X_is_raw(adata: AnnData, X: bool = True) -> bool:
    """
    Simplified check if AnnData contains raw data.

    Args:
        adata: AnnData object to check.
        X: If True, check .X; if False, check layers["counts"].

    Returns:
        True if data appears to be raw integer counts.
    """
    if X:
        return np.array_equal(adata.X.sum(axis=0).astype(int), adata.X.sum(axis=0))
    else:
        return np.array_equal(adata.layers["counts"].sum(axis=0).astype(int), adata.layers["counts"].sum(axis=0))
