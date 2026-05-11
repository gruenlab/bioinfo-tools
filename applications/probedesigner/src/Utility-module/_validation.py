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

    # Handle sparse matrices
    if scipy.sparse.issparse(adata.X):
        # For sparse matrices, check the data array
        sample_data = adata.X.data
        if len(sample_data) > SAMPLE_SIZE_FOR_RAW_CHECK:
            sample_data = sample_data[:SAMPLE_SIZE_FOR_RAW_CHECK]
    else:
        # For dense matrices, sample a subset
        flat_data = adata.X.flatten()
        if len(flat_data) > SAMPLE_SIZE_FOR_RAW_CHECK:
            sample_indices = np.random.choice(len(flat_data), SAMPLE_SIZE_FOR_RAW_CHECK, replace=False)
            sample_data = flat_data[sample_indices]
        else:
            sample_data = flat_data

    # Remove zeros for the check
    non_zero_data = sample_data[sample_data != 0]

    if len(non_zero_data) == 0:
        # All zeros - could be either, but likely raw
        return True

    # Check if values are close to integers (within floating point precision)
    is_integer_data = np.allclose(non_zero_data, np.round(non_zero_data))

    return is_integer_data


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

    # Handle sparse matrices
    if scipy.sparse.issparse(layer_data):
        # For sparse matrices, check the data array
        sample_data = layer_data.data
        if len(sample_data) > SAMPLE_SIZE_FOR_RAW_CHECK:
            sample_data = sample_data[:SAMPLE_SIZE_FOR_RAW_CHECK]
    else:
        # For dense matrices, sample a subset
        flat_data = layer_data.flatten()
        if len(flat_data) > SAMPLE_SIZE_FOR_RAW_CHECK:
            sample_indices = np.random.choice(len(flat_data), SAMPLE_SIZE_FOR_RAW_CHECK, replace=False)
            sample_data = flat_data[sample_indices]
        else:
            sample_data = flat_data

    # Remove zeros for the check
    non_zero_data = sample_data[sample_data != 0]

    if len(non_zero_data) == 0:
        # All zeros - could be either, but likely raw
        return True

    # Check if values are close to integers (within floating point precision)
    is_integer_data = np.allclose(non_zero_data, np.round(non_zero_data))

    return is_integer_data


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
