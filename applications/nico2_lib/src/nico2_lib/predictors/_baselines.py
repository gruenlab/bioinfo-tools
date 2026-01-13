"""
Single-cell data neighbor shuffling utilities.

This module provides functions for shuffling single-cell count data based on
nearest neighbors in a low-dimensional embedding space. This is useful as a baseline
for assessing feature prediction performance.
"""

from typing import Optional, Union

import numpy as np
from numpy import intp, number
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from sklearn.decomposition import NMF, PCA, FastICA, KernelPCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.neighbors import NearestNeighbors

SklearnEmbedder = Union[NMF, PCA, FastICA, KernelPCA, TruncatedSVD, MDS, TSNE, Isomap]


def _sample_neighbours(
    adjacency_matrix: NDArray, rng: Optional[Union[int, Generator]] = None
) -> NDArray[intp]:
    """Sample one random neighbor per cell from an adjacency matrix (vectorized).

    Args:
        adjacency_matrix: Square 2D array (n_cells, n_cells). Nonzero entries
            indicate neighbors.
        rng: Random number generator, integer seed, or None.

    Returns:
        Array of shape (n_cells,) with one randomly selected neighbor per cell.

    Raises:
        ValueError: If adjacency_matrix is not square or any cell has no neighbors.
    """
    if rng is None:
        rng = default_rng()
    elif isinstance(rng, int):
        rng = default_rng(rng)

    if (
        adjacency_matrix.ndim != 2
        or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]
    ):
        raise ValueError("adjacency_matrix must be a square 2D array")

    n_cells = adjacency_matrix.shape[0]
    adj = adjacency_matrix.astype(bool)
    degrees = adj.sum(axis=1)
    if np.any(degrees == 0):
        bad = np.flatnonzero(degrees == 0)[0]
        raise ValueError(f"Cell {bad} has no neighbors")
    _, neighbors = np.nonzero(adj)
    row_starts = np.cumsum(degrees) - degrees
    offsets = (rng.random(n_cells) * degrees).astype(intp)
    return neighbors[row_starts + offsets]


def shuffle_by_embedding_neighbors(
    X: NDArray[number],
    rng: Optional[Union[int, Generator]] = None,
    sklearn_embedder: Optional[SklearnEmbedder] = None,
    sklearn_neighbors: Optional[NearestNeighbors] = None,
) -> NDArray[number]:
    """Shuffle rows by randomly swapping with embedding-space neighbors.

    Cells are embedded into a low-dimensional space, a neighborhood graph is
    constructed using an sklearn neighbors model, and each cell is replaced
    by one of its returned neighbors. Inclusion of self-neighbors is entirely
    controlled by the neighbors model.

    Args:
        X: Array of shape (n_cells, n_features) containing cell features.
        rng: Random number generator, seed, or None.
        sklearn_embedder: sklearn embedder used to compute embeddings.
            Defaults to PCA(n_components=10).
        sklearn_neighbors: sklearn neighbors model defining neighborhoods.
            Defaults to NearestNeighbors(n_neighbors=6).

    Returns:
        Shuffled array of shape (n_cells, n_features).

    Raises:
        ValueError: If a cell has no neighbors.
    """
    embedder = sklearn_embedder or PCA(n_components=10)
    embeddings: NDArray[number] = embedder.fit_transform(X)  # type: ignore

    neighbors_model = sklearn_neighbors or NearestNeighbors(n_neighbors=6)
    neighbors_model.fit(embeddings)

    _, indices = neighbors_model.kneighbors(embeddings)

    n_cells, k = indices.shape

    adjacency_matrix = np.zeros((n_cells, n_cells), dtype=bool)
    adjacency_matrix[
        np.repeat(np.arange(n_cells), k),
        indices.ravel(),
    ] = True

    shuffled_index = _sample_neighbours(adjacency_matrix, rng)
    return X[shuffled_index]
