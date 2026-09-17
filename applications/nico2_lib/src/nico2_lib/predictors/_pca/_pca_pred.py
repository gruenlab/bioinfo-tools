from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from nico2_lib.typing import NumericArray
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PcaPredictor:
    n_components: int | None = None
    # Optional: fit() previously always used sklearn's default (random_state=None),
    # which can invoke the randomized SVD solver non-deterministically for typical
    # matrix sizes. Left unset by default so this is purely additive.
    random_state: int | None = None
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None

    _dtype: np.dtype | None = None
    _feature_embeddings: NumericArray | None = None
    _mean: NumericArray | None = None
    _explained_variance: NumericArray | None = None

    @property
    def embedding_size(self) -> int:
        return self.n_components or 3

    def fit(self, x: NumericArray) -> "PcaPredictor":
        if self.preprocessing_steps is not None:
            for step in self.preprocessing_steps:
                x = step(x)
        pca = PCA(n_components=self.embedding_size, random_state=self.random_state).fit(x)
        return replace(
            self,
            _dtype=x.dtype,
            _feature_embeddings=pca.components_,
            _mean=pca.mean_,
            _explained_variance=pca.explained_variance_,
        )

    def predict(
        self, x: NumericArray, indexer: NumericArray
    ) -> tuple[NumericArray, NumericArray]:
        """
        X: Partial feature matrix of shape (n_samples, n_features_subset)
        indexer: Indices of features in X relative to the original 'fit' matrix
        """
        assert self._feature_embeddings is not None and self._mean is not None, (
            "fit must be called before predict"
        )
        x = x.astype(self._dtype)
        if self.preprocessing_steps is not None:
            for step in self.preprocessing_steps:
                x = step(x)
        components_subset = self._feature_embeddings[:, indexer]  # type: ignore
        mean_subset = self._mean[indexer]  # type: ignore
        centered_X = x - mean_subset
        # Least-squares solve, not a direct projection: `centered_X @ components_subset.T`
        # is only the least-squares-optimal reconstruction when components_subset has
        # orthonormal columns, which holds when `indexer` covers every fit-time feature
        # but not in general for a genuine subset (e.g. scoring a probe gene panel
        # against loadings learned on the full transcriptome). This lstsq solve is exact
        # in both cases, and reduces to the same formula as before whenever the subset
        # happens to be orthonormal -- so it shouldn't change anything for full-feature
        # -set callers, only genuine subset-feature predictions.
        cell_embeddings, _, _, _ = np.linalg.lstsq(
            components_subset.T, centered_X.T, rcond=None
        )
        cell_embeddings = cell_embeddings.T

        full_reconstruction = (
            np.dot(cell_embeddings, self._feature_embeddings) + self._mean
        )

        return cell_embeddings, full_reconstruction

    @property
    def feature_embedding(self) -> NumericArray:
        assert self._feature_embeddings is not None, (
            "fit must be called before accessing"
        )
        return self._feature_embeddings
