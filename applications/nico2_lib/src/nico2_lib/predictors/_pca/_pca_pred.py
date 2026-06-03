from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from nico2_lib.typing import NumericArray
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PcaPredictor:
    embedding_size: int | None = None
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None

    _feature_embeddings: NumericArray | None = None
    _mean: NumericArray | None = None
    _explained_variance: NumericArray | None = None

    def fit(self, x: NumericArray) -> "PcaPredictor":
        if self.preprocessing_steps is not None:
            for step in self.preprocessing_steps:
                x = step(x)
        pca = PCA(n_components=self.embedding_size).fit(x)
        return replace(
            self,
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
        if self.preprocessing_steps is not None:
            for step in self.preprocessing_steps:
                x = step(x)
        components_subset = self._feature_embeddings[:, indexer]
        mean_subset = self._mean[indexer]
        centered_X = x - mean_subset
        cell_embeddings = np.dot(centered_X, components_subset.T)

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
