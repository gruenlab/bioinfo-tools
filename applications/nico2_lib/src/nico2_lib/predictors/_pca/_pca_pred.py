from dataclasses import dataclass, replace

from nico2_lib.typing import NumericArray
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PcaPredictor:
    embedding_size: int | None = None

    _feature_embeddings: NumericArray | None = None
    _mean: NumericArray | None = None
    _explained_variance: NumericArray | None = None

    def fit(self, X: NumericArray) -> "PcaPredictor":
        pca = PCA(
            n_components=self.embedding_size,
        ).fit(X)
        feature_embeddings: NumericArray = pca.components_  # type: ignore
        mean = pca.mean_  # type: ignore
        explained_variance = pca.explained_variance_  # type: ignore
        return replace(
            self,
            _feature_embeddings=feature_embeddings,
            _mean=mean,
            _explained_variance=explained_variance,
        )

    def predict(
        self, X: NumericArray, indexer: NumericArray
    ) -> tuple[NumericArray, NumericArray]:
        assert (
            self._feature_embeddings is not None
            and self._mean is not None
            and self._explained_variance is not None
        ), "fit must be called before predict"
        pca = PCA(
            n_components=self.embedding_size,
        )
        pca.components_ = self._feature_embeddings[:, indexer]  # type: ignore
        pca.mean_ = self._mean[indexer]  # type: ignore
        pca.explained_variance_ = self._explained_variance
        cell_embeddings = pca.transform(X)
        return cell_embeddings, pca.inverse_transform(cell_embeddings)

    @property
    def feature_embedding(self) -> NumericArray | None:
        assert self._feature_embeddings is not None, (
            "fit must be called before accessing feature_embeddings"
        )
        return self._feature_embeddings
