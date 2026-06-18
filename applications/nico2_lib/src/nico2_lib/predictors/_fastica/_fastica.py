"""
FastICA predictor following the PredictorProtocol.
Uses sklearn.decomposition.FastICA for Independent Component Analysis.
Finds latent factors that are statistically independent (non-Gaussian).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from nico2_lib.typing import IndexArray, NumericArray
from sklearn.decomposition import FastICA as SklearnFastICA


@dataclass(frozen=True)
class FastIcaPredictor:
    """FastICA predictor using ProtocolN (fit on X, predict all fit-time features).

    FastICA finds a linear transformation that maximizes the statistical
    independence between output components. Unlike PCA which finds orthogonal
    directions of maximum variance, ICA finds components that are statistically
    independent (non-Gaussian).

    Useful for:
    - Separating mixed signals (e.g., cell types from expression programs)
    - Finding interpretable, biologically meaningful factors
    - Removing confounders assumed to be Gaussian

    Args:
        n_components: Number of ICA components (embedding dimension).
        algorithm: Algorithm type ('parallel' or 'deflation').
        fun: Function used to approximate negentropy ('logcosh', 'exp', 'cube').
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
        random_state: Random seed for reproducibility.
        preprocessing_steps: Optional preprocessing pipeline.
    """

    n_components: int | None = None
    algorithm: Literal["parallel", "deflation"] = "parallel"
    fun: Literal["logcosh", "exp", "cube"] = "logcosh"
    max_iter: int = 200
    tol: float = 0.0001
    random_state: int | None = 42
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None

    _dtype: np.dtype | None = None
    _mixing: NumericArray | None = None  # A: unmixing matrix (n_features, n_components)
    _mean: NumericArray | None = None
    _n_iter: int | None = None

    @property
    def embedding_size(self) -> int | None:
        return self.n_components

    def fit(self, x: NumericArray) -> "FastIcaPredictor":
        """Fits FastICA on X.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            The fitted predictor instance.
        """
        if self.preprocessing_steps is not None:
            for step in self.preprocessing_steps:
                x = step(x)

        ica = SklearnFastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            fun=self.fun,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        # fit_transform returns S = X @ A.T where A is the unmixing matrix
        ica.fit_transform(x)

        # sklearn stores the unmixing matrix in ica.components_ (n_components, n_features)
        # The mixing matrix for reconstruction is the pseudoinverse: A @ (S.T @ S)^-1
        # or more simply, for whitened data: A @ S.T
        #
        # Standard ICA model: X = S @ A where S are sources, A is mixing
        # sklearn stores A.T in components_ (n_components, n_features)
        components = ica.components_  # (n_components, n_features)

        return replace(
            self,
            _dtype=x.dtype,
            _mixing=components.T if components is not None else None,  # (n_features, n_components)
            _mean=ica.mean_ if hasattr(ica, 'mean_') else None,
            _n_iter=ica.n_iter_ if hasattr(ica, 'n_iter_') else None,
        )

    def predict(
        self, x: NumericArray, indexer: IndexArray
    ) -> tuple[NumericArray, NumericArray]:
        """Predicts all fit-time features using X and a feature index map.

        For ICA, we use the unmixing matrix to project data into the
        independent component space, then reconstruct via the mixing matrix.

        Args:
            x: Feature matrix for prediction, shape (n_samples, n_features_subset).
            indexer: Indices mapping fit-time features to predict-time features.

        Returns:
            Tuple of (cell_embeddings, full_reconstruction).
        """
        assert self._mixing is not None, "fit must be called before predict"

        x = x.astype(self._dtype)
        if self.preprocessing_steps is not None:
            for step in self.preprocessing_steps:
                x = step(x)

        # Get the unmixing matrix columns corresponding to indexed features
        mixing_subset = self._mixing[indexer]  # (n_features_subset, n_components)

        # Project to independent component space (embeddings)
        # S = X @ A where A is the unmixing matrix
        if self._mean is not None:
            x_centered = x - self._mean[indexer] if self._mean is not None else x
        else:
            x_centered = x

        # Solve least squares for embeddings: x = S @ mixing_subset.T
        # S = x @ pinv(mixing_subset.T)
        cell_embeddings, _, _, _ = np.linalg.lstsq(mixing_subset, x_centered.T, rcond=None)
        cell_embeddings = cell_embeddings.T  # (n_samples, n_components)

        # Reconstruct all features using full mixing matrix
        # X_rec = S @ A.T where A is the full unmixing matrix
        full_reconstruction = cell_embeddings @ self._mixing.T

        if self._mean is not None:
            full_reconstruction = full_reconstruction + self._mean

        return cell_embeddings, full_reconstruction

    @property
    def feature_embedding(self) -> NumericArray | None:
        """Returns the unmixing matrix (feature loadings for independent components)."""
        return self._mixing.T
