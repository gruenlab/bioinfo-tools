from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
from kneed import KneeLocator
from numpy import number
from numpy.typing import NDArray
from sklearn.decomposition import non_negative_factorization


@dataclass
class NmfPredictor:
    n_components: Optional[Union[int, Literal["auto"]]] = None
    H_query: Optional[NDArray[number]] = None
    H_predicted: Optional[NDArray[number]] = None
    n_shared_features: Optional[int] = None

    def _explained_variance(
        self,
        X: NDArray[number],
        X_reconstructed: NDArray[number],
    ) -> float:
        """R²-style explained variance."""
        residual = np.linalg.norm(X - X_reconstructed) ** 2
        total = np.linalg.norm(X - X.mean(axis=0)) ** 2
        return 1.0 - residual / total

    def _select_n_components(
        self,
        X: NDArray[number],
        component_range=range(2, 11),
    ) -> int:
        scores = []

        for k in component_range:
            W, H, _ = non_negative_factorization(
                X,
                n_components=k,
                init="nndsvda",
                random_state=0,
            )
            X_hat = W @ H
            scores.append(self._explained_variance(X, X_hat))

        knee = KneeLocator(
            list(component_range),
            scores,
            curve="concave",
            direction="increasing",
        )

        return knee.knee or component_range[-1]

    def fit(
        self,
        X: NDArray[number],
        y: NDArray[number],
    ) -> "NmfPredictor":
        """
        Fits the NMF basis on the reference matrix X.
        X is assumed to have shape (n_samples, n_shared + n_predicted).
        """
        reference_matrix = X
        self.n_shared_features = y.shape[1]

        reference_matrix = np.concatenate([X, y], axis=1)
        self.n_shared_features = X.shape[1]

        if self.n_components is None or self.n_components == "auto":
            self.n_components = self._select_n_components(X)

        _, H_ref, _ = non_negative_factorization(
            reference_matrix,
            n_components=self.n_components,
        )

        self.H_query, self.H_predicted = np.hsplit(H_ref, [self.n_shared_features])

        return self

    def predict(
        self,
        X: NDArray[number],
    ) -> NDArray[number]:
        """
        Predicts the missing features for the query matrix X.
        """
        if self.H_query is None or self.H_predicted is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=self.H_query.dtype)
        W_query, _, _ = non_negative_factorization(
            X=X,
            H=self.H_query,
            init="custom",
            update_H=False,
        )

        predicted = W_query @ self.H_predicted
        return predicted
