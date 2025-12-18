from dataclasses import dataclass
from typing import Optional
from sklearn.decomposition import non_negative_factorization
import numpy as np
from numpy.typing import NDArray
from numpy import number


@dataclass
class NmfPredictor:
    n_components: Optional[int] = None
    H_query: Optional[NDArray[number]] = None
    H_predicted: Optional[NDArray[number]] = None
    n_shared_features: Optional[int] = None

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
