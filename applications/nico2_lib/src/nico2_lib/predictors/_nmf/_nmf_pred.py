from dataclasses import dataclass, replace
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy import number
from numpy.typing import NDArray
from sklearn.decomposition import non_negative_factorization


@dataclass(frozen=True)
class NmfPredictor:
    """NMF-based predictor using ProtocolN (fit on X, predict all fit-time features)."""

    n_components: Optional[Union[int, Literal["auto"]]] = None
    seed: int = 0
    solver: Literal["cd", "mu"] = "cd"
    h_reference: Optional[NDArray[number]] = None
    n_shared_features: Optional[int] = None
    ref_embedding: Optional[NDArray[number]] = None

    def fit(self, X: NDArray[number]) -> "NmfPredictor":
        """Fit NMF on X to learn the reference component matrix.

        Args:
            X: Feature matrix used for fitting, shape (n_samples, n_features_fit).

        Returns:
            The fitted predictor instance.
        """
        w_reference, h_reference, _ = non_negative_factorization(
            X, n_components=self.n_components, solver=self.solver
        )
        return replace(self, h_reference=h_reference, ref_embedding=w_reference)

    def predict(
        self, X: NDArray[number], indexer: NDArray[np.intp]
    ) -> Tuple[NDArray[number], NDArray[number]]:
        """Predict all fit-time features using X and a feature index map.

        Args:
            X: Feature matrix for prediction, shape (n_samples, n_features_pred).
            indexer: Sequence or array of indices, length n_features_fit, where
                each value points to the corresponding feature column in X.

        Returns:
            Predicted outputs containing all fit-time features in the original
            fit order, shape (n_samples, n_features_fit).
        """
        assert self.h_reference is not None
        w_query, _, _ = non_negative_factorization(
            X=X, H=self.h_reference[:, indexer], init="custom", update_H=False
        )
        return w_query, w_query @ self.h_reference
