from typing import Protocol, Sequence

import numpy as np
from numpy import intp, number
from numpy.typing import NDArray


class PredictorProtocol(Protocol):
    """Protocol for predictors fit on X, y and predicting y from X.

    Usage:
    - fit(X, y): learn a model that predicts y from X.
    - predict(X): return predictions with the same feature dimensionality as y.
    """

    def fit(
        self,
        X: NDArray[number],
        y: NDArray[number],
    ) -> "PredictorProtocol":
        """Fits a model that predicts y from X.

        Args:
            X: Feature matrix used for fitting, shape
                (n_samples, n_features_x).
            y: Target matrix to learn to predict, shape
                (n_samples, n_features_y).

        Returns:
            The fitted predictor instance.
        """
        ...

    def predict(
        self,
        X: NDArray[number],
    ) -> NDArray[number]:
        """Predicts y-like outputs from X.

        Args:
            X: Feature matrix for prediction, shape
                (n_samples, n_features_x). n_features_x must match the feature
                dimensionality used in fit.

        Returns:
            Predicted outputs, shape (n_samples, n_features_y).
        """
        ...


class PredictorProtocolN(Protocol):
    """Protocol for predictors fit on X and predicting X across feature sets.

    Usage:
    - fit(X): learn structure from X only.
    - predict(X, indexer): indexer maps fit-time feature indices to predict-time
      feature indices; return all fit-time features in their original order.
    """

    def fit(
        self,
        X: NDArray[number],
    ) -> "PredictorProtocolN":
        """Fits a model using X only.

        Args:
            X: Feature matrix used for fitting, shape
                (n_samples, n_features_fit).

        Returns:
            The fitted predictor instance.
        """
        ...

    def predict(
        self,
        X: NDArray[number],
        indexer: NDArray[intp],
    ) -> NDArray[number]:
        """Predicts all fit-time features using X and a feature index map.

        Args:
            X: Feature matrix for prediction, shape
                (n_samples, n_features_pred).
            indexer: Sequence or array of indices, length n_features_fit, where
                each value points to the corresponding feature column in X. This
                maps fit-time feature indices to predict-time feature indices.

        Returns:
            Predicted outputs containing all fit-time features in the original
            fit order, shape (n_samples, n_features_fit).
        """
        ...
