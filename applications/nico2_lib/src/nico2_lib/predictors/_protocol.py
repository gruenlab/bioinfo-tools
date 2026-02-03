from typing import Protocol

from numpy import intp, number
from numpy.typing import NDArray



class PredictorProtocol(Protocol):
    """Protocol for predictors fit on X and predicting X across feature sets.

    Usage:
    - fit(X): learn structure from X only.
    - predict(X, indexer): indexer maps fit-time feature indices to predict-time
      feature indices; return all fit-time features in their original order.
    """

    def fit(
        self,
        X: NDArray[number],
    ) -> "PredictorProtocol":
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
