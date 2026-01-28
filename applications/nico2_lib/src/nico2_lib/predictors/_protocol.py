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
        indexer: Sequence[int] | NDArray[intp],
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


def _adapt_protocol_n_to_protocol(
    predictor_protocol: PredictorProtocolN,
) -> PredictorProtocol:
    """Adapt a ProtocolN predictor to the legacy PredictorProtocol interface."""

    class _ProtocolAdapter:
        def __init__(self, predictor: PredictorProtocolN) -> None:
            self._predictor = predictor
            self._n_features_fit: int | None = None

        def fit(
            self,
            X: NDArray[number],
            y: NDArray[number],
        ) -> "_ProtocolAdapter":
            if hasattr(self._predictor, "_set_fit_targets"):
                self._predictor._set_fit_targets(y)
            self._predictor = self._predictor.fit(X)
            self._n_features_fit = X.shape[1]
            return self

        def predict(
            self,
            X: NDArray[number],
        ) -> NDArray[number]:
            if self._n_features_fit is None:
                raise ValueError("Predictor must be fit before calling predict.")
            indexer = np.arange(self._n_features_fit, dtype=intp)
            return self._predictor.predict(X, indexer)

    return _ProtocolAdapter(predictor_protocol)


def _adapt_protocol_to_protocol_n(
    predictor_protocol: PredictorProtocol,
) -> PredictorProtocolN:
    """Adapt a legacy PredictorProtocol to the ProtocolN interface."""

    class _ProtocolNAdapter:
        def __init__(self, predictor: PredictorProtocol) -> None:
            self._predictor = predictor
            self._fit_targets: NDArray[number] | None = None

        def _set_fit_targets(self, y: NDArray[number]) -> None:
            self._fit_targets = y

        def fit(
            self,
            X: NDArray[number],
        ) -> "_ProtocolNAdapter":
            if self._fit_targets is None:
                raise ValueError("Targets must be provided before calling fit.")
            self._predictor = self._predictor.fit(X, self._fit_targets)
            return self

        def predict(
            self,
            X: NDArray[number],
            indexer: Sequence[int] | NDArray[intp],
        ) -> NDArray[number]:
            indexer_arr = np.asarray(indexer, dtype=intp)
            return self._predictor.predict(X[:, indexer_arr])

    return _ProtocolNAdapter(predictor_protocol)
