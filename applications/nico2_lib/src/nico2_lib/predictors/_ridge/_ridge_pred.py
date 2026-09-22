"""
Ridge-regression predictor following the PredictorProtocol.
Uses sklearn.linear_model.Ridge/RidgeCV to map probe-gene expression to
full-transcriptome expression via a standardised multioutput linear regression.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Optional

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from nico2_lib.predictors.utils import preprocess_counts
from nico2_lib.typing import IndexArray, NumericArray

_DEFAULT_ALPHA_VALUES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class RidgePredictor:
    """Ridge-regression predictor conforming to PredictorProtocol.

    Unlike ``PcaPredictor``/``FastIcaPredictor``/``NmfPredictor`` -- which fit one
    indexer-independent basis once on the full gene set, then cheaply evaluate *any*
    later probe-gene subset by column-selecting that fixed basis -- ``RidgePredictor``
    cannot do this, for a genuine mathematical reason (contrast with
    ``ScviPredictor``, where the equivalent limitation is an implementation choice,
    not a requirement of the method): a ridge regression's coefficients map
    specifically from a chosen set of predictor (probe) genes to the full
    transcriptome, so they are only defined once those predictor genes are known.
    There is no "fit on everything, subset later" formulation for a supervised
    regression.

    Consequently:
    - ``fit(x)`` only stores the reference data (after preprocessing) -- it does not
      fit any regression yet, since the predictor (probe) genes aren't known at
      ``fit()`` time.
    - ``predict(x, indexer)`` fits a fresh ``StandardScaler`` + ``Ridge``/``RidgeCV``
      regressor from ``self._reference[:, indexer]`` (the probe-gene columns of the
      fit-time reference data) to ``self._reference`` (all genes), every single call
      -- even two calls with the same ``indexer`` refit independently. This is cheap
      for a linear model (unlike ``ScviPredictor``'s VAE retraining), but the same
      architectural pattern: no cached/reusable model persists between calls.
    - ``embedding_size``/``feature_embedding`` always return ``None``: there is no
      low-dimensional embedding or fixed, indexer-independent gene-loading matrix to
      expose here -- the model *is* the probe-to-full mapping, and it is specific to
      whichever ``indexer`` the most recent ``predict()`` call used.

    Args:
        alpha: Ridge regularisation strength. Ignored when ``use_ridgecv`` is True.
        use_ridgecv: If True, select alpha via cross-validation from ``alpha_values``
            instead of fitting with a single fixed ``alpha``.
        alpha_values: Candidate alphas for ``RidgeCV``. Defaults to
            ``(0.01, 0.1, 1.0, 10.0, 100.0)`` when None.
        cv_n_splits: Number of folds for ``RepeatedKFold`` when ``use_ridgecv`` is True.
        cv_n_repeats: Number of repeats for ``RepeatedKFold`` when ``use_ridgecv`` is True.
        random_state: Random seed passed to ``RepeatedKFold`` when ``use_ridgecv`` is True.
        preprocessing_steps: Optional preprocessing pipeline.
    """

    alpha: float = 1.0
    use_ridgecv: bool = False
    alpha_values: Sequence[float] | None = None
    cv_n_splits: int = 5
    cv_n_repeats: int = 3
    random_state: Optional[int] = 42
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None

    _reference: NumericArray | None = None

    def fit(self, x: NumericArray) -> "RidgePredictor":
        x = preprocess_counts(x, pipeline=self.preprocessing_steps)
        return replace(self, _reference=x)

    def predict(
        self, x: NumericArray, indexer: IndexArray
    ) -> tuple[NumericArray, NumericArray]:
        assert self._reference is not None, "fit must be called before predict"

        x = preprocess_counts(x, pipeline=self.preprocessing_steps)

        X_train = self._reference[:, indexer]
        y_train = self._reference

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        if self.use_ridgecv:
            alpha_values = (
                tuple(self.alpha_values) if self.alpha_values is not None else _DEFAULT_ALPHA_VALUES
            )
            cv = RepeatedKFold(
                n_splits=self.cv_n_splits,
                n_repeats=self.cv_n_repeats,
                random_state=self.random_state,
            )
            reg = RidgeCV(alphas=list(alpha_values), cv=cv)
        else:
            reg = Ridge(alpha=self.alpha)
        reg.fit(X_train_scaled, y_train)

        # Not a low-dimensional embedding -- see class docstring. Returned for
        # PredictorProtocol shape-parity; it is the standardised probe-gene input
        # actually used as the regression input for this call.
        cell_embeddings = scaler.transform(x)
        full_reconstruction = reg.predict(cell_embeddings)

        return cell_embeddings, full_reconstruction

    @property
    def embedding_size(self) -> int | None:
        return None

    @property
    def feature_embedding(self) -> NumericArray | None:
        return None
