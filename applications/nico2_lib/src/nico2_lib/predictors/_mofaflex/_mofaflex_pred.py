from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import anndata as ad
import mofaflex
import pandas as pd
from mofaflex import priors

from nico2_lib.predictors.utils import preprocess_counts
from nico2_lib.typing import IndexArray, NumericArray


def slice_mofaflex_weights(
    weights: dict[str, pd.DataFrame],
    indexer: IndexArray,
) -> dict[str, pd.DataFrame]:
    sliced_weights: dict[str, pd.DataFrame] = {}
    for key, df in weights.items():
        sliced_weights[key] = df.iloc[indexer]
    return weights


def slice_mofaflex_model(
    model: mofaflex.MOFAFLEX,
    indexer: IndexArray,
) -> mofaflex.MOFAFLEX:
    weights: dict[str, pd.DataFrame] = model.get_weights()
    sliced_weights = slice_mofaflex_weights(
        weights=weights,
        indexer=indexer,
    )
    weight_prior = priors.Constant(
        const_values=sliced_weights,
    )
    return mofaflex.terms.MofaFlex(
        n_factors=model.n_factors,
        weight_prior=weight_prior,
    )


@dataclass(frozen=True)
class MofaFlexPredictor:
    embedding_size: int | None
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None
    seed: int | None = None

    _model: mofaflex.MOFAFLEX | None = None

    def fit(
        self,
        x: NumericArray,
    ) -> "MofaFlexPredictor":
        x = preprocess_counts(x, self.preprocessing_steps)
        model: mofaflex.MOFAFLEX = mofaflex.terms.MofaFlex(  # type: ignore
            n_factors=self.embedding_size,
        )
        model.fit(  # type: ignore
            ad.AnnData(X=x),
            save_path=False,
        )

        return replace(self, _model=model)

    def predict(
        self,
        x: NumericArray,
        indexer: IndexArray,
    ) -> tuple[NumericArray, NumericArray]:
        assert self._model is not None, "Model not fitted"
        x = preprocess_counts(x, self.preprocessing_steps)
        model_query = slice_mofaflex_model(
            model=self._model,
            indexer=indexer,
        )
        model_query.fit(
            ad.AnnData(X=x),
            save_path=False,
        )
        h_reference: NumericArray = self._model.get_weights()["view_1"].values.T
        w_query: NumericArray = model_query.get_factors()["group_1"].values
        return w_query, w_query @ h_reference
