from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import anndata as ad
import mofaflex
import numpy as np
import pandas as pd
from anndata.typing import AnnData
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
    return sliced_weights


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
            save_path=None,
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
            save_path=None,
        )
        h_reference: NumericArray = self._model.get_weights()["view_1"].values.T
        w_query: NumericArray = model_query.get_factors()["group_1"].values
        return w_query, w_query @ h_reference

    @property
    def feature_embeddings(self) -> NumericArray | None:
        assert self._model is not None, "Model not fitted"
        return self._model.get_weights()["view_1"].values.T


def prepare_mofaflex_input(
    query: AnnData,
    reference: AnnData,
) -> dict[str, dict[str, AnnData]]:
    """Returns: Nested dict with group names as keys, view names as subkeys and AnnData objects as values (incompatible with `.group_by`)"""

    target_features = sorted(list(set(reference.var_names).union(set(query.var_names))))

    def reindex_adata(
        adata: AnnData,
        target_features: Sequence[str],
    ) -> AnnData:
        x_df = pd.DataFrame(
            adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray(),  # type: ignore
            index=adata.obs_names,
            columns=adata.var_names,
        )
        x_reindexed = x_df.reindex(columns=target_features, fill_value=np.nan)
        new_var = pd.DataFrame(index=target_features).join(adata.var, how="left")  # type: ignore
        return ad.AnnData(
            X=x_reindexed.values,
            obs=adata.obs.copy(),  # type: ignore
            var=new_var,
        )

    return {
        "group_1": {
            "view_1": reindex_adata(query, target_features),
            "view_2": reindex_adata(reference, target_features),
        }
    }


@dataclass(frozen=True)
class MofaFlexClassicPredictor:
    n_components: int
    max_epochs: int = 200

    _reference_anndata: AnnData | None = None

    def fit(
        self,
        x: NumericArray,
    ) -> "MofaFlexClassicPredictor":
        return replace(
            self,
            _reference_anndata=ad.AnnData(X=x),
        )

    def predict(
        self,
        x: NumericArray,
        indexer: IndexArray,
    ) -> tuple[NumericArray, NumericArray]:
        assert self._reference_anndata is not None, "Reference not fitted"
        data = prepare_mofaflex_input(
            query=ad.AnnData(X=x),
            reference=self._reference_anndata,
        )
        mofaflex_model = mofaflex.terms.MofaFlex(  # type: ignore
            n_factors=self.n_components,
        )
        mofaflex_model.fit(
            data,
            max_epochs=self.max_epochs,
            save_path=None,
        )
        cell_embeddings: NumericArray = (
            mofaflex_model.get_factors()["group_1"].iloc[: x.shape[0]].values
        )
        feature_embeddings: NumericArray = mofaflex_model.get_weights(views="view_1")[
            "view_2"
        ].values.T
        return cell_embeddings, cell_embeddings @ feature_embeddings

    @property
    def feature_embedding(self) -> NumericArray | None:
        return None

    @property
    def embedding_size(self) -> int | None:
        return None

    @property
    def preprocessing_steps(
        self,
    ) -> Sequence[Callable[[NumericArray], NumericArray]] | None:
        return None
