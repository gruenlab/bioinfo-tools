from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import non_negative_factorization

from nico2_lib.predictors.utils import preprocess_counts
from nico2_lib.typing import IndexArray, NumericArray


def init_nmf_matrices(
    X: NumericArray,
    n_components: int,
) -> tuple[NumericArray, NumericArray]:
    labels = KMeans(n_clusters=n_components).fit_predict(X)
    adata = sc.AnnData(X)
    adata.obs["kmeans"] = labels
    sc.tl.rank_genes_groups(adata, "kmeans")
    groups_scores = adata.uns["rank_genes_groups"]["scores"]
    groups_names = adata.uns["rank_genes_groups"]["names"]
    gsa = np.array(groups_scores.tolist())
    gna = np.array(groups_names.tolist())
    df = adata.X.transpose()  # type: ignore
    w_init = np.zeros((df.shape[0], n_components), dtype=float)
    for i in range(n_components):
        gsd = pd.DataFrame(gsa.transpose(), columns=gna[:, i])
        w_init[:, i] = gsd[adata.var.index].iloc[i]
        w_init[w_init[:, i] < 0, i] = 0.1
        w_init[:, i] = w_init[:, i] + 0.01
    h_init = np.zeros((n_components, df.shape[1]), dtype=float)
    for i in range(n_components):
        h_init[i, adata.obs["kmeans"].astype("float") == i] = 1
        h_init[i, adata.obs["kmeans"].astype("float") != i] = 0.1
    return w_init, h_init


@dataclass(frozen=True)
class NmfPredictor:
    """NMF-based predictor using ProtocolN (fit on X, predict all fit-time features)."""

    embedding_size: int | None = None
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None
    pre_init: bool = False
    seed: int = 0
    solver: Literal["cd", "mu"] = "cd"
    h_reference: NumericArray | None = None
    n_shared_features: int | None = None
    ref_embedding: NumericArray | None = None

    def fit(self, x: NumericArray) -> "NmfPredictor":
        """Fit NMF on X to learn the reference component matrix.

        Args:
            X: Feature matrix used for fitting, shape (n_samples, n_features_fit).

        Returns:
            The fitted predictor instance.
        """
        x = preprocess_counts(x, self.preprocessing_steps)

        w_init, h_init = (
            init_nmf_matrices(x, self.embedding_size) if self.pre_init else (None, None)
        )
        w_reference, h_reference, _ = non_negative_factorization(
            x,
            n_components=self.embedding_size,
            solver=self.solver,
            W=w_init,
            H=h_init,
        )
        return replace(self, h_reference=h_reference, ref_embedding=w_reference)

    def predict(
        self, x: NumericArray, indexer: IndexArray
    ) -> tuple[NumericArray, NumericArray]:
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
        x = (
            apply_pipeline(x, pipeline=self.preprocessing_steps)
            if self.preprocessing_steps is not None
            else x
        )
        w_query, _, _ = non_negative_factorization(
            X=x, H=self.h_reference[:, indexer], init="custom", update_H=False
        )
        return w_query, w_query @ self.h_reference
