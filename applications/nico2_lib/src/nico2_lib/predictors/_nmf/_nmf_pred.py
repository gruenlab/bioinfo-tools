from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from nico2_lib.predictors.utils import preprocess_counts
from nico2_lib.typing import IndexArray, NumericArray
from numpy.random import RandomState
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, non_negative_factorization


def init_nmf_matrices(
    X: NumericArray,
    n_components: int,
) -> tuple[NumericArray, NumericArray]:
    n_obs, n_vars = X.shape
    labels = KMeans(n_clusters=n_components).fit_predict(X)
    adata = sc.AnnData(X)
    adata.obs["kmeans"] = pd.Series(labels).astype(str).astype("category").values
    sc.tl.rank_genes_groups(adata, "kmeans")
    groups_scores = adata.uns["rank_genes_groups"]["scores"]

    h_init = np.zeros((n_components, n_vars), dtype=np.float64)
    for i in range(n_components):
        scores = groups_scores[str(i)]
        h_init[i, :] = scores
    h_init[h_init < 0] = 0.1
    h_init += 0.01
    w_init = np.zeros((n_obs, n_components), dtype=np.float64)
    for i in range(n_components):
        mask = labels == i
        w_init[mask, i] = 1.0
        w_init[~mask, i] = 0.1

    return w_init, h_init


def robust_init_nmf_matrices(X, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    n_obs, n_vars = X.shape

    kmeans = KMeans(n_clusters=n_components, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    counts = pd.Series(labels).value_counts()
    if counts.min() < 2:
        h_init = np.full((n_components, n_vars), 0.1)
        for i in range(n_components):
            mask = labels == i
            if mask.any():
                # Use the mean profile of the cluster if possible
                h_init[i, :] = np.array(X[mask].mean(axis=0)).flatten()
    else:
        adata = sc.AnnData(X.astype(np.float64))
        adata.obs["kmeans"] = pd.Series(labels).astype(str).astype("category").values

        try:
            sc.tl.rank_genes_groups(adata, "kmeans", method="t-test")
            h_init = np.zeros((n_components, n_vars))
            for i in range(n_components):
                scores_df = sc.get.rank_genes_groups_df(adata, group=str(i))
                scores_df = scores_df.set_index("names").reindex(adata.var_names)
                h_init[i, :] = scores_df["scores"].values
        except Exception:
            h_init = np.random.rand(n_components, n_vars) * 0.1

    h_init = np.nan_to_num(h_init, nan=0.1)
    h_init[h_init < 0] = 0.1
    h_init += 0.01

    w_init = np.full((n_obs, n_components), 0.1)
    for i in range(n_components):
        w_init[labels == i, i] = 1.0

    return w_init, h_init


@dataclass(frozen=True)
class NmfPredictor:
    """NMF-based predictor using ProtocolN (fit on X, predict all fit-time features)."""

    embedding_size: int | None = None
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"] | None = None
    random_state: int | RandomState | None = None
    max_iter: int = 200
    alpha_W: float = 0.0
    alpha_H: float | Literal["same"] = "same"
    l1_ratio: float = 0.0
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None
    pre_init: bool = False
    solver: Literal["cd", "mu"] = "cd"
    beta_loss: str = "frobenius"
    max_iter: int = 1000
    init: str | None = None
    alpha_W: float = 0.0
    alpha_H: float = 0.0
    l1_ratio: float = 0.0
    h_reference: NumericArray | None = None
    n_shared_features: int | None = None
    ref_embedding: NumericArray | None = None

    def fit(self, x: NumericArray) -> "NmfPredictor":
        x = preprocess_counts(x, self.preprocessing_steps)
        w_init, h_init = (None, None)
        if self.pre_init and self.embedding_size is not None:
            w_init, h_init = robust_init_nmf_matrices(x, self.embedding_size)

        model = NMF(
            n_components=self.embedding_size or 3,
            init="custom" if w_init is not None else "nndsvd",
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            beta_loss="frobenius",
        )

        if w_init is not None and h_init is not None:
            w_reference = model.fit_transform(
                x,
                W=w_init,
                H=h_init,
            )
        else:
            w_reference = model.fit_transform(x)

        h_reference = model.components_

        return replace(
            self,
            h_reference=h_reference,
            ref_embedding=w_reference,
        )

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
        if self.preprocessing_steps is not None:
            for step in self.preprocessing_steps:
                x = step(x)
        w_query, _, _ = non_negative_factorization(
            X=x, H=self.h_reference[:, indexer], init="custom", update_H=False,
            n_components=self.embedding_size, max_iter=self.max_iter,
        )
        return w_query, w_query @ self.h_reference

    @property
    def feature_embedding(self) -> NumericArray | None:
        """Returns the feature embedding matrix."""
        assert self.h_reference is not None, (
            "Embedding not available; fit must be called first."
        )
        return self.h_reference
