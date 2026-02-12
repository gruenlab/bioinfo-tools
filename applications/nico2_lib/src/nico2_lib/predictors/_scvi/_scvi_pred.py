from dataclasses import dataclass
from typing import Optional

import anndata as ad
import kneed as kn
import numpy as np
import scanpy as sc
import sklearn as sk
import torch
from anndata.typing import AnnData
from numpy import intp, number
from numpy.typing import NDArray
from scvi import REGISTRY_KEYS
from torch.distributions import NegativeBinomial

from nico2_lib.predictors._scvi._scvi import SCVI


@dataclass(frozen=True)
class ScviPredictor:
    n_factors: Optional[int] = None
    adata_reference: Optional[AnnData] = None

    def fit(self, X: NDArray[number]) -> "ScviPredictor":
        adata_reference = ad.AnnData(X)
        if self.n_factors is None:
            adata_reference, n_factors = _find_components(adata_reference)
        else:
            n_factors = self.n_factors
        return ScviPredictor(n_factors=n_factors, adata_reference=adata_reference)

    def predict(
        self, X: NDArray[number], indexer: NDArray[intp]
    ) -> tuple[NDArray[number], NDArray[number]]:
        if self.adata_reference is None or self.n_factors is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n_query_features = X_arr.shape[1]
        n_reference_features = self.adata_reference.n_vars
        indexer_valid = _validate_indexer(
            indexer=indexer,
            n_reference_features=n_reference_features,
            n_query_features=n_query_features,
        )

        adata_reference = self.adata_reference.copy()
        model = _train_scvi(
            adata=adata_reference,
            indexer=indexer_valid,
            n_factors=self.n_factors,
        )

        adata_query = _build_query_anndata(
            X=X_arr,
            indexer=indexer_valid,
            adata_reference=self.adata_reference,
        )
        predicted_counts, embeddings = _get_reconstruction_and_embeddings(
            model=model, adata_query=adata_query
        )

        scale = _get_global_scaling_factor(self.adata_reference, indexer_valid)
        predicted_counts = predicted_counts * scale
        predicted_counts = np.nan_to_num(predicted_counts)
        embeddings = np.nan_to_num(embeddings)
        return embeddings, predicted_counts


def _validate_indexer(
    indexer: NDArray[intp],
    n_reference_features: int,
    n_query_features: int,
) -> NDArray[intp]:
    indexer_arr = np.asarray(indexer)

    if indexer_arr.ndim != 1:
        raise ValueError("indexer must be a 1D array.")
    if not np.issubdtype(indexer_arr.dtype, np.integer):
        raise TypeError("indexer must contain integer feature indices.")
    if indexer_arr.shape[0] != n_query_features:
        raise ValueError(
            "indexer length must match the number of columns in X. "
            f"Got {indexer_arr.shape[0]} and {n_query_features}."
        )

    indexer_int = indexer_arr.astype(np.intp, copy=False)
    if np.any(indexer_int < 0) or np.any(indexer_int >= n_reference_features):
        raise ValueError(
            "indexer entries must be within [0, n_reference_features). "
            f"Got n_reference_features={n_reference_features}."
        )
    if np.unique(indexer_int).size != indexer_int.size:
        raise ValueError("indexer contains duplicate reference feature indices.")

    return indexer_int


def _train_scvi(adata: AnnData, indexer: NDArray[intp], n_factors: int) -> SCVI:
    SCVI.setup_anndata(adata)
    model = SCVI(
        adata=adata,
        idx=indexer.tolist(),
        n_latent=n_factors,
        beta=1,
        C=0,
    )
    model.train(max_epochs=200)
    return model


def _build_query_anndata(
    X: NDArray[number], indexer: NDArray[intp], adata_reference: AnnData
) -> AnnData:
    n_query_cells = X.shape[0]
    n_reference_features = adata_reference.n_vars
    query_full = np.zeros((n_query_cells, n_reference_features), dtype=np.float32)
    query_full[:, indexer] = np.asarray(X, dtype=np.float32)
    adata_query = ad.AnnData(X=query_full)
    adata_query.var_names = [str(name) for name in adata_reference.var_names]
    return adata_query


@torch.inference_mode()
def _get_reconstruction_and_embeddings(
    model: SCVI, adata_query: AnnData
) -> tuple[NDArray[number], NDArray[number]]:
    if model.is_trained_ is False:
        raise RuntimeError("Please train the model first.")

    adata_valid = model._validate_anndata(adata_query)
    dataloader = model._make_data_loader(adata=adata_valid)

    reconstructions = []
    embeddings = []
    for tensors in dataloader:
        inference_inputs = model.module._get_inference_input(tensors)
        inference_outputs = model.module.inference(**inference_inputs)
        generative_inputs = model.module._get_generative_input(
            tensors, inference_outputs
        )
        generative_outputs = model.module.generative(**generative_inputs)

        theta = generative_outputs["theta"]
        px_rate = generative_outputs["px_rate"]
        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        mean = NegativeBinomial(total_count=theta, logits=nb_logits).mean

        reconstructions.append(mean.cpu())
        embeddings.append(inference_outputs["qz_m"].cpu())

    reconstructed_counts = torch.cat(reconstructions).numpy()
    latent_embeddings = torch.cat(embeddings).numpy()
    return reconstructed_counts, latent_embeddings


def _to_dense_array(X: object) -> NDArray[number]:
    toarray = getattr(X, "toarray", None)
    if callable(toarray):
        return np.asarray(toarray())
    return np.asarray(X)


def _get_global_scaling_factor(
    adata_reference: AnnData, indexer: NDArray[intp]
) -> float:
    reference_counts = _to_dense_array(adata_reference.X).astype(np.float64, copy=False)
    reference_total = reference_counts.sum(axis=1)
    reference_subset = reference_counts[:, indexer].sum(axis=1)

    ratio = np.divide(
        reference_total,
        reference_subset,
        out=np.ones_like(reference_total, dtype=np.float64),
        where=reference_subset != 0,
    )
    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size == 0:
        return 1.0
    return float(finite_ratio.mean())


def _process_data(
    ad: AnnData, ct: str, ct_labels: str, min_counts: int = 3
) -> tuple[AnnData, AnnData]:
    adr = ad[ad.obs[ct_labels] == ct].copy()
    sc.pp.filter_genes(adr, min_counts=min_counts)
    adc = adr.copy()
    sc.pp.normalize_total(adr)
    sc.pp.log1p(adr)
    sc.tl.pca(adr)
    return adr, adc


def _find_components(
    adata: AnnData,
    mink: int = 1,
    maxk: int = 10,
    opt: bool = True,
    nf_init: int = 3,
) -> tuple[AnnData, int]:
    sc.pp.pca(adata)
    if opt:
        sse = []
        for k in range(mink, maxk):
            kmeans = sk.cluster.KMeans(n_clusters=k).fit(adata.obsm["X_pca"])
            sse.append(kmeans.inertia_)
        kln = kn.KneeLocator(
            range(mink, maxk), sse, curve="convex", direction="decreasing"
        )
        if kln.elbow is None:
            nf = nf_init
        else:
            nf = max(kln.elbow, 2)
    else:
        nf = nf_init

    kmeans = sk.cluster.KMeans(n_clusters=nf).fit(adata.obsm["X_pca"])
    labels = kmeans.labels_
    labels = labels.astype("str", copy=False)
    lab_size = []
    for i in set(labels):
        lab_size.append(sum(labels == i))

    while (min(lab_size) < 5) and (nf > 1):
        nf = nf - 1
        kmeans = sk.cluster.KMeans(n_clusters=nf).fit(adata.obsm["X_pca"])
        labels = kmeans.labels_
        labels = labels.astype("str", copy=False)
        lab_size = []
        for i in set(labels):
            lab_size.append(sum(labels == i))

    adata.obs["kmeans"] = labels

    return adata, nf
