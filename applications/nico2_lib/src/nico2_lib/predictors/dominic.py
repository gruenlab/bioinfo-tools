import random
import statistics
from typing import List, Tuple

import kneed as kn
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from numpy import number
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn import cluster as sk_cluster
from sklearn import decomposition as sk_decomposition


def _to_1d_dense(values) -> np.ndarray:
    """Convert dense or sparse inputs to a flat NumPy array.

    Args:
        values (array-like or sparse matrix): Vector-like data that may be
            sparse and needs to be densified.

    Returns:
        np.ndarray: 1D dense array of shape ``(n,)``.
    """
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).ravel()


def _init_nmf_matrices(
    rank_scores_array: NDArray[number],
    rank_names_array: NDArray[number],
    expression_matrix: NDArray[number],
    gene_index,
    kmeans_labels,
    nf: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic NMF initialization matrices from ranked genes.

    Args:
        rank_scores_array (np.ndarray): Ranked gene scores (e.g., from Scanpy
            ``rank_genes_groups``) with rows as genes and columns as clusters.
        rank_names_array (np.ndarray): Ranked gene names aligned with
            ``rank_scores_array``; columns correspond to clusters.
        expression_matrix (np.ndarray): Gene-by-cell matrix used for sizing
            the initialization matrices.
        gene_index (pandas.Index or sequence): Gene index aligning names to the
            expression matrix (typically ``AnnData.var.index``).
        kmeans_labels (array-like): Cluster labels per cell used to seed ``H``.
        nf (int): Number of NMF components.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(w_init, h_init)`` where ``w_init`` has
        shape ``(n_genes, nf)`` and ``h_init`` has shape ``(nf, n_cells)``.
    """
    w_init = np.zeros((expression_matrix.shape[0], nf), dtype=float)
    for i in range(nf):
        # Deterministic iteration over genes
        scores_df = pd.DataFrame(
            rank_scores_array.transpose(), columns=rank_names_array[:, i]
        )
        w_init[:, i] = scores_df[gene_index].iloc[i]
        w_init[w_init[:, i] < 0, i] = 0.1
        w_init[:, i] += 0.01

    h_init = np.zeros((nf, expression_matrix.shape[1]), dtype=float)
    for i in range(nf):
        cluster_mask = np.asarray(kmeans_labels, dtype=float) == i
        h_init[i, cluster_mask] = 1
        h_init[i, ~cluster_mask] = 0.1

    return w_init, h_init


# -----------------------------
# Helper: deterministic component finder
# -----------------------------
def _find_components(
    embedding: NDArray[number],
    mink: int = 1,
    maxk: int = 10,
    opt: bool = True,
    nf_init: int = 3,
    seed: int = 42,
) -> tuple[int, np.ndarray]:
    """Choose a component count and KMeans labels for an embedding.

    Uses an elbow heuristic on KMeans SSE when ``opt`` is True, then enforces
    a minimum cluster size by reducing the component count as needed.

    Args:
        embedding (np.ndarray): PCA embedding array of shape
            ``(n_samples, n_components)``.
        mink (int): Minimum number of clusters to try.
        maxk (int): Maximum number of clusters to try (exclusive).
        opt (bool): Whether to optimize the component count using the elbow
            method.
        nf_init (int): Default component count when optimization is disabled or
            inconclusive.
        seed (int): Random seed for deterministic KMeans fitting.

    Returns:
        tuple[int, np.ndarray]: ``(nf, labels)`` where ``labels`` is a string
        array of shape ``(n_samples,)``.
    """
    n_samples, _ = embedding.shape
    if n_samples < 1:
        return 1, np.array([], dtype=str)

    max_clusters_exclusive = min(maxk, n_samples + 1)
    min_clusters = min(mink, n_samples)

    def fit_kmeans(n_clusters: int) -> sk_cluster.KMeans:
        return sk_cluster.KMeans(n_clusters=n_clusters, random_state=seed).fit(
            embedding
        )

    if opt:
        cluster_range = range(min_clusters, max_clusters_exclusive)
        inertias: List[sk_cluster.KMeans] = [
            fit_kmeans(k).inertia_ for k in cluster_range
        ]

        elbow_finder = kn.KneeLocator(
            cluster_range, inertias, curve="convex", direction="decreasing"
        )
        elbow = elbow_finder.elbow
        nf = max(elbow, 2) if elbow is not None else nf_init
    else:
        nf = nf_init

    nf = min(nf, n_samples)

    def labels_and_sizes(n_clusters: int) -> tuple[np.ndarray, List[int]]:
        labels = fit_kmeans(n_clusters).labels_.astype(str)
        ordered_labels = np.unique(labels)  # deterministic ordering
        sizes = [np.sum(labels == label) for label in ordered_labels]
        return labels, sizes

    # Iteratively reduce nf until clusters have at least 5 samples.
    while True:
        labels, label_sizes = labels_and_sizes(nf)
        if min(label_sizes) >= 5 or nf <= 1:
            break
        nf -= 1

    return nf, labels


def _prepare_cluster_nmf(
    ad_sp: AnnData,
    ad: AnnData,
    cluster_id,
    seed: int,
    init: bool,
    opt_nf: bool,
    nf_init: int,
) -> tuple[
    int,
    List[int],
    NDArray[number],
    NDArray[number],
    NDArray[number],
]:
    """Preprocess a cluster and fit NMF models for shared genes.

    Args:
        ad_sp (AnnData): Spatial AnnData with expression and ``cluster`` labels.
        ad (AnnData): scRNA-seq AnnData with expression and ``cluster`` labels.
        cluster_id (object): Cluster label to subset from ``ad`` and ``ad_sp``.
        seed (int): Random seed for deterministic NMF fitting.
        init (bool): Whether to use deterministic initialization from ranked
            genes.
        opt_nf (bool): Whether to optimize the number of components via elbow
            detection.
        nf_init (int): Default number of components when optimization is off.

    Returns:
        tuple[int, list[int], np.ndarray, np.ndarray, np.ndarray]:
        ``(nf, cluster_sizes, spatial_w, rna_shared_w, spatial_expression)``.
        ``spatial_w`` has shape ``(n_shared_spatial_genes, nf)``,
        ``rna_shared_w`` has shape ``(n_shared_spatial_genes, nf)``, and
        ``spatial_expression`` is gene-by-cell with shape
        ``(n_shared_spatial_genes, n_cells)``. ``cluster_sizes`` is populated
        only when ``init`` is True.
    """
    cluster_ad = ad[ad.obs["cluster"] == cluster_id].copy()
    cluster_ad = sc.pp.filter_genes(cluster_ad, min_counts=1, copy=True)
    cluster_ad = sc.pp.normalize_total(cluster_ad, copy=True)
    cluster_ad = sc.pp.log1p(cluster_ad, copy=True)
    cluster_ad = sc.tl.pca(cluster_ad, copy=True)

    expression_matrix = cluster_ad.X.transpose()
    shared_spatial_genes_mask = np.isin(
        cluster_ad.var.index.to_list(), ad_sp.var.index.to_list()
    )
    shared_genes_mask = np.isin(cluster_ad.var.index.to_list(), ad.var.index.to_list())

    nf, kmeans_labels = _find_components(
        cluster_ad.obsm["X_pca"], 1, 10, opt=opt_nf, nf_init=nf_init, seed=seed
    )
    cluster_ad.obs["kmeans"] = kmeans_labels

    if not init:
        nf = nf_init

    cluster_sizes: List[int] = []
    if init:
        cluster_sizes = [sum(kmeans_labels == l) for l in sorted(set(kmeans_labels))]
        cluster_ad = sc.tl.rank_genes_groups(cluster_ad, "kmeans", copy=True)

        w_init, h_init = _init_nmf_matrices(
            rank_scores_array=np.array(
                cluster_ad.uns["rank_genes_groups"]["scores"].tolist()
            ),
            rank_names_array=np.array(
                cluster_ad.uns["rank_genes_groups"]["names"].tolist()
            ),
            expression_matrix=cluster_ad.X.transpose(),
            gene_index=cluster_ad.var.index,
            kmeans_labels=cluster_ad.obs["kmeans"],
            nf=nf,
        )
        nmf_model_rna = sk_decomposition.NMF(
            n_components=nf,
            init="custom",
            alpha_W=0,
            alpha_H=0,
            l1_ratio=1.0,
            random_state=seed,
        )
        nmf_model_spatial = sk_decomposition.NMF(
            n_components=nf,
            init="custom",
            alpha_W=0,
            alpha_H=0,
            l1_ratio=1.0,
            random_state=seed,
        )
        rna_w = nmf_model_rna.fit_transform(
            expression_matrix[shared_genes_mask, :],
            W=w_init[shared_genes_mask, :].astype("float32"),
            H=h_init.astype("float32"),
        )
        spatial_w = nmf_model_spatial.fit_transform(
            expression_matrix[shared_spatial_genes_mask, :],
            W=w_init[shared_spatial_genes_mask, :].astype("float32"),
            H=h_init.astype("float32"),
        )
    else:
        nmf_model_rna = sk_decomposition.NMF(
            n_components=nf, alpha_W=0, alpha_H=0, l1_ratio=1.0, random_state=seed
        )
        nmf_model_spatial = sk_decomposition.NMF(
            n_components=nf, alpha_W=0, alpha_H=0, l1_ratio=1.0, random_state=seed
        )
        rna_w = nmf_model_rna.fit_transform(expression_matrix[shared_genes_mask, :])
        spatial_w = nmf_model_spatial.fit_transform(
            expression_matrix[shared_spatial_genes_mask, :]
        )

    shared_spatial_mask = np.isin(
        cluster_ad.var.index.to_list(), ad_sp.var.index.to_list()
    )
    shared_spatial_genes = cluster_ad.var.index[shared_spatial_mask].to_list()
    spatial_cluster = ad_sp[ad_sp.obs["cluster"] == cluster_id, shared_spatial_genes]
    spatial_expression = spatial_cluster.X.transpose()
    rna_shared_w = rna_w[shared_spatial_genes_mask, :]

    return nf, cluster_sizes, spatial_w, rna_shared_w, spatial_expression


def _predict_expression_nmf(
    rna_shared_w: NDArray[number],
    spatial_expression: NDArray[number],
    train_gene_idx: NDArray[number],
    test_gene_idx: NDArray[number],
    max_iter_nmf: int,
    eps: float = 1e-8,
) -> NDArray[number]:
    """Estimate spatial gene expression using NMF multiplicative updates.

    Fits the NMF H matrix by iterating multiplicative updates over the training
    genes, then projects the held-out genes using the learned H.

    Args:
        rna_shared_w (np.ndarray): W matrix for shared genes from the RNA NMF
            model with shape ``(n_shared_genes, n_components)``.
        spatial_expression (np.ndarray): Spatial gene-by-cell matrix for shared
            genes with shape ``(n_shared_genes, n_cells)``.
        train_gene_idx (np.ndarray): Indices of genes used to estimate ``H``.
        test_gene_idx (np.ndarray): Indices of genes to predict after fitting
            ``H``.
        max_iter_nmf (int): Number of multiplicative update iterations for
            estimating ``H``.
        eps (float): Small constant to avoid divide-by-zero.

    Returns:
        np.ndarray: Predicted expression for held-out genes with shape
        ``(n_test_genes, n_cells)``.
    """
    train_w = rna_shared_w[train_gene_idx, :]
    h_est = np.ones((rna_shared_w.shape[1], spatial_expression.shape[1]))

    for _ in range(max_iter_nmf):
        h_est *= (train_w.T @ spatial_expression[train_gene_idx, :]) / (
            train_w.T @ train_w @ h_est + eps
        )

    return rna_shared_w[test_gene_idx, :] @ h_est


def _collect_correlations(
    predicted_expression: NDArray[number],
    spatial_expression: NDArray[number],
    test_gene_idx: NDArray[number],
) -> tuple[List[float], List[float]]:
    """Compute Spearman and Pearson correlations per cell.

    Args:
        predicted_expression (np.ndarray): Predicted gene-by-cell matrix for
            held-out genes with shape ``(n_test_genes, n_cells)``.
        spatial_expression (np.ndarray): Observed gene-by-cell matrix for shared
            genes with shape ``(n_shared_genes, n_cells)``.
        test_gene_idx (np.ndarray): Indices of held-out genes in
            ``spatial_expression``.

    Returns:
        tuple[list[float], list[float]]: Spearman and Pearson correlation values
        per cell, skipping cells with all-zero observed expression.
    """
    cluster_spearman: List[float] = []
    cluster_pearson: List[float] = []

    for cell_idx in range(predicted_expression.shape[1]):
        predicted_vector = np.nan_to_num(predicted_expression)[:, cell_idx]
        observed_vector = _to_1d_dense(spatial_expression[test_gene_idx, cell_idx])
        if np.max(observed_vector) == 0:
            continue
        cluster_spearman.append(spearmanr(predicted_vector, observed_vector)[0])
        cluster_pearson.append(pearsonr(predicted_vector, observed_vector)[0])

    return cluster_spearman, cluster_pearson


def dominic_experiment_refactor_v1(
    ad_sp: AnnData,
    ad: AnnData,
    seed: int = 42,
    init=True,
    opt_nf=True,
    nf_init=3,
    test_genes=20,
    max_iter_nmf=100,
    gene_samples=10,
) -> Tuple[List[float], List[float]]:
    """Run the Dominic NMF-based spatial prediction experiment.

    This routine iterates over clusters in the scRNA-seq AnnData, fits NMF
    models with deterministic initialization, and evaluates prediction quality
    on spatial data using Spearman and Pearson correlations.

    Args:
        ad_sp (AnnData): Spatial AnnData with gene expression and cluster labels.
        ad (AnnData): scRNA-seq AnnData with gene expression and cluster labels.
        seed (int): Random seed for deterministic behavior across NumPy and
            Python RNGs.
        init (bool): Whether to use custom deterministic NMF initialization.
        opt_nf (bool): Whether to optimize the number of NMF components via
            elbow detection.
        nf_init (int): Default number of NMF components when optimization is disabled
            or fails to find an elbow.
        test_genes (int): Number of genes held out for each prediction test.
        max_iter_nmf (int): Number of multiplicative update iterations for
            estimating ``H``.
        gene_samples (int): Number of random gene-subsampling trials per cluster.

    Returns:
        tuple[list[float], list[float]]: ``(SP, PE)`` lists of Spearman and
        Pearson correlation values across all clusters, cells, and subsamples.
    """
    # -----------------------------
    # 1. Setup deterministic RNGs
    # -----------------------------
    rng = np.random.default_rng(
        seed
    )  # NumPy RNG for all deterministic random operations
    random.seed(seed)  # Seed Python's random
    np.random.seed(seed)  # Seed NumPy legacy functions

    # -----------------------------
    # Helper: deterministic NMF initialization
    # -----------------------------

    spearman_scores = []
    pearson_scores = []

    for cluster_id in sorted(set(ad.obs["cluster"])):
        (
            nf,
            cluster_sizes,
            spatial_w,
            rna_shared_w,
            spatial_expression,
        ) = _prepare_cluster_nmf(
            ad_sp=ad_sp,
            ad=ad,
            cluster_id=cluster_id,
            seed=seed,
            init=init,
            opt_nf=opt_nf,
            nf_init=nf_init,
        )

        print(cluster_id, "Optimal nf:", nf)

        if init:
            print(cluster_id, "Cluster sizes:", cluster_sizes)

        cluster_spearman: List[float] = []
        cluster_pearson: List[float] = []

        for _ in range(gene_samples):
            train_gene_idx = rng.choice(
                np.arange(1, spatial_w.shape[0]),
                spatial_w.shape[0] - test_genes,
                replace=False,
            )
            test_gene_idx = np.setdiff1d(np.arange(spatial_w.shape[0]), train_gene_idx)

            predicted_expression = _predict_expression_nmf(
                rna_shared_w,
                spatial_expression,
                train_gene_idx,
                test_gene_idx,
                max_iter_nmf,
            )
            sp_vals, pe_vals = _collect_correlations(
                predicted_expression, spatial_expression, test_gene_idx
            )
            cluster_spearman.extend(sp_vals)
            cluster_pearson.extend(pe_vals)

        spearman_scores.extend(cluster_spearman)
        pearson_scores.extend(cluster_pearson)

        print(cluster_id, "Spearman median:", statistics.median(cluster_spearman))
        print(
            cluster_id, "Pearson median:", statistics.median(cluster_pearson), end="\n"
        )

    return spearman_scores, pearson_scores
