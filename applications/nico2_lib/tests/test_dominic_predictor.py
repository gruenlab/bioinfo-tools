import random
import statistics
from typing import List, Tuple

import kneed as kn
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import read_h5ad
from anndata.typing import AnnData
from joblib import Memory
from scipy.stats import pearsonr, spearmanr
from sklearn import cluster as sk_cluster
from sklearn import decomposition as sk_decomposition
from sklearn import metrics as sk_metrics

from nico2_lib.predictors.dominic import _to_1d_dense, dominic_experiment_refactor_v1

memory = Memory("./tests/cache")


@memory.cache
def dominic_experiment(
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
        ad_sp: Spatial AnnData with gene expression and cluster labels.
        ad: scRNA-seq AnnData with gene expression and cluster labels.
        seed: Random seed for deterministic behavior across NumPy and Python RNGs.
        init: Whether to use custom deterministic NMF initialization.
        opt_nf: Whether to optimize the number of NMF components via elbow detection.
        nf_init: Default number of NMF components when optimization is disabled
            or fails to find an elbow.
        test_genes: Number of genes held out for each prediction test.
        max_iter_nmf: Number of multiplicative update iterations for H estimation.
        gene_samples: Number of random gene-subsampling trials per cluster.

    Returns:
        A tuple containing:
            - SP: List of Spearman correlation values across all tests.
            - PE: List of Pearson correlation values across all tests.
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
    def init_nmf_matrices(
        adata: AnnData, nf: int, ngenes: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize NMF W/H matrices deterministically from ranked genes.

        Args:
            adata: AnnData with a "kmeans" cluster annotation in `obs`.
            nf: Number of NMF components.
            ngenes: Number of top-ranked genes to consider.

        Returns:
            A tuple containing:
                - Winit: Initialized W matrix of shape (n_genes, nf).
                - Hinit: Initialized H matrix of shape (nf, n_cells).
        """
        sc.tl.rank_genes_groups(adata, "kmeans")
        gs = adata.uns["rank_genes_groups"]["scores"]
        gn = adata.uns["rank_genes_groups"]["names"]
        gsa = np.array(gs.tolist())
        gna = np.array(gn.tolist())
        df = adata.X.transpose()

        Winit = np.zeros((df.shape[0], nf), dtype=float)
        for i in range(nf):
            # Deterministic iteration over genes
            gsd = pd.DataFrame(gsa.transpose(), columns=gna[:, i])
            Winit[:, i] = gsd[adata.var.index].iloc[i]
            Winit[Winit[:, i] < 0, i] = 0.1
            Winit[:, i] += 0.01

        Hinit = np.zeros((nf, df.shape[1]), dtype=float)
        for i in range(nf):
            mask = adata.obs["kmeans"].astype("float") == i
            Hinit[i, mask] = 1
            Hinit[i, ~mask] = 0.1

        return Winit, Hinit

    # -----------------------------
    # Helper: deterministic component finder
    # -----------------------------
    def find_components(
        adata: AnnData,
        mink: int = 1,
        maxk: int = 10,
        opt: bool = True,
        nf_init: int = 3,
    ) -> tuple[int, np.ndarray]:
        """Choose a component count and KMeans labels for an AnnData object.

        Uses an elbow heuristic on KMeans SSE when `opt` is True and then enforces
        a minimum cluster size by reducing the component count as needed.

        Args:
            adata: AnnData with PCA embedding stored in `obsm["X_pca"]`.
            mink: Minimum number of clusters to try.
            maxk: Maximum number of clusters to try (exclusive).
            opt: Whether to optimize the component count using the elbow method.
            nf_init: Default component count when optimization is disabled or
                inconclusive.

        Returns:
            A tuple containing:
                - nf: Selected number of components.
                - labels: KMeans labels as a NumPy array of strings.
        """
        n_samples = adata.obsm["X_pca"].shape[0]
        if n_samples < 1:
            return 1, np.array([], dtype=str)

        maxk = min(maxk, n_samples + 1)
        mink = min(mink, n_samples)

        if opt:
            sse = []
            for k in range(mink, maxk):
                kmeans = sk_cluster.KMeans(n_clusters=k, random_state=seed).fit(
                    adata.obsm["X_pca"]
                )
                sse.append(kmeans.inertia_)

            kln = kn.KneeLocator(
                range(mink, maxk), sse, curve="convex", direction="decreasing"
            )
            nf = max(kln.elbow, 2) if kln.elbow is not None else nf_init
        else:
            nf = nf_init

        nf = min(nf, n_samples)

        # Iteratively reduce nf until clusters have at least 5 samples
        while True:
            kmeans = sk_cluster.KMeans(n_clusters=nf, random_state=seed).fit(
                adata.obsm["X_pca"]
            )
            labels = kmeans.labels_.astype(str)
            lab_size = [
                sum(labels == l) for l in sorted(set(labels))
            ]  # deterministic ordering
            if min(lab_size) >= 5 or nf <= 1:
                break
            nf -= 1

        return nf, labels

    SP = []
    PE = []

    # Deterministic iteration over clusters
    for ict in sorted(set(ad.obs["cluster"])):
        adr = ad[ad.obs["cluster"] == ict].copy()
        sc.pp.filter_genes(adr, min_counts=1)
        sc.pp.normalize_total(adr)
        # sc.pp.log1p(adr)
        sc.tl.pca(adr)

        df = adr.X.transpose()
        shared_genes_mask_sp = np.isin(
            adr.var.index.to_list(), ad_sp.var.index.to_list()
        )
        shared_genes_mask = np.isin(adr.var.index.to_list(), ad.var.index.to_list())

        nf, labels = find_components(adr, 1, 10, opt=opt_nf, nf_init=nf_init)
        adr.obs["kmeans"] = labels

        if not init:
            nf = nf_init

        print(ict, "Optimal nf:", nf)

        if init:
            lab_size = [sum(labels == l) for l in sorted(set(labels))]
            print(ict, "Cluster sizes:", lab_size)
            Winit, Hinit = init_nmf_matrices(adr, nf, 50)
            modelH = sk_decomposition.NMF(
                n_components=nf,
                init="custom",
                alpha_W=0,
                alpha_H=0,
                l1_ratio=1.0,
                random_state=seed,
            )
            modelHsp = sk_decomposition.NMF(
                n_components=nf,
                init="custom",
                alpha_W=0,
                alpha_H=0,
                l1_ratio=1.0,
                random_state=seed,
            )
            W = modelH.fit_transform(
                df[shared_genes_mask, :],
                W=Winit[shared_genes_mask, :].astype("float32"),
                H=Hinit.astype("float32"),
            )
            Wsp = modelHsp.fit_transform(
                df[shared_genes_mask_sp, :],
                W=Winit[shared_genes_mask_sp, :].astype("float32"),
                H=Hinit.astype("float32"),
            )
        else:
            modelH = sk_decomposition.NMF(
                n_components=nf, alpha_W=0, alpha_H=0, l1_ratio=1.0, random_state=seed
            )
            modelHsp = sk_decomposition.NMF(
                n_components=nf, alpha_W=0, alpha_H=0, l1_ratio=1.0, random_state=seed
            )
            W = modelH.fit_transform(df[shared_genes_mask, :])
            Wsp = modelHsp.fit_transform(df[shared_genes_mask_sp, :])

        H = modelH.components_

        mask = np.isin(adr.var.index.to_list(), ad_sp.var.index.to_list())
        genes = adr.var.index[mask].to_list()
        spr = ad_sp[ad_sp.obs["cluster"] == ict, genes]
        v = spr.X.transpose()

        sp = []
        pe = []

        # Deterministic gene sampling using NumPy RNG
        for tr in range(gene_samples):
            subidx = rng.choice(
                np.arange(1, Wsp.shape[0]), Wsp.shape[0] - test_genes, replace=False
            )
            compidx = np.setdiff1d(
                np.arange(Wsp.shape[0]), subidx
            )  # deterministic complement

            vl = v[subidx, :]
            w_sample = W[shared_genes_mask_sp, :][subidx, :]
            h = np.ones((W.shape[1], vl.shape[1]))

            for i in range(max_iter_nmf):
                h *= (w_sample.T @ vl) / (
                    w_sample.T @ w_sample @ h + 1e-8
                )  # avoid divide-by-zero

            v1 = W[shared_genes_mask_sp, :][compidx, :] @ h

            for k in range(v1.shape[1]):
                x = np.nan_to_num(v1)[:, k]
                y = _to_1d_dense(v[compidx, k])
                if np.max(y) == 0:
                    continue
                sp_val = spearmanr(x, y)[0]
                pe_val = pearsonr(x, y)[0]
                sp.append(sp_val)
                pe.append(pe_val)
                SP.append(sp_val)
                PE.append(pe_val)

        print(ict, "Spearman median:", statistics.median(sp))
        print(ict, "Pearson median:", statistics.median(pe), end="\n")

    return SP, PE


def test_refactor_equality():
    ad = read_h5ad(
        "./tests/data/mouse_small_intestine_sc/mouse_small_intestine_sc.h5ad"
    )
    ad_sp = read_h5ad(
        "./tests/data/mouse_small_intestine_merfish/mouse_small_intestine_merfish.h5ad"
    )
    sp, pe = dominic_experiment(ad.copy(), ad_sp.copy())
    sp2, pe2 = dominic_experiment_refactor_v1(ad.copy(), ad_sp.copy())
    assert sp == sp2
    assert sp != pe
    assert pe == pe2
    assert sp2 != pe2
