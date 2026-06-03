import nico2_lib as n2l
from functools import partial
import numpy as np
import pytest
from nico2_lib.typing import NumericArray
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n_obs, n_var = 100, 500
counts = rng.poisson(lam=10, size=(n_obs, n_var))
n_test_genes = 20
gene_train_idx, gene_test_idx = train_test_split(
    np.arange(n_var),
    test_size=n_test_genes,
    random_state=0,
)
n_test_cells = 10
cell_train_idx, cell_test_idx = train_test_split(
    np.arange(n_obs),
    test_size=n_test_cells,
    random_state=0,
)


def _run_predictor(
    predictor: n2l.pd.PredictorProtocol,
) -> tuple[n2l.pd.PredictorProtocol, tuple[NumericArray, NumericArray]]:
    predictor = predictor.fit(counts[cell_train_idx])
    return predictor, predictor.predict(
        counts[cell_test_idx][:, gene_train_idx],
        indexer=gene_train_idx,  # type: ignore
    )


def _assert_reconstruction_shape(full_reconstruction: NumericArray):

    assert full_reconstruction.shape == (len(cell_test_idx), n_var), (
        f"Expected shape: {(len(cell_test_idx), n_var)}, got: {full_reconstruction.shape}"
    )


def _assert_embedding_size(
    cell_embedding: NumericArray,
    gene_embedding: NumericArray | None,
    embedding_size: int,
):
    assert cell_embedding.shape[1] == embedding_size, (
        f"Expected cell embedding size: {embedding_size}, got: {cell_embedding.shape[1]}"
    )
    if gene_embedding is not None:
        assert gene_embedding.shape[0] == embedding_size, (
            f"Expected gene embedding size: {embedding_size}, got: {gene_embedding.shape[0]}"
        )


@pytest.mark.parametrize("embedding_size", [3, 5])
def test_pca_predictor(embedding_size: int):
    pca_predictor = n2l.pd.PcaPredictor(n_components=embedding_size).fit(
        counts[cell_train_idx]
    )
    pca_predictor, (cell_embedding, full_reconstruction) = _run_predictor(
        pca_predictor,
    )
    _assert_reconstruction_shape(
        full_reconstruction,
    )
    assert pca_predictor.embedding_size is not None, (
        f"Expected embedding size to be set, got: {pca_predictor.embedding_size}"
    )
    _assert_embedding_size(
        cell_embedding,
        pca_predictor.feature_embedding,
        pca_predictor.embedding_size,
    )


@pytest.mark.parametrize(
    "embedding_size",
    [
        3,
        5,
        partial(n2l.pd.consensus_nmf, k_range=range(2, 11), n_runs=5, max_iter=500),
        lambda x: partial(n2l.pd.find_k_by_inflection, k_range=range(2, 11), max_iter=500)(x)[0],
    ],
)
def test_nmf_predictor(embedding_size: int):
    nmf_predictor = n2l.pd.NmfPredictor(n_components=embedding_size).fit(
        counts[cell_train_idx]
    )
    nmf_predictor, (cell_embedding, full_reconstruction) = _run_predictor(
        nmf_predictor,
    )
    _assert_reconstruction_shape(
        full_reconstruction,
    )
    assert nmf_predictor.embedding_size is not None, (
        f"Expected embedding size to be set, got: {nmf_predictor.embedding_size}"
    )
    _assert_embedding_size(
        cell_embedding,
        nmf_predictor.feature_embedding,
        nmf_predictor.embedding_size,
    )
