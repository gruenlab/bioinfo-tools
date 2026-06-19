from functools import partial

import nico2_lib as n2l
import numpy as np
import pytest
from nico2_lib.typing import NumericArray
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n_obs, n_var = 100, 500
counts = rng.poisson(lam=10, size=(n_obs, n_var)).astype(np.float64)
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


def run_predictor_test_suite(predictor: n2l.pd.PredictorProtocol) -> None:
    predictor = predictor.fit(counts[cell_train_idx])
    predictor, (cell_embedding, full_reconstruction) = _run_predictor(predictor)
    _assert_reconstruction_shape(full_reconstruction)
    match predictor.embedding_size:
        case int():
            _assert_embedding_size(
                cell_embedding,
                predictor.feature_embedding,
                predictor.embedding_size,
            )
        case None:
            pass


@pytest.mark.parametrize("embedding_size", [3, 5])
def test_pca_predictor(embedding_size: int):
    pca_predictor = n2l.pd.PcaPredictor(n_components=embedding_size)
    run_predictor_test_suite(pca_predictor)


@pytest.mark.parametrize(
    "embedding_size",
    [
        3,
        5,
        lambda x: partial(
            n2l.pd.find_k_by_inflection, k_range=range(2, 11), max_iter=500
        )(x)[0],
    ],
)
def test_nmf_predictor(embedding_size: int):
    nmf_predictor = n2l.pd.NmfPredictor(n_components=embedding_size)
    run_predictor_test_suite(nmf_predictor)


@pytest.mark.parametrize(
    "embedding_size",
    [3],
)
def test_scvi_predictor(embedding_size: int):
    scvi_predictor = n2l.pd.ScviPredictor(n_factors=embedding_size)
    run_predictor_test_suite(scvi_predictor)


@pytest.mark.parametrize(
    "n_components",
    [3],
)
def test_fastica_predictor(n_components: int):
    fastica_predictor = n2l.pd.FastIcaPredictor(n_components=n_components)
    run_predictor_test_suite(fastica_predictor)


@pytest.mark.parametrize(
    "n_factors",
    [3],
)
def test_mofaflex_classic(n_factors: int):
    import mofaflex

    mofaflex_predictor = n2l.pd.MofaFlexClassicPredictor(
        mofaflex_model=mofaflex.terms.MofaFlex(n_factors=n_factors),  # type: ignore
        max_epochs=1,
    )
    run_predictor_test_suite(mofaflex_predictor)


def test_tangram_predictor():
    tangram_predictor = n2l.pd.TangramPredictor()
    run_predictor_test_suite(tangram_predictor)
