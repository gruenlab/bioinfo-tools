from nico2_lib.predictors._baselines import shuffle_by_embedding_neighbors
from nico2_lib.predictors._fastica._fastica import FastIcaPredictor
from nico2_lib.predictors._mofaflex._mofaflex_pred import (
    MofaFlexClassicPredictor,
    MofaFlexPredictor,
)
from nico2_lib.predictors._nmf._nmf_pred import (
    NmfPredictor,
    consensus_nmf,
    find_k_by_inflection,
)
from nico2_lib.predictors._pca._pca_pred import PcaPredictor
from nico2_lib.predictors._protocol import PredictorProtocol
from nico2_lib.predictors._scvi._scvi_pred import ScviPredictor
from nico2_lib.predictors._tangram._tangram_pred import TangramPredictor

__all__ = [
    "PcaPredictor",
    "NmfPredictor",
    "consensus_nmf",
    "find_k_by_inflection",
    "TangramPredictor",
    "PredictorProtocol",
    "shuffle_by_embedding_neighbors",
    "MofaFlexPredictor",
    "ScviPredictor",
    "FastIcaPredictor",
    "MofaFlexClassicPredictor",
]
