from nico2_lib.predictors._baselines import shuffle_by_embedding_neighbors
from nico2_lib.predictors._fastica._fastica import FastIcaPredictor
from nico2_lib.predictors._nmf._nmf_pred import (
    NmfPredictor,
    consensus_nmf,
    find_k_by_inflection,
)
from nico2_lib.predictors._pca._pca_pred import PcaPredictor
from nico2_lib.predictors._protocol import PredictorProtocol

__all__ = [
    "PcaPredictor",
    "NmfPredictor",
    "consensus_nmf",
    "find_k_by_inflection",
    "PredictorProtocol",
    "shuffle_by_embedding_neighbors",
    "FastIcaPredictor",
]

# These predictors pull in heavy, optional dependencies (mofaflex, scvi-tools,
# tangram-sc respectively) that aren't installed in every environment that only
# needs NmfPredictor/PcaPredictor/FastIcaPredictor above -- don't let a missing
# optional dependency break importing the whole package.
try:
    from nico2_lib.predictors._mofaflex._mofaflex_pred import (
        MofaFlexClassicPredictor,
        MofaFlexPredictor,
    )
    __all__ += ["MofaFlexPredictor", "MofaFlexClassicPredictor"]
except ImportError:
    pass

try:
    from nico2_lib.predictors._scvi._scvi_pred import ScviPredictor
    __all__.append("ScviPredictor")
except ImportError:
    pass

try:
    from nico2_lib.predictors._tangram._tangram_pred import TangramPredictor
    __all__.append("TangramPredictor")
except ImportError:
    pass
