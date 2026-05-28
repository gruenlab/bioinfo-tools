from __future__ import annotations

__all__ = [
    "PcaPredictor",
    "NmfPredictor",
    "TangramPredictor",
    "PredictorProtocol",
    "shuffle_by_embedding_neighbors",
    "MofaFlexPredictor",
    "ScviPredictor",
]


def __getattr__(name: str):
    if name == "NmfPredictor":
        from nico2_lib.predictors._nmf._nmf_pred import NmfPredictor
        return NmfPredictor
    if name == "TangramPredictor":
        from nico2_lib.predictors._tangram._tangram_pred import TangramPredictor
        return TangramPredictor
    if name in ("PredictorProtocol", "shuffle_by_embedding_neighbors", "MofaFlexPredictor", "ScviPredictor"):
        raise ImportError(
            f"'{name}' requires optional dependencies not installed in this environment."
        )
    raise AttributeError(f"module 'nico2_lib.predictors' has no attribute {name!r}")
