import numpy as np

from nico2_lib.predictors._nmf._nmf_pred import NmfPredictor
from nico2_lib.predictors._protocol import (
    _adapt_protocol_n_to_protocol,
    _adapt_protocol_to_protocol_n,
)


def test_nmf_predictor_roundtrip_protocol_adapters() -> None:
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [0.5, 1.5],
            [2.5, 3.5],
            [4.5, 5.5],
            [6.5, 7.5],
        ],
        dtype=float,
    )

    np.random.seed(0)
    direct = NmfPredictor(n_components=2)
    direct.fit(X, y)
    direct_pred = direct.predict(X)

    np.random.seed(0)
    roundtrip = _adapt_protocol_n_to_protocol(
        _adapt_protocol_to_protocol_n(NmfPredictor(n_components=2))
    )
    roundtrip.fit(X, y)
    roundtrip_pred = roundtrip.predict(X)

    assert np.array_equal(direct_pred, roundtrip_pred)
