import numpy as np

from nico2_lib.predictors._nmf._nmf_pred import NmfPredictor, NmfPredictorN

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


def test_nmf_new_protocol() -> None:
    # Arrange
    np.random.seed(0)
    old = NmfPredictor(n_components=2)
    old.fit(X, y)

    # Act
    old_pred = old.predict(X)

    np.random.seed(0)
    new = NmfPredictorN(n_components=2)
    new.fit(np.hstack([X, y]))

    new_pred = new.predict(X, [0, 1, 2])[:, [3, 4]]

    # Assert
    assert old_pred.shape == new_pred.shape, (
        f"prediction shapes differ: old={old_pred.shape}, new={new_pred.shape}"
    )
    np.testing.assert_allclose(
        old_pred,
        new_pred,
        err_msg="new protocol predictions are not close to legacy protocol outputs",
    )
