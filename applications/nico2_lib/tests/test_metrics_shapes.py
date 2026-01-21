import numpy as np

import nico2_lib.metrics as metrics


def _exported_metrics():
    return [getattr(metrics, name) for name in metrics.__all__]


def _as_array(result):
    return np.asarray(result)


def test_metrics_return_scalar_for_1d_inputs():
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    y = np.array([1.1, 1.9, 3.2], dtype=float)

    for fn in _exported_metrics():
        result = fn(x, y)
        result_array = _as_array(result)
        assert result_array.shape == (), (
            f"{fn.__name__} should return a single float for 1d inputs, "
            f"got shape {result_array.shape}"
        )


def test_metrics_return_axis0_values_for_2d_inputs():
    x = np.array(
        [
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 5.0],
            [7.0, 8.0, 9.0, 2.0],
        ]
    )
    y = x + 0.1
    expected_shape = (x.shape[1],)

    for fn in _exported_metrics():
        result = fn(x, y)
        result_array = _as_array(result)
        assert result_array.shape == expected_shape, (
            f"{fn.__name__} should return one value per axis=0 column for 2d inputs, "
            f"got shape {result_array.shape}"
        )
