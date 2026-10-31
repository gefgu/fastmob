import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fastmob.utils import haversine


def test_haversine_array_likes_units_and_formats():
    args = (pd.Series([0, 0], index=[7, 8]), [0, 0], np.array([1, 2], dtype=np.float32), (0, 0))
    metres = haversine(*args)
    np.testing.assert_allclose(metres, [111195, 222390], rtol=0.001)
    np.testing.assert_allclose(haversine(*args, unit="km"), metres / 1000)
    series = haversine(*args, output="pandas")
    assert isinstance(series, pd.Series)
    np.testing.assert_allclose(series.to_numpy(), metres)
    arrow = haversine(*args, output="arrow")
    assert isinstance(arrow, pa.Array)
    np.testing.assert_allclose(arrow.to_numpy(), metres)
    np.testing.assert_allclose(haversine(np.arange(4)[::2], [0, 0], [0, 0], [0, 0]), [0, 222390], rtol=0.001)
    assert haversine([], [], [], []).size == 0
    assert np.isnan(haversine([float("nan")], [0], [1], [0])[0])


def test_haversine_rejects_invalid_options_and_shapes():
    with pytest.raises(ValueError, match="same length"):
        haversine([0], [], [0], [0])
    with pytest.raises(ValueError, match="one-dimensional"):
        haversine([[0]], [0], [0], [0])
    with pytest.raises(ValueError, match="unit"):
        haversine([], [], [], [], unit="miles")
    with pytest.raises(ValueError, match="output"):
        haversine([], [], [], [], output="invalid")


def test_legacy_network_entry_point():
    from fastmob.network import haversine_m_batch

    np.testing.assert_allclose(haversine_m_batch([0], [0], [1], [0]), haversine([0], [0], [1], [0]))
