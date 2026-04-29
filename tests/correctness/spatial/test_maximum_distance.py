"""Correctness tests for skmob2/measures/spatial/maximum_distance.py."""

from __future__ import annotations

import pytest
import narwhals as nw
import numpy as np
import pandas as pd

# Pre-computed expected maximum distances for the shared synthetic fixture
# (3 users, 5 GPS points each, 1-degree steps equator/meridian, Paris cluster).
EXPECTED_MAX_DIST: dict[str, float] = {
    "user_a": 111.1950802335329,  # 1-degree step on equator
    "user_b": 111.1950802335329,  # 1-degree step on meridian
    "user_c": 0.9188177472926384,  # largest consecutive step in Paris cluster
}


def _to_dict(df) -> dict[str, float]:
    """Convert a maximum_distance result DataFrame to {uid: distance_km}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["maximum_distance"] for row in nw_df.rows(named=True)}


def test_maximum_distance_known_values(synthetic_tdf):
    """Known-value check against pre-computed Haversine results."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.maximum_distance import maximum_distance

    result = maximum_distance(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST.keys())
    for uid, expected in EXPECTED_MAX_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_maximum_distance_single_user():
    """Without a uid column the whole frame is treated as one individual."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.maximum_distance import maximum_distance

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [0.0, 1.0, 0.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = maximum_distance(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "maximum_distance" in nw_result.columns
    assert len(nw_result) == 1
    val = nw_result.rows(named=True)[0]["maximum_distance"]
    # Both jumps are ~111.195 km; max = 111.195 km
    assert abs(val - 111.1950802335329) < 1e-5


def test_maximum_distance_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.maximum_distance import maximum_distance

    result = maximum_distance(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST.keys())
    for uid, expected in EXPECTED_MAX_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"


def test_maximum_distance_numpy_and_arrow_helpers_match_batch_helper():
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from skmob2._core import maximum_distance_arrow, maximum_distance_batch_km, maximum_distance_numpy

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ranges = [(0, 3), (3, 5)]

    expected = maximum_distance_batch_km(lats.tolist(), lngs.tolist(), ranges)
    result_numpy = maximum_distance_numpy(lats, lngs, ranges)
    result_arrow = maximum_distance_arrow(pl.Series(lats).to_arrow(), pl.Series(lngs).to_arrow(), ranges)

    np.testing.assert_allclose(result_numpy, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result_arrow, expected, rtol=0.0, atol=1e-12)


def test_maximum_distance_numpy_helper_validation_errors():
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import maximum_distance_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        maximum_distance_numpy(arr, arr[:1], [(0, 1)])
    with pytest.raises(ValueError, match="range end"):
        maximum_distance_numpy(arr, arr, [(0, 3)])


@pytest.mark.skmob
def test_maximum_distance_matches_skmob(brightkite_skmob):
    """skmob2 result matches skmob on the Brightkite dataset."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob.measures.individual import maximum_distance as skmob_md
    from skmob2.measures.spatial.maximum_distance import maximum_distance as skmob2_md

    skmob_result = skmob_md(brightkite_skmob)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_md(skmob2_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["maximum_distance"].tolist()))
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        assert abs(skmob_dict[uid] - skmob2_dict[uid]) < skmob_dict[uid] * 1e-5 + 1e-5, (
            f"uid={uid}: skmob={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )
