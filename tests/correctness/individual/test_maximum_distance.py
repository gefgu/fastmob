"""Correctness tests for fkmob/measures/spatial/maximum_distance.py."""

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


def _arrow_to_numpy(values) -> np.ndarray:
    if hasattr(values, "to_pyarrow"):
        values = values.to_pyarrow()
    try:
        return values.to_numpy(zero_copy_only=False)
    except TypeError:
        return values.to_numpy()


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
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.maximum_distance import maximum_distance

    result = maximum_distance(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST.keys())
    for uid, expected in EXPECTED_MAX_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_maximum_distance_single_user():
    """Without a uid column the whole frame is treated as one individual."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.maximum_distance import maximum_distance

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


def test_maximum_distance_single_point_returns_nan():
    """A single-point user matches skmob's undefined maximum-distance result."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.maximum_distance import maximum_distance

    df = pd.DataFrame(
        {
            "uid": ["a"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [0.0],
            "lng": [0.0],
        }
    )
    result = maximum_distance(df)
    mapping = _to_dict(result)
    assert np.isnan(mapping["a"])


def test_maximum_distance_sorted_single_point_group_returns_nan():
    """The presorted backend emits NaN for groups with fewer than two points."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.maximum_distance import maximum_distance

    df = pd.DataFrame(
        {
            "uid": ["a", "b", "b"],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "lat": [0.0, 0.0, 1.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = maximum_distance(df, presorted=True)
    mapping = _to_dict(result)

    assert np.isnan(mapping["a"])
    assert abs(mapping["b"] - 111.1950802335329) < 1e-5


def test_maximum_distance_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.maximum_distance import maximum_distance

    result = maximum_distance(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST.keys())
    for uid, expected in EXPECTED_MAX_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"


def test_maximum_distance_polars_single_point_returns_nan():
    """Arrow-backed high-level path keeps NaN semantics without Python patching."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from fkmob.measures.individual.maximum_distance import maximum_distance

    df = pl.DataFrame(
        {
            "uid": ["a"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [0.0],
            "lng": [0.0],
        }
    )
    result = maximum_distance(df)
    mapping = _to_dict(result)

    assert np.isnan(mapping["a"])


def test_maximum_distance_numpy_and_arrow_helpers_match_batch_helper():
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from fkmob._core import maximum_distance_arrow, maximum_distance_batch_km, maximum_distance_numpy

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ranges = [(0, 3), (3, 5)]
    ends = np.array([3, 5], dtype=np.uintp)

    expected = maximum_distance_batch_km(lats.tolist(), lngs.tolist(), ranges)
    result_numpy = maximum_distance_numpy(lats, lngs, ends)
    result_arrow = maximum_distance_arrow(pl.Series(lats).to_arrow(), pl.Series(lngs).to_arrow(), ends)

    np.testing.assert_allclose(result_numpy, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_arrow_to_numpy(result_arrow), expected, rtol=0.0, atol=1e-12)
    assert hasattr(result_arrow, "__arrow_c_array__")


def test_maximum_distance_helpers_return_nan_for_short_presorted_groups():
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from fkmob._core import maximum_distance_arrow, maximum_distance_batch_km, maximum_distance_numpy

    lats = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lngs = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ranges = [(0, 1), (1, 3)]
    ends = np.array([1, 3], dtype=np.uintp)

    expected = maximum_distance_batch_km(lats.tolist(), lngs.tolist(), ranges)
    result_numpy = maximum_distance_numpy(lats, lngs, ends)
    result_arrow = maximum_distance_arrow(pl.Series(lats).to_arrow(), pl.Series(lngs).to_arrow(), ends)

    assert np.isnan(expected[0])
    assert np.isnan(result_numpy[0])
    assert np.isnan(_arrow_to_numpy(result_arrow)[0])
    np.testing.assert_allclose(result_numpy[1:], expected[1:], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_arrow_to_numpy(result_arrow)[1:], expected[1:], rtol=0.0, atol=1e-12)


def test_maximum_distance_indexed_arrow_returns_nan_when_nulls_leave_one_valid_point():
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fkmob._core import maximum_distance_indexed_arrow

    lats = pa.array([0.0, None, float("nan")])
    lngs = pa.array([0.0, 1.0, 2.0])
    indices = np.array([0, 1, 2], dtype=np.uintp)
    ends = np.array([3], dtype=np.uintp)

    result = maximum_distance_indexed_arrow(lats, lngs, indices, ends)

    values = _arrow_to_numpy(result)
    assert values.shape == (1,)
    assert np.isnan(values[0])


def test_maximum_distance_numpy_helper_validation_errors():
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob._core import maximum_distance_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        maximum_distance_numpy(arr, arr[:1], np.array([1], dtype=np.uintp))
    with pytest.raises(ValueError, match="range end"):
        maximum_distance_numpy(arr, arr, np.array([3], dtype=np.uintp))
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        maximum_distance_numpy(arr, arr, np.array([2, 1], dtype=np.uintp))


@pytest.mark.skmob
def test_maximum_distance_matches_skmob(comparison_skmob):
    """fkmob result matches skmob on each comparison dataset."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from skmob.measures.individual import maximum_distance as skmob_md
    from fkmob.measures.individual.maximum_distance import maximum_distance as fkmob_md

    skmob_result = skmob_md(comparison_skmob)
    fkmob_input = pd.DataFrame(comparison_skmob).copy()
    fkmob_result = fkmob_md(fkmob_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["maximum_distance"].tolist()))
    fkmob_dict = _to_dict(fkmob_result)

    common = set(skmob_dict) & set(fkmob_dict)
    assert len(common) > 0
    for uid in common:
        if np.isnan(skmob_dict[uid]) and np.isnan(fkmob_dict[uid]):
            continue
        assert abs(skmob_dict[uid] - fkmob_dict[uid]) < skmob_dict[uid] * 1e-5 + 1e-5, (
            f"uid={uid}: skmob={skmob_dict[uid]}, fkmob={fkmob_dict[uid]}"
        )


def test_maximum_distance_matches_cached_reference(comparison_skmob_reference):
    """maximum_distance matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.maximum_distance import maximum_distance as fkmob_md

    ref = comparison_skmob_reference
    skmob_result = ref.result("maximum_distance")
    fkmob_result = fkmob_md(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["maximum_distance"].tolist()))
    fkmob_dict = _to_dict(fkmob_result)

    common = set(skmob_dict) & set(fkmob_dict)
    assert len(common) > 0
    for uid in common:
        if np.isnan(skmob_dict[uid]) and np.isnan(fkmob_dict[uid]):
            continue
        assert abs(skmob_dict[uid] - fkmob_dict[uid]) < skmob_dict[uid] * 1e-5 + 1e-5, (
            f"uid={uid}: cached={skmob_dict[uid]}, fkmob={fkmob_dict[uid]}"
        )
