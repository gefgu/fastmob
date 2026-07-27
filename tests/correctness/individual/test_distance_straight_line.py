"""Correctness tests for fastmob/measures/spatial/distance_straight_line.py."""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

# Pre-computed expected total distances for the shared synthetic fixture
# (3 users, 5 GPS points each, 1-degree steps equator/meridian, Paris cluster).
EXPECTED_TOTAL_DIST: dict[str, float] = {
    "user_a": 444.7803209341316,  # 4 × ~111.195 km along equator
    "user_b": 444.7803209341316,  # 4 × ~111.195 km along meridian
    "user_c": 3.440787203829053,  # sum of 4 small Paris steps
}


def _to_dict(df) -> dict[str, float]:
    """Convert a distance_straight_line result DataFrame to {uid: distance_km}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["distance_straight_line"] for row in nw_df.rows(named=True)}


def test_distance_straight_line_known_values(synthetic_tdf):
    """Known-value check against pre-computed summed Haversine results."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.distance_straight_line import distance_straight_line

    result = distance_straight_line(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_TOTAL_DIST.keys())
    for uid, expected in EXPECTED_TOTAL_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_distance_straight_line_single_user():
    """Without a uid column the whole frame is treated as one individual."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.distance_straight_line import distance_straight_line

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [0.0, 1.0, 0.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = distance_straight_line(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "distance_straight_line" in nw_result.columns
    assert len(nw_result) == 1
    val = nw_result.rows(named=True)[0]["distance_straight_line"]
    # Two jumps of ~111.195 km each = ~222.390 km
    assert abs(val - 2 * 111.1950802335329) < 1e-5


def test_distance_straight_line_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.distance_straight_line import distance_straight_line

    result = distance_straight_line(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_TOTAL_DIST.keys())
    for uid, expected in EXPECTED_TOTAL_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"


def _core_result_array(values):
    if hasattr(values, "to_pyarrow"):
        values = values.to_pyarrow()
    return np.asarray(values)


def test_total_distance_presorted_numpy_and_arrow_match_batch_helper():
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import total_distance_presorted

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ends = np.array([3, 5], dtype=np.uintp)

    expected = [222.3901604670658, 109.50573519924356]
    result_numpy = total_distance_presorted(lats, lngs, ends)
    result_arrow = total_distance_presorted(
        pa.array(lats, type=pa.float64()),
        pa.array(lngs, type=pa.float64()),
        ends,
    )

    np.testing.assert_allclose(_core_result_array(result_numpy), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_core_result_array(result_arrow), expected, rtol=0.0, atol=1e-12)


def test_total_distance_indexed_backends_match_presorted():
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import total_distance_indexed, total_distance_presorted

    lats = np.array([10.0, 0.0, 11.0, 20.0, 1.0, 21.0], dtype=np.float64)
    lngs = np.array([30.0, 0.0, 31.0, 40.0, 1.0, 41.0], dtype=np.float64)
    indices = np.array([1, 4, 0, 2, 3, 5], dtype=np.uintp)
    ends = np.array([2, 4, 6], dtype=np.uintp)

    expected_lats = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0], dtype=np.float64)
    expected_lngs = np.array([0.0, 1.0, 30.0, 31.0, 40.0, 41.0], dtype=np.float64)
    expected = _core_result_array(total_distance_presorted(expected_lats, expected_lngs, ends))

    result_numpy = total_distance_indexed(lats, lngs, indices, ends)
    result_arrow = total_distance_indexed(
        pa.array(lats, type=pa.float64()),
        pa.array(lngs, type=pa.float64()),
        indices,
        ends,
    )

    np.testing.assert_allclose(_core_result_array(result_numpy), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_core_result_array(result_arrow), expected, rtol=0.0, atol=1e-12)


def test_total_distance_presorted_validation_errors():
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob._core import total_distance_presorted

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        total_distance_presorted(arr, arr[:1], np.array([1], dtype=np.uintp))
    with pytest.raises(ValueError, match="range end"):
        total_distance_presorted(arr, arr, np.array([3], dtype=np.uintp))


def test_total_distance_indexed_validation_errors():
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob._core import total_distance_indexed

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        total_distance_indexed(
            arr,
            arr[:1],
            np.array([0], dtype=np.uintp),
            np.array([1], dtype=np.uintp),
        )
    with pytest.raises(ValueError, match="index array bounds"):
        total_distance_indexed(
            arr,
            arr,
            np.array([0], dtype=np.uintp),
            np.array([2], dtype=np.uintp),
        )
    with pytest.raises(ValueError, match="monotonically"):
        total_distance_indexed(
            arr,
            arr,
            np.array([0, 1], dtype=np.uintp),
            np.array([2, 1], dtype=np.uintp),
        )
    with pytest.raises(ValueError, match="coordinate array bounds"):
        total_distance_indexed(
            arr,
            arr,
            np.array([0, 2], dtype=np.uintp),
            np.array([2], dtype=np.uintp),
        )


def test_total_distance_indexed_filters_arrow_nulls():
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import total_distance_indexed

    lats = pa.array([0.0, None, 0.0], type=pa.float64())
    lngs = pa.array([0.0, 1.0, 2.0], type=pa.float64())
    indices = np.array([0, 1, 2], dtype=np.uintp)
    ends = np.array([3], dtype=np.uintp)

    result = total_distance_indexed(lats, lngs, indices, ends)

    np.testing.assert_allclose(_core_result_array(result), [222.3901604670658], rtol=0.0, atol=1e-12)


def test_total_distance_presorted_rejects_mixed_backends():
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import total_distance_presorted

    lats = np.array([0.0, 1.0], dtype=np.float64)
    lngs = pa.array([0.0, 1.0], type=pa.float64())
    ends = np.array([2], dtype=np.uintp)

    with pytest.raises(TypeError, match="NumPy arrays or both be Arrow arrays"):
        total_distance_presorted(lats, lngs, ends)


@pytest.mark.skmob
def test_distance_straight_line_matches_skmob(comparison_skmob):
    """fastmob result closely matches skmob on each comparison dataset."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.distance_straight_line import (
        distance_straight_line as fastmob_dsl,
    )
    from skmob.measures.individual import distance_straight_line as skmob_dsl

    skmob_result = skmob_dsl(comparison_skmob)
    fastmob_input = pd.DataFrame(comparison_skmob).copy()
    fastmob_result = fastmob_dsl(fastmob_input)

    skmob_dict = dict(
        zip(
            skmob_result["uid"].tolist(),
            skmob_result["distance_straight_line"].tolist(),
        )
    )
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        # skmob uses skmob.utils.gislib with earth radius 6371.0 km; fastmob
        # uses Rust geo::Haversine. The accumulated total can differ slightly.
        assert abs(skmob_dict[uid] - fastmob_dict[uid]) < skmob_dict[uid] * 2e-5 + 1e-5, (
            f"uid={uid}: skmob={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"
        )


def test_distance_straight_line_matches_cached_reference(comparison_skmob_reference):
    """distance_straight_line matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.distance_straight_line import distance_straight_line as fastmob_dsl

    ref = comparison_skmob_reference
    skmob_result = ref.result("distance_straight_line")
    fastmob_result = fastmob_dsl(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["distance_straight_line"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert abs(skmob_dict[uid] - fastmob_dict[uid]) < skmob_dict[uid] * 2e-5 + 1e-5, (
            f"uid={uid}: cached={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"
        )
