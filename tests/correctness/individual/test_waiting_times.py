"""Correctness tests for fkmob/measures/spatial/waiting_times.py."""

from __future__ import annotations

from typing import Any

import pytest
import narwhals as nw
import numpy as np
import pandas as pd

# The synthetic fixture has points spaced 1 hour apart, so all waiting times
# should be 3600.0 seconds.
EXPECTED_WAITING_TIME_S = 3600.0
EXPECTED_N_WAITS = 4  # 5 points → 4 intervals


def _to_dict(df) -> dict[str, Any]:
    """Convert a waiting_times result DataFrame to {uid: [wait_seconds, ...]}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["waiting_times"] for row in nw_df.rows(named=True)}


def test_waiting_times_known_values(synthetic_tdf):
    """Each interval in the synthetic fixture is exactly 3600 seconds."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.waiting_times import waiting_times

    result = waiting_times(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, wt_list in mapping.items():
        assert len(wt_list) == EXPECTED_N_WAITS, (
            f"uid={uid!r}: expected {EXPECTED_N_WAITS} intervals, got {len(wt_list)}"
        )
        for wt in wt_list:
            assert abs(wt - EXPECTED_WAITING_TIME_S) < 1.0, (
                f"uid={uid!r}: waiting time {wt} != {EXPECTED_WAITING_TIME_S}"
            )


def test_waiting_times_single_user():
    """Without a uid column the whole frame is treated as one individual."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.waiting_times import waiting_times

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [0.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = waiting_times(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "waiting_times" in nw_result.columns
    assert len(nw_result) == 1
    wt_list = nw_result.rows(named=True)[0]["waiting_times"]
    assert len(wt_list) == 2
    for wt in wt_list:
        assert abs(wt - 3600.0) < 1.0


def test_waiting_times_single_point_returns_empty_list():
    """A user with only one point produces an empty waiting-times list."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.waiting_times import waiting_times

    df = pd.DataFrame(
        {
            "uid": ["a"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [0.0],
            "lng": [0.0],
        }
    )
    result = waiting_times(df)
    mapping = _to_dict(result)
    assert len(mapping["a"]) == 0


def test_waiting_times_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.waiting_times import waiting_times

    result = waiting_times(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, wt_list in mapping.items():
        assert len(wt_list) == EXPECTED_N_WAITS
        for wt in wt_list:
            assert abs(wt - EXPECTED_WAITING_TIME_S) < 1.0, (
                f"uid={uid!r}: waiting time {wt} != {EXPECTED_WAITING_TIME_S} (Polars)"
            )


def test_waiting_times_merge_returns_flat_array(synthetic_tdf):
    """merge=True returns flat backend-native waiting times."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.waiting_times import waiting_times

    result = waiting_times(synthetic_tdf, merge=True)
    values = np.asarray(result, dtype=np.float64)

    # 3 users × 4 intervals = 12 waiting times total.
    assert len(values) == 3 * EXPECTED_N_WAITS
    for wt in values:
        assert abs(wt - EXPECTED_WAITING_TIME_S) < 1.0


def test_waiting_times_numpy_and_arrow_helpers_return_offsets_and_flat_values():
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from fkmob._core import (
        waiting_times_arrow,
        waiting_times_flat_numpy,
        waiting_times_indexed_numpy,
        waiting_times_numpy,
        waiting_times_seconds,
    )

    timestamps = np.array([0.0, 60.0, 90.0, 1000.0, 1060.0], dtype=np.float64)
    ends = np.array([3, 5], dtype=np.uintp)

    expected_starts = np.array([0, 2], dtype=np.uintp)
    expected_ends = np.array([2, 3], dtype=np.uintp)
    expected_values = np.array([60.0, 30.0, 60.0], dtype=np.float64)

    for starts, value_ends, values in (
        waiting_times_seconds(timestamps.tolist(), ends.tolist()),
        waiting_times_numpy(timestamps, ends),
        waiting_times_arrow(pl.Series(timestamps).to_arrow(), ends),
        waiting_times_indexed_numpy(timestamps, np.arange(len(timestamps), dtype=np.uintp), ends),
    ):
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(value_ends, expected_ends)
        np.testing.assert_allclose(np.asarray(values), expected_values)

    np.testing.assert_allclose(
        waiting_times_flat_numpy(timestamps, ends),
        expected_values,
    )


def test_waiting_times_helper_offsets_include_empty_groups():
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob._core import waiting_times_numpy

    timestamps = np.array([0.0, 60.0, 120.0, 1000.0], dtype=np.float64)
    ends = np.array([1, 3, 4], dtype=np.uintp)

    starts, value_ends, values = waiting_times_numpy(timestamps, ends)

    np.testing.assert_array_equal(starts, np.array([0, 0, 1], dtype=np.uintp))
    np.testing.assert_array_equal(value_ends, np.array([0, 1, 1], dtype=np.uintp))
    np.testing.assert_allclose(values, np.array([60.0], dtype=np.float64))


def test_waiting_times_polars_result_uses_list_dtype(synthetic_tdf_polars):
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from fkmob.measures.individual.waiting_times import waiting_times

    result = waiting_times(synthetic_tdf_polars)

    assert result.schema["waiting_times"] == pl.List(pl.Float64)


def test_waiting_times_numpy_helper_validation_errors():
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob._core import waiting_times_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="range end"):
        waiting_times_numpy(arr, np.array([3], dtype=np.uintp))


@pytest.mark.skmob
def test_waiting_times_matches_skmob(comparison_skmob):
    """fkmob result matches skmob on each comparison dataset."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from skmob.measures.individual import waiting_times as skmob_wt
    from fkmob.measures.individual.waiting_times import waiting_times as fkmob_wt

    skmob_result = skmob_wt(comparison_skmob)
    fkmob_input = pd.DataFrame(comparison_skmob).copy()
    fkmob_result = fkmob_wt(fkmob_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["waiting_times"].tolist()))
    fkmob_dict = _to_dict(fkmob_result)

    common = set(skmob_dict) & set(fkmob_dict)
    assert len(common) > 0
    for uid in common:
        left = sorted(skmob_dict[uid])
        right = sorted(fkmob_dict[uid])
        assert len(left) == len(right), f"uid={uid}: skmob n={len(left)}, fkmob n={len(right)}"
        for a, b in zip(left, right):
            assert abs(a - b) < max(abs(a), 1.0) * 1e-5 + 1.0, f"uid={uid}: skmob wt={a}, fkmob wt={b}"


def test_waiting_times_matches_cached_reference(comparison_skmob_reference):
    """waiting_times matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    from fkmob.measures.individual.waiting_times import waiting_times as fkmob_wt

    ref = comparison_skmob_reference
    skmob_result = ref.result("waiting_times")
    fkmob_result = fkmob_wt(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["waiting_times"].tolist()))
    fkmob_dict = _to_dict(fkmob_result)

    common = set(skmob_dict) & set(fkmob_dict)
    assert len(common) > 0
    for uid in common:
        left = sorted(skmob_dict[uid])
        right = sorted(fkmob_dict[uid])
        assert len(left) == len(right), f"uid={uid}: cached n={len(left)}, fkmob n={len(right)}"
        for a, b in zip(left, right):
            assert abs(a - b) < max(abs(a), 1.0) * 1e-5 + 1.0, f"uid={uid}: cached wt={a}, fkmob wt={b}"
