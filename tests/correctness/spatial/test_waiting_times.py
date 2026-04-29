"""Correctness tests for skmob2/measures/spatial/waiting_times.py."""

from __future__ import annotations

import pytest
import narwhals as nw
import numpy as np
import pandas as pd

# The synthetic fixture has points spaced 1 hour apart, so all waiting times
# should be 3600.0 seconds.
EXPECTED_WAITING_TIME_S = 3600.0
EXPECTED_N_WAITS = 4  # 5 points → 4 intervals


def _to_dict(df) -> dict[str, list[float]]:
    """Convert a waiting_times result DataFrame to {uid: [wait_seconds, ...]}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["waiting_times"] for row in nw_df.rows(named=True)}


def test_waiting_times_known_values(synthetic_tdf):
    """Each interval in the synthetic fixture is exactly 3600 seconds."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.waiting_times import waiting_times

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
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.waiting_times import waiting_times

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
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.waiting_times import waiting_times

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
    assert mapping["a"] == []


def test_waiting_times_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.waiting_times import waiting_times

    result = waiting_times(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, wt_list in mapping.items():
        assert len(wt_list) == EXPECTED_N_WAITS
        for wt in wt_list:
            assert abs(wt - EXPECTED_WAITING_TIME_S) < 1.0, (
                f"uid={uid!r}: waiting time {wt} != {EXPECTED_WAITING_TIME_S} (Polars)"
            )


def test_waiting_times_merge_returns_flat_list(synthetic_tdf):
    """merge=True returns a single flat list of all waiting times."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.waiting_times import waiting_times

    result = waiting_times(synthetic_tdf, merge=True)

    assert isinstance(result, list)
    # 3 users × 4 intervals = 12 waiting times total.
    assert len(result) == 3 * EXPECTED_N_WAITS
    for wt in result:
        assert abs(wt - EXPECTED_WAITING_TIME_S) < 1.0


def test_waiting_times_numpy_and_arrow_helpers_match_batch_helper():
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from skmob2._core import waiting_times_arrow, waiting_times_flat_numpy, waiting_times_numpy, waiting_times_seconds

    timestamps = np.array([0.0, 60.0, 90.0, 1000.0, 1060.0], dtype=np.float64)
    ranges = [(0, 3), (3, 5)]

    expected = waiting_times_seconds(timestamps.tolist(), ranges)
    result_numpy = waiting_times_numpy(timestamps, ranges)
    result_arrow = waiting_times_arrow(pl.Series(timestamps).to_arrow(), ranges)

    assert result_numpy == expected
    assert result_arrow == expected
    assert waiting_times_flat_numpy(timestamps, ranges) == expected[0] + expected[1]


def test_waiting_times_numpy_helper_validation_errors():
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import waiting_times_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="range end"):
        waiting_times_numpy(arr, [(0, 3)])


@pytest.mark.skmob
def test_waiting_times_matches_skmob(brightkite_skmob):
    """skmob2 result matches skmob on the Brightkite dataset."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob.measures.individual import waiting_times as skmob_wt
    from skmob2.measures.spatial.waiting_times import waiting_times as skmob2_wt

    skmob_result = skmob_wt(brightkite_skmob)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_wt(skmob2_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["waiting_times"].tolist()))
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        left = sorted(skmob_dict[uid])
        right = sorted(skmob2_dict[uid])
        assert len(left) == len(right), f"uid={uid}: skmob n={len(left)}, skmob2 n={len(right)}"
        for a, b in zip(left, right):
            assert abs(a - b) < max(abs(a), 1.0) * 1e-5 + 1.0, f"uid={uid}: skmob wt={a}, skmob2 wt={b}"
