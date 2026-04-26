"""Correctness tests for skmob2/measures/spatial/waiting_times.py."""

from __future__ import annotations

import pytest
import narwhals as nw
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
