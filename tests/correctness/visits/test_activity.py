"""Correctness tests for skmob2.measures.visits.activity."""

from __future__ import annotations

import pandas as pd
import pytest
from skmob2.measures.visits.activity import activity_transition_matrix


def _simple_visits():
    """Two users with predictable activity sequences."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 2],
            "start_timestamp": pd.date_range("2020-01-01", periods=7, freq="2h"),
            "purpose": ["HOME", "WORK", "HOME", "WORK", "HOME", "WORK", "HOME"],
            "day_of_week": ["monday"] * 7,
        }
    )


def test_transition_matrix_shape_and_labels():
    df = _simple_visits()
    result = activity_transition_matrix(df)
    assert isinstance(result, pd.DataFrame)
    # Both HOME and WORK are present
    assert set(result.index) == {"HOME", "WORK"}
    assert set(result.columns) == {"HOME", "WORK"}


def test_transition_matrix_polars_backend_matches_input():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    result = activity_transition_matrix(pl.from_pandas(_simple_visits()))
    assert isinstance(result, pl.DataFrame)
    assert "activity" in result.columns
    assert set(result.get_column("activity").to_list()) == {"HOME", "WORK"}


def test_transition_matrix_no_self_loops():
    """With alternating sequences there are only HOME->WORK and WORK->HOME."""
    df = _simple_visits()
    result = activity_transition_matrix(df)
    # All mass on off-diagonal (HOME->WORK and WORK->HOME); diagonals are 0
    assert result.loc["HOME", "HOME"] == pytest.approx(0.0)
    assert result.loc["WORK", "WORK"] == pytest.approx(0.0)


def test_transition_matrix_sums_to_100():
    df = _simple_visits()
    result = activity_transition_matrix(df)
    # Percentages sum to 100
    assert result.values.sum() == pytest.approx(100.0, abs=1e-6)


def test_transition_matrix_day_filter_weekdays():
    df = _simple_visits()
    df["day_of_week"] = ["monday", "monday", "monday", "monday", "saturday", "saturday", "saturday"]
    result_all = activity_transition_matrix(df)
    result_weekdays = activity_transition_matrix(df, day_filter="weekdays")
    # Weekday-only result excludes Saturday transitions
    assert result_weekdays.values.sum() == pytest.approx(100.0, abs=1e-6)
    # The two should differ because Saturday rows are excluded
    assert not result_all.equals(result_weekdays)


def test_transition_matrix_column_autodetection_agent_id():
    df = _simple_visits().rename(columns={"user_id": "agent_id"})
    result = activity_transition_matrix(df)
    assert result.values.sum() == pytest.approx(100.0, abs=1e-6)


def test_transition_matrix_missing_day_col_raises_when_day_filter_set():
    df = _simple_visits().drop(columns=["day_of_week"])
    with pytest.raises(ValueError, match="day_col"):
        activity_transition_matrix(df, day_filter="weekdays")


def test_transition_matrix_weekend_filter():
    """Weekend filter keeps only Saturday/Sunday rows."""
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "start_timestamp": pd.date_range("2020-01-01", periods=5, freq="2h"),
            "purpose": ["HOME", "WORK", "HOME", "SHOP", "HOME"],
            "day_of_week": ["monday", "saturday", "saturday", "sunday", "sunday"],
        }
    )
    result = activity_transition_matrix(df, day_filter="weekends")
    # Should not contain monday transitions
    assert result.values.sum() == pytest.approx(100.0, abs=1e-6)


def test_transition_matrix_single_activity():
    """With a single activity type, no transitions exist — empty result or zero matrix."""
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "start_timestamp": pd.date_range("2020-01-01", periods=3, freq="1h"),
            "purpose": ["HOME", "HOME", "HOME"],
        }
    )
    result = activity_transition_matrix(df)
    # Only HOME -> HOME transitions (self-loop only)
    assert set(result.index) == {"HOME"}
    assert set(result.columns) == {"HOME"}
    assert result.values.sum() == pytest.approx(100.0, abs=1e-6)
