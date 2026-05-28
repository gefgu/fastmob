"""Correctness tests for skmob2.measures.visits.activity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skmob2.measures.visits.activity import (
    activity_transition_matrix,
    daily_activity_distribution,
    visit_purpose_distribution,
)


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


def test_visit_purpose_distribution_percentages_sum_to_100():
    result = visit_purpose_distribution(_simple_visits())

    assert result["percentage"].sum() == pytest.approx(100.0, abs=1e-6)
    assert set(result["activity"]) == {"HOME", "WORK"}


def test_visit_purpose_distribution_missing_activity_warns_unknown():
    df = _simple_visits().drop(columns=["purpose"])

    with pytest.warns(UserWarning, match="activity column"):
        result = visit_purpose_distribution(df)

    assert result["activity"].tolist() == ["UNKNOWN"]
    assert result["percentage"].tolist() == [100.0]


def test_visit_purpose_distribution_null_activity_warns_unknown():
    df = _simple_visits()
    df.loc[0, "purpose"] = None

    with pytest.warns(UserWarning, match="null activity"):
        result = visit_purpose_distribution(df)

    assert "UNKNOWN" in result["activity"].tolist()


def test_transition_matrix_missing_activity_column_warns_unknown_self_loop():
    df = _simple_visits().drop(columns=["purpose"])

    with pytest.warns(UserWarning, match="activity column"):
        result = activity_transition_matrix(df)

    assert result.index.tolist() == ["UNKNOWN"]
    assert result.columns.tolist() == ["UNKNOWN"]
    assert result.loc["UNKNOWN", "UNKNOWN"] == pytest.approx(100.0)


def test_daily_activity_distribution_shape():
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "start_timestamp": pd.to_datetime(["2020-01-01 08:00", "2020-01-01 08:10"]),
            "end_timestamp": pd.to_datetime(["2020-01-01 08:20", "2020-01-01 08:30"]),
            "purpose": ["HOME", "WORK"],
        }
    )

    matrix, categories, n_bins = daily_activity_distribution(df)

    assert n_bins == 144
    assert categories == ["HOME", "WORK"]
    assert matrix.shape == (2, 144)


def test_daily_activity_distribution_missing_activity_warns_unknown():
    df = pd.DataFrame(
        {
            "start_timestamp": pd.to_datetime(["2020-01-01 08:00"]),
            "end_timestamp": pd.to_datetime(["2020-01-01 08:10"]),
        }
    )

    with pytest.warns(UserWarning, match="activity column"):
        matrix, categories, n_bins = daily_activity_distribution(df)

    assert n_bins == 144
    assert categories == ["UNKNOWN"]
    assert matrix[0, 48] == pytest.approx(100.0)


def test_daily_activity_distribution_overnight_visit_spans_late_and_early_bins():
    df = pd.DataFrame(
        {
            "start_timestamp": pd.to_datetime(["2020-01-01 23:50"]),
            "end_timestamp": pd.to_datetime(["2020-01-02 00:10"]),
            "purpose": ["HOME"],
        }
    )

    matrix, categories, _ = daily_activity_distribution(df)

    assert categories == ["HOME"]
    assert matrix[0, 143] == pytest.approx(100.0)
    assert matrix[0, 0] == pytest.approx(100.0)
    assert matrix[0, 1] == pytest.approx(100.0)
    assert np.isnan(matrix[0, 2])
