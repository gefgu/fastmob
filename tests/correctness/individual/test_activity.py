"""Correctness tests for fastmob.measures.individual.activity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.measures.individual.activity import (
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


def _matrix_sum(result: pd.DataFrame) -> float:
    """Sum of all transition percentages, excluding the ``activity`` label column."""
    return result.drop(columns=["activity"]).to_numpy().sum()


def _cell(result: pd.DataFrame, from_activity: str, to_activity: str) -> float:
    """Look up a single transition percentage by (from, to) activity labels."""
    row = result.loc[result["activity"] == from_activity]
    return row[to_activity].iloc[0]


def test_transition_matrix_shape_and_labels():
    df = _simple_visits()
    result = activity_transition_matrix(df)
    assert isinstance(result, pd.DataFrame)
    # Both HOME and WORK are present
    assert set(result["activity"]) == {"HOME", "WORK"}
    assert set(result.columns) == {"activity", "HOME", "WORK"}


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
    assert _cell(result, "HOME", "HOME") == pytest.approx(0.0)
    assert _cell(result, "WORK", "WORK") == pytest.approx(0.0)


def test_transition_matrix_sums_to_100():
    df = _simple_visits()
    result = activity_transition_matrix(df)
    # Percentages sum to 100
    assert _matrix_sum(result) == pytest.approx(100.0, abs=1e-6)


def test_transition_matrix_day_filter_weekdays():
    df = _simple_visits()
    df["day_of_week"] = ["monday", "monday", "monday", "monday", "saturday", "saturday", "saturday"]
    result_all = activity_transition_matrix(df)
    result_weekdays = activity_transition_matrix(df, day_filter="weekdays")
    # Weekday-only result excludes Saturday transitions
    assert _matrix_sum(result_weekdays) == pytest.approx(100.0, abs=1e-6)
    # The two should differ because Saturday rows are excluded
    assert not result_all.equals(result_weekdays)


def test_transition_matrix_column_autodetection_agent_id():
    df = _simple_visits().rename(columns={"user_id": "agent_id"})
    result = activity_transition_matrix(df)
    assert _matrix_sum(result) == pytest.approx(100.0, abs=1e-6)


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
    assert _matrix_sum(result) == pytest.approx(100.0, abs=1e-6)


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
    assert set(result["activity"]) == {"HOME"}
    assert set(result.columns) == {"activity", "HOME"}
    assert _matrix_sum(result) == pytest.approx(100.0, abs=1e-6)


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

    assert result["activity"].tolist() == ["UNKNOWN"]
    assert set(result.columns) == {"activity", "UNKNOWN"}
    assert _cell(result, "UNKNOWN", "UNKNOWN") == pytest.approx(100.0)


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


def _simple_visits_arrow():
    """A pyarrow.Table equivalent of `_simple_visits`, with a null activity value."""
    import pyarrow as pa
    import pyarrow.compute as pc

    return pa.table(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 2],
            "start_timestamp": pc.cast(
                pa.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T02:00:00",
                        "2020-01-01T04:00:00",
                        "2020-01-01T06:00:00",
                        "2020-01-01T00:00:00",
                        "2020-01-01T02:00:00",
                        "2020-01-01T04:00:00",
                    ]
                ),
                pa.timestamp("us"),
            ),
            "purpose": ["HOME", "WORK", "HOME", "WORK", "HOME", "WORK", None],
            "day_of_week": ["monday"] * 7,
        }
    )


def test_activity_functions_work_on_pyarrow_backed_input_without_pandas():
    """This file's logic must not depend on pandas: exercise it against a raw
    pyarrow.Table (not a Narwhals-pandas frame), including a null activity
    value that exercises the fallback path previously implemented via
    ``pd.isna``.
    """
    pytest.importorskip("pyarrow")
    visits = _simple_visits_arrow()

    with pytest.warns(UserWarning, match="null activity"):
        transitions = activity_transition_matrix(visits)
    assert transitions.__class__.__module__.startswith("pyarrow")
    transitions_pd = transitions.to_pandas()
    assert _matrix_sum(transitions_pd) == pytest.approx(100.0, abs=1e-6)
    assert "UNKNOWN" in set(transitions_pd["activity"])

    with pytest.warns(UserWarning, match="null activity"):
        purposes = visit_purpose_distribution(visits)
    purposes_pd = purposes.to_pandas()
    assert purposes_pd["percentage"].sum() == pytest.approx(100.0, abs=1e-6)

    with pytest.warns(UserWarning, match="null activity"):
        matrix, categories, n_bins = daily_activity_distribution(visits)
    assert n_bins == 144
    assert "UNKNOWN" in categories
    assert matrix.shape == (len(categories), 144)
