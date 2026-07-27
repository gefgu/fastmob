"""Correctness tests for intermittance_and_degree_of_return in fastmob.measures.individual.intermittance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.measures.individual.mobility_profiling import intermittance_and_degree_of_return


def _agent_visits():
    """Three visits: HOME, unknown_place, HOME.
    Expected: one exploration (unknown_place), surrounded by returns.
    """
    return pd.DataFrame(
        {
            "agent_id": ["u1", "u1", "u1"],
            "location_id": ["home_loc", "work_loc", "home_loc"],
            "purpose": ["HOME", "LEISURE", "HOME"],
            "duration_steps": [10, 5, 10],
        }
    )


def test_pure_return_agent():
    """Agent who only visits HOME: all time is return time, exploration=0."""
    df = pd.DataFrame(
        {
            "agent_id": ["u1", "u1", "u1"],
            "location_id": ["h", "h", "h"],
            "purpose": ["HOME", "HOME", "HOME"],
            "duration_steps": [10, 10, 10],
        }
    )
    result = intermittance_and_degree_of_return(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    assert row["mean_exploration"] == pytest.approx(0.0)
    # degree_of_return = pi/2 when mean_exploration == 0
    assert row["degree_of_return"] == pytest.approx(np.pi / 2, abs=1e-9)


def test_exploration_and_return():
    """One exploration flanked by returns."""
    df = _agent_visits()
    result = intermittance_and_degree_of_return(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    # mean_exploration > 0 (the LEISURE visit)
    assert row["mean_exploration"] > 0
    # mean_return > 0 (the HOME visits)
    assert row["mean_return"] > 0
    # intermittency = mean_exploration + mean_return
    assert row["intermittency"] == pytest.approx(row["mean_exploration"] + row["mean_return"], abs=1e-9)
    # degree_of_return is arctan(mean_return / mean_exploration)
    expected_dor = np.arctan(row["mean_return"] / row["mean_exploration"])
    assert row["degree_of_return"] == pytest.approx(expected_dor, abs=1e-9)


def test_multi_user():
    df = pd.DataFrame(
        {
            "agent_id": ["u1", "u1", "u2", "u2"],
            "location_id": ["h", "other", "h", "h"],
            "purpose": ["HOME", "LEISURE", "HOME", "HOME"],
            "duration_steps": [5, 5, 10, 10],
        }
    )
    result = intermittance_and_degree_of_return(df)
    assert len(result) == 2
    assert set(result["agent_id"]) == {"u1", "u2"}


def test_column_autodetection_user_id():
    """Auto-detects 'user_id' when 'agent_id' is absent."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "location_id": ["h", "other"],
            "purpose": ["HOME", "LEISURE"],
            "duration_steps": [10, 5],
        }
    )
    result = intermittance_and_degree_of_return(df)
    assert "user_id" in result.columns
    assert len(result) == 1


def test_result_columns():
    """Result must contain the expected columns."""
    df = _agent_visits()
    result = intermittance_and_degree_of_return(df)
    for col in ["intermittency", "degree_of_return", "mean_return", "mean_exploration"]:
        assert col in result.columns


def test_polars_backend_matches_input():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    result = intermittance_and_degree_of_return(pl.from_pandas(_agent_visits()))
    assert isinstance(result, pl.DataFrame)
    assert "degree_of_return" in result.columns


def test_single_visit_user():
    """User with a single visit: no transitions, should still return a row."""
    df = pd.DataFrame(
        {
            "agent_id": ["u1"],
            "location_id": ["h"],
            "purpose": ["HOME"],
            "duration_steps": [10],
        }
    )
    result = intermittance_and_degree_of_return(df)
    assert len(result) == 1


def test_pure_exploration_agent():
    """Agent who only visits new places: exploration > 0, degree_of_return near 0.

    cold_start_strategy="none" is required so that all-equal-frequency locations
    are not pre-labelled as known, letting each unique visit count as exploration.
    """
    df = pd.DataFrame(
        {
            "agent_id": ["u1", "u1", "u1"],
            "location_id": ["a", "b", "c"],
            "purpose": ["SHOP", "LEISURE", "OTHER"],
            "duration_steps": [10, 10, 10],
        }
    )
    result = intermittance_and_degree_of_return(df, cold_start_strategy="none")
    row = result.iloc[0]
    assert row["mean_exploration"] > 0
    assert row["mean_return"] == pytest.approx(0.0)
    assert row["degree_of_return"] == pytest.approx(0.0, abs=1e-9)


def test_use_trajectory_expands_aligned_stay_to_5_minute_slices():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "location_id": ["a"],
            "start_timestamp": [pd.Timestamp("2020-01-01 08:00")],
            "end_timestamp": [pd.Timestamp("2020-01-01 08:10")],
        }
    )

    result = intermittance_and_degree_of_return(df, cold_start_strategy="none")
    row = result.iloc[0]

    assert row["mean_exploration"] == pytest.approx(1.0)
    assert row["mean_return"] == pytest.approx(2.0)
    assert row["intermittency"] == pytest.approx(3.0)
    assert row["degree_of_return"] == pytest.approx(np.arctan2(2.0, 1.0))


def test_use_trajectory_expands_non_aligned_stay_to_inner_5_minute_slices():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "location_id": ["a"],
            "start_timestamp": [pd.Timestamp("2020-01-01 08:02")],
            "end_timestamp": [pd.Timestamp("2020-01-01 08:13")],
        }
    )

    result = intermittance_and_degree_of_return(df, cold_start_strategy="none")
    row = result.iloc[0]

    assert row["mean_exploration"] == pytest.approx(1.0)
    assert row["mean_return"] == pytest.approx(1.0)
    assert row["intermittency"] == pytest.approx(2.0)
    assert row["degree_of_return"] == pytest.approx(np.arctan2(1.0, 1.0))


def test_use_trajectory_falls_back_to_rows_without_end_timestamp():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "location_id": ["a"],
            "start_timestamp": [pd.Timestamp("2020-01-01 08:00")],
        }
    )

    result = intermittance_and_degree_of_return(df, cold_start_strategy="none")
    row = result.iloc[0]

    assert row["mean_exploration"] == pytest.approx(1.0)
    assert row["mean_return"] == pytest.approx(0.0)
    assert row["degree_of_return"] == pytest.approx(0.0, abs=1e-9)


def test_use_trajectory_false_ignores_end_timestamp():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "location_id": ["a"],
            "start_timestamp": [pd.Timestamp("2020-01-01 08:00")],
            "end_timestamp": [pd.Timestamp("2020-01-01 08:10")],
        }
    )

    result = intermittance_and_degree_of_return(df, cold_start_strategy="none", use_trajectory=False)
    row = result.iloc[0]

    assert row["mean_exploration"] == pytest.approx(1.0)
    assert row["mean_return"] == pytest.approx(0.0)
    assert row["degree_of_return"] == pytest.approx(0.0, abs=1e-9)


def test_impute_gaps_false_preserves_observed_slices_only():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "location_id": ["home", "shop"],
            "start_timestamp": [
                pd.Timestamp("2020-01-01 02:00"),
                pd.Timestamp("2020-01-01 02:15"),
            ],
            "end_timestamp": [
                pd.Timestamp("2020-01-01 02:05"),
                pd.Timestamp("2020-01-01 02:15"),
            ],
        }
    )

    result = intermittance_and_degree_of_return(df, cold_start_strategy="none", impute_gaps=False)
    row = result.iloc[0]

    assert row["mean_exploration"] == pytest.approx(1.0)
    assert row["mean_return"] == pytest.approx(1.0)
    assert row["intermittency"] == pytest.approx(2.0)
    assert row["degree_of_return"] == pytest.approx(np.arctan2(1.0, 1.0))


def test_impute_gaps_fills_missing_nighttime_slice_with_home_anchor():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "location_id": ["home", "shop"],
            "start_timestamp": [
                pd.Timestamp("2020-01-01 02:00"),
                pd.Timestamp("2020-01-01 02:15"),
            ],
            "end_timestamp": [
                pd.Timestamp("2020-01-01 02:05"),
                pd.Timestamp("2020-01-01 02:15"),
            ],
        }
    )

    result = intermittance_and_degree_of_return(df, cold_start_strategy="none", impute_gaps=True)
    row = result.iloc[0]

    assert row["mean_exploration"] == pytest.approx(1.0)
    assert row["mean_return"] == pytest.approx(2.0)
    assert row["intermittency"] == pytest.approx(3.0)
    assert row["degree_of_return"] == pytest.approx(np.arctan2(2.0, 1.0))


def test_impute_gaps_leaves_missing_slice_absent_without_anchor():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "location_id": ["home", "shop"],
            "start_timestamp": [
                pd.Timestamp("2020-01-01 08:00"),
                pd.Timestamp("2020-01-01 08:15"),
            ],
            "end_timestamp": [
                pd.Timestamp("2020-01-01 08:05"),
                pd.Timestamp("2020-01-01 08:15"),
            ],
        }
    )

    result = intermittance_and_degree_of_return(df, cold_start_strategy="none", impute_gaps=True)
    row = result.iloc[0]

    assert row["mean_exploration"] == pytest.approx(1.0)
    assert row["mean_return"] == pytest.approx(1.0)
    assert row["intermittency"] == pytest.approx(2.0)
    assert row["degree_of_return"] == pytest.approx(np.arctan2(1.0, 1.0))


def test_impute_gaps_polars_backend_matches_input():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "location_id": ["home", "shop"],
            "start_timestamp": [
                pd.Timestamp("2020-01-01 02:00"),
                pd.Timestamp("2020-01-01 02:15"),
            ],
            "end_timestamp": [
                pd.Timestamp("2020-01-01 02:05"),
                pd.Timestamp("2020-01-01 02:15"),
            ],
        }
    )

    result = intermittance_and_degree_of_return(
        pl.from_pandas(df),
        cold_start_strategy="none",
        impute_gaps=True,
    )

    assert isinstance(result, pl.DataFrame)
    assert result["mean_return"].to_list() == pytest.approx([2.0])
