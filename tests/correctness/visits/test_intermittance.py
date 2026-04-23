"""Correctness tests for intermittance_and_degree_of_return in skmob2.measures.visits.intermittance."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from skmob2.measures.visits.intermittance_and_degree_of_return import intermittance_and_degree_of_return


def _agent_visits():
    """Three visits: HOME, unknown_place, HOME.
    Expected: one exploration (unknown_place), surrounded by returns.
    """
    return pd.DataFrame({
        "agent_id":   ["u1", "u1", "u1"],
        "location_id": ["home_loc", "work_loc", "home_loc"],
        "purpose":    ["HOME", "LEISURE", "HOME"],
        "duration_steps": [10, 5, 10],
    })


def test_pure_return_agent():
    """Agent who only visits HOME: all time is return time, exploration=0."""
    df = pd.DataFrame({
        "agent_id":     ["u1", "u1", "u1"],
        "location_id":  ["h", "h", "h"],
        "purpose":      ["HOME", "HOME", "HOME"],
        "duration_steps": [10, 10, 10],
    })
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
    assert row["intermittency"] == pytest.approx(
        row["mean_exploration"] + row["mean_return"], abs=1e-9
    )
    # degree_of_return is arctan(mean_return / mean_exploration)
    expected_dor = np.arctan(row["mean_return"] / row["mean_exploration"])
    assert row["degree_of_return"] == pytest.approx(expected_dor, abs=1e-9)


def test_multi_user():
    df = pd.DataFrame({
        "agent_id":     ["u1", "u1", "u2", "u2"],
        "location_id":  ["h", "other", "h", "h"],
        "purpose":      ["HOME", "LEISURE", "HOME", "HOME"],
        "duration_steps": [5, 5, 10, 10],
    })
    result = intermittance_and_degree_of_return(df)
    assert len(result) == 2
    assert set(result["agent_id"]) == {"u1", "u2"}


def test_column_autodetection_user_id():
    """Auto-detects 'user_id' when 'agent_id' is absent."""
    df = pd.DataFrame({
        "user_id":      ["u1", "u1"],
        "location_id":  ["h", "other"],
        "purpose":      ["HOME", "LEISURE"],
        "duration_steps": [10, 5],
    })
    result = intermittance_and_degree_of_return(df)
    assert "user_id" in result.columns
    assert len(result) == 1


def test_combine_purpose_false():
    """When combine_purpose_with_location=False, location key is just location_id."""
    df = pd.DataFrame({
        "agent_id":     ["u1", "u1", "u1"],
        "location_id":  ["h", "other", "h"],
        "purpose":      ["HOME", "LEISURE", "HOME"],
        "duration_steps": [10, 5, 10],
    })
    result_combined = intermittance_and_degree_of_return(
        df, combine_purpose_with_location=True
    )
    result_plain = intermittance_and_degree_of_return(
        df, combine_purpose_with_location=False
    )
    # Both should produce a result; exact values differ only in edge cases
    assert len(result_combined) == 1
    assert len(result_plain) == 1


def test_result_columns():
    """Result must contain the expected columns."""
    df = _agent_visits()
    result = intermittance_and_degree_of_return(df)
    for col in ["intermittency", "degree_of_return", "mean_return", "mean_exploration"]:
        assert col in result.columns


def test_single_visit_user():
    """User with a single visit: no transitions, should still return a row."""
    df = pd.DataFrame({
        "agent_id":     ["u1"],
        "location_id":  ["h"],
        "purpose":      ["HOME"],
        "duration_steps": [10],
    })
    result = intermittance_and_degree_of_return(df)
    assert len(result) == 1


def test_pure_exploration_agent():
    """Agent who only visits new places: exploration > 0, degree_of_return near 0."""
    df = pd.DataFrame({
        "agent_id":     ["u1", "u1", "u1"],
        "location_id":  ["a", "b", "c"],
        "purpose":      ["SHOP", "LEISURE", "OTHER"],
        "duration_steps": [10, 10, 10],
    })
    result = intermittance_and_degree_of_return(df)
    row = result.iloc[0]
    assert row["mean_exploration"] > 0
    assert row["mean_return"] == pytest.approx(0.0)
    assert row["degree_of_return"] == pytest.approx(0.0, abs=1e-9)
