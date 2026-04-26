"""Correctness tests for skmob2.measures.flows.visits_per_location."""

from __future__ import annotations

import pandas as pd
import pytest

from skmob2.measures.flows.visits_per_location import visits_per_location


@pytest.fixture()
def simple_traj():
    """Three users; location A appears 3 times, location B twice, C once."""
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                ]
            ),
            "lat": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_visits_per_location_counts(simple_traj):
    """Location (0,0) has 4 visits, (1,0) has 1, (2,0) has 1."""
    result = visits_per_location(simple_traj)
    counts = {(row["lat"], row["lng"]): row["n_visits"] for _, row in result.iterrows()}
    assert counts[(0.0, 0.0)] == 4
    assert counts[(1.0, 0.0)] == 1
    assert counts[(2.0, 0.0)] == 1


def test_visits_per_location_columns(simple_traj):
    """Result has lat, lng, n_visits columns."""
    result = visits_per_location(simple_traj)
    assert "lat" in result.columns
    assert "lng" in result.columns
    assert "n_visits" in result.columns


def test_visits_per_location_sorted_descending(simple_traj):
    """Rows are sorted by n_visits descending."""
    result = visits_per_location(simple_traj)
    counts = list(result["n_visits"])
    assert counts == sorted(counts, reverse=True)


def test_visits_per_location_total_equals_rows(simple_traj):
    """Sum of n_visits equals the total number of trajectory rows."""
    result = visits_per_location(simple_traj)
    assert result["n_visits"].sum() == len(simple_traj)


def test_visits_per_location_returns_pandas(simple_traj):
    """Result is pandas DataFrame when input is pandas."""
    result = visits_per_location(simple_traj)
    assert isinstance(result, pd.DataFrame)


def test_visits_per_location_no_uid():
    """Function works without a uid column."""
    traj = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
            "lat": [0.0, 0.0, 1.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = visits_per_location(traj)
    counts = {(row["lat"], row["lng"]): row["n_visits"] for _, row in result.iterrows()}
    assert counts[(0.0, 0.0)] == 2
    assert counts[(1.0, 0.0)] == 1
