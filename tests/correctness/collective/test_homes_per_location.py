"""Correctness tests for fastmob.measures.collective.homes_per_location."""

from __future__ import annotations

import pandas as pd
import pytest

from fastmob.measures.collective.homes_per_location import homes_per_location


@pytest.fixture()
def night_traj():
    """Two users; both have nighttime records at (0,0), u2 also visits (1,0)."""
    # Nighttime is 22:00-07:00; these records are at 23:00.
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u2", "u2"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 23:00",
                    "2020-01-02 00:00",
                    "2020-01-01 23:00",
                    "2020-01-02 00:00",
                    "2020-01-02 01:00",
                ]
            ),
            "lat": [0.0, 0.0, 0.0, 0.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_homes_per_location_shared_home(night_traj):
    """Both users home at (0,0) -> n_homes = 2."""
    result = homes_per_location(night_traj)
    row = result[(result["lat"] == 0.0) & (result["lng"] == 0.0)]
    assert len(row) == 1
    assert row["n_homes"].iloc[0] == 2


def test_homes_per_location_columns(night_traj):
    """Result has lat, lng, n_homes columns."""
    result = homes_per_location(night_traj)
    assert "lat" in result.columns
    assert "lng" in result.columns
    assert "n_homes" in result.columns


def test_homes_per_location_total_equals_n_users(night_traj):
    """Sum of n_homes equals the number of distinct users."""
    result = homes_per_location(night_traj)
    n_users = night_traj["uid"].nunique()
    assert result["n_homes"].sum() == n_users


def test_homes_per_location_returns_pandas(night_traj):
    """Result is pandas DataFrame when input is pandas."""
    result = homes_per_location(night_traj)
    assert isinstance(result, pd.DataFrame)


def test_homes_per_location_sorted_descending(night_traj):
    """Rows sorted by n_homes descending."""
    result = homes_per_location(night_traj)
    counts = list(result["n_homes"])
    assert counts == sorted(counts, reverse=True)
