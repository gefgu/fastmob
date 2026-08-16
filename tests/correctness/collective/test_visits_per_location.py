"""Correctness tests for fastmob.measures.collective.visits_per_location."""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.measures.collective.visits_per_location import visits_per_location


@pytest.fixture()
def simple_traj():
    """Three users sharing global location IDs."""
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
            "location_id": ["A", "B", "A", "A", "C", "A"],
        }
    )


def test_visits_per_location_counts(simple_traj):
    """Location A has 4 visits, while B and C have one each."""
    result = visits_per_location(simple_traj)
    counts = dict(zip(result["location_id"], result["n_visits"]))
    assert counts == {"A": 4, "B": 1, "C": 1}


def test_visits_per_location_columns(simple_traj):
    """Result has location_id and n_visits columns."""
    result = visits_per_location(simple_traj)
    assert "location_id" in result.columns
    assert "lat" not in result.columns
    assert "lng" not in result.columns
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
            "location_id": ["A", "A", "B"],
        }
    )
    result = visits_per_location(traj)
    counts = dict(zip(result["location_id"], result["n_visits"]))
    assert counts == {"A": 2, "B": 1}


def test_visits_per_location_counts_unsorted_input():
    """Collective location counts do not depend on chronological ordering."""
    traj = pd.DataFrame(
        {
            "uid": ["u2", "u1", "u2", "u1"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-02 02:00",
                    "2020-01-01 03:00",
                    "2020-01-01 01:00",
                    "2020-01-02 01:00",
                ]
            ),
            "location_id": ["B", "A", "B", "A"],
        }
    )

    result = visits_per_location(traj)
    counts = dict(zip(result["location_id"], result["n_visits"]))

    assert counts == {"A": 2, "B": 2}


def test_visits_per_location_explicit_location_id_column():
    """A custom location-ID column can be selected explicitly."""
    traj = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "place": ["A", "A"],
        }
    )
    result = visits_per_location(traj, datetime_col="timestamp", location_id_col="place")
    assert result.to_dict("records") == [{"place": "A", "n_visits": 2}]
