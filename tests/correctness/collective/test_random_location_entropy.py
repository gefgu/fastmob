"""Correctness tests for fastmob.measures.collective.random_location_entropy."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fastmob.measures.collective.random_location_entropy import random_location_entropy


@pytest.fixture()
def traj_multi_user():
    """Two users share one location, each has one unique location."""
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u2", "u2"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 01:00",
                    "2020-01-01 03:00",
                    "2020-01-01 04:00",
                ]
            ),
            "lat": [0.0, 1.0, 0.0, 2.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_random_location_entropy_shared_location(traj_multi_user):
    """A location visited by 2 users has random entropy log2(2) = 1.0."""
    result = random_location_entropy(traj_multi_user)
    row_shared = result[(result["lat"] == 0.0) & (result["lng"] == 0.0)]
    assert len(row_shared) == 1
    assert abs(row_shared["random_entropy"].iloc[0] - 1.0) < 1e-10


def test_random_location_entropy_unique_location(traj_multi_user):
    """A location visited by exactly 1 user has random entropy 0.0."""
    result = random_location_entropy(traj_multi_user)
    row_unique = result[(result["lat"] == 1.0) & (result["lng"] == 0.0)]
    assert len(row_unique) == 1
    assert abs(row_unique["random_entropy"].iloc[0] - 0.0) < 1e-10


def test_random_location_entropy_columns(traj_multi_user):
    """Result contains lat, lng, random_entropy columns."""
    result = random_location_entropy(traj_multi_user)
    assert "lat" in result.columns
    assert "lng" in result.columns
    assert "random_entropy" in result.columns


def test_random_location_entropy_no_uid():
    """Without a uid column all rows are one user; entropy is 0 everywhere."""
    traj = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "lat": [0.0, 1.0],
            "lng": [0.0, 1.0],
        }
    )
    result = random_location_entropy(traj)
    assert (result["random_entropy"] == 0.0).all()


def test_random_location_entropy_returns_pandas(traj_multi_user):
    """Result is a pandas DataFrame when input is pandas."""
    result = random_location_entropy(traj_multi_user)
    assert isinstance(result, pd.DataFrame)


def test_random_location_entropy_three_users():
    """Location visited by 3 distinct users has entropy log2(3)."""
    traj = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u3"],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
            "lat": [0.0, 0.0, 0.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = random_location_entropy(traj)
    assert len(result) == 1
    assert abs(result["random_entropy"].iloc[0] - math.log2(3)) < 1e-10
