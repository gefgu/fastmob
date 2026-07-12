"""Correctness tests for fastmob.measures.collective.uncorrelated_location_entropy."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fastmob.measures.collective.uncorrelated_location_entropy import uncorrelated_location_entropy


@pytest.fixture()
def traj_equal_visits():
    """Two users each visit location A once (equal share -> max entropy)."""
    return pd.DataFrame(
        {
            "uid": ["u1", "u2"],
            "datetime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00"]),
            "lat": [0.0, 0.0],
            "lng": [0.0, 0.0],
        }
    )


def test_uncorrelated_location_entropy_equal_visits(traj_equal_visits):
    """Equal visits from 2 users gives Shannon entropy = 1.0 bit."""
    result = uncorrelated_location_entropy(traj_equal_visits)
    assert len(result) == 1
    # p1 = p2 = 0.5; H = -2*(0.5*log2(0.5)) = 1.0
    assert abs(result["uncorrelated_entropy"].iloc[0] - 1.0) < 1e-10


def test_uncorrelated_location_entropy_single_user():
    """A location visited only by one user has entropy 0."""
    traj = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "lat": [0.0, 0.0],
            "lng": [0.0, 0.0],
        }
    )
    result = uncorrelated_location_entropy(traj)
    assert len(result) == 1
    assert abs(result["uncorrelated_entropy"].iloc[0] - 0.0) < 1e-10


def test_uncorrelated_location_entropy_columns():
    """Result contains lat, lng, uncorrelated_entropy columns."""
    traj = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": pd.to_datetime(["2020-01-01"]),
            "lat": [1.0],
            "lng": [2.0],
        }
    )
    result = uncorrelated_location_entropy(traj)
    assert "lat" in result.columns
    assert "lng" in result.columns
    assert "uncorrelated_entropy" in result.columns


def test_uncorrelated_location_entropy_no_uid():
    """Without uid column entropy is 0 everywhere (single user)."""
    traj = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "lat": [0.0, 1.0],
            "lng": [0.0, 1.0],
        }
    )
    result = uncorrelated_location_entropy(traj)
    assert (result["uncorrelated_entropy"] == 0.0).all()


def test_uncorrelated_location_entropy_known_value():
    """Verify against hand-computed value: u1 visits A 3x, u2 visits A 1x."""
    # p_u1 = 3/4, p_u2 = 1/4
    # H = -(3/4)*log2(3/4) - (1/4)*log2(1/4)
    traj = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1", "u2"],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = uncorrelated_location_entropy(traj)
    expected = -(3 / 4) * math.log2(3 / 4) - (1 / 4) * math.log2(1 / 4)
    assert abs(result["uncorrelated_entropy"].iloc[0] - expected) < 1e-10


def test_uncorrelated_location_entropy_returns_pandas():
    """Result is a pandas DataFrame when input is pandas."""
    traj = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": pd.to_datetime(["2020-01-01"]),
            "lat": [0.0],
            "lng": [0.0],
        }
    )
    result = uncorrelated_location_entropy(traj)
    assert isinstance(result, pd.DataFrame)
