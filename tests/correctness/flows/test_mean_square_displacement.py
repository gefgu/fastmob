"""Correctness tests for skmob2.measures.flows.mean_square_displacement."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from skmob2.measures.flows.mean_square_displacement import mean_square_displacement
from skmob2._core import square_displacement_km2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_user_traj():
    """Two users: u1 stays at origin within 1 h; u2 moves to (0, 1) at 0:30."""
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u2"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 01:30",  # outside 1-h window
                    "2020-01-01 00:00",
                    "2020-01-01 00:30",
                ]
            ),
            "lat": [0.0, 1.0, 0.0, 0.0],
            "lng": [0.0, 0.0, 0.0, 1.0],
        }
    )


@pytest.fixture()
def single_user_stationary():
    """Single user that does not move."""
    return pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30"]),
            "lat": [0.0, 0.0],
            "lng": [0.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# Known-value tests
# ---------------------------------------------------------------------------


def test_msd_two_users_known_value(two_user_traj):
    """MSD equals (0 + haversine(0,0->0,1)^2) / 2 for the two-user fixture."""
    expected = (0.0 + square_displacement_km2(0.0, 0.0, 0.0, 1.0)) / 2.0
    result = mean_square_displacement(two_user_traj, hours=1)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_msd_stationary_user_is_zero(single_user_stationary):
    """A user who does not leave their starting point has MSD = 0."""
    result = mean_square_displacement(single_user_stationary, hours=1)
    assert result == pytest.approx(0.0)


def test_msd_returns_float(two_user_traj):
    """Return type is Python float."""
    result = mean_square_displacement(two_user_traj, hours=1)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Behaviour tests
# ---------------------------------------------------------------------------


def test_msd_no_uid_column():
    """Works when there is no uid column (treats whole trajectory as one user)."""
    traj = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30"]),
            "lat": [0.0, 0.0],
            "lng": [0.0, 1.0],
        }
    )
    expected = square_displacement_km2(0.0, 0.0, 0.0, 1.0)
    result = mean_square_displacement(traj, hours=1)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_msd_window_too_narrow_displacement_zero():
    """When delta_t is 0, rt == r0 and MSD is 0 for all users."""
    traj = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
            "lat": [0.0, 1.0],
            "lng": [0.0, 1.0],
        }
    )
    result = mean_square_displacement(traj, days=0, hours=0, minutes=0)
    assert result == pytest.approx(0.0)


def test_msd_window_covers_full_trajectory():
    """When delta_t is large enough to cover all points, rt is the last point."""
    traj = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 23:00"]),
            "lat": [0.0, 0.0],
            "lng": [0.0, 1.0],
        }
    )
    expected = square_displacement_km2(0.0, 0.0, 0.0, 1.0)
    result = mean_square_displacement(traj, hours=24)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_msd_explicit_column_names():
    """Explicit column name overrides are respected."""
    traj = pd.DataFrame(
        {
            "user": ["u1"],
            "time": pd.to_datetime(["2020-01-01 00:00"]),
            "latitude": [0.0],
            "longitude": [0.0],
        }
    )
    result = mean_square_displacement(
        traj,
        hours=1,
        datetime_col="time",
        lat_col="latitude",
        lng_col="longitude",
        uid_col="user",
    )
    assert result == pytest.approx(0.0)
