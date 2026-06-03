"""Correctness tests for skmob2.measures.collective.visits_per_time_unit."""

from __future__ import annotations

import pandas as pd
import pytest

from skmob2.measures.collective.visits_per_time_unit import visits_per_time_unit


@pytest.fixture()
def hourly_traj():
    """6 records spread across 3 hours (2 per hour)."""
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u2", "u1", "u2"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 00:30",
                    "2020-01-01 01:00",
                    "2020-01-01 01:30",
                    "2020-01-01 02:00",
                    "2020-01-01 02:30",
                ]
            ),
            "lat": [0.0] * 6,
            "lng": [0.0] * 6,
        }
    )


def test_visits_per_time_unit_hourly_counts(hourly_traj):
    """Each hour bin contains exactly 2 visits."""
    result = visits_per_time_unit(hourly_traj, freq="1h")
    assert len(result) == 3
    assert (result["n_visits"] == 2).all()


def test_visits_per_time_unit_total_visits(hourly_traj):
    """Sum of n_visits equals total trajectory rows."""
    result = visits_per_time_unit(hourly_traj, freq="1h")
    assert result["n_visits"].sum() == len(hourly_traj)


def test_visits_per_time_unit_columns(hourly_traj):
    """Result has datetime and n_visits columns."""
    result = visits_per_time_unit(hourly_traj, freq="1h")
    assert "n_visits" in result.columns
    assert "datetime" in result.columns


def test_visits_per_time_unit_polars_backend_matches_input(hourly_traj):
    pl = pytest.importorskip("polars", reason="Polars not installed")
    result = visits_per_time_unit(pl.from_pandas(hourly_traj), freq="1h")
    assert isinstance(result, pl.DataFrame)
    assert result.get_column("n_visits").to_list() == [2, 2, 2]


def test_visits_per_time_unit_polars_supports_pandas_minute_alias(hourly_traj):
    pl = pytest.importorskip("polars", reason="Polars not installed")
    result = visits_per_time_unit(pl.from_pandas(hourly_traj), freq="15min")
    assert isinstance(result, pl.DataFrame)
    assert result.get_column("n_visits").to_list() == [1, 1, 1, 1, 1, 1]


def test_visits_per_time_unit_no_empty_bins(hourly_traj):
    """Bins with zero visits are not included in the result."""
    result = visits_per_time_unit(hourly_traj, freq="1h")
    assert (result["n_visits"] > 0).all()


def test_visits_per_time_unit_daily_freq():
    """Daily frequency bins all records from one day into one row."""
    traj = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 08:00",
                    "2020-01-01 12:00",
                    "2020-01-01 18:00",
                ]
            ),
            "lat": [0.0, 0.0, 0.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = visits_per_time_unit(traj, freq="1D")
    assert len(result) == 1
    assert result["n_visits"].iloc[0] == 3


def test_visits_per_time_unit_counts_unsorted_input():
    """Collective time counts do not depend on input row order."""
    traj = pd.DataFrame(
        {
            "uid": ["u2", "u1", "u1", "u2"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 01:30",
                    "2020-01-01 00:30",
                    "2020-01-01 01:00",
                    "2020-01-01 00:00",
                ]
            ),
            "lat": [0.0] * 4,
            "lng": [0.0] * 4,
        }
    )

    result = visits_per_time_unit(traj, freq="1h")

    assert result["n_visits"].to_list() == [2, 2]


def test_visits_per_time_unit_pandas_supports_minute_alias(hourly_traj):
    result = visits_per_time_unit(hourly_traj, freq="15min")
    assert result["n_visits"].to_list() == [1, 1, 1, 1, 1, 1]


def test_visits_per_time_unit_falls_back_for_pandas_week_alias(hourly_traj):
    result = visits_per_time_unit(hourly_traj, freq="1W")
    assert result["n_visits"].to_list() == [6]


def test_visits_per_time_unit_falls_back_for_pandas_month_start_alias(hourly_traj):
    result = visits_per_time_unit(hourly_traj, freq="1MS")
    assert result["n_visits"].to_list() == [6]


def test_visits_per_time_unit_polars_fallback_preserves_backend(hourly_traj):
    pl = pytest.importorskip("polars", reason="Polars not installed")
    result = visits_per_time_unit(pl.from_pandas(hourly_traj), freq="1W")
    assert isinstance(result, pl.DataFrame)
    assert result.get_column("n_visits").to_list() == [6]


def test_visits_per_time_unit_sorted_chronologically(hourly_traj):
    """Bins are in chronological order."""
    result = visits_per_time_unit(hourly_traj, freq="1h")
    datetimes = list(result["datetime"])
    assert datetimes == sorted(datetimes)
