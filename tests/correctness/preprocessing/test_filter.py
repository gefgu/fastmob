"""Correctness tests for skmob2.preprocessing.filter."""

from __future__ import annotations

import pandas as pd
import pytest

from skmob2.preprocessing import filter as traj_filter


def test_filter_clean_trajectory_unchanged(filter_tdf):
    """Clean trajectory (no high-speed points) should come back intact."""
    clean = filter_tdf[filter_tdf["uid"] == "user_clean"].copy()
    result = traj_filter(clean, max_speed_kmh=500.0)
    assert len(result) == 5


def test_filter_removes_high_speed_point(filter_tdf):
    """Point teleported to null island must be removed from user_noisy."""
    noisy = filter_tdf[filter_tdf["uid"] == "user_noisy"].copy()
    result = traj_filter(noisy, max_speed_kmh=500.0)
    # The null-island point (lat=0, lng=0) causes impossible speed → removed
    assert len(result) == 4
    assert not ((result["lat"] == 0.0) & (result["lng"] == 0.0)).any()


def test_filter_full_df_clean_user_unchanged(filter_tdf):
    """With multi-user df, clean user keeps all 5 points."""
    result = traj_filter(filter_tdf, max_speed_kmh=500.0)
    clean_result = result[result["uid"] == "user_clean"]
    assert len(clean_result) == 5


def test_filter_full_df_noisy_user_filtered(filter_tdf):
    """With multi-user df, noisy user loses the high-speed point."""
    result = traj_filter(filter_tdf, max_speed_kmh=500.0)
    noisy_result = result[result["uid"] == "user_noisy"]
    assert len(noisy_result) == 4


def test_filter_single_point_user():
    """A single-point trajectory is returned unchanged."""
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = traj_filter(df, max_speed_kmh=500.0)
    assert len(result) == 1


def test_filter_returns_same_backend_type(filter_tdf):
    """Output backend must match input (pandas in → pandas out)."""
    result = traj_filter(filter_tdf, max_speed_kmh=500.0)
    assert isinstance(result, type(filter_tdf))


def test_filter_polars_backend(filter_tdf_polars):
    """Polars input produces a Polars output with the same filtering result."""
    import polars as pl

    result = traj_filter(filter_tdf_polars, max_speed_kmh=500.0)
    assert isinstance(result, pl.DataFrame)
    clean = result.filter(pl.col("uid") == "user_clean")
    noisy = result.filter(pl.col("uid") == "user_noisy")
    assert len(clean) == 5
    assert len(noisy) == 4


def test_filter_zero_dt_removes_duplicate_timestamps():
    """Two points at the same timestamp trigger ZeroDivisionError → remove second."""
    df = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-01 00:00:00"),  # same timestamp as prev → ZeroDivision
                pd.Timestamp("2020-01-01 01:00:00"),
            ],
            "lat": [48.856, 48.900, 48.856],  # big jump at same timestamp
            "lng": [2.352, 2.352, 2.352],
        }
    )
    result = traj_filter(df, max_speed_kmh=500.0)
    # The duplicate-timestamp point should be removed
    assert len(result) == 2


@pytest.mark.skmob
def test_filter_matches_skmob(brightkite_skmob):
    """Results must closely match skmob despite tiny Haversine threshold drift."""
    from skmob.preprocessing import filtering as skmob_filtering
    import pandas as pd

    skmob_result = skmob_filtering.filter(brightkite_skmob, max_speed_kmh=500.0)
    skmob_result_df = pd.DataFrame(skmob_result)

    our_result = traj_filter(
        pd.DataFrame(brightkite_skmob),
        max_speed_kmh=500.0,
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )

    # skmob uses its Python gislib Haversine implementation while skmob2 uses
    # the Rust geo kernel. Points whose speed is exactly near the threshold can
    # fall on different sides, so compare the retained count with a tiny budget.
    assert abs(len(our_result) - len(skmob_result_df)) <= 5
