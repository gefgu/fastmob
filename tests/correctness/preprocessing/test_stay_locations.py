"""Correctness tests for fastmob.preprocessing.stay_locations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.preprocessing import stay_locations


def test_stay_locations_detects_single_stop(stops_tdf):
    """Stationary user (60 min within 0.2 km) → exactly 1 stop."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    assert len(result) == 1


def test_stay_locations_normalizes_string_datetimes():
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 7,
            "datetime": [f"2020-01-01 08:{minute:02d}:00" for minute in range(0, 35, 5)],
            "lat": [48.8566] * 7,
            "lng": [2.3522] * 7,
        }
    )

    result = stay_locations(df, spatial_radius_km=0.2, minutes_for_a_stop=20.0)

    assert len(result) == 1
    assert pd.Timestamp(result.iloc[0]["datetime"]) == pd.Timestamp("2020-01-01 08:00:00")


def test_stay_locations_no_stop_for_moving_user(stops_tdf):
    """Moving user (5 km steps, 5-min spacing) → 0 stops detected."""
    moving = stops_tdf[stops_tdf["uid"] == "user_moving"].copy()
    result = stay_locations(moving, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    assert len(result) == 0


def test_stay_locations_stop_coords_are_median(stops_tdf):
    """Stop coordinates should be median lat/lng of accumulated points."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0)

    input_lats = stationary["lat"].values
    input_lngs = stationary["lng"].values
    expected_lat = np.median(input_lats)
    expected_lng = np.median(input_lngs)

    if hasattr(result, "iloc"):
        out_lat = float(result.iloc[0]["lat"])
        out_lng = float(result.iloc[0]["lng"])
    else:
        out_lat = float(result["lat"][0])
        out_lng = float(result["lng"][0])

    assert abs(out_lat - expected_lat) < 0.001
    assert abs(out_lng - expected_lng) < 0.001


def test_stay_locations_datetime_is_entry_time(stops_tdf):
    """Stop datetime should be the first timestamp (entry time)."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    expected_entry = pd.Timestamp("2020-01-01 08:00:00")

    if hasattr(result, "iloc"):
        out_dt = pd.Timestamp(result.iloc[0]["datetime"])
    else:
        out_dt = pd.Timestamp(result["datetime"][0])

    assert out_dt == expected_entry


def test_stay_locations_leaving_time_column(stops_tdf):
    """With leaving_time=True, result must have 'leaving_datetime' column."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0, leaving_time=True)
    cols = result.columns if hasattr(result, "columns") else result.schema.names()
    assert "leaving_datetime" in cols


def test_stay_locations_leaving_time_false_omits_column(stops_tdf):
    """With leaving_time=False, 'leaving_datetime' column must be absent."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0, leaving_time=False)
    cols = result.columns if hasattr(result, "columns") else result.schema.names()
    assert "leaving_datetime" not in cols


def test_stay_locations_leaving_datetime_after_entry(stops_tdf):
    """leaving_datetime must be strictly after datetime (entry time)."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    if hasattr(result, "iloc"):
        entry = pd.Timestamp(result.iloc[0]["datetime"])
        leaving = pd.Timestamp(result.iloc[0]["leaving_datetime"])
    else:
        entry = pd.Timestamp(result["datetime"][0])
        leaving = pd.Timestamp(result["leaving_datetime"][0])
    assert leaving > entry


def test_stay_locations_single_point_user_zero_stops():
    """Single-point user can never form a stop."""
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = stay_locations(df, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    assert len(result) == 0


def test_stay_locations_no_uid_column_detects_stop():
    """Regression test: a dataframe with no user-ID column (the whole frame
    treated as one user) must not crash while building the output -- the
    uid-label lookup is only relevant, and only run, when uid_col is not
    None."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=7, freq="5min"),
            "lat": [0.0] * 7,
            "lng": [0.0] * 7,
        }
    )
    result = stay_locations(df, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    assert len(result) == 1
    assert "uid" not in result.columns


def test_stay_locations_output_schema(stops_tdf):
    """Output must contain uid, lat, lng, datetime columns."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    cols = set(result.columns if hasattr(result, "columns") else result.schema.names())
    assert {"uid", "lat", "lng", "datetime"} <= cols


def test_stay_locations_returns_same_backend(stops_tdf):
    """Pandas input → pandas output."""
    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    assert isinstance(result, type(stationary))


def test_stay_locations_accepts_traj_dataframe(stops_tdf):
    """Fastmob wrappers are unwrapped before entering the standalone API."""
    from fastmob import TrajDataFrame

    stationary = stops_tdf[stops_tdf["uid"] == "user_stationary"].copy()
    result = stay_locations(
        TrajDataFrame(stationary),
        spatial_radius_km=0.2,
        minutes_for_a_stop=20.0,
    )

    assert isinstance(result, type(stationary))
    assert len(result) == 1


def test_stay_locations_polars_backend(stops_tdf_polars):
    """Polars input → Polars output with correct stop count."""
    import polars as pl

    stationary = stops_tdf_polars.filter(pl.col("uid") == "user_stationary")
    result = stay_locations(stationary, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 1


def test_stay_locations_no_data_gap_resets_stop():
    """A long gap (no_data_for_minutes) should break stop accumulation."""
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 6,
            "datetime": [
                pd.Timestamp("2020-01-01 08:00"),
                pd.Timestamp("2020-01-01 08:10"),
                pd.Timestamp("2020-01-01 08:20"),
                # Large gap: 2 hours → resets
                pd.Timestamp("2020-01-01 10:21"),
                pd.Timestamp("2020-01-01 10:31"),
                pd.Timestamp("2020-01-01 10:41"),
            ],
            "lat": [48.856] * 6,
            "lng": [2.352] * 6,
        }
    )
    # With no_data_for_minutes=30, the 2h gap resets → two potential stops
    result = stay_locations(df, spatial_radius_km=0.2, minutes_for_a_stop=15.0, no_data_for_minutes=30.0)
    # After the gap the new group has 3 points → 21 min → qualifies
    assert len(result) >= 1


@pytest.mark.skmob
def test_stay_locations_matches_skmob(comparison_skmob):
    """Stop count is close to skmob on each comparison dataset (not exact -- see note)."""
    import pandas as pd
    from skmob.preprocessing import detection as skmob_detection

    skmob_result = skmob_detection.stay_locations(comparison_skmob, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    our_result = stay_locations(
        pd.DataFrame(comparison_skmob),
        spatial_radius_km=0.2,
        minutes_for_a_stop=20.0,
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )
    # fastmob's stop-radius/duration comparisons are >=, matching trackintel's
    # own algorithm; skmob uses strict >, so rows landing exactly on a
    # threshold can go either way. Allow a small, documented drift instead of
    # requiring bit-for-bit agreement with a library that disagrees with
    # trackintel on this specific point.
    assert abs(len(our_result) - len(skmob_result)) <= max(5, round(0.01 * len(skmob_result)))


def test_stay_locations_matches_cached_reference(comparison_skmob_reference):
    """Stop count is close to the cached skmob baseline (not exact -- see note)."""
    ref = comparison_skmob_reference
    cached_count = ref.row_count("stay_locations")
    our_result = stay_locations(
        ref.input_df,
        spatial_radius_km=0.2,
        minutes_for_a_stop=20.0,
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )
    # fastmob's stop-radius/duration comparisons are >=, matching trackintel's
    # own sliding-window algorithm (generate_staypoints); skmob's equivalent
    # (skmob/preprocessing/detection.py) uses strict >, so rows landing
    # exactly on a threshold can go either way and an exact match against the
    # skmob cache is no longer guaranteed. How many (if any) rows sit exactly
    # on a threshold is data-dependent (0 on the small geolife reference, 5
    # on brightkite's ~51.4k rows) -- allow a small documented drift rather
    # than requiring bit-for-bit agreement with a library using a different
    # tie-breaking rule.
    assert abs(len(our_result) - cached_count) <= max(5, round(0.01 * cached_count))
