"""Correctness tests for CDR visitation/trip converters."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fastmob.preprocessing import cdr_to_trips_df, cdr_to_visitation_df


def _cdr_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2"],
            "timestamp": [
                pd.Timestamp("2020-01-01 00:00"),
                pd.Timestamp("2020-01-01 00:10"),
                pd.Timestamp("2020-01-01 00:20"),
                pd.Timestamp("2020-01-01 00:40"),
                pd.Timestamp("2020-01-01 00:50"),
                pd.Timestamp("2020-01-01 00:00"),
                pd.Timestamp("2020-01-01 00:30"),
            ],
            "venueId": ["A", "A", "B", "B", "A", "X", "Y"],
            "lat": [0.0, 2.0, 10.0, 12.0, 20.0, 30.0, 40.0],
            "long": [0.0, 2.0, 10.0, 12.0, 20.0, 30.0, 40.0],
        }
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _travel_minutes(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    avg_speed_kmh: float = 50.0,
    circuity: float = 1.2,
) -> float:
    return _haversine_km(lat1, lon1, lat2, lon2) * circuity / avg_speed_kmh * 60.0


def test_cdr_to_visitation_collapses_consecutive_same_venue_runs():
    visits = cdr_to_visitation_df(_cdr_fixture())

    assert list(visits[visits["user_id"] == "u1"]["area"]) == ["A", "B", "A"]
    assert list(visits[visits["user_id"] == "u2"]["area"]) == ["X", "Y"]


def test_cdr_to_visitation_travel_time_end_duration_and_final_stay():
    visits = cdr_to_visitation_df(_cdr_fixture())
    u1 = visits[visits["user_id"] == "u1"].reset_index(drop=True)

    assert u1.loc[0, "end_timestamp"] == pd.Timestamp("2020-01-01 00:10")
    assert u1.loc[0, "duration_minutes"] == 10.0
    assert u1.loc[0, "lat"] == 1.0
    assert u1.loc[0, "lon"] == 1.0
    assert pd.isna(u1.loc[2, "end_timestamp"])
    assert math.isnan(u1.loc[2, "duration_minutes"])
    assert u1.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert u1.loc[0, "day_of_week"] == "wednesday"
    assert u1.loc[0, "purpose"] == "UNKNOWN"
    assert u1.loc[0, "main_mode"] == "UNKNOWN"
    assert u1.loc[0, "weight"] == 1.0


def test_cdr_to_visitation_uses_default_travel_minutes_when_not_clamped():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "timestamp": [pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 01:00")],
            "venueId": ["A", "B"],
            "lat": [0.0, 0.0],
            "long": [0.0, 0.01],
        }
    )

    visits = cdr_to_visitation_df(df)
    trips = cdr_to_trips_df(visits)
    expected_travel_minutes = _travel_minutes(0.0, 0.0, 0.0, 0.01)

    assert visits.loc[0, "duration_minutes"] == pytest.approx(60.0 - expected_travel_minutes, abs=1e-3)
    assert trips.loc[0, "duration_minutes"] == pytest.approx(expected_travel_minutes, abs=1e-3)
    assert trips.loc[0, "duration_minutes"] < 2.0


def test_cdr_to_visitation_clamps_travel_time_to_observed_boundary_gap():
    visits = cdr_to_visitation_df(_cdr_fixture())
    u2 = visits[visits["user_id"] == "u2"].reset_index(drop=True)

    assert u2.loc[0, "end_timestamp"] == pd.Timestamp("2020-01-01 00:00")
    assert u2.loc[0, "duration_minutes"] == 0.0


def test_cdr_custom_travel_parameters_change_visitation_and_trip_durations():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "timestamp": [pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 01:00")],
            "venueId": ["A", "B"],
            "lat": [0.0, 0.0],
            "long": [0.0, 0.01],
        }
    )

    visits = cdr_to_visitation_df(df, avg_speed_kmh=100.0, circuity=1.0)
    trips = cdr_to_trips_df(visits, avg_speed_kmh=100.0, circuity=1.0)
    expected_travel_minutes = _travel_minutes(0.0, 0.0, 0.0, 0.01, avg_speed_kmh=100.0, circuity=1.0)

    assert visits.loc[0, "duration_minutes"] == pytest.approx(60.0 - expected_travel_minutes, abs=1e-3)
    assert trips.loc[0, "duration_minutes"] == pytest.approx(expected_travel_minutes, abs=1e-3)


def test_cdr_to_trips_builds_consecutive_stay_trips_and_numbers_per_user():
    visits = cdr_to_visitation_df(_cdr_fixture())
    trips = cdr_to_trips_df(visits)

    u1 = trips[trips["user_id"] == "u1"].reset_index(drop=True)
    u2 = trips[trips["user_id"] == "u2"].reset_index(drop=True)

    assert list(u1["trip_number"]) == [1, 2]
    assert list(u1["origin_area"]) == ["A", "B"]
    assert list(u1["destination_area"]) == ["B", "A"]
    assert u1.loc[0, "start_timestamp"] == pd.Timestamp("2020-01-01 00:10")
    assert u1.loc[0, "end_timestamp"] == pd.Timestamp("2020-01-01 00:20")
    assert u1.loc[0, "duration_minutes"] == 10.0
    assert u1.loc[0, "Purpose_D"] == "UNKNOWN"
    assert list(u2["trip_number"]) == [1]
    assert u2.loc[0, "origin_area"] == "X"
    assert u2.loc[0, "destination_area"] == "Y"


def test_cdr_custom_columns_default_longitude_maps_to_lon():
    df = _cdr_fixture().rename(
        columns={
            "user_id": "person",
            "timestamp": "when",
            "venueId": "tower",
            "lat": "latitude",
            "long": "longitude",
        }
    )

    visits = cdr_to_visitation_df(
        df,
        user_id_column="person",
        timestamp_column="when",
        venue_column="tower",
        lat_column="latitude",
        lon_column="longitude",
    )

    assert "lon" in visits.columns
    assert "user_id" in visits.columns
    assert visits.iloc[0]["area"] == "A"


def test_cdr_pandas_returns_pandas_backend():
    visits = cdr_to_visitation_df(_cdr_fixture())
    trips = cdr_to_trips_df(visits)

    assert isinstance(visits, pd.DataFrame)
    assert isinstance(trips, pd.DataFrame)


def test_cdr_polars_returns_polars_backend():
    import polars as pl

    visits = cdr_to_visitation_df(pl.from_pandas(_cdr_fixture()))
    trips = cdr_to_trips_df(visits)

    assert isinstance(visits, pl.DataFrame)
    assert isinstance(trips, pl.DataFrame)
    assert visits.select(pl.len()).item() == 5
    assert trips.select(pl.len()).item() == 3


def test_cdr_to_trips_empty_output_has_expected_schema():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "timestamp": [pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 00:10")],
            "venueId": ["A", "A"],
            "lat": [0.0, 1.0],
            "long": [0.0, 1.0],
        }
    )

    visits = cdr_to_visitation_df(df)
    trips = cdr_to_trips_df(visits)

    assert trips.empty
    assert list(trips.columns) == [
        "user_id",
        "trip_number",
        "origin_lat",
        "origin_lon",
        "origin_area",
        "destination_lat",
        "destination_lon",
        "destination_area",
        "start_timestamp",
        "end_timestamp",
        "duration_minutes",
        "Purpose_D",
        "main_mode",
        "date",
        "day_of_week",
    ]
