"""Correctness tests for fastmob.preprocessing.create_activity_flag /
identify_locations.

Uses hand-built `Staypoints`/`Locations`-shaped tables (rather than the full
pipeline) for deterministic, unambiguous ground truth: one user with an
obvious "always home overnight, always at work on weekday daytime" pattern.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.core.locations_dataframe import Locations
from fastmob.core.staypoints_dataframe import Staypoints
from fastmob.preprocessing import create_activity_flag, identify_locations


def _staypoints_df():
    # started_at/finished_at chosen to straddle the 15-minute activity
    # threshold: rows 0/2 are long (activity), row 1 is short (not activity).
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "started_at": pd.to_datetime(["2020-01-06 00:00", "2020-01-06 12:00", "2020-01-06 12:20"]),
            "finished_at": pd.to_datetime(["2020-01-06 00:30", "2020-01-06 12:10", "2020-01-06 13:00"]),
            "lat": [0.0, 0.0, 0.0],
            "lng": [0.0, 0.001, 0.002],
        }
    )


def test_create_activity_flag_thresholds_correctly():
    sp = Staypoints(_staypoints_df(), started_at_col="started_at", finished_at_col="finished_at")
    flagged = sp.create_activity_flag(time_threshold_min=15.0)
    assert isinstance(flagged, Staypoints)
    assert flagged.df["activity"].tolist() == [True, False, True]


def test_create_activity_flag_on_plain_dataframe():
    result = create_activity_flag(_staypoints_df(), time_threshold_min=15.0)
    assert result["activity"].tolist() == [True, False, True]


def test_create_activity_flag_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown activity-flag method"):
        create_activity_flag(_staypoints_df(), method="bogus")


def test_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = _staypoints_df()
    result_pd = create_activity_flag(df, time_threshold_min=15.0)
    result_pl = create_activity_flag(pl.from_pandas(df), time_threshold_min=15.0).to_pandas()
    assert result_pl["activity"].tolist() == result_pd["activity"].tolist()


# ---------------------------------------------------------------------------
# identify_locations: obvious "home at night, work on weekday daytime" pattern.
# ---------------------------------------------------------------------------


def _home_work_staypoints():
    rows = []
    # 5 weekdays: home overnight (location A, lat/lng 0,0), work daytime
    # (location B, lat/lng 0,1). Monday 2020-01-06.
    for day in range(5):
        date = pd.Timestamp("2020-01-06") + pd.Timedelta(days=day)
        rows.append(
            {
                "uid": "u1",
                "started_at": date + pd.Timedelta(hours=22),
                "finished_at": date + pd.Timedelta(hours=30),  # to 06:00 next day
                "lat": 0.0,
                "lng": 0.0,
                "location_id": 0,
            }
        )
        rows.append(
            {
                "uid": "u1",
                "started_at": date + pd.Timedelta(hours=9),
                "finished_at": date + pd.Timedelta(hours=17),
                "lat": 0.0,
                "lng": 1.0,
                "location_id": 1,
            }
        )
    # A brief, infrequent third location (a coffee shop) -- neither home nor work.
    rows.append(
        {
            "uid": "u1",
            "started_at": pd.Timestamp("2020-01-06 12:30"),
            "finished_at": pd.Timestamp("2020-01-06 12:45"),
            "lat": 0.0,
            "lng": 0.5,
            "location_id": 2,
        }
    )
    return pd.DataFrame(rows)


def _locations_for(sp_df):
    grouped = sp_df.groupby("location_id").agg(center_lat=("lat", "mean"), center_lng=("lng", "mean")).reset_index()
    grouped["uid"] = "u1"
    grouped["n_staypoints"] = sp_df.groupby("location_id").size().to_numpy()
    return grouped[["uid", "location_id", "center_lat", "center_lng", "n_staypoints"]]


def test_identify_locations_labels_home_and_work():
    sp_df = _home_work_staypoints()
    sp = Staypoints(sp_df, started_at_col="started_at", finished_at_col="finished_at")
    locations = Locations(_locations_for(sp_df))

    labeled = identify_locations(locations, sp)
    purpose_by_location = dict(zip(labeled.df["location_id"], labeled.df["purpose"]))
    assert purpose_by_location[0] == "home"
    assert purpose_by_location[1] == "work"
    assert purpose_by_location[2] == "other"


def test_locations_identify_convenience_method():
    sp_df = _home_work_staypoints()
    sp = Staypoints(sp_df, started_at_col="started_at", finished_at_col="finished_at")
    locations = Locations(_locations_for(sp_df))

    labeled = locations.identify(sp)
    assert isinstance(labeled, Locations)
    purpose_by_location = dict(zip(labeled.df["location_id"], labeled.df["purpose"]))
    assert purpose_by_location[0] == "home"


def test_identify_locations_unknown_method_raises():
    sp_df = _home_work_staypoints()
    sp = Staypoints(sp_df, started_at_col="started_at", finished_at_col="finished_at")
    locations = Locations(_locations_for(sp_df))
    with pytest.raises(ValueError, match="unknown location-identification method"):
        identify_locations(locations, sp, method="bogus")


def test_identify_locations_requires_uid_column():
    sp_df = _home_work_staypoints().drop(columns=["uid"])
    sp = Staypoints(sp_df, started_at_col="started_at", finished_at_col="finished_at")
    locations = Locations(_locations_for(_home_work_staypoints()).drop(columns=["uid"]), uid_col=None)
    with pytest.raises(ValueError, match="requires staypoints to have a uid column"):
        identify_locations(locations, sp)
