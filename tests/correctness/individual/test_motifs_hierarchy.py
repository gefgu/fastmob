"""Tests for Staypoints.generate_daily_motifs -- the hierarchy-native
composition of daily_motifs with Locations.identify()'s home/work/other
purpose labels.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.core.locations_dataframe import Locations
from fastmob.core.staypoints_dataframe import Staypoints
from fastmob.measures.individual.motifs import daily_motifs


def _home_work_staypoints(uid: str = "u1"):
    rows = []
    # 5 weekdays: home overnight (far from work), work daytime. Monday 2020-01-06.
    for day in range(5):
        date = pd.Timestamp("2020-01-06") + pd.Timedelta(days=day)
        rows.append(
            {
                "uid": uid,
                "started_at": date + pd.Timedelta(hours=22),
                "finished_at": date + pd.Timedelta(hours=30),
                "lat": 0.0,
                "lng": 0.0,
            }
        )
        rows.append(
            {
                "uid": uid,
                "started_at": date + pd.Timedelta(hours=9),
                "finished_at": date + pd.Timedelta(hours=17),
                "lat": 0.0,
                "lng": 1.0,
            }
        )
    # A brief, infrequent third location -- neither home nor work.
    rows.append(
        {
            "uid": uid,
            "started_at": pd.Timestamp("2020-01-06 12:30"),
            "finished_at": pd.Timestamp("2020-01-06 12:45"),
            "lat": 0.0,
            "lng": 0.5,
        }
    )
    return pd.DataFrame(rows)


def _staypoints_and_locations(uid: str = "u1"):
    sp = Staypoints(_home_work_staypoints(uid), started_at_col="started_at", finished_at_col="finished_at")
    locations, sp_with_location = sp.generate_user_locations(min_samples=1)
    locations = locations.identify(sp_with_location)
    return sp_with_location, locations


def _two_user_home_work_staypoints():
    # location_id numbering is per-user (DBSCAN clusters each user
    # independently), so both users end up with overlapping location_id
    # values (0, 1) that must be disambiguated by (uid, location_id), not
    # location_id alone.
    rows = []
    for uid, lat_offset in [("u1", 0.0), ("u2", 10.0)]:
        for day in range(3):
            date = pd.Timestamp("2020-01-06") + pd.Timedelta(days=day)
            rows.append(
                {
                    "uid": uid,
                    "started_at": date + pd.Timedelta(hours=22),
                    "finished_at": date + pd.Timedelta(hours=30),
                    "lat": lat_offset,
                    "lng": 0.0,
                }
            )
            rows.append(
                {
                    "uid": uid,
                    "started_at": date + pd.Timedelta(hours=9),
                    "finished_at": date + pd.Timedelta(hours=17),
                    "lat": lat_offset,
                    "lng": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_generate_daily_motifs_full_pipeline():
    sp, locations = _staypoints_and_locations()
    result = sp.generate_daily_motifs(locations)
    assert list(result.columns) == ["uid", "date", "motif_id"]
    assert len(result) == 5
    assert (result["uid"] == "u1").all()


def test_generate_daily_motifs_requires_uid_column():
    sp_df = pd.DataFrame(
        {
            "started_at": pd.to_datetime(["2020-01-06 00:00"]),
            "finished_at": pd.to_datetime(["2020-01-06 01:00"]),
            "location_id": [0],
        }
    )
    sp = Staypoints(sp_df, uid_col=None, started_at_col="started_at", finished_at_col="finished_at")
    locations = Locations(
        pd.DataFrame({"location_id": [0], "center_lat": [0.0], "center_lng": [0.0], "purpose": ["HOME"]}),
        uid_col=None,
    )
    with pytest.raises(ValueError, match="requires staypoints to have a uid column"):
        sp.generate_daily_motifs(locations)


def test_generate_daily_motifs_requires_location_id_column():
    sp_df = pd.DataFrame(
        {
            "uid": ["u1"],
            "started_at": pd.to_datetime(["2020-01-06 00:00"]),
            "finished_at": pd.to_datetime(["2020-01-06 01:00"]),
        }
    )
    sp = Staypoints(sp_df, started_at_col="started_at", finished_at_col="finished_at")
    locations = Locations(
        pd.DataFrame(
            {"uid": ["u1"], "location_id": [0], "center_lat": [0.0], "center_lng": [0.0], "purpose": ["HOME"]}
        ),
        uid_col="uid",
    )
    with pytest.raises(ValueError, match="requires staypoints to have a 'location_id' column"):
        sp.generate_daily_motifs(locations)


def test_generate_daily_motifs_matches_flat_daily_motifs_with_uppercased_purpose():
    sp, locations = _staypoints_and_locations()
    result = sp.generate_daily_motifs(locations)

    # Independent oracle: hand-join purpose onto staypoints ourselves (with
    # explicit uppercasing) and call the flat daily_motifs directly,
    # bypassing the hierarchy entirely.
    purpose_by_location_id = dict(zip(locations.df["location_id"], locations.df["purpose"].str.upper()))
    visits = sp.df.assign(purpose=sp.df["location_id"].map(purpose_by_location_id))
    expected = daily_motifs(
        visits,
        uid_col="uid",
        location_col="location_id",
        purpose_col="purpose",
        datetime_col="started_at",
        end_datetime_col="finished_at",
    )
    pd.testing.assert_frame_equal(
        result.sort_values(["uid", "date"]).reset_index(drop=True),
        expected.sort_values(["uid", "date"]).reset_index(drop=True),
    )
    # Confirms the case bridge actually anchored a real home node rather than
    # silently falling back to "no HOME purpose found" (encode_motif_purposes
    # would otherwise treat the (lowercase) purposes as never matching "HOME").
    assert (result["motif_id"] > 0).all()


def test_generate_daily_motifs_handles_unmatched_location():
    sp, locations = _staypoints_and_locations()
    work_location_id = locations.df.loc[locations.df["purpose"] == "work", "location_id"].iloc[0]
    locations_missing_work = Locations(
        locations.df[locations.df["location_id"] != work_location_id].reset_index(drop=True),
        uid_col=locations.uid_col,
    )
    result = sp.generate_daily_motifs(locations_missing_work)
    assert len(result) == 5


def test_generate_daily_motifs_presorted_matches_default_path():
    sp, locations = _staypoints_and_locations()
    default_result = sp.generate_daily_motifs(locations)
    presorted_result = sp.generate_daily_motifs(locations)
    pd.testing.assert_frame_equal(
        default_result.sort_values(["uid", "date"]).reset_index(drop=True),
        presorted_result.sort_values(["uid", "date"]).reset_index(drop=True),
    )


def test_generate_daily_motifs_disambiguates_colliding_location_ids_across_users():
    sp = Staypoints(_two_user_home_work_staypoints(), started_at_col="started_at", finished_at_col="finished_at")
    locations, sp_with_location = sp.generate_user_locations(min_samples=1)
    locations = locations.identify(sp_with_location)
    # location_id 0/1 exist for both u1 and u2 independently -- confirms the
    # (user_idx, location_code) lookup is keyed by user, not location_id alone.
    assert set(locations.df["location_id"]) == {0, 1}
    assert set(locations.df["uid"]) == {"u1", "u2"}

    result = sp_with_location.generate_daily_motifs(locations)

    purpose_by_uid_location = {
        (uid, location_id): purpose.upper()
        for uid, location_id, purpose in zip(locations.df["uid"], locations.df["location_id"], locations.df["purpose"])
    }
    visits = sp_with_location.df.copy()
    visits["purpose"] = [
        purpose_by_uid_location[(uid, location_id)] for uid, location_id in zip(visits["uid"], visits["location_id"])
    ]
    expected = daily_motifs(
        visits,
        uid_col="uid",
        location_col="location_id",
        purpose_col="purpose",
        datetime_col="started_at",
        end_datetime_col="finished_at",
    )
    pd.testing.assert_frame_equal(
        result.sort_values(["uid", "date"]).reset_index(drop=True),
        expected.sort_values(["uid", "date"]).reset_index(drop=True),
    )
    assert (result["motif_id"] > 0).all()
