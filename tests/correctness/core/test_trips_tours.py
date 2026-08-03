"""Correctness tests for the Trips/Tours hierarchy levels
(`fastmob.core.{trips,tours}_dataframe`).

`_home_work_home_rows` builds one user's day: a long overnight stop at
(0, 0) ("home"), a short tripleg, a long daytime stop at (0, 1) ("work"),
another short tripleg, then back to (0, 0) overnight again -- an obvious
round trip with two triplegs merging into two trips (home->work,
work->home) and one tour (home -> work -> home).

`_home_wait_work_rows` adds a deliberately short (non-activity) "waiting"
staypoint between two triplegs, which must merge them into a single trip
rather than splitting them (the case Trips.from_triplegs is built to
handle).
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob import Positionfixes
from fastmob.core.trips_dataframe import Trips


def _home_work_home_rows(uid: str | None = "u1"):
    rows = []
    base = pd.Timestamp("2020-01-06 00:00:00")

    def add(dt, lat, lng):
        row = {"datetime": dt, "lat": lat, "lng": lng}
        if uid is not None:
            row["uid"] = uid
        rows.append(row)

    for m in range(0, 601, 30):
        add(base + pd.Timedelta(minutes=m), 0.0, 0.0)
    t0 = base + pd.Timedelta(minutes=610)
    for i in range(1, 4):
        add(t0 + pd.Timedelta(minutes=i * 3), 0.0, 0.1 * i)
    t3 = t0 + pd.Timedelta(minutes=19)
    for m in range(0, 601, 30):
        add(t3 + pd.Timedelta(minutes=m), 0.0, 1.0)
    t4 = t3 + pd.Timedelta(minutes=610)
    for i in range(1, 4):
        add(t4 + pd.Timedelta(minutes=i * 3), 0.0, 1.0 - 0.3333 * i)
    t5 = t4 + pd.Timedelta(minutes=19)
    for m in range(0, 601, 30):
        add(t5 + pd.Timedelta(minutes=m), 0.0, 0.0)

    return pd.DataFrame(rows)


STOP_KWARGS = {"minutes_for_a_stop": 10.0, "spatial_radius_km": 0.2}


def _build_trips(rows: pd.DataFrame):
    pf = Positionfixes(rows)
    sp = pf.generate_staypoints(**STOP_KWARGS)
    tl = pf.generate_triplegs(sp)
    sp_act = sp.create_activity_flag(time_threshold_min=15.0)
    trips = tl.generate_trips(sp_act)
    return pf, sp, tl, sp_act, trips


def test_generate_trips_requires_activity_column():
    pf = Positionfixes(_home_work_home_rows())
    sp = pf.generate_staypoints(**STOP_KWARGS)
    tl = pf.generate_triplegs(sp)
    with pytest.raises(ValueError, match="requires staypoints to have an 'activity' column"):
        tl.generate_trips(sp)


def test_home_work_home_produces_two_trips():
    _, _, _, _, trips = _build_trips(_home_work_home_rows())
    assert isinstance(trips, Trips)
    assert len(trips.df) == 2
    assert trips.df["tripleg_ids"].apply(len).tolist() == [1, 1]


def test_trip_bounds_are_bracketed_by_the_full_door_to_door_span():
    """Regression test: `segment(method="stop")` attributes a stop's own
    entry/leaving transition rows to the stop's segment, not the moving
    segment before/after it, which used to truncate a tripleg's own
    min/max datetime (sometimes to a single degenerate point, 0 duration
    despite a real nonzero length). Trips (and Triplegs itself) must
    recover the true door-to-door span from the bracketing staypoints."""
    _, _, tl, sp_act, trips = _build_trips(_home_work_home_rows())

    assert (tl.df["duration_s"] > 0).all()
    assert (tl.df["length_km"] > 0).all()

    sp_sorted = sp_act.df.sort_values("started_at").reset_index(drop=True)
    trips_sorted = trips.df.sort_values("started_at").reset_index(drop=True)
    # Each trip's started_at must exactly match the finished_at of the
    # staypoint it left, and its finished_at must match the started_at of
    # the staypoint it arrives at.
    assert trips_sorted["started_at"].iloc[0] == sp_sorted["finished_at"].iloc[0]
    assert trips_sorted["finished_at"].iloc[0] == sp_sorted["started_at"].iloc[1]
    assert trips_sorted["started_at"].iloc[1] == sp_sorted["finished_at"].iloc[1]
    assert trips_sorted["finished_at"].iloc[1] == sp_sorted["started_at"].iloc[2]


def test_origin_and_destination_staypoint_ids_resolve():
    _, _, _, sp_act, trips = _build_trips(_home_work_home_rows())
    trips_sorted = trips.df.sort_values("started_at").reset_index(drop=True)
    valid_staypoint_ids = set(sp_act.df["staypoint_id"])
    assert set(trips_sorted["origin_staypoint_id"].dropna()) <= valid_staypoint_ids
    assert set(trips_sorted["destination_staypoint_id"].dropna()) <= valid_staypoint_ids
    # The very first trip has no preceding staypoint recorded as "activity"
    # before it in this fixture's timeline -- actually it does (the initial
    # overnight stop), so origin should be populated for both trips here.
    assert trips_sorted["origin_staypoint_id"].notna().all()
    assert trips_sorted["destination_staypoint_id"].notna().all()


def test_no_uid_column():
    _, _, _, _, trips = _build_trips(_home_work_home_rows(uid=None))
    assert trips.uid_col is None
    assert len(trips.df) == 2
    assert "uid" not in trips.df.columns


def test_two_users_produce_independent_trips():
    rows_a = _home_work_home_rows(uid="u1")
    rows_b = _home_work_home_rows(uid="u2")
    df = pd.concat([rows_a, rows_b], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    _, _, _, _, trips = _build_trips(df)
    assert len(trips.df) == 4
    assert set(trips.df["uid"]) == {"u1", "u2"}


def test_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    rows = _home_work_home_rows()

    _, _, _, _, trips_pd = _build_trips(rows)
    pf_pl = Positionfixes(pl.from_pandas(rows))
    sp_pl = pf_pl.generate_staypoints(**STOP_KWARGS)
    tl_pl = pf_pl.generate_triplegs(sp_pl)
    sp_pl_act = sp_pl.create_activity_flag(time_threshold_min=15.0)
    trips_pl = tl_pl.generate_trips(sp_pl_act)

    trips_pl_pd = trips_pl.df.to_pandas().sort_values("started_at").reset_index(drop=True)
    trips_pd_sorted = trips_pd.df.sort_values("started_at").reset_index(drop=True)
    assert trips_pl_pd["started_at"].tolist() == trips_pd_sorted["started_at"].tolist()
    assert trips_pl_pd["finished_at"].tolist() == trips_pd_sorted["finished_at"].tolist()
    assert trips_pl_pd["origin_staypoint_id"].tolist() == trips_pd_sorted["origin_staypoint_id"].tolist()
    assert trips_pl_pd["destination_staypoint_id"].tolist() == trips_pd_sorted["destination_staypoint_id"].tolist()


# ---------------------------------------------------------------------------
# Waiting-stop merge: a short non-activity stop between two triplegs must not
# split them into separate trips.
# ---------------------------------------------------------------------------


def _home_wait_work_rows():
    rows = []
    base = pd.Timestamp("2020-01-06 00:00:00")

    def add(dt, lat, lng):
        rows.append({"uid": "u1", "datetime": dt, "lat": lat, "lng": lng})

    for m in range(0, 601, 30):
        add(base + pd.Timedelta(minutes=m), 0.0, 0.0)
    t0 = base + pd.Timedelta(minutes=610)
    for i in range(1, 4):
        add(t0 + pd.Timedelta(minutes=i * 3), 0.0, 0.1 * i)
    # short (non-activity) waiting stop: detected duration ~12 minutes,
    # above the 10-minute stay_locations threshold (so it IS detected as a
    # staypoint) but below the 15-minute activity threshold (so it must NOT
    # split the trip).
    t1 = t0 + pd.Timedelta(minutes=9)
    for m in range(0, 9, 4):
        add(t1 + pd.Timedelta(minutes=m), 0.0, 0.3)
    t2 = t1 + pd.Timedelta(minutes=9)
    for i in range(1, 4):
        add(t2 + pd.Timedelta(minutes=i * 3), 0.0, 0.3 + 0.2333 * i)
    t3 = t2 + pd.Timedelta(minutes=10)
    for m in range(0, 601, 30):
        add(t3 + pd.Timedelta(minutes=m), 0.0, 1.0)

    return pd.DataFrame(rows)


def test_short_non_activity_stop_merges_two_triplegs_into_one_trip():
    _, _, tl, sp_act, trips = _build_trips(_home_wait_work_rows())
    assert len(tl.df) == 2  # home->wait, wait->work
    assert (~sp_act.df["activity"]).sum() == 1  # exactly one non-activity (waiting) staypoint
    assert len(trips.df) == 1
    assert sorted(trips.df["tripleg_ids"].iloc[0]) == sorted(tl.df["tripleg_id"].tolist())


# ---------------------------------------------------------------------------
# Tours: a home -> work -> home round trip must form exactly one tour.
# ---------------------------------------------------------------------------


def _build_tours(rows: pd.DataFrame, **location_kwargs):
    _pf, _sp, _tl, sp_act, trips = _build_trips(rows)
    locations, sp_with_location = sp_act.generate_user_locations(**{"epsilon_km": 0.15, "min_samples": 1, **location_kwargs})
    tours = trips.generate_tours(sp_with_location)
    return locations, sp_with_location, trips, tours


def test_generate_tours_requires_location_id_column():
    _, _, _, sp_act, trips = _build_trips(_home_work_home_rows())
    with pytest.raises(ValueError, match="requires staypoints_with_location to have a 'location_id' column"):
        trips.generate_tours(sp_act)


def test_home_work_home_forms_exactly_one_tour():
    locations, _sp_with_location, trips, tours = _build_tours(_home_work_home_rows())
    assert len(tours.df) == 1
    home_location_id = locations.df.loc[locations.df["center_lng"].round(3) == 0.0, "location_id"].iloc[0]
    assert tours.df["location_id"].iloc[0] == home_location_id
    assert sorted(tours.df["journey"].iloc[0]) == sorted(trips.df["trip_id"].tolist())
    assert tours.df["started_at"].iloc[0] == trips.df["started_at"].min()
    assert tours.df["finished_at"].iloc[0] == trips.df["finished_at"].max()


def test_no_uid_tours():
    _locations, _sp_with_location, _trips, tours = _build_tours(_home_work_home_rows(uid=None))
    assert tours.uid_col is None
    assert len(tours.df) == 1
    assert "uid" not in tours.df.columns


def test_two_users_produce_independent_tours():
    rows_a = _home_work_home_rows(uid="u1")
    rows_b = _home_work_home_rows(uid="u2")
    df = pd.concat([rows_a, rows_b], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    _locations, _sp_with_location, _trips, tours = _build_tours(df)
    assert len(tours.df) == 2
    assert set(tours.df["uid"]) == {"u1", "u2"}


def test_tours_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    rows = _home_work_home_rows()

    _, _, _, tours_pd = _build_tours(rows)

    pf_pl = Positionfixes(pl.from_pandas(rows))
    sp_pl = pf_pl.generate_staypoints(**STOP_KWARGS)
    tl_pl = pf_pl.generate_triplegs(sp_pl)
    sp_pl_act = sp_pl.create_activity_flag(time_threshold_min=15.0)
    trips_pl = tl_pl.generate_trips(sp_pl_act)
    _loc_pl, sp_pl_loc = sp_pl_act.generate_user_locations(epsilon_km=0.15, min_samples=1)
    tours_pl = trips_pl.generate_tours(sp_pl_loc)

    tours_pl_pd = tours_pl.df.to_pandas()
    assert len(tours_pl_pd) == len(tours_pd.df)
    assert tours_pl_pd["started_at"].tolist() == tours_pd.df["started_at"].tolist()
    assert tours_pl_pd["finished_at"].tolist() == tours_pd.df["finished_at"].tolist()
