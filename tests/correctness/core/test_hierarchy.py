"""Correctness tests for the Positionfixes -> Staypoints -> Triplegs ->
Locations hierarchy (`fastmob.core.{positionfixes,staypoints,triplegs,
locations}_dataframe`).

The synthetic fixture (`_two_stop_rows`) builds one user with a known,
hand-computed ground truth: a 30-minute stop at (0, 0), a ~10-minute move to
(0, 0.1), then a 30-minute stop at (0, 0.1). `minutes_for_a_stop=20` and
`spatial_radius_km=0.2` are chosen so both stops qualify and the move does
not. Real-dataset (Brightkite) tests only check structural invariants, since
there is no independent ground truth for real stop/tripleg boundaries.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob import Locations, Positionfixes, Staypoints, Triplegs


def _two_stop_rows(uid: str | None = "u1"):
    base = pd.Timestamp("2020-01-01 00:00:00")
    rows = []

    def add(dt, lat, lng):
        row = {"datetime": dt, "lat": lat, "lng": lng}
        if uid is not None:
            row["uid"] = uid
        rows.append(row)

    for m in range(0, 31, 5):
        add(base + pd.Timedelta(minutes=m), 0.0, 0.0)
    for i in range(1, 11):
        add(base + pd.Timedelta(minutes=30 + i), 0.0, 0.01 * i)
    stop2_start = base + pd.Timedelta(minutes=41)
    for m in range(0, 31, 5):
        add(stop2_start + pd.Timedelta(minutes=m), 0.0, 0.1)

    return pd.DataFrame(rows)


STOP_KWARGS = {"minutes_for_a_stop": 20.0, "spatial_radius_km": 0.2}


# ---------------------------------------------------------------------------
# Positionfixes / Staypoints
# ---------------------------------------------------------------------------


def test_generate_staypoints_detects_both_stops():
    pf = Positionfixes(_two_stop_rows())
    sp = pf.generate_staypoints(**STOP_KWARGS)
    assert isinstance(sp, Staypoints)
    assert len(sp.df) == 2
    assert sp.df["lat"].tolist() == pytest.approx([0.0, 0.0])
    assert sp.df["lng"].tolist() == pytest.approx([0.0, 0.1])
    assert (sp.df["finished_at"] >= sp.df["started_at"]).all()


def test_staypoints_validate_rejects_finished_before_started():
    bad = pd.DataFrame(
        {
            "uid": ["u1"],
            "started_at": [pd.Timestamp("2020-01-01 01:00:00")],
            "finished_at": [pd.Timestamp("2020-01-01 00:00:00")],
            "lat": [0.0],
            "lng": [0.0],
        }
    )
    with pytest.raises(ValueError, match="finished_at >= started_at"):
        Staypoints(bad)


def test_staypoints_no_uid_column():
    pf = Positionfixes(_two_stop_rows(uid=None))
    sp = pf.generate_staypoints(**STOP_KWARGS)
    assert sp.uid_col is None
    assert len(sp.df) == 2
    assert "uid" not in sp.df.columns


def test_staypoints_records_generation_parameters():
    pf = Positionfixes(_two_stop_rows())
    sp = pf.generate_staypoints(**STOP_KWARGS)
    assert sp.parameters["minutes_for_a_stop"] == 20.0
    assert sp.parameters["spatial_radius_km"] == 0.2


# ---------------------------------------------------------------------------
# Triplegs
# ---------------------------------------------------------------------------


def test_generate_triplegs_finds_exactly_one_moving_segment():
    pf = Positionfixes(_two_stop_rows())
    sp = pf.generate_staypoints(**STOP_KWARGS)
    tl = pf.generate_triplegs(sp)
    assert isinstance(tl, Triplegs)
    assert len(tl.df) == 1
    row = tl.df.iloc[0]
    assert row["length_km"] > 0
    assert row["duration_s"] > 0
    assert row["mean_speed_kmh"] == pytest.approx(row["length_km"] / (row["duration_s"] / 3600.0))
    # The tripleg must be temporally bracketed by the two staypoints.
    assert row["started_at"] >= sp.df["finished_at"].iloc[0]
    assert row["finished_at"] <= sp.df["started_at"].iloc[1]


def test_triplegs_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        Triplegs(pd.DataFrame({"tripleg_id": [1]}))


def test_triplegs_no_uid_column():
    pf = Positionfixes(_two_stop_rows(uid=None))
    sp = pf.generate_staypoints(**STOP_KWARGS)
    tl = pf.generate_triplegs(sp)
    assert tl.uid_col is None
    assert len(tl.df) == 1
    assert "uid" not in tl.df.columns


def test_two_users_are_segmented_independently():
    rows_a = _two_stop_rows(uid="u1")
    rows_b = _two_stop_rows(uid="u2")
    df = pd.concat([rows_a, rows_b], ignore_index=True)

    pf = Positionfixes(df)
    sp = pf.generate_staypoints(**STOP_KWARGS)
    tl = pf.generate_triplegs(sp)

    assert len(sp.df) == 4
    assert set(sp.df["uid"]) == {"u1", "u2"}
    assert len(tl.df) == 2
    assert set(tl.df["uid"]) == {"u1", "u2"}


def test_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = _two_stop_rows()

    pf_pd = Positionfixes(df)
    sp_pd = pf_pd.generate_staypoints(**STOP_KWARGS)
    tl_pd = pf_pd.generate_triplegs(sp_pd)

    pf_pl = Positionfixes(pl.from_pandas(df))
    sp_pl = pf_pl.generate_staypoints(**STOP_KWARGS)
    tl_pl = pf_pl.generate_triplegs(sp_pl)

    assert sp_pl.df.to_pandas()["lat"].tolist() == pytest.approx(sp_pd.df["lat"].tolist())
    assert sp_pl.df.to_pandas()["lng"].tolist() == pytest.approx(sp_pd.df["lng"].tolist())
    tl_pl_pd = tl_pl.df.to_pandas()
    assert tl_pl_pd["length_km"].tolist() == pytest.approx(tl_pd.df["length_km"].tolist())
    assert tl_pl_pd["duration_s"].tolist() == pytest.approx(tl_pd.df["duration_s"].tolist())


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def test_generate_user_locations_clusters_each_stop_separately():
    pf = Positionfixes(_two_stop_rows())
    sp = pf.generate_staypoints(**STOP_KWARGS)
    locations, sp_with_location = sp.generate_user_locations(epsilon_km=0.1, min_samples=1)

    assert isinstance(locations, Locations)
    assert len(locations.df) == 2
    assert sorted(locations.df["center_lng"].round(6).tolist()) == pytest.approx([0.0, 0.1])
    assert (sp_with_location.df["location_id"].notna()).all()


def test_locations_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        Locations(pd.DataFrame({"location_id": [0]}))


def test_generate_global_locations_uses_shared_h3_ids():
    pf = Positionfixes(_two_stop_rows())
    sp = pf.generate_staypoints(**STOP_KWARGS)
    locations, assigned = sp.generate_global_locations()
    assert locations.scope == "global"
    assert locations.scheme == "h3"
    assert "location_id" in assigned.df.columns


def test_associate_global_locations_validates_exact_ids():
    pf = Positionfixes(_two_stop_rows())
    sp = pf.generate_staypoints(**STOP_KWARGS)
    locations, assigned = sp.generate_global_locations()
    reassigned = assigned.associate_global_locations(locations, location_id_col="location_id")
    assert reassigned.df["location_id"].equals(assigned.df["location_id"])


# ---------------------------------------------------------------------------
# Real dataset: structural invariants only.
# ---------------------------------------------------------------------------


def test_brightkite_pipeline_structural_invariants(brightkite_sample_df):
    df = brightkite_sample_df.rename(
        columns={"user": "uid", "check-in_time": "datetime", "latitude": "lat", "longitude": "lng"}
    )
    pf = Positionfixes(df)
    sp = pf.generate_staypoints(minutes_for_a_stop=20.0, spatial_radius_km=0.2)
    tl = pf.generate_triplegs(sp)

    assert (sp.df["finished_at"] >= sp.df["started_at"]).all()
    assert (tl.df["length_km"] >= 0).all()
    assert (tl.df["duration_s"] >= 0).all()
    assert (tl.df["mean_speed_kmh"] >= 0).all()

    if len(sp.df) > 0:
        locations, _sp_with_location = sp.generate_user_locations(epsilon_km=0.1, min_samples=1)
        assert (locations.df["n_staypoints"] >= 1).all()
