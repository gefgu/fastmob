"""Correctness tests for fastmob.preprocessing.trajectory_to_od."""

from __future__ import annotations

import pandas as pd
import pytest

from fastmob.preprocessing import trajectory_to_od

# Two well-separated locations (different H3 cells at any reasonable resolution):
SF_DOWNTOWN = (37.7793, -122.4194)
SF_MISSION = (37.6007, -122.3824)
SF_NORTH = (37.9, -122.1)


def _rows(uid_points):
    """uid_points: list of (uid, [(lat, lng), ...]) with 1h spacing starting 2020-01-01."""
    rows = []
    for uid, points in uid_points:
        for i, (lat, lng) in enumerate(points):
            rows.append(
                {
                    "uid": uid,
                    "datetime": pd.Timestamp(f"2020-01-01 {i:02d}:00:00"),
                    "lat": lat,
                    "lng": lng,
                }
            )
    return pd.DataFrame(rows)


def test_two_users_one_trip_each():
    df = _rows(
        [
            ("u1", [SF_DOWNTOWN, SF_MISSION, SF_MISSION]),  # 1 real trip, 1 self-loop
            ("u2", [SF_DOWNTOWN, SF_DOWNTOWN]),  # self-loop only
        ]
    )
    result = trajectory_to_od(df, resolution=7)
    assert len(result) == 1
    assert result["count"].iloc[0] == 1


def test_matches_wide_pivot_shape():
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH])])
    result = trajectory_to_od(df, resolution=7)
    wide = result.pivot(index="origin", columns="destination", values="count").fillna(0.0)
    assert wide.shape == (2, 2)
    assert wide.to_numpy().sum() == 2


def test_polars_backend_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH])])
    pandas_result = trajectory_to_od(df, resolution=7)
    polars_result = trajectory_to_od(pl.from_pandas(df), resolution=7)
    assert isinstance(polars_result, pl.DataFrame)

    pandas_sorted = pandas_result.sort_values(["origin", "destination"]).reset_index(drop=True)
    polars_sorted = polars_result.sort(["origin", "destination"]).to_pandas().reset_index(drop=True)
    pd.testing.assert_frame_equal(pandas_sorted, polars_sorted, check_dtype=False)


def test_precision_preserved_for_full_width_h3_cells():
    # H3 res-9 cell indices are ~60-bit -- past float64's 53-bit exact-integer
    # range. A naive shift-based implementation upcasts to float64 and
    # silently corrupts the last 1-2 digits; this must stay exact.
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION])])
    result = trajectory_to_od(df, resolution=9)
    assert result["origin"].dtype == "uint64"
    assert result["destination"].dtype == "uint64"


def test_self_loops_dropped_by_default():
    df = _rows([("u1", [SF_DOWNTOWN, SF_DOWNTOWN, SF_DOWNTOWN])])
    result = trajectory_to_od(df, resolution=7)
    assert len(result) == 0


def test_self_loops_kept_when_requested():
    df = _rows([("u1", [SF_DOWNTOWN, SF_DOWNTOWN])])
    result = trajectory_to_od(df, resolution=7, drop_self_loops=False)
    assert len(result) == 1
    assert result["origin"].iloc[0] == result["destination"].iloc[0]


def test_single_row_returns_empty():
    df = _rows([("u1", [SF_DOWNTOWN])])
    result = trajectory_to_od(df, resolution=7)
    assert len(result) == 0
    assert list(result.columns) == ["origin", "destination", "count"]


def test_empty_input_returns_empty():
    df = pd.DataFrame({"uid": [], "datetime": pd.to_datetime([]), "lat": [], "lng": []})
    result = trajectory_to_od(df, resolution=7)
    assert len(result) == 0
    assert list(result.columns) == ["origin", "destination", "count"]


def test_no_uid_column_treats_frame_as_single_user():
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH])]).drop(columns=["uid"])
    result = trajectory_to_od(df, resolution=7)
    assert len(result) == 2
    assert result["count"].sum() == 2


def test_out_of_range_coordinates_dropped():
    df = _rows([("u1", [SF_DOWNTOWN, (200.0, 300.0), SF_MISSION])])
    result = trajectory_to_od(df, resolution=7)
    # the out-of-range middle row is dropped entirely (not converted to a
    # sentinel/null cell), closing the gap so the remaining two rows become
    # directly consecutive -> one SF_DOWNTOWN -> SF_MISSION trip survives,
    # matching citybehavex's original drop-then-sort-then-shift semantics.
    assert len(result) == 1
    assert result["count"].iloc[0] == 1


def test_explicit_column_overrides():
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION])]).rename(
        columns={"uid": "user", "datetime": "ts", "lat": "latitude", "lng": "longitude"}
    )
    result = trajectory_to_od(
        df, resolution=7, uid_col="user", datetime_col="ts", lat_col="latitude", lng_col="longitude"
    )
    assert len(result) == 1
