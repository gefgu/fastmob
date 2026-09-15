"""Correctness tests for fastmob.preprocessing.trajectory_to_od."""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob import FlowDataFrame
from fastmob.measures.evaluation import common_part_of_commuters
from fastmob.preprocessing import latlng_to_h3, trajectory_to_od, trajectory_to_trips

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


def test_trajectory_to_trips_pairs_chronologically_within_each_user():
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH]), ("u2", [SF_NORTH, SF_DOWNTOWN])])
    df = df.sample(frac=1, random_state=7)

    trips = trajectory_to_trips(df, resolution=7)

    assert len(trips.df) == 3
    assert list(trips.df.columns) == [
        "trip_id",
        "started_at",
        "finished_at",
        "origin_location_id",
        "destination_location_id",
    ]
    assert (trips.df["finished_at"] > trips.df["started_at"]).all()
    points = (SF_DOWNTOWN, SF_MISSION, SF_NORTH)
    location_df = pd.DataFrame({"lat": [point[0] for point in points], "lng": [point[1] for point in points]})
    locations = latlng_to_h3(location_df, resolution=7)["h3_cell"].tolist()
    expected_pairs = {
        (locations[0], locations[1]),
        (locations[1], locations[2]),
        (locations[2], locations[0]),
    }
    observed_pairs = set(zip(trips.df["origin_location_id"], trips.df["destination_location_id"]))
    assert observed_pairs == expected_pairs


def test_trajectory_to_trips_drops_self_loops_by_default_and_can_keep_them():
    df = _rows([("u1", [SF_DOWNTOWN, SF_DOWNTOWN, SF_MISSION])])

    trips = trajectory_to_trips(df, resolution=7)
    with_loops = trajectory_to_trips(df, resolution=7, drop_self_loops=False)

    assert len(trips.df) == 1
    assert len(with_loops.df) == 2
    assert with_loops.df["origin_location_id"].iloc[0] == with_loops.df["destination_location_id"].iloc[0]


def test_trajectory_to_trips_polars_backend_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH])])

    pandas_trips = trajectory_to_trips(df, resolution=7)
    polars_trips = trajectory_to_trips(pl.from_pandas(df), resolution=7)

    assert isinstance(polars_trips.df, pl.DataFrame)
    pandas_sorted = pandas_trips.df.sort_values("trip_id").reset_index(drop=True)
    polars_sorted = polars_trips.df.sort("trip_id").to_pandas().reset_index(drop=True)
    pd.testing.assert_frame_equal(pandas_sorted, polars_sorted, check_dtype=False)


def test_trajectory_to_trips_empty_and_single_ping_inputs():
    single = trajectory_to_trips(_rows([("u1", [SF_DOWNTOWN])]), resolution=7)
    empty = trajectory_to_trips(
        pd.DataFrame({"uid": [], "datetime": pd.to_datetime([]), "lat": [], "lng": []}), resolution=7
    )

    assert len(single.df) == 0
    assert len(empty.df) == 0
    assert {"origin_location_id", "destination_location_id"}.issubset(single.df.columns)


def test_presorted_matches_default_sorted_path_for_od():
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH]), ("u2", [SF_NORTH, SF_DOWNTOWN])])
    default_result = trajectory_to_od(df, resolution=7)
    presorted_result = trajectory_to_od(df, resolution=7, presorted=True)

    default_sorted = default_result.sort_values(["origin", "destination"]).reset_index(drop=True)
    presorted_sorted = presorted_result.sort_values(["origin", "destination"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(default_sorted, presorted_sorted)


def test_presorted_matches_default_sorted_path_for_trips():
    df = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH]), ("u2", [SF_NORTH, SF_DOWNTOWN])])
    default_trips = trajectory_to_trips(df, resolution=7)
    presorted_trips = trajectory_to_trips(df, resolution=7, presorted=True)

    default_sorted = default_trips.df.sort_values("trip_id").reset_index(drop=True)
    presorted_sorted = presorted_trips.df.sort_values("trip_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(default_sorted, presorted_sorted)


def test_od_edges_aggregate_across_interleaved_users():
    # Three users share the same SF_DOWNTOWN -> SF_MISSION edge; interleaving
    # them (via a non-chronological uid ordering in the source rows) exercises
    # the fused Rust kernel's global (not per-user) edge counting.
    df = _rows(
        [
            ("u1", [SF_DOWNTOWN, SF_MISSION]),
            ("u2", [SF_DOWNTOWN, SF_MISSION]),
            ("u3", [SF_DOWNTOWN, SF_MISSION, SF_NORTH]),
        ]
    )
    result = trajectory_to_od(df, resolution=7)
    downtown, mission = latlng_to_h3(
        pd.DataFrame({"lat": [SF_DOWNTOWN[0], SF_MISSION[0]], "lng": [SF_DOWNTOWN[1], SF_MISSION[1]]}),
        resolution=7,
    )["h3_cell"].tolist()
    row = result[(result["origin"] == downtown) & (result["destination"] == mission)]
    assert len(row) == 1
    assert row["count"].iloc[0] == 3


@pytest.mark.parametrize("resolution", [7, 8, 9])
def test_direct_trip_cpc_matches_materialized_od_cpc(resolution):
    real = _rows([("u1", [SF_DOWNTOWN, SF_MISSION, SF_NORTH, SF_DOWNTOWN])])
    predicted = _rows([("u2", [SF_DOWNTOWN, SF_MISSION, SF_DOWNTOWN])])

    real_trips = trajectory_to_trips(real, resolution=resolution)
    predicted_trips = trajectory_to_trips(predicted, resolution=resolution)
    real_od = trajectory_to_od(real, resolution=resolution)
    predicted_od = trajectory_to_od(predicted, resolution=resolution)
    real_flows = FlowDataFrame(real_od, origin="origin", destination="destination", flow="count")
    predicted_flows = FlowDataFrame(predicted_od, origin="origin", destination="destination", flow="count")

    assert common_part_of_commuters(real_trips, predicted_trips) == pytest.approx(
        common_part_of_commuters(real_flows, predicted_flows)
    )
