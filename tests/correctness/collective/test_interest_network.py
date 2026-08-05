"""Correctness tests for collective location interest networks."""

from __future__ import annotations

import fastmob
import pandas as pd
import pytest
from fastmob.core import Locations, Staypoints
from fastmob.measures.collective import interest_network


def _locations() -> Locations:
    # Deliberately non-sorted IDs: edge endpoints follow this catalogue order.
    return Locations(
        pd.DataFrame(
            {
                "location_id": [90, 10, 70],
                "center_lat": [0.0, 1.0, 2.0],
                "center_lng": [0.0, 1.0, 2.0],
            }
        ),
        scope="global",
    )


def _staypoint_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=7, freq="h")
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1", "u2", "u2", "u3", "u3"],
            "started_at": timestamps,
            "finished_at": timestamps,
            "lat": [0.0] * 7,
            "lng": [0.0] * 7,
            "location_id": [90, 10, 90, 90, 70, 10, 70],
        }
    )


def test_counts_shared_people_once_and_uses_catalogue_order():
    result = interest_network(Staypoints(_staypoint_frame()), _locations())
    assert result.to_dict("records") == [
        {"location_id_a": 90, "location_id_b": 10, "n_people": 1},
        {"location_id_a": 90, "location_id_b": 70, "n_people": 1},
        {"location_id_a": 10, "location_id_b": 70, "n_people": 1},
    ]


def test_raw_input_and_staypoints_method_delegate_to_the_same_measure():
    frame = _staypoint_frame()
    locations = _locations()
    expected = interest_network(frame, locations)
    actual = Staypoints(frame).interest_network(locations)
    assert actual.equals(expected)
    assert fastmob.interest_network(frame, locations).equals(expected)


def test_empty_network_has_a_stable_schema():
    frame = _staypoint_frame().iloc[[0, 3]].copy()
    result = interest_network(frame, _locations())
    assert list(result.columns) == ["location_id_a", "location_id_b", "n_people"]
    assert result.empty


def test_string_location_ids_follow_catalogue_order_without_id_sorting():
    frame = _staypoint_frame().iloc[:2].copy()
    frame["location_id"] = ["zebra", "ant"]
    locations = Locations(
        pd.DataFrame(
            {
                "location_id": ["zebra", "ant"],
                "center_lat": [0.0, 1.0],
                "center_lng": [0.0, 1.0],
            }
        ),
        scope="global",
    )
    result = interest_network(frame, locations)
    assert result.to_dict("records") == [{"location_id_a": "zebra", "location_id_b": "ant", "n_people": 1}]


def test_rejects_non_global_locations_and_unknown_assignments():
    frame = _staypoint_frame()
    user_locations = Locations(
        pd.DataFrame({"uid": ["u1"], "location_id": [1], "center_lat": [0.0], "center_lng": [0.0]}),
        uid_col="uid",
        scope="user",
    )
    with pytest.raises(ValueError, match="global"):
        interest_network(frame, user_locations)

    frame.loc[0, "location_id"] = 999
    with pytest.raises(ValueError, match="absent"):
        interest_network(frame, _locations())


def test_coerce_rejects_non_staypoint_data():
    with pytest.raises(ValueError, match="finished_at"):
        Staypoints.coerce(pd.DataFrame({"uid": ["u1"], "started_at": [pd.Timestamp("2024-01-01")]}))
