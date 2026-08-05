from __future__ import annotations

import fastmob
import pandas as pd
import pytest


def _trips(rows):
    return fastmob.Trips(
        pd.DataFrame(
            {
                "trip_id": range(len(rows)),
                "started_at": pd.Timestamp("2026-01-01"),
                "finished_at": pd.Timestamp("2026-01-01 01:00"),
                "origin_staypoint_id": [None] * len(rows),
                "destination_staypoint_id": [None] * len(rows),
                "tripleg_ids": [[] for _ in rows],
                "origin_location_id": [row[0] for row in rows],
                "destination_location_id": [row[1] for row in rows],
            }
        )
    )


def test_trip_cpc_uses_global_location_ids():
    left = _trips([("home", "work"), ("home", "work"), ("work", "home")])
    right = _trips([("home", "work"), ("park", "work")])
    assert left.common_part_of_commuters(right) == pytest.approx(0.4)
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)


def test_flow_cpc_aggregates_duplicate_edges_and_ignores_self_loops():
    left = fastmob.FlowDataFrame({"origin": [1, 1, 2], "destination": [2, 2, 2], "flow": [2, 3, 5]})
    right = fastmob.FlowDataFrame({"origin": [1], "destination": [2], "flow": [4]})
    assert left.common_part_of_commuters(right) == pytest.approx(8 / 9)


def test_trips_can_be_explicitly_materialized_as_flows():
    flows = _trips([(1, 2), (1, 2), (2, 2), (None, 1)]).to_flow_dataframe()
    assert flows.get_flow(1, 2) == 2
