from __future__ import annotations

import fastmob
import numpy as np
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


def _numeric_trips(rows, dtype=np.uint64):
    return fastmob.Trips(
        pd.DataFrame(
            {
                "trip_id": range(len(rows)),
                "started_at": pd.Timestamp("2026-01-01"),
                "finished_at": pd.Timestamp("2026-01-01 01:00"),
                "origin_staypoint_id": [None] * len(rows),
                "destination_staypoint_id": [None] * len(rows),
                "tripleg_ids": [[] for _ in rows],
                "origin_location_id": np.array([row[0] for row in rows], dtype=dtype),
                "destination_location_id": np.array([row[1] for row in rows], dtype=dtype),
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


def test_trip_cpc_numeric_ids_skip_factorization_and_match_string_equivalent():
    # Same topology as test_trip_cpc_uses_global_location_ids (score 0.4), but
    # with H3-cell-scale uint64 ids -- past uint32::MAX -- to exercise the
    # numeric fast path that skips _joint_factorize_arrow_values entirely.
    big = 1 << 40
    home, work, park = big, big + 1, big + 2
    left = _numeric_trips([(home, work), (home, work), (work, home)])
    right = _numeric_trips([(home, work), (park, work)])
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)
    # left has 2 unique links {(home,work),(work,home)}, right has 2 unique
    # links {(home,work),(park,work)}; only (home,work) is shared -> 2*1/4.
    assert fastmob.common_part_of_links(left, right) == pytest.approx(0.5)


def test_trip_cpc_mixed_unsigned_width_uses_numeric_fast_path():
    # Differently-widthed unsigned columns (uint32 vs uint64) both qualify for
    # the numeric fast path once cast to a common uint64 type -- this used to
    # be exactly the case _joint_factorize_arrow_values rejects outright
    # ("jointly factorized columns must use the same Arrow data type").
    left = _numeric_trips([(1, 2), (1, 2), (2, 1)], dtype=np.uint32)
    right = _numeric_trips([(1, 2), (3, 2)], dtype=np.uint64)
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)


def test_trip_cpc_string_ids_still_use_factorization_fallback():
    # Non-numeric ids (e.g. "home"/"work") cannot skip factorization -- this
    # is the same scenario as test_trip_cpc_uses_global_location_ids, kept
    # here to make the fallback path explicit alongside the fast-path tests.
    left = _trips([("home", "work"), ("home", "work"), ("work", "home")])
    right = _trips([("home", "work"), ("park", "work")])
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)
