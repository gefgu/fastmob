"""Correctness tests for skmob2/measures/visits/individual_mobility_network.py."""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_edge_dict(df) -> dict:
    """Convert a mobility network DataFrame to {uid: {(lat_o,lng_o,lat_d,lng_d): n_trips}}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)

    result: dict = {}
    for row in nw_df.rows(named=True):
        uid = row[uid_col] if uid_col else "__single__"
        edge = (row["lat_origin"], row["lng_origin"], row["lat_dest"], row["lng_dest"])
        result.setdefault(uid, {})[edge] = row["n_trips"]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_imn_known_values(synthetic_tdf):
    """user_a moves along 5 distinct equator points: 4 sequential transitions."""
    from skmob2.measures.visits.individual_mobility_network import (
        individual_mobility_network,
    )

    result = individual_mobility_network(synthetic_tdf)
    edge_dict = _to_edge_dict(result)

    # All 3 users must be present.
    assert set(edge_dict.keys()) == {"user_a", "user_b", "user_c"}

    # user_a: 5 equatorial points -> 4 edges, each traversed once.
    user_a_edges = edge_dict["user_a"]
    assert len(user_a_edges) == 4
    assert all(n == 1 for n in user_a_edges.values())

    # user_b: same structure along a meridian.
    user_b_edges = edge_dict["user_b"]
    assert len(user_b_edges) == 4
    assert all(n == 1 for n in user_b_edges.values())


def test_imn_self_loops_excluded_by_default():
    """Consecutive visits to the same location produce no edge by default."""
    from skmob2.measures.visits.individual_mobility_network import (
        individual_mobility_network,
    )

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]
            ),
            "lat": [1.0, 1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = individual_mobility_network(df, self_loops=False)
    edge_dict = _to_edge_dict(result)
    edges = edge_dict["a"]

    # No (1.0, 0.0) -> (1.0, 0.0) self-loop.
    assert (1.0, 0.0, 1.0, 0.0) not in edges
    # Transitions (1->2) and (2->1) are present.
    assert (1.0, 0.0, 2.0, 0.0) in edges
    assert (2.0, 0.0, 1.0, 0.0) in edges


def test_imn_self_loops_included():
    """When self_loops=True, staying at the same location creates an edge."""
    from skmob2.measures.visits.individual_mobility_network import (
        individual_mobility_network,
    )

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]),
            "lat": [1.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = individual_mobility_network(df, self_loops=True)
    edge_dict = _to_edge_dict(result)
    edges = edge_dict["a"]

    assert (1.0, 0.0, 1.0, 0.0) in edges
    assert edges[(1.0, 0.0, 1.0, 0.0)] == 1


def test_imn_repeated_transition():
    """A transition that occurs multiple times accumulates in n_trips."""
    from skmob2.measures.visits.individual_mobility_network import (
        individual_mobility_network,
    )

    # A -> B -> A -> B: edge A->B and B->A each appear twice.
    df = pd.DataFrame(
        {
            "uid": ["x", "x", "x", "x"],
            "datetime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]
            ),
            "lat": [1.0, 2.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = individual_mobility_network(df)
    edge_dict = _to_edge_dict(result)
    edges = edge_dict["x"]

    assert edges[(1.0, 0.0, 2.0, 0.0)] == 2
    assert edges[(2.0, 0.0, 1.0, 0.0)] == 1


def test_imn_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.visits.individual_mobility_network import (
        individual_mobility_network,
    )

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 2.0, 3.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = individual_mobility_network(df)
    nw_result = nw.from_native(result, eager_only=True)

    assert "n_trips" in nw_result.columns
    # 3 sequential distinct points -> 2 edges.
    assert len(nw_result) == 2


def test_imn_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same edge structure as pandas."""
    from skmob2.measures.visits.individual_mobility_network import (
        individual_mobility_network,
    )

    result = individual_mobility_network(synthetic_tdf_polars)
    edge_dict = _to_edge_dict(result)

    assert set(edge_dict.keys()) == {"user_a", "user_b", "user_c"}
    assert len(edge_dict["user_a"]) == 4


@pytest.mark.skmob
def test_imn_matches_skmob(comparison_skmob):
    """skmob2 result matches skmob on each comparison dataset."""
    import pandas as pd
    from skmob.measures.individual import individual_mobility_network as skmob_imn
    from skmob2.measures.visits.individual_mobility_network import (
        individual_mobility_network as skmob2_imn,
    )

    skmob_result = skmob_imn(comparison_skmob, show_progress=False)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
    skmob2_result = skmob2_imn(skmob2_input)

    # Build comparable dicts: {uid: {(lat_o,lng_o,lat_d,lng_d): n_trips}}.
    skmob_dict: dict = {}
    skmob_result = skmob_result.reset_index()
    for _, row in skmob_result.iterrows():
        uid = row["uid"]
        edge = (row["lat_origin"], row["lng_origin"], row["lat_dest"], row["lng_dest"])
        skmob_dict.setdefault(uid, {})[edge] = int(row["n_trips"])

    skmob2_dict = _to_edge_dict(skmob2_result)

    common_uids = set(skmob_dict) & set(skmob2_dict)
    assert len(common_uids) > 0
    for uid in common_uids:
        common_edges = set(skmob_dict[uid]) & set(skmob2_dict[uid])
        for edge in common_edges:
            assert skmob_dict[uid][edge] == skmob2_dict[uid][edge], (
                f"uid={uid}, edge={edge}: skmob={skmob_dict[uid][edge]}, skmob2={skmob2_dict[uid][edge]}"
            )


def test_imn_matches_cached_reference(comparison_skmob_reference):
    """individual_mobility_network matches the cached skmob baseline without requiring the skmob environment."""
    from skmob2.measures.visits.individual_mobility_network import individual_mobility_network as skmob2_imn

    ref = comparison_skmob_reference
    skmob_result = ref.result("individual_mobility_network")
    skmob2_result = skmob2_imn(ref.input_df)

    skmob_dict: dict = {}
    for _, row in skmob_result.iterrows():
        uid = row["uid"]
        edge = (row["lat_origin"], row["lng_origin"], row["lat_dest"], row["lng_dest"])
        skmob_dict.setdefault(uid, {})[edge] = int(row["n_trips"])

    skmob2_dict = _to_edge_dict(skmob2_result)

    common_uids = set(skmob_dict) & set(skmob2_dict)
    assert len(common_uids) > 0
    for uid in common_uids:
        common_edges = set(skmob_dict[uid]) & set(skmob2_dict[uid])
        for edge in common_edges:
            assert skmob_dict[uid][edge] == skmob2_dict[uid][edge], (
                f"uid={uid}, edge={edge}: cached={skmob_dict[uid][edge]}, skmob2={skmob2_dict[uid][edge]}"
            )
