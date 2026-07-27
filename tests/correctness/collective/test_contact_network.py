"""Correctness tests for fastmob.measures.collective.contact_network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.measures.collective.contact_network import (
    NetworkGraph,
    clustering_coefficients,
    co_presence_graph_from_visits,
    degree_preserving_random_graph,
    distribution_summary,
    graph_from_edges,
    infer_social_ties,
    random_persistence,
    safe_wasserstein,
    topological_overlap,
)

# ---------------------------------------------------------------------------
# co_presence_graph_from_visits
# ---------------------------------------------------------------------------
# Day 0: users 1,2,3 co-present at venueA -> edges (1,2),(1,3),(2,3).
# Day 1: users 1,2 co-present again at venueA -> edge (1,2) persists.
# Day 0: users 4,5 co-present at venueB -> edge (4,5), independent component.
# Cross-checked against citybehavex-core's Rust unit test
# `co_presence_edges_from_two_day_groups`.


def _visits_rows():
    return [
        {"uid": 1, "datetime": pd.Timestamp("2020-01-01"), "location_id": "A"},
        {"uid": 2, "datetime": pd.Timestamp("2020-01-01"), "location_id": "A"},
        {"uid": 3, "datetime": pd.Timestamp("2020-01-01"), "location_id": "A"},
        {"uid": 1, "datetime": pd.Timestamp("2020-01-02"), "location_id": "A"},
        {"uid": 2, "datetime": pd.Timestamp("2020-01-02"), "location_id": "A"},
        {"uid": 4, "datetime": pd.Timestamp("2020-01-01"), "location_id": "B"},
        {"uid": 5, "datetime": pd.Timestamp("2020-01-01"), "location_id": "B"},
    ]


def _edges_by_uid(graph: NetworkGraph, uid_by_node: dict[int, int]) -> dict[tuple[int, int], int]:
    """Map node-index edges back to original uid pairs for assertion."""
    out = {}
    for u, v in zip(graph.edge_from.tolist(), graph.edge_to.tolist()):
        a, b = uid_by_node[u], uid_by_node[v]
        out[(a, b) if a < b else (b, a)] = (u, v)
    return out


def test_co_presence_graph_matches_rust_reference_pandas():
    df = pd.DataFrame(_visits_rows())
    graph, persistence, time_steps, skip_info = co_presence_graph_from_visits(df)
    assert graph.node_count == 5
    assert time_steps == 2
    assert graph.edge_count == 4
    assert skip_info == {"skipped_groups": 0, "skipped_rows": 0}

    # node index i corresponds to the i-th smallest uid (1,2,3,4,5) -> node i == uid i+1
    persistence_by_uid_pair = {}
    for (u, v), p in zip(zip(graph.edge_from.tolist(), graph.edge_to.tolist()), persistence.tolist()):
        persistence_by_uid_pair[(u + 1, v + 1)] = p

    assert persistence_by_uid_pair[(1, 2)] == pytest.approx(1.0)
    assert persistence_by_uid_pair[(1, 3)] == pytest.approx(0.5)
    assert persistence_by_uid_pair[(2, 3)] == pytest.approx(0.5)
    assert persistence_by_uid_pair[(4, 5)] == pytest.approx(0.5)


def test_co_presence_graph_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = pd.DataFrame(_visits_rows())
    pandas_graph, pandas_persistence, pandas_steps, _pandas_skip = co_presence_graph_from_visits(df)
    polars_graph, polars_persistence, polars_steps, _polars_skip = co_presence_graph_from_visits(pl.from_pandas(df))

    assert polars_steps == pandas_steps
    assert polars_graph.node_count == pandas_graph.node_count
    assert polars_graph.edge_count == pandas_graph.edge_count
    np.testing.assert_array_equal(np.sort(polars_persistence), np.sort(pandas_persistence))


def test_co_presence_graph_explicit_day_col():
    df = pd.DataFrame(_visits_rows())
    df["day"] = df["datetime"].dt.normalize()
    graph, _persistence, time_steps, _skip_info = co_presence_graph_from_visits(df, day_col="day", datetime_col=None)
    assert time_steps == 2
    assert graph.edge_count == 4


def test_co_presence_graph_oversized_group_skipped():
    df = pd.DataFrame(_visits_rows())
    graph, _persistence, _time_steps, skip_info = co_presence_graph_from_visits(df, max_group_size=2)
    # The 3-user group at venue A on day 0 (size 3 > max_group_size=2) is
    # skipped entirely; only the (1,2) pair from day 1 and (4,5) from venue B
    # survive.
    assert graph.edge_count == 2
    assert skip_info == {"skipped_groups": 1, "skipped_rows": 3}


def test_co_presence_graph_empty_input():
    df = pd.DataFrame({"uid": [], "datetime": pd.to_datetime([]), "location_id": []})
    graph, _persistence, time_steps, skip_info = co_presence_graph_from_visits(df)
    assert graph.node_count == 0
    assert graph.edge_count == 0
    assert time_steps == 0
    assert skip_info == {"skipped_groups": 0, "skipped_rows": 0}


def test_co_presence_graph_missing_columns_raises():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError):
        co_presence_graph_from_visits(df)


def test_co_presence_graph_max_group_size_validation():
    df = pd.DataFrame(_visits_rows())
    with pytest.raises(ValueError, match="max_group_size"):
        co_presence_graph_from_visits(df, max_group_size=1)


# ---------------------------------------------------------------------------
# graph_from_edges
# ---------------------------------------------------------------------------


def test_graph_from_edges_normalizes_and_dedupes():
    graph = graph_from_edges(4, {(0, 1), (1, 0), (2, 3)})
    assert graph.edge_count == 2
    assert graph.edges == {(0, 1), (2, 3)}


def test_graph_from_edges_drops_self_loops_and_out_of_range():
    graph = graph_from_edges(3, {(0, 0), (0, 5), (-1, 1), (1, 2)})
    assert graph.edges == {(1, 2)}


# ---------------------------------------------------------------------------
# clustering_coefficients / topological_overlap
# ---------------------------------------------------------------------------
# 0-1-2 triangle plus a pendant 3 attached to 0 -- cross-checked against the
# Rust reference test `clustering_coefficient_of_a_triangle_is_one`.


def test_clustering_coefficient_matches_rust_reference():
    graph = NetworkGraph(
        node_count=4,
        edge_from=np.array([0, 1, 0, 0], dtype=np.uint32),
        edge_to=np.array([1, 2, 2, 3], dtype=np.uint32),
    )
    cc = clustering_coefficients(graph)
    assert cc[0] == pytest.approx(1.0 / 3.0)
    assert cc[1] == pytest.approx(1.0)
    assert cc[2] == pytest.approx(1.0)
    assert cc[3] == 0.0


def test_topological_overlap_matches_hand_computed_jaccard():
    graph = NetworkGraph(
        node_count=4,
        edge_from=np.array([0, 1, 2, 0], dtype=np.uint32),
        edge_to=np.array([1, 2, 3, 2], dtype=np.uint32),
    )
    overlap = topological_overlap(graph)
    by_edge = {(u, v): o for u, v, o in zip(graph.edge_from.tolist(), graph.edge_to.tolist(), overlap.tolist())}
    assert by_edge[(0, 1)] == pytest.approx(1.0 / 3.0)
    assert by_edge[(1, 2)] == pytest.approx(1.0 / 4.0)
    assert by_edge[(2, 3)] == pytest.approx(0.0)
    assert by_edge[(0, 2)] == pytest.approx(1.0 / 4.0)


def test_empty_graph_produces_empty_metrics():
    graph = NetworkGraph(node_count=0, edge_from=np.array([], dtype=np.uint32), edge_to=np.array([], dtype=np.uint32))
    assert clustering_coefficients(graph).size == 0
    assert topological_overlap(graph).size == 0


# ---------------------------------------------------------------------------
# degree_preserving_random_graph / random_persistence
# ---------------------------------------------------------------------------


def test_degree_preserving_random_graph_is_deterministic():
    degrees = np.array([5.0, 3.0, 2.0, 4.0, 1.0])
    g1 = degree_preserving_random_graph(degrees, seed=7)
    g2 = degree_preserving_random_graph(degrees, seed=7)
    np.testing.assert_array_equal(g1.edge_from, g2.edge_from)
    np.testing.assert_array_equal(g1.edge_to, g2.edge_to)


def test_degree_preserving_random_graph_zero_degrees_produces_empty_graph():
    degrees = np.zeros(4)
    g = degree_preserving_random_graph(degrees, seed=1)
    assert g.edge_count == 0
    assert g.node_count == 4


def test_random_persistence_empty_graph_returns_empty():
    empty = NetworkGraph(node_count=3, edge_from=np.array([], dtype=np.uint32), edge_to=np.array([], dtype=np.uint32))
    result = random_persistence(empty, np.array([1.0, 2.0, 3.0]), time_steps=5, seed=1)
    assert result.size == 0


# ---------------------------------------------------------------------------
# distribution_summary / safe_wasserstein
# ---------------------------------------------------------------------------


def test_distribution_summary_basic():
    summary = distribution_summary(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert summary["count"] == 5
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(3.0)


def test_distribution_summary_empty_after_filtering_non_finite():
    summary = distribution_summary(np.array([np.nan, np.inf]))
    assert summary["count"] == 0
    assert summary["mean"] is None


def test_safe_wasserstein_identical_distributions_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert safe_wasserstein(a, a) == pytest.approx(0.0)


def test_safe_wasserstein_empty_returns_none():
    assert safe_wasserstein(np.array([]), np.array([1.0])) is None


# ---------------------------------------------------------------------------
# infer_social_ties
# ---------------------------------------------------------------------------
# Two tight-knit communities {0,1,2,3} and {4,5,6,7}, each internally
# fully connected (high topological overlap among members), plus one
# "incidental contact" bridge edge (3,4) with low persistence and low
# overlap. A correct implementation should promote all intra-community
# edges and reject the bridge.


def _planted_community_graph() -> tuple[NetworkGraph, np.ndarray]:
    edges = []
    for community in ([0, 1, 2, 3], [4, 5, 6, 7]):
        for i in range(len(community)):
            for j in range(i + 1, len(community)):
                edges.append((community[i], community[j]))
    edges.append((3, 4))  # incidental bridge
    edge_from = np.array([e[0] for e in edges], dtype=np.uint32)
    edge_to = np.array([e[1] for e in edges], dtype=np.uint32)
    graph = NetworkGraph(node_count=8, edge_from=edge_from, edge_to=edge_to)
    persistence = np.full(len(edges), 0.9)
    persistence[-1] = 0.05  # bridge is rarely observed together
    return graph, persistence


def test_infer_social_ties_separates_planted_communities_from_bridge():
    graph, persistence = _planted_community_graph()
    inferred = infer_social_ties(graph, persistence, regularity_threshold=0.5, random_chance_probability=0.5, seed=3)
    inferred_edges = set(zip(inferred.edge_from.tolist(), inferred.edge_to.tolist()))
    assert (3, 4) not in inferred_edges
    assert (0, 1) in inferred_edges
    assert (4, 5) in inferred_edges
    assert inferred.edge_count == graph.edge_count - 1


def test_infer_social_ties_is_deterministic():
    graph, persistence = _planted_community_graph()
    a = infer_social_ties(graph, persistence, regularity_threshold=0.5, seed=11)
    b = infer_social_ties(graph, persistence, regularity_threshold=0.5, seed=11)
    np.testing.assert_array_equal(a.edge_from, b.edge_from)
    np.testing.assert_array_equal(a.edge_to, b.edge_to)


def test_infer_social_ties_empty_graph_returns_empty():
    empty = NetworkGraph(node_count=5, edge_from=np.array([], dtype=np.uint32), edge_to=np.array([], dtype=np.uint32))
    result = infer_social_ties(empty, np.array([]), regularity_threshold=0.5)
    assert result.edge_count == 0


def test_infer_social_ties_high_threshold_rejects_everything():
    graph, persistence = _planted_community_graph()
    result = infer_social_ties(graph, persistence, regularity_threshold=0.99)
    assert result.edge_count == 0
