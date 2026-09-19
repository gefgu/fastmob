"""RECAST social-contact classification tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.core import Locations, Staypoints
from fastmob.social import (
    RecastClass,
    clustering_coefficients,
    co_presence_graph_from_staypoints,
    degree_preserving_random_graph,
    graph_from_edges,
    recast_from_staypoints,
    rnd,
    t_rnd,
    temporal_graph_from_staypoints,
    topological_overlap,
    validate_recast_from_staypoints,
)


def _locations() -> Locations:
    return Locations(
        pd.DataFrame({"location_id": ["A", "B"], "center_lat": [0.0, 1.0], "center_lng": [0.0, 1.0]}),
        scope="global",
    )


def _staypoints() -> Staypoints:
    base = pd.Timestamp("2020-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "uid": ["alice", "bob", "alice", "bob", "carol", "carol"],
            "lat": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "started_at": [
                base,
                base + pd.Timedelta(minutes=5),
                base + pd.Timedelta(hours=1),
                base + pd.Timedelta(hours=1, minutes=5),
                base,
                base + pd.Timedelta(hours=1),
            ],
            "finished_at": [
                base + pd.Timedelta(minutes=20),
                base + pd.Timedelta(minutes=25),
                base + pd.Timedelta(hours=1, minutes=20),
                base + pd.Timedelta(hours=1, minutes=25),
                base + pd.Timedelta(minutes=20),
                base + pd.Timedelta(hours=1, minutes=20),
            ],
            "location_id": ["A", "A", "A", "A", "B", "B"],
        }
    )
    return Staypoints(frame)


def test_recast_uses_interval_overlap_and_restores_user_ids():
    result = recast_from_staypoints(_staypoints(), _locations(), min_minutes_for_encounter=5, p_rnd=0.5, seed=7)
    assert result.time_steps == 1
    assert list(zip(result.source_users.to_pylist(), result.target_users.to_pylist())) == [("alice", "bob")]
    assert result.edge_persistence[0].as_py() == pytest.approx(1.0)
    assert result.classes[0].as_py() in set(RecastClass)
    assert result.class_names[0].as_py() in {"Friends", "Bridges", "Acquaintances", "Random"}


def test_recast_is_seed_deterministic():
    first = recast_from_staypoints(_staypoints(), _locations(), min_minutes_for_encounter=5, p_rnd=0.5, seed=9)
    second = recast_from_staypoints(_staypoints(), _locations(), min_minutes_for_encounter=5, p_rnd=0.5, seed=9)
    assert first.classes.equals(second.classes)
    assert first.edge_persistence.equals(second.edge_persistence)
    assert first.persistence_threshold == second.persistence_threshold
    assert first.overlap_threshold == second.overlap_threshold


def test_recast_requires_global_locations_and_intervals():
    with pytest.raises(ValueError, match="global Locations"):
        recast_from_staypoints(_staypoints(), None)
    with pytest.raises(ValueError, match="positive"):
        recast_from_staypoints(_staypoints(), _locations(), min_minutes_for_encounter=0)


def test_recast_minimum_encounter_duration_filters_short_overlap():
    result = recast_from_staypoints(_staypoints(), _locations(), min_minutes_for_encounter=16, p_rnd=0.5)
    assert len(result.classes) == 0


def test_temporal_graph_and_paper_random_generators_are_arrow_backed_and_seeded():
    graph = temporal_graph_from_staypoints(_staypoints(), _locations(), min_minutes_for_encounter=5)
    assert graph.time_steps == 1
    assert graph.event(0).edge_from.to_pylist() == [0]
    assert graph.node_ids.to_pylist() == ["alice", "bob", "carol"]
    first = rnd(graph.event(0), seed=11)
    second = rnd(graph.event(0), seed=11)
    assert first.edge_from.equals(second.edge_from)
    replicas = t_rnd(graph, random_replicates=2, seed=11)
    assert len(replicas) == 2
    assert all(replica.window_starts_ms.equals(graph.window_starts_ms) for replica in replicas)


def test_validation_exposes_paper_ccdf_and_clustering_diagnostics():
    validation = validate_recast_from_staypoints(
        _staypoints(), _locations(), min_minutes_for_encounter=5, p_rnd=0.5, random_replicates=2, seed=3
    )
    assert validation.classification.time_steps == 1
    assert validation.persistence_observed.equals(validation.classification.edge_persistence)
    assert len(validation.persistence_null) >= 0
    assert len(validation.full_clustering.observed) == validation.graph.time_steps
    assert len(validation.full_clustering.random_mean) == validation.graph.time_steps
    assert len(validation.full_clustering.random_std) == validation.graph.time_steps
    assert len(validation.random_only_clustering.observed) == validation.graph.time_steps


def test_raw_contact_graph_reuses_recast_daily_event_construction():
    frame = _staypoints().to_native().copy()
    second_day = frame.copy()
    second_day["started_at"] += pd.Timedelta(days=1)
    second_day["finished_at"] += pd.Timedelta(days=1)
    staypoints = Staypoints(pd.concat([frame, second_day], ignore_index=True))
    temporal = temporal_graph_from_staypoints(staypoints, _locations(), min_minutes_for_encounter=5)
    result = co_presence_graph_from_staypoints(staypoints, _locations(), min_minutes_for_encounter=5)

    assert result.temporal_graph.edge_offsets.equals(temporal.edge_offsets)
    assert result.temporal_graph.edge_from.equals(temporal.edge_from)
    assert result.temporal_graph.edge_to.equals(temporal.edge_to)
    assert result.graph.node_count == 3
    assert result.graph.edges == {(0, 1)}
    np.testing.assert_allclose(result.edge_persistence, [1.0])
    assert result.time_steps == 2
    assert result.temporal_graph.event(0).edge_from.equals(result.temporal_graph.event(1).edge_from)
    np.testing.assert_allclose(clustering_coefficients(result.graph), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(topological_overlap(result.graph), [0.0])


def test_raw_contact_graph_uses_minimum_duration_and_has_no_group_cap():
    frame = _staypoints().to_native().copy()
    frame = frame.loc[frame["location_id"] == "A"].copy()
    frame = frame.iloc[:3].copy()
    frame["uid"] = ["alice", "bob", "carol"]
    frame["started_at"] = pd.Timestamp("2020-01-01T00:00:00Z")
    frame["finished_at"] = pd.Timestamp("2020-01-01T00:20:00Z")
    three_users = Staypoints(frame)

    result = co_presence_graph_from_staypoints(three_users, _locations(), min_minutes_for_encounter=5)
    assert result.temporal_graph.event(0).edge_from.to_pylist() == [0, 0, 1]
    assert result.temporal_graph.event(0).edge_to.to_pylist() == [1, 2, 2]
    assert result.graph.edge_count == 3

    too_short = co_presence_graph_from_staypoints(three_users, _locations(), min_minutes_for_encounter=21)
    assert too_short.graph.edge_count == 0


def test_raw_contact_diagnostics_use_seeded_recast_graph_kernels():
    graph = graph_from_edges(4, [(0, 1), (0, 2), (1, 2), (2, 3)])
    np.testing.assert_allclose(clustering_coefficients(graph), [1.0, 1.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(topological_overlap(graph), [1.0 / 3.0, 0.25, 0.25, 0.0])

    first = degree_preserving_random_graph(graph.degrees(), seed=13)
    second = degree_preserving_random_graph(graph.degrees(), seed=13)
    assert first.edges == second.edges
    assert all(source < target for source, target in first.edges)
