"""RECAST social-contact classification tests."""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.core import Locations, Staypoints
from fastmob.social import (
    RecastClass,
    recast_from_staypoints,
    rnd,
    t_rnd,
    temporal_graph_from_staypoints,
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
