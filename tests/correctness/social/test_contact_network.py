"""RECAST social-contact classification tests."""

from __future__ import annotations

import pandas as pd
import pytest

from fastmob.core import Locations, Staypoints
from fastmob.social import RecastClass, recast_from_staypoints


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
            "started_at": [base, base + pd.Timedelta(minutes=5), base + pd.Timedelta(hours=1), base + pd.Timedelta(hours=1, minutes=5), base, base + pd.Timedelta(hours=1)],
            "finished_at": [base + pd.Timedelta(minutes=20), base + pd.Timedelta(minutes=25), base + pd.Timedelta(hours=1, minutes=20), base + pd.Timedelta(hours=1, minutes=25), base + pd.Timedelta(minutes=20), base + pd.Timedelta(hours=1, minutes=20)],
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
