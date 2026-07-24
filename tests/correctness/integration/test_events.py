"""Correctness tests for fastmob.integration.events."""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.integration.events import join_with_events

_TRAJ = pd.DataFrame(
    {
        "uid": ["u1", "u1"],
        "datetime": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:20:00"]),
        "lat": [0.0, 0.0],
        "lng": [0.0, 0.0],
    }
)

_EVENTS = pd.DataFrame(
    {
        "lat": [0.0, 5.0],
        "lng": [0.0, 5.0],
        "datetime": pd.to_datetime(["2020-01-01 00:05:00", "2020-01-01 00:05:00"]),
        "event_id": ["e1", "e2"],
        "event_type": ["show", "feira"],
    }
)


def test_join_with_events_picks_spatially_nearest_in_window():
    result = join_with_events(_TRAJ, _EVENTS, time_window_s=900)
    # traj point 0 (t=00:00) is within 900s of both events (both at 00:05);
    # e1 is co-located (dist=0), so it must win over the far-away e2.
    assert result["event_id"].iloc[0] == "e1"
    assert result["event_type"].iloc[0] == "show"
    assert result["dist_event"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_join_with_events_window_boundary_is_inclusive():
    # traj point 1 (t=00:20) is exactly 900s after the events (00:05).
    result = join_with_events(_TRAJ, _EVENTS, time_window_s=900)
    assert result["event_id"].iloc[1] == "e1"
    assert result["dist_event"].iloc[1] < 1.0


def test_join_with_events_outside_window_reports_null_and_inf():
    result = join_with_events(_TRAJ, _EVENTS, time_window_s=100)
    assert result["event_id"].isna().all()
    assert result["event_type"].isna().all()
    assert (result["dist_event"] == float("inf")).all()


def test_join_with_events_empty_events_returns_null_and_inf():
    empty_events = _EVENTS.iloc[:0]
    result = join_with_events(_TRAJ, empty_events)
    assert result["event_id"].isna().all()
    assert (result["dist_event"] == float("inf")).all()


def test_join_with_events_preserves_original_columns():
    result = join_with_events(_TRAJ, _EVENTS, time_window_s=900)
    for col in _TRAJ.columns:
        assert col in result.columns


def test_join_with_events_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    pandas_result = join_with_events(_TRAJ, _EVENTS, time_window_s=900)
    polars_result = join_with_events(pl.from_pandas(_TRAJ), pl.from_pandas(_EVENTS), time_window_s=900).to_pandas()
    assert list(pandas_result["event_id"]) == list(polars_result["event_id"])
    assert list(pandas_result["dist_event"]) == pytest.approx(list(polars_result["dist_event"]))
