"""Correctness tests for skmob2.measures.od."""
from __future__ import annotations

import pandas as pd
import pytest
from skmob2.measures.od import od_matrix, od_metrics_per_area


def _lookup(od, origin, dest, origin_col="origin_area", dest_col="destination_area"):
    """Return the count for an (origin, dest) pair; 0 if the pair is absent."""
    mask = (od[origin_col] == origin) & (od[dest_col] == dest)
    rows = od[mask]
    return int(rows["count"].iloc[0]) if len(rows) > 0 else 0


def test_od_matrix_basic_counts():
    """OD matrix groups and counts correctly."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "A", "B", "A", "C"],
        "destination_area": ["B", "B", "A", "A", "A"],
    })
    result = od_matrix(trips)
    assert _lookup(result, "A", "B") == 2
    assert _lookup(result, "A", "A") == 1
    assert _lookup(result, "B", "A") == 1
    assert _lookup(result, "C", "B") == 0  # absent pair → 0


def test_od_matrix_drops_nulls():
    trips = pd.DataFrame({
        "origin_area":      ["A", None, "B"],
        "destination_area": ["B", "B",  None],
    })
    result = od_matrix(trips)
    assert len(result) == 1
    assert _lookup(result, "A", "B") == 1


def test_od_matrix_long_format_columns():
    """od_matrix result has origin, destination, and count columns."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "B"],
        "destination_area": ["B", "A"],
    })
    result = od_matrix(trips)
    assert "origin_area" in result.columns
    assert "destination_area" in result.columns
    assert "count" in result.columns


def test_od_matrix_returns_native_backend():
    """od_matrix returns the same backend as the input."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "B"],
        "destination_area": ["B", "A"],
    })
    result = od_matrix(trips)
    assert isinstance(result, pd.DataFrame)


def test_od_matrix_column_autodetection():
    """Auto-detect works with non-default column names."""
    trips = pd.DataFrame({
        "Origin_Area": ["X", "Y"],
        "Dest_Area":   ["Y", "X"],
    })
    result = od_matrix(trips)
    assert _lookup(result, "X", "Y", origin_col="Origin_Area", dest_col="Dest_Area") == 1
    assert _lookup(result, "Y", "X", origin_col="Origin_Area", dest_col="Dest_Area") == 1


def test_od_metrics_per_area_move_inside():
    """MoveInside equals self-loops; InComing/OutGoing are cross-zone sums."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "A", "A", "B", "B"],
        "destination_area": ["A", "B", "C", "A", "B"],
    })
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od).set_index("area_code")

    # A: 1 self-loop, 2 outgoing (to B and C), 1 incoming (from B)
    assert metrics.loc["A", "MoveInside"] == 1
    assert metrics.loc["A", "OutGoing"] == 2
    assert metrics.loc["A", "InComing"] == 1

    # B: 1 self-loop, 1 outgoing (to A), 1 incoming (from A)
    assert metrics.loc["B", "MoveInside"] == 1
    assert metrics.loc["B", "OutGoing"] == 1
    assert metrics.loc["B", "InComing"] == 1


def test_od_metrics_total_column():
    """Total column equals MoveInside + InComing + OutGoing."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "A", "B"],
        "destination_area": ["A", "B", "A"],
    })
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od).set_index("area_code")
    for area in metrics.index:
        row = metrics.loc[area]
        assert row["Total"] == row["MoveInside"] + row["InComing"] + row["OutGoing"]


def test_od_metrics_columns():
    """od_metrics_per_area output has expected columns."""
    trips = pd.DataFrame({
        "origin_area":      ["A"],
        "destination_area": ["A"],
    })
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od)
    assert list(metrics.columns) == ["area_code", "MoveInside", "InComing", "OutGoing", "Total"]


def test_od_metrics_area_with_only_incoming():
    """An area that only receives trips has OutGoing=0."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "A"],
        "destination_area": ["B", "C"],
    })
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od).set_index("area_code")
    assert metrics.loc["B", "OutGoing"] == 0
    assert metrics.loc["B", "InComing"] == 1
    assert metrics.loc["B", "MoveInside"] == 0
