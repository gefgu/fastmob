"""Correctness tests for skmob2.measures.od."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from skmob2.measures.od import od_matrix, od_metrics_per_area


def test_od_matrix_basic_counts():
    """OD matrix groups and counts correctly."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "A", "B", "A", "C"],
        "destination_area": ["B", "B", "A", "A", "A"],
    })
    result = od_matrix(trips)
    # A->B appears twice
    assert result.loc["A", "B"] == 2
    # A->A appears once (self-loop)
    assert result.loc["A", "A"] == 1
    # B->A appears once
    assert result.loc["B", "A"] == 1
    # C->B does not exist → 0
    assert result.loc["C", "B"] == 0


def test_od_matrix_drops_nulls():
    trips = pd.DataFrame({
        "origin_area":      ["A", None, "B"],
        "destination_area": ["B", "B",  None],
    })
    result = od_matrix(trips)
    # Only A->B survives
    assert result.loc["A", "B"] == 1
    assert result.shape == (1, 1)


def test_od_metrics_per_area_move_inside():
    """MoveInside equals the diagonal; InComing/OutGoing are cross-zone sums."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "A", "A", "B", "B"],
        "destination_area": ["A", "B", "C", "A", "B"],
    })
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od)
    metrics = metrics.set_index("area_code")

    # A: 1 self-loop (MoveInside), 2 outgoing (to B and C), 1 incoming (from B)
    assert metrics.loc["A", "MoveInside"] == 1
    assert metrics.loc["A", "OutGoing"] == 2
    assert metrics.loc["A", "InComing"] == 1

    # B: 1 self-loop, 1 outgoing (to A), 1 incoming (from A)
    assert metrics.loc["B", "MoveInside"] == 1
    assert metrics.loc["B", "OutGoing"] == 1
    assert metrics.loc["B", "InComing"] == 1


def test_od_matrix_column_autodetection():
    """Auto-detect works with non-default column names."""
    trips = pd.DataFrame({
        "Origin_Area":  ["X", "Y"],
        "Dest_Area":    ["Y", "X"],
    })
    result = od_matrix(trips)
    assert result.loc["X", "Y"] == 1
    assert result.loc["Y", "X"] == 1


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


def test_od_matrix_returns_pandas_dataframe():
    """od_matrix must always return a plain pandas DataFrame."""
    trips = pd.DataFrame({
        "origin_area":      ["A", "B"],
        "destination_area": ["B", "A"],
    })
    result = od_matrix(trips)
    assert isinstance(result, pd.DataFrame)


def test_od_metrics_columns():
    """od_metrics_per_area output has expected columns."""
    trips = pd.DataFrame({
        "origin_area":      ["A"],
        "destination_area": ["A"],
    })
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od)
    assert list(metrics.columns) == ["area_code", "MoveInside", "InComing", "OutGoing", "Total"]
