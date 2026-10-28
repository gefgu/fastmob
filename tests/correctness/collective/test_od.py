"""Correctness tests for fastmob.measures.collective.od."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from fastmob.measures.collective.od import od_matrix, od_metrics_per_area


def _lookup(od, origin, dest, origin_col="origin_area", dest_col="destination_area"):
    """Return the count for an (origin, dest) pair; 0 if the pair is absent."""
    mask = (od[origin_col] == origin) & (od[dest_col] == dest)
    rows = od[mask]
    return int(rows["count"].iloc[0]) if len(rows) > 0 else 0


def test_od_matrix_basic_counts():
    """OD matrix groups and counts correctly."""
    trips = pd.DataFrame(
        {
            "origin_area": ["A", "A", "B", "A", "C"],
            "destination_area": ["B", "B", "A", "A", "A"],
        }
    )
    result = od_matrix(trips)
    assert _lookup(result, "A", "B") == 2
    assert _lookup(result, "A", "A") == 1
    assert _lookup(result, "B", "A") == 1
    assert _lookup(result, "C", "B") == 0  # absent pair → 0


def test_od_matrix_drops_nulls():
    trips = pd.DataFrame(
        {
            "origin_area": ["A", None, "B"],
            "destination_area": ["B", "B", None],
        }
    )
    result = od_matrix(trips)
    assert len(result) == 1
    assert _lookup(result, "A", "B") == 1


def test_od_matrix_long_format_columns():
    """od_matrix result has origin, destination, and count columns."""
    trips = pd.DataFrame(
        {
            "origin_area": ["A", "B"],
            "destination_area": ["B", "A"],
        }
    )
    result = od_matrix(trips)
    assert "origin_area" in result.columns
    assert "destination_area" in result.columns
    assert "count" in result.columns


def test_od_matrix_returns_native_backend():
    """od_matrix returns the same backend as the input."""
    trips = pd.DataFrame(
        {
            "origin_area": ["A", "B"],
            "destination_area": ["B", "A"],
        }
    )
    result = od_matrix(trips)
    assert isinstance(result, pd.DataFrame)


def test_od_matrix_column_autodetection():
    """Auto-detect works with non-default column names."""
    trips = pd.DataFrame(
        {
            "Origin_Area": ["X", "Y"],
            "Dest_Area": ["Y", "X"],
        }
    )
    result = od_matrix(trips)
    assert _lookup(result, "X", "Y", origin_col="Origin_Area", dest_col="Dest_Area") == 1
    assert _lookup(result, "Y", "X", origin_col="Origin_Area", dest_col="Dest_Area") == 1


def test_od_matrix_sort_false_matches_sort_true_rows():
    """sort=False must return the same rows/counts, just not necessarily ordered."""
    trips = pd.DataFrame(
        {
            "origin_area": ["C", "A", "A", "B", "A"],
            "destination_area": ["A", "B", "B", "A", "A"],
        }
    )
    sorted_result = od_matrix(trips, sort=True)
    unsorted_result = od_matrix(trips, sort=False)

    sorted_cmp = sorted_result.sort_values(["origin_area", "destination_area"]).reset_index(drop=True)
    unsorted_cmp = unsorted_result.sort_values(["origin_area", "destination_area"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sorted_cmp, unsorted_cmp)


@pytest.mark.parametrize("backend", [pd.DataFrame, pl.DataFrame])
def test_od_matrix_weighted(backend):
    trips = backend(
        {
            "origin_area": ["A", "A", "B", None],
            "destination_area": ["B", "B", "A", "B"],
            "weight": [1.5, 2.25, 4.0, 100.0],
        }
    )
    result = od_matrix(trips, weight_col="weight")
    assert isinstance(result, backend)
    rows = result.to_dict("records") if backend is pd.DataFrame else result.to_dicts()
    assert rows == [
        {"origin_area": "A", "destination_area": "B", "count": 3.75},
        {"origin_area": "B", "destination_area": "A", "count": 4.0},
    ]
    unweighted = od_matrix(trips)
    rows = unweighted.to_dict("records") if backend is pd.DataFrame else unweighted.to_dicts()
    assert rows[0]["count"] == 2


@pytest.mark.parametrize("backend", [pd.DataFrame, pl.DataFrame])
def test_od_matrix_missing_weights_and_invalid_dtype(backend):
    trips = backend({
        "origin_area": ["A", "A", "B", "B"],
        "destination_area": ["B", "B", "A", "A"],
        "weight": [1.5, float("nan"), None, float("nan")],
    })
    result = od_matrix(trips, weight_col="weight")
    rows = result.to_dict("records") if backend is pd.DataFrame else result.to_dicts()
    assert [row["count"] for row in rows] == [1.5, 0.0]

    invalid = backend({"origin_area": ["A"], "destination_area": ["B"], "weight": ["2"]})
    with pytest.raises(ValueError, match="numeric dtype"):
        od_matrix(invalid, weight_col="weight")
    with pytest.raises(ValueError, match="not found"):
        od_matrix(invalid, weight_col="missing")


def test_od_metrics_per_area_move_inside():
    """MoveInside equals self-loops; InComing/OutGoing are cross-zone sums."""
    trips = pd.DataFrame(
        {
            "origin_area": ["A", "A", "A", "B", "B"],
            "destination_area": ["A", "B", "C", "A", "B"],
        }
    )
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
    trips = pd.DataFrame(
        {
            "origin_area": ["A", "A", "B"],
            "destination_area": ["A", "B", "A"],
        }
    )
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od).set_index("area_code")
    for area in metrics.index:
        row = metrics.loc[area]
        assert row["Total"] == row["MoveInside"] + row["InComing"] + row["OutGoing"]


def test_od_metrics_columns():
    """od_metrics_per_area output has expected columns."""
    trips = pd.DataFrame(
        {
            "origin_area": ["A"],
            "destination_area": ["A"],
        }
    )
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od)
    assert list(metrics.columns) == ["area_code", "MoveInside", "InComing", "OutGoing", "Total"]


def test_od_metrics_area_with_only_incoming():
    """An area that only receives trips has OutGoing=0."""
    trips = pd.DataFrame(
        {
            "origin_area": ["A", "A"],
            "destination_area": ["B", "C"],
        }
    )
    od = od_matrix(trips)
    metrics = od_metrics_per_area(od).set_index("area_code")
    assert metrics.loc["B", "OutGoing"] == 0
    assert metrics.loc["B", "InComing"] == 1
    assert metrics.loc["B", "MoveInside"] == 0
