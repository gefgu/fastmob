"""Correctness tests for BaseDataFrame.compare_to() (fastmob.measures.evaluation.compare)."""

from __future__ import annotations

import fastmob
import pandas as pd
import pytest
from fastmob.measures.evaluation import ComparisonResult, jensen_shannon_divergence
from fastmob.measures.evaluation.compare import compare_to


def _trips(values, groups=None):
    n = len(values)
    data = {
        "trip_id": range(n),
        "started_at": pd.Timestamp("2026-01-01"),
        "finished_at": pd.Timestamp("2026-01-01 01:00"),
        "origin_staypoint_id": [None] * n,
        "destination_staypoint_id": [None] * n,
        "tripleg_ids": [[] for _ in range(n)],
        "metric_value": values,
    }
    if groups is not None:
        data["bucket"] = groups
    return fastmob.Trips(pd.DataFrame(data))


def test_compare_to_ungrouped_uses_wasserstein_by_default():
    left = _trips([1.0, 2.0, 3.0])
    right = _trips([1.0, 2.0, 3.0])
    result = left.compare_to(right, "metric_value")
    assert isinstance(result, ComparisonResult)
    assert result.overall == pytest.approx(0.0)
    assert result.groups is None


def test_compare_to_ungrouped_matches_free_function():
    left = _trips([1.0, 2.0, 3.0])
    right = _trips([2.0, 3.0, 4.0])
    result = left.compare_to(right, "metric_value")
    expected = compare_to(left.df, right.df, "metric_value")
    assert result.overall == pytest.approx(expected.overall)
    assert result.overall > 0.0


def test_compare_to_grouped_returns_per_group_breakdown():
    left = _trips([1.0, 2.0, 10.0, 11.0], groups=["morning", "morning", "evening", "evening"])
    right = _trips([1.0, 2.0, 20.0, 21.0], groups=["morning", "morning", "evening", "evening"])
    result = left.compare_to(right, "metric_value", group_col="bucket")
    assert result.groups is not None
    labels = dict(result.groups)
    assert set(labels) == {"evening", "morning"}
    assert labels["morning"] == pytest.approx(0.0)
    assert labels["evening"] > 0.0
    assert result.overall == pytest.approx(sum(labels.values()) / len(labels))


def test_compare_to_grouped_only_compares_shared_labels():
    left = _trips([1.0, 2.0, 10.0], groups=["morning", "morning", "evening"])
    right = _trips([1.0, 2.0], groups=["morning", "morning"])
    result = left.compare_to(right, "metric_value", group_col="bucket")
    assert result.groups == [("morning", pytest.approx(0.0))]


def test_compare_to_accepts_custom_metric_callable():
    left = _trips([1.0, 2.0, 3.0])
    right = _trips([1.0, 2.0, 3.0, 3.0])

    def mean_abs_diff(a, b):
        import pyarrow.compute as pc

        return abs(pc.mean(a).as_py() - pc.mean(b).as_py())

    result = left.compare_to(right, "metric_value", metric=mean_abs_diff)
    assert result.overall == pytest.approx(abs(2.0 - 2.25))


def test_compare_to_accepts_jensen_shannon_divergence_as_metric():
    left = _trips([1.0, 0.0])
    right = _trips([0.0, 1.0])
    result = left.compare_to(right, "metric_value", metric=jensen_shannon_divergence)
    assert result.overall > 0.0


def test_compare_to_missing_column_raises():
    left = _trips([1.0, 2.0])
    right = _trips([1.0, 2.0])
    with pytest.raises(ValueError, match="not_a_column"):
        left.compare_to(right, "not_a_column")


def test_compare_to_missing_group_column_raises():
    left = _trips([1.0, 2.0])
    right = _trips([1.0, 2.0])
    with pytest.raises(ValueError, match="not_a_group"):
        left.compare_to(right, "metric_value", group_col="not_a_group")


def test_compare_to_works_on_a_different_hierarchy_class():
    """compare_to() is inherited generically from BaseDataFrame, not Trips-specific."""
    left = fastmob.FlowDataFrame({"origin": [1, 2], "destination": [2, 3], "flow": [1.0, 5.0]})
    right = fastmob.FlowDataFrame({"origin": [1, 2], "destination": [2, 3], "flow": [1.0, 5.0]})
    result = left.compare_to(right, "flow")
    assert result.overall == pytest.approx(0.0)
