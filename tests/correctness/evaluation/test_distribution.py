"""Correctness tests for fkmob.measures.evaluation.distribution."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fkmob.measures.evaluation import (
    column_distribution_jensen_shannon_divergence,
    column_distribution_wasserstein_distance,
    visits_per_user_jensen_shannon_divergence,
    visits_per_user_wasserstein_distance,
)


def _comparison_df():
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "value": [1.0, 2.0, 4.0, 8.0, 2.0, 4.0],
            "purpose": ["HOME", "WORK", "HOME", "SHOP", "WORK", "SHOP"],
            "day_of_week": ["monday", "monday", "saturday", "saturday", "monday", "saturday"],
            "start_timestamp": pd.to_datetime(
                [
                    "2020-01-01 08:00",
                    "2020-01-01 13:00",
                    "2020-01-04 20:00",
                    "2020-01-04 23:00",
                    "2020-01-06 09:00",
                    "2020-01-04 03:00",
                ]
            ),
        }
    )


def test_grouped_column_comparisons_for_supported_hues():
    left = _comparison_df()
    right = _comparison_df()
    right["value"] = right["value"] + 1

    for hue in [None, "day_of_week", "day_period", "purpose"]:
        js_mean, js_parts = column_distribution_jensen_shannon_divergence(
            left,
            right,
            "value",
            hue=hue,
            trip_start_col1="start_timestamp",
            trip_start_col2="start_timestamp",
            day_col1="day_of_week",
            day_col2="day_of_week",
            purpose_col="purpose",
        )
        w_mean, w_parts = column_distribution_wasserstein_distance(
            left,
            right,
            "value",
            hue=hue,
            trip_start_col1="start_timestamp",
            trip_start_col2="start_timestamp",
            day_col1="day_of_week",
            day_col2="day_of_week",
            purpose_col="purpose",
        )
        assert math.isfinite(js_mean)
        assert len(js_parts) > 0
        assert math.isfinite(w_mean)
        assert len(w_parts) > 0


def test_visits_per_user_comparisons():
    left = _comparison_df()
    right = pd.concat([_comparison_df(), _comparison_df().iloc[[0]]], ignore_index=True)
    js_mean, js_parts = visits_per_user_jensen_shannon_divergence(left, right)
    w_mean, w_parts = visits_per_user_wasserstein_distance(left, right)
    assert math.isfinite(js_mean)
    assert len(js_parts) == 1
    assert math.isfinite(w_mean)
    assert len(w_parts) == 1


def test_polars_dataframe_input_for_grouped_comparison():
    pl = pytest.importorskip("polars")
    left = pl.from_pandas(_comparison_df())
    right_pd = _comparison_df()
    right_pd["value"] = right_pd["value"] + 1
    right = pl.from_pandas(right_pd)
    mean, parts = column_distribution_wasserstein_distance(left, right, "value")
    assert math.isfinite(mean)
    assert len(parts) == 1
