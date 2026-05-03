"""Correctness tests for skmob2.measures.comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from skmob2.measures.comparison import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix_jensen_shannon_divergence,
    column_distribution_jensen_shannon_divergence,
    column_distribution_wasserstein_distance,
    histogram_jensen_shannon_divergence,
    jensen_shannon_divergence,
    od_matrix_common_part_of_commuters,
    time_bin_matrix_jensen_shannon_divergence,
    visits_per_user_jensen_shannon_divergence,
    visits_per_user_wasserstein_distance,
    wasserstein_distance,
)

try:
    from scipy.spatial.distance import jensenshannon as _scipy_jensenshannon
    from scipy.stats import wasserstein_distance as _scipy_wasserstein

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy is not installed")


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


@requires_scipy
def test_jensen_shannon_matches_old_scipy_squared_semantics():
    p = np.array([0.2, 0.8])
    q = np.array([0.7, 0.3])
    assert jensen_shannon_divergence(p, q) == pytest.approx(float(_scipy_jensenshannon(p, q) ** 2))


def test_jensen_shannon_identical_distribution_is_zero():
    assert jensen_shannon_divergence([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_activity_distribution_jensen_shannon_aligns_categories():
    left = pd.DataFrame({"purpose": ["HOME", "WORK", "WORK"]})
    right = pd.DataFrame({"purpose": ["HOME", "SHOP", "SHOP"]})
    value = activity_distribution_jensen_shannon_divergence(left, right)
    assert value > 0.0
    assert math.isfinite(value)


def test_time_bin_matrix_jensen_shannon_aligns_categories():
    matrix1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    matrix2 = np.array([[1.0, 1.0], [0.0, 0.0]])
    value = time_bin_matrix_jensen_shannon_divergence(
        matrix1,
        matrix2,
        categories1=["HOME", "WORK"],
        categories2=["HOME", "SHOP"],
    )
    assert value > 0.0
    assert math.isfinite(value)


def test_transition_matrix_jensen_shannon_aligns_pandas_index():
    left = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["HOME", "WORK"], columns=["HOME", "WORK"])
    right = pd.DataFrame([[0.5, 0.5], [0.0, 0.0]], index=["HOME", "SHOP"], columns=["HOME", "SHOP"])
    value = activity_transition_matrix_jensen_shannon_divergence(left, right)
    assert value > 0.0
    assert math.isfinite(value)


def test_histogram_jensen_shannon_identical_and_partial_overlap():
    assert histogram_jensen_shannon_divergence([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert histogram_jensen_shannon_divergence([1, 2, 3], [3, 4, 5]) > 0.0


@requires_scipy
def test_wasserstein_distance_matches_scipy_for_equal_and_unequal_lengths():
    assert wasserstein_distance([0.0, 1.0], [1.0, 2.0]) == pytest.approx(
        float(_scipy_wasserstein([0.0, 1.0], [1.0, 2.0])),
        rel=1e-6,
    )
    assert wasserstein_distance([0.0, 10.0, 20.0], [5.0]) == pytest.approx(
        float(_scipy_wasserstein([0.0, 10.0, 20.0], [5.0])),
        rel=1e-6,
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


def test_od_matrix_common_part_aligns_origins_and_destinations():
    left = pd.DataFrame([[10, 0], [0, 5]], index=["a", "b"], columns=["x", "y"])
    right = pd.DataFrame([[5, 5], [0, 5]], index=["a", "c"], columns=["x", "z"])
    assert od_matrix_common_part_of_commuters(left, right) == pytest.approx(1.0 / 3.0)


def test_polars_dataframe_input_for_grouped_comparison():
    pl = pytest.importorskip("polars")
    left = pl.from_pandas(_comparison_df())
    right_pd = _comparison_df()
    right_pd["value"] = right_pd["value"] + 1
    right = pl.from_pandas(right_pd)
    mean, parts = column_distribution_wasserstein_distance(left, right, "value")
    assert math.isfinite(mean)
    assert len(parts) == 1
