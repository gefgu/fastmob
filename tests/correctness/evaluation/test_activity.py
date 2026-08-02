"""Correctness tests for fastmob.measures.evaluation.activity."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from fastmob.measures.evaluation import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix_jensen_shannon_divergence,
    motif_distribution_jensen_shannon_divergence,
    time_bin_matrix_jensen_shannon_divergence,
)


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


def test_motif_distribution_jensen_shannon_divergence_compares_precomputed_results():
    daily1 = pd.DataFrame({"uid": ["u1", "u1"], "date": ["2020-01-01", "2020-01-02"], "motif_id": [1, 1]})
    daily2 = pd.DataFrame({"uid": ["u1", "u1"], "date": ["2020-01-01", "2020-01-02"], "motif_id": [1, 2]})
    value = motif_distribution_jensen_shannon_divergence(daily1, daily2)
    assert value > 0.0
    assert math.isfinite(value)


def test_motif_distribution_jensen_shannon_divergence_identical_distributions_is_zero():
    daily = pd.DataFrame({"uid": ["u1", "u1"], "date": ["2020-01-01", "2020-01-02"], "motif_id": [1, 2]})
    value = motif_distribution_jensen_shannon_divergence(daily, daily.copy())
    assert value == 0.0


def test_motif_distribution_jensen_shannon_divergence_supports_custom_motif_id_col():
    daily1 = pd.DataFrame({"motif": [1, 1]})
    daily2 = pd.DataFrame({"motif": [1, 2]})
    value = motif_distribution_jensen_shannon_divergence(daily1, daily2, motif_id_col="motif")
    assert value > 0.0
    assert math.isfinite(value)


def test_motif_distribution_jensen_shannon_divergence_raises_on_missing_column():
    daily1 = pd.DataFrame({"motif_id": [1, 2]})
    daily2 = pd.DataFrame({"not_motif_id": [1, 2]})
    with pytest.raises(ValueError, match="Motif ID column"):
        motif_distribution_jensen_shannon_divergence(daily1, daily2)
