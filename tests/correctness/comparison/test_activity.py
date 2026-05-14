"""Correctness tests for skmob2.comparison.activity."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from skmob2.comparison import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix_jensen_shannon_divergence,
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
