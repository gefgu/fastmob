"""Correctness tests for fastmob.measures.evaluation.information."""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastmob.measures.evaluation.information import information_gain, kullback_leibler_divergence


# ---------------------------------------------------------------------------
# information_gain
# ---------------------------------------------------------------------------


def test_information_gain_identical():
    """Information gain is 0 when true == pred (log(1) = 0)."""
    v = [1.0, 2.0, 3.0]
    assert information_gain(v, v) == pytest.approx(0.0)


def test_information_gain_known_value():
    """information_gain matches hand computation for a two-element case."""
    true = [2.0, 4.0]
    pred = [1.0, 8.0]
    expected = (2.0 / 6.0) * np.log(2.0 / 1.0) + (4.0 / 6.0) * np.log(4.0 / 8.0)
    assert information_gain(true, pred) == pytest.approx(float(expected), rel=1e-9)


# ---------------------------------------------------------------------------
# kullback_leibler_divergence
# ---------------------------------------------------------------------------


def test_kl_divergence_identical():
    """KL divergence of identical distributions is 0."""
    v = [0.5, 0.5]
    assert kullback_leibler_divergence(v, v) == pytest.approx(0.0)


def test_kl_divergence_normalizes_unnormalized_inputs():
    true = [0.4, 0.6]
    pred = [0.5, 0.5]
    expected = 0.4 * math.log(0.4 / 0.5) + 0.6 * math.log(0.6 / 0.5)
    assert kullback_leibler_divergence(true, pred) == pytest.approx(expected, rel=1e-9)
    assert kullback_leibler_divergence([4.0, 6.0], [5.0, 5.0]) == pytest.approx(expected, rel=1e-9)


def test_kl_divergence_ignores_zero_reference_mass():
    assert kullback_leibler_divergence([0.0, 2.0], [0.0, 2.0]) == pytest.approx(0.0)


def test_kl_divergence_is_infinite_for_missing_predicted_support():
    assert kullback_leibler_divergence([1.0, 1.0], [0.0, 1.0]) == float("inf")


def test_kl_divergence_requires_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        kullback_leibler_divergence([1.0, 2.0], [1.0])
