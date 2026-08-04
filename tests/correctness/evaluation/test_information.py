"""Correctness tests for fastmob.measures.evaluation.information."""

from __future__ import annotations

import numpy as np
import pytest
from fastmob.measures.evaluation.information import information_gain, kullback_leibler_divergence

try:
    import scipy  # noqa: F401

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(
    not _HAS_SCIPY,
    reason="scipy is not installed — install with: pip install scipy",
)


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
# kullback_leibler_divergence (requires scipy)
# ---------------------------------------------------------------------------


@requires_scipy
def test_kl_divergence_identical():
    """KL divergence of identical distributions is 0."""
    v = [0.5, 0.5]
    assert kullback_leibler_divergence(v, v) == pytest.approx(0.0)


@requires_scipy
def test_kl_divergence_known_value():
    """KL divergence matches scipy.stats.entropy for a known input."""
    from scipy import stats

    true = [0.4, 0.6]
    pred = [0.5, 0.5]
    expected = float(stats.entropy(true, pred))
    assert kullback_leibler_divergence(true, pred) == pytest.approx(expected, rel=1e-9)
