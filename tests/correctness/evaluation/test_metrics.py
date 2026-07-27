"""Correctness tests for fastmob.measures.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from fastmob.measures.evaluation import (
    histogram_jensen_shannon_divergence,
    jensen_shannon_divergence,
    wasserstein_distance,
)

try:
    from scipy.spatial.distance import jensenshannon as _scipy_jensenshannon
    from scipy.stats import wasserstein_distance as _scipy_wasserstein

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy is not installed")


@requires_scipy
def test_jensen_shannon_matches_old_scipy_squared_semantics():
    p = np.array([0.2, 0.8])
    q = np.array([0.7, 0.3])
    assert jensen_shannon_divergence(p, q) == pytest.approx(float(_scipy_jensenshannon(p, q) ** 2))


def test_jensen_shannon_identical_distribution_is_zero():
    assert jensen_shannon_divergence([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


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
