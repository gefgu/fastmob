"""Correctness tests for fastmob.measures.evaluation.correlation."""

from __future__ import annotations

import pytest
from fastmob.measures.evaluation.correlation import pearson_correlation, spearman_correlation

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
# pearson_correlation (requires scipy)
# ---------------------------------------------------------------------------


@requires_scipy
def test_pearson_perfect_positive_correlation():
    """Pearson r=1.0 for perfectly correlated arrays."""
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    r, _p = pearson_correlation(v, v)
    assert r == pytest.approx(1.0)


@requires_scipy
def test_pearson_returns_tuple():
    """pearson_correlation returns a 2-tuple of floats."""
    result = pearson_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert isinstance(result, tuple)
    assert len(result) == 2


@requires_scipy
def test_pearson_known_value():
    """Pearson result matches scipy.stats.pearsonr for a known input."""
    from scipy import stats

    true = [1.0, 2.0, 3.0, 4.0]
    pred = [1.1, 1.9, 3.1, 3.9]
    expected_r, expected_p = stats.pearsonr(true, pred)
    r, p = pearson_correlation(true, pred)
    assert r == pytest.approx(float(expected_r), rel=1e-9)
    assert p == pytest.approx(float(expected_p), rel=1e-9)


# ---------------------------------------------------------------------------
# spearman_correlation (requires scipy)
# ---------------------------------------------------------------------------


@requires_scipy
def test_spearman_perfect_monotonic():
    """Spearman rho=1.0 for a perfectly monotonic relationship."""
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, _p = spearman_correlation(v, v)
    assert rho == pytest.approx(1.0)


@requires_scipy
def test_spearman_returns_tuple():
    """spearman_correlation returns a 2-tuple of floats."""
    result = spearman_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert isinstance(result, tuple)
    assert len(result) == 2


@requires_scipy
def test_spearman_known_value():
    """Spearman result matches scipy.stats.spearmanr for a known input."""
    from scipy import stats

    true = [1.0, 2.0, 3.0, 4.0]
    pred = [1.0, 3.0, 2.0, 4.0]
    expected_rho, expected_p = stats.spearmanr(true, pred)
    rho, p = spearman_correlation(true, pred)
    assert rho == pytest.approx(float(expected_rho), rel=1e-9)
    assert p == pytest.approx(float(expected_p), rel=1e-9)
