"""Correctness tests for skmob2.measures.evaluation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from skmob2.measures.evaluation import (
    common_part_of_commuters,
    common_part_of_links,
    information_gain,
    kullback_leibler_divergence,
    max_error,
    mse,
    nrmse,
    pearson_correlation,
    r_squared,
    rmse,
    spearman_correlation,
)

# Tests that call into scipy are skipped when scipy is not installed.
# Use a module-level flag so each test can reference it with a simple decorator.
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
# common_part_of_commuters
# ---------------------------------------------------------------------------


def test_cpc_identical_arrays():
    """CPC of identical arrays is 1.0."""
    v = [10.0, 20.0, 30.0]
    assert common_part_of_commuters(v, v) == pytest.approx(1.0)


def test_cpc_disjoint_arrays():
    """CPC of arrays that share no commuters is 0.0 when one is all-zero."""
    v1 = [10.0, 0.0]
    v2 = [0.0, 10.0]
    # numerator = 2 * sum(min(10,0), min(0,10)) = 0
    assert common_part_of_commuters(v1, v2) == pytest.approx(0.0)


def test_cpc_known_value():
    """CPC matches hand-computed value."""
    v1 = [3.0, 1.0]
    v2 = [1.0, 3.0]
    # min sums = 1 + 1 = 2; total = 4 + 4 = 8; CPC = 2*2/8 = 0.5
    assert common_part_of_commuters(v1, v2) == pytest.approx(0.5)


def test_cpc_zero_inputs_returns_zero():
    """CPC of all-zero arrays returns 0.0 without division error."""
    assert common_part_of_commuters([0.0], [0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# common_part_of_links
# ---------------------------------------------------------------------------


def test_cpl_identical_arrays():
    """CPL of identical non-zero arrays is 1.0."""
    v = [1.0, 2.0, 3.0]
    assert common_part_of_links(v, v) == pytest.approx(1.0)


def test_cpl_no_shared_links():
    """CPL when no link is active in both arrays is 0.0."""
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    # active1 = [1, 0], active2 = [0, 1]; numerator = 0; denom = 2; CPL = 0
    assert common_part_of_links(v1, v2) == pytest.approx(0.0)


def test_cpl_partial_overlap():
    """CPL with partial link overlap matches hand computation."""
    v1 = [1.0, 1.0, 0.0]
    v2 = [1.0, 0.0, 1.0]
    # active1 = [1,1,0], active2 = [1,0,1]; shared = 1; denom = 2+2=4; CPL = 2/4 = 0.5
    assert common_part_of_links(v1, v2) == pytest.approx(0.5)


def test_cpl_zero_inputs_returns_zero():
    """CPL of all-zero arrays returns 0.0 without division error."""
    assert common_part_of_links([0.0], [0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# r_squared
# ---------------------------------------------------------------------------


def test_r_squared_perfect_prediction():
    """R² is 1.0 for a perfect prediction."""
    y = [1.0, 2.0, 3.0]
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_baseline_prediction():
    """R² is 0.0 when pred is always the mean of true."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert r_squared(y_true, y_pred) == pytest.approx(0.0)


def test_r_squared_known_value():
    """R² matches hand-computed value for a simple case."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.5, 2.0, 2.5, 4.0]
    # ss_res = 0.25 + 0 + 0.25 + 0 = 0.5; mean=2.5; ss_tot=1.25+0.25+0.25+2.25=5.0
    # R² = 1 - 0.5/5.0 = 0.9
    assert r_squared(y_true, y_pred) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# mse / rmse / nrmse
# ---------------------------------------------------------------------------


def test_mse_perfect_prediction():
    """MSE is 0.0 for identical arrays."""
    y = [1.0, 2.0, 3.0]
    assert mse(y, y) == pytest.approx(0.0)


def test_mse_known_value():
    """MSE matches hand computation."""
    # errors = [1, 0, -1] → squared = [1, 0, 1] → mean = 2/3
    assert mse([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(2.0 / 3.0)


def test_rmse_is_sqrt_of_mse():
    """RMSE equals sqrt(MSE) for a non-trivial example."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(mse(y_true, y_pred)))


def test_nrmse_normalises_by_sum():
    """NRMSE equals RMSE / sum(true)."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]
    expected = rmse(y_true, y_pred) / sum(y_true)
    assert nrmse(y_true, y_pred) == pytest.approx(expected)


def test_nrmse_zero_sum_returns_zero():
    """NRMSE returns 0.0 when sum(true) is 0 to avoid division by zero."""
    assert nrmse([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# max_error
# ---------------------------------------------------------------------------


def test_max_error_positive():
    """max_error returns largest (true - pred)."""
    assert max_error([3.0, 5.0, 2.0], [1.0, 2.0, 2.0]) == pytest.approx(3.0)


def test_max_error_all_equal():
    """max_error is 0 when true == pred."""
    assert max_error([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_max_error_negative():
    """max_error can be negative when pred always exceeds true."""
    assert max_error([1.0, 2.0], [2.0, 3.0]) == pytest.approx(-1.0)


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
    # N=6; IG = (2/6)*log(2/1) + (4/6)*log(4/8)
    #         = (1/3)*ln(2) + (2/3)*ln(0.5)
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


# ---------------------------------------------------------------------------
# pearson_correlation (requires scipy)
# ---------------------------------------------------------------------------


@requires_scipy
def test_pearson_perfect_positive_correlation():
    """Pearson r=1.0 for perfectly correlated arrays."""
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    r, p = pearson_correlation(v, v)
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
    rho, p = spearman_correlation(v, v)
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
