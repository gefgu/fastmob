"""Correctness tests for fastmob.measures.evaluation.regression."""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastmob.measures.evaluation.regression import max_error, mse, nrmse, r_squared, rmse

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
