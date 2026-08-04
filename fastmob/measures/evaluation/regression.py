"""Regression loss functions for comparing predicted vs. true mobility flows.

All functions operate on plain Python sequences or numpy arrays; they do not
accept DataFrame inputs and do not use the Narwhals API.
"""

from __future__ import annotations

import numpy as np


def r_squared(true, pred) -> float:
    """Return the coefficient of determination R².

    \\[
    R^2 = 1 - \\frac{SS_\\mathrm{res}}{SS_\\mathrm{tot}}
    \\]

    where

    \\[
    SS_\\mathrm{tot} = \\sum_i (y_i - \\bar{y})^2
    \\]

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        R² score. Best possible value is 1.0; can be negative.

    Examples
    --------
    >>> from fastmob import r_squared
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> print(round(r_squared(observed, predicted), 3))
    0.948
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def mse(true, pred) -> float:
    """Return the mean squared error between true and predicted values.

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Non-negative MSE; 0.0 is the best possible value.

    Examples
    --------
    >>> from fastmob import mse
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> print(round(mse(observed, predicted), 3))
    6.5
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return float(np.mean((t - p) ** 2))


def rmse(true, pred) -> float:
    """Return the root mean squared error between true and predicted values.

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Non-negative RMSE; 0.0 is the best possible value.

    Examples
    --------
    >>> from fastmob import rmse
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> print(round(rmse(observed, predicted), 3))
    2.55
    """
    return float(np.sqrt(mse(true, pred)))


def nrmse(true, pred) -> float:
    """Return the normalized root mean squared error (RMSE / sum(true)).

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Non-negative NRMSE; 0.0 is the best possible value.

    Examples
    --------
    >>> from fastmob import nrmse
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> print(round(nrmse(observed, predicted), 3))
    0.025
    """
    t = np.asarray(true, dtype=float)
    total = float(np.sum(t))
    if total == 0.0:
        return 0.0
    return rmse(true, pred) / total


def max_error(true, pred) -> float:
    """Return the maximum signed error max(true_i - pred_i).

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Maximum element-wise difference true - pred.

    Examples
    --------
    >>> from fastmob import max_error
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> print(round(max_error(observed, predicted), 3))
    3.0
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return float(np.max(t - p))
