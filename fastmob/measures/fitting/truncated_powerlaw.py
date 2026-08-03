"""Truncated power-law fitting utilities."""

from __future__ import annotations

import numpy as np

from fastmob._core import fit_truncated_powerlaw_grid


def log_truncated_powerlaw(x: np.ndarray, c: float, r0: float, beta: float, kappa: float) -> np.ndarray:
    """Evaluate ``log(c * (x + r0)**(-beta) * exp(-x / kappa))``."""
    return np.log(c) - beta * np.log(x + r0) - (x / kappa)


def fit_values_to_truncated_powerlaw(
    values: list[float] | np.ndarray,
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a truncated power law to positive values using a log histogram.

    Fitting uses a deterministic, parallel coarse-to-fine grid search over
    ``(r0, beta, kappa)`` implemented in Rust; the optimal ``c`` is solved in
    closed form in log space for every candidate. The log-spaced histogram
    the search optimizes over is also computed in Rust and returned alongside
    the fitted parameters.
    """
    values_array = np.asarray(values, dtype=np.float64)
    (c, r0, beta, kappa), x_data, y_data = fit_truncated_powerlaw_grid(values_array, bins)
    return np.array([c, r0, beta, kappa]), np.asarray(x_data), np.asarray(y_data)
