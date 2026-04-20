"""Mobility law measures: power-law fitting and related utilities."""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
except ImportError:
    _scipy_curve_fit = None


def log_truncated_powerlaw(
    x: np.ndarray,
    c: float,
    r0: float,
    beta: float,
    kappa: float,
) -> np.ndarray:
    """Log of a truncated power-law (Gonzalez et al. 2008).

    f(x) = c * (x + r0)^{-beta} * exp(-x / kappa)
    log f(x) = log(c) - beta * log(x + r0) - x / kappa

    Parameters
    ----------
    x:
        Input values (must be positive for meaningful results).
    c:
        Scale parameter.
    r0:
        Offset parameter (shifts the origin to avoid singularity at x=0).
    beta:
        Power-law exponent.
    kappa:
        Exponential cutoff scale.

    Returns
    -------
    np.ndarray
        log f(x) evaluated at each element of x.
    """
    return np.log(c) - beta * np.log(x + r0) - (x / kappa)


def fit_values_to_truncated_powerlaw(
    values: "list[float] | np.ndarray",
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a truncated power-law to a 1-D array of positive values.

    Builds a log-spaced histogram of ``values``, then fits
    :func:`log_truncated_powerlaw` to the log-density using nonlinear
    least-squares (scipy.optimize.curve_fit).

    Parameters
    ----------
    values:
        1-D array of positive float values (e.g. jump lengths in km).
    bins:
        Number of log-spaced histogram bins. Default 100.

    Returns
    -------
    popt : np.ndarray
        Fitted parameters ``[c, r0, beta, kappa]``.
    x_data : np.ndarray
        Geometric bin centers used in the fit (all positive).
    y_data : np.ndarray
        Density values at bin centers (all positive).

    Raises
    ------
    ImportError
        When scipy is not installed.
    """
    if _scipy_curve_fit is None:
        raise ImportError(
            "scipy is required for power-law fitting: pip install skmob2[fitting]"
        )

    values_array = np.asarray(values, dtype=float)
    values_array = values_array[values_array > 0]

    bins_edges = np.logspace(
        np.log10(values_array.min()), np.log10(values_array.max()), num=bins
    )
    hist, bin_edges = np.histogram(values_array, bins=bins_edges, density=True)

    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    valid = hist > 0
    x_data = bin_centers[valid]
    y_data = hist[valid]

    log_y_data = np.log(y_data)

    initial_guess = [1.0, 1.0, 1.75, 400.0]
    bounds = ([1e-5, 1e-5, 0, 1e-5], [np.inf, np.inf, np.inf, np.inf])

    popt, _pcov = _scipy_curve_fit(
        log_truncated_powerlaw, x_data, log_y_data,
        p0=initial_guess, bounds=bounds,
    )

    return popt, x_data, y_data
