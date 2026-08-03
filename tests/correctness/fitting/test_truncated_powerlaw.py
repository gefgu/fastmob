"""Correctness tests for fastmob.measures.fitting.truncated_powerlaw."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: rejection sampler for the Gonzalez truncated power-law
# ---------------------------------------------------------------------------


def _sample_truncated_powerlaw(n, beta=1.75, r0=1.5, kappa=400.0, x_max=1000.0, seed=0):
    """Rejection sampler for the Gonzalez truncated power-law.

    Proposal: Pareto with shape (beta-1) shifted by r0, truncated at x_max.
    Accept with probability exp(-x / kappa).
    """
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n:
        # Sample from power-law: x = r0 * (u^(-1/(beta-1)) - 1) for Pareto-like
        u = rng.uniform(size=n * 10)
        x = r0 * (u ** (-1.0 / (beta - 1.0)) - 1.0)
        # Truncate
        x = x[x < x_max]
        # Rejection: accept with exp(-x/kappa)
        accept_prob = np.exp(-x / kappa)
        u2 = rng.uniform(size=len(x))
        x = x[u2 < accept_prob]
        samples.extend(x.tolist())
    return np.array(samples[:n])


# ---------------------------------------------------------------------------
# Tests for log_truncated_powerlaw
# ---------------------------------------------------------------------------


def test_log_truncated_powerlaw_is_log_of_powerlaw():
    """log_truncated_powerlaw(x, c, r0, b, k) == log(c*(x+r0)^-b * exp(-x/k))."""
    from fastmob.measures.fitting import log_truncated_powerlaw

    x = np.array([1.0, 10.0, 100.0])
    c, r0, beta, kappa = 1.5, 1.5, 1.75, 400.0

    log_f = log_truncated_powerlaw(x, c, r0, beta, kappa)
    expected = np.log(c) - beta * np.log(x + r0) - x / kappa

    np.testing.assert_allclose(log_f, expected, rtol=1e-12)


def test_log_truncated_powerlaw_scalar_inputs():
    """Works with scalar x as well."""
    from fastmob.measures.fitting import log_truncated_powerlaw

    val = log_truncated_powerlaw(np.array([5.0]), 1.0, 1.5, 1.75, 400.0)
    expected = np.log(1.0) - 1.75 * np.log(5.0 + 1.5) - 5.0 / 400.0
    assert abs(val[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests for fit_values_to_truncated_powerlaw (Rust parallel grid search)
# ---------------------------------------------------------------------------


def test_fit_truncated_powerlaw_recovers_beta():
    """Fit on a synthetic Gonzalez sample recovers beta within a documented
    tolerance. The grid search is an approximation, not a gradient-based
    nonlinear least squares solver, so the tolerance is looser than a
    curve_fit-based estimator would need."""
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw

    TRUE_BETA = 1.75
    TRUE_KAPPA = 400.0
    TRUE_R0 = 1.5
    N = 50_000

    samples = _sample_truncated_powerlaw(N, beta=TRUE_BETA, r0=TRUE_R0, kappa=TRUE_KAPPA, seed=42)
    popt, _x_data, _y_data = fit_values_to_truncated_powerlaw(samples, bins=100)
    _c_fit, _r0_fit, beta_fit, _kappa_fit = popt

    assert abs(beta_fit - TRUE_BETA) < 0.3, f"Recovered beta={beta_fit:.4f}, expected {TRUE_BETA} ± 0.3"


def test_fit_truncated_powerlaw_returns_correct_shapes():
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw

    samples = _sample_truncated_powerlaw(5_000, seed=0)
    popt, x_data, y_data = fit_values_to_truncated_powerlaw(samples, bins=50)

    assert len(popt) == 4, "popt must have 4 parameters: c, r0, beta, kappa"
    assert len(x_data) == len(y_data), "x_data and y_data must have equal length"
    assert len(x_data) > 0, "x_data must be non-empty"
    assert np.all(x_data > 0), "all bin centers must be positive"
    assert np.all(y_data > 0), "all density values must be positive"


def test_fit_truncated_powerlaw_no_print(capsys):
    """fit_values_to_truncated_powerlaw must not print to stdout."""
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw

    samples = _sample_truncated_powerlaw(1_000, seed=1)
    fit_values_to_truncated_powerlaw(samples, bins=20)
    captured = capsys.readouterr()
    assert captured.out == "", "fit_values_to_truncated_powerlaw must not print"


def test_fit_truncated_powerlaw_is_deterministic():
    """The Rust grid search is parallelized across r0 candidates; repeated
    calls on the same data must still return bit-identical results."""
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw

    samples = _sample_truncated_powerlaw(5_000, seed=7)
    popt_first, x_first, y_first = fit_values_to_truncated_powerlaw(samples, bins=60)
    popt_second, x_second, y_second = fit_values_to_truncated_powerlaw(samples, bins=60)

    np.testing.assert_array_equal(popt_first, popt_second)
    np.testing.assert_array_equal(x_first, x_second)
    np.testing.assert_array_equal(y_first, y_second)


# ---------------------------------------------------------------------------
# Rust grid search vs. scipy curve_fit -- similarity check
#
# fastmob's production fit is a coarse-to-fine grid search implemented in
# Rust (see fastmob-core::measures::fitting::truncated_powerlaw). scipy is no
# longer part of the production fitting path at all; these tests use it only
# as an independent nonlinear least squares reference to check how close the
# grid search's answer is on the same histogram data.
# ---------------------------------------------------------------------------


def test_rust_grid_fit_close_to_independent_scipy_curve_fit():
    pytest.importorskip("scipy", reason="scipy not installed")
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw, log_truncated_powerlaw
    from scipy.optimize import curve_fit

    samples = _sample_truncated_powerlaw(20_000, seed=7)
    popt_rust, x_data, y_data = fit_values_to_truncated_powerlaw(samples, bins=60)

    popt_scipy, _pcov = curve_fit(
        log_truncated_powerlaw,
        x_data,
        np.log(y_data),
        p0=[1.0, 1.0, 1.75, 400.0],
        bounds=([1e-5, 1e-5, 0, 1e-5], [np.inf, np.inf, np.inf, np.inf]),
    )

    _c_rust, _r0_rust, beta_rust, _kappa_rust = popt_rust
    _c_scipy, _r0_scipy, beta_scipy, _kappa_scipy = popt_scipy
    assert abs(beta_rust - beta_scipy) < 0.3, (
        f"Rust beta={beta_rust:.4f} vs scipy beta={beta_scipy:.4f} differ by more than the "
        "documented grid-search-vs-NLS tolerance"
    )


def test_rust_grid_fit_curve_matches_scipy_curve_reasonably_well():
    """Beyond the single beta parameter, check the two fitted curves produce
    similar log-densities across the fitted x range (a broader similarity
    check than comparing individual parameters, which can trade off against
    each other)."""
    pytest.importorskip("scipy", reason="scipy not installed")
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw, log_truncated_powerlaw
    from scipy.optimize import curve_fit

    samples = _sample_truncated_powerlaw(20_000, beta=1.75, r0=1.5, kappa=400.0, seed=11)
    popt_rust, x_data, y_data = fit_values_to_truncated_powerlaw(samples, bins=60)

    popt_scipy, _pcov = curve_fit(
        log_truncated_powerlaw,
        x_data,
        np.log(y_data),
        p0=[1.0, 1.0, 1.75, 400.0],
        bounds=([1e-5, 1e-5, 0, 1e-5], [np.inf, np.inf, np.inf, np.inf]),
    )

    log_y_rust = log_truncated_powerlaw(x_data, *popt_rust)
    log_y_scipy = log_truncated_powerlaw(x_data, *popt_scipy)
    max_abs_diff = float(np.max(np.abs(log_y_rust - log_y_scipy)))
    assert max_abs_diff < 1.0, (
        f"Rust and scipy fitted log-densities differ by up to {max_abs_diff:.4f} "
        "over the histogram range, more than the documented tolerance"
    )
