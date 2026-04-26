"""Correctness tests for skmob2.measures.fitting.mobility_laws — power-law fitting."""

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
    from skmob2.measures.fitting.mobility_laws import log_truncated_powerlaw

    x = np.array([1.0, 10.0, 100.0])
    c, r0, beta, kappa = 1.5, 1.5, 1.75, 400.0

    log_f = log_truncated_powerlaw(x, c, r0, beta, kappa)
    expected = np.log(c) - beta * np.log(x + r0) - x / kappa

    np.testing.assert_allclose(log_f, expected, rtol=1e-12)


def test_log_truncated_powerlaw_scalar_inputs():
    """Works with scalar x as well."""
    from skmob2.measures.fitting.mobility_laws import log_truncated_powerlaw

    val = log_truncated_powerlaw(np.array([5.0]), 1.0, 1.5, 1.75, 400.0)
    expected = np.log(1.0) - 1.75 * np.log(5.0 + 1.5) - 5.0 / 400.0
    assert abs(val[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests for fit_values_to_truncated_powerlaw
# ---------------------------------------------------------------------------


def test_fit_truncated_powerlaw_recovers_beta():
    """Fit on synthetic Gonzalez sample recovers beta within ±0.05."""
    pytest.importorskip("scipy", reason="scipy not installed")
    from skmob2.measures.fitting.mobility_laws import fit_values_to_truncated_powerlaw

    TRUE_BETA = 1.75
    TRUE_KAPPA = 400.0
    TRUE_R0 = 1.5
    N = 50_000

    samples = _sample_truncated_powerlaw(N, beta=TRUE_BETA, r0=TRUE_R0, kappa=TRUE_KAPPA, seed=42)

    popt, x_data, y_data = fit_values_to_truncated_powerlaw(samples, bins=100)
    _c_fit, _r0_fit, beta_fit, _kappa_fit = popt

    assert abs(beta_fit - TRUE_BETA) < 0.05, f"Recovered beta={beta_fit:.4f}, expected {TRUE_BETA} ± 0.05"


def test_fit_truncated_powerlaw_returns_correct_shapes():
    pytest.importorskip("scipy", reason="scipy not installed")
    from skmob2.measures.fitting.mobility_laws import fit_values_to_truncated_powerlaw

    samples = _sample_truncated_powerlaw(5_000, seed=0)
    popt, x_data, y_data = fit_values_to_truncated_powerlaw(samples, bins=50)

    assert len(popt) == 4, "popt must have 4 parameters: c, r0, beta, kappa"
    assert len(x_data) == len(y_data), "x_data and y_data must have equal length"
    assert len(x_data) > 0, "x_data must be non-empty"
    assert np.all(x_data > 0), "all bin centers must be positive"
    assert np.all(y_data > 0), "all density values must be positive"


def test_fit_truncated_powerlaw_raises_without_scipy(monkeypatch):
    """fit_values_to_truncated_powerlaw raises ImportError when scipy is absent."""
    import skmob2.measures.fitting.mobility_laws as mod

    original = mod._scipy_curve_fit
    monkeypatch.setattr(mod, "_scipy_curve_fit", None)
    with pytest.raises(ImportError, match="scipy"):
        mod.fit_values_to_truncated_powerlaw([1.0, 2.0, 3.0])
    monkeypatch.setattr(mod, "_scipy_curve_fit", original)


def test_fit_truncated_powerlaw_no_print(capsys):
    """fit_values_to_truncated_powerlaw must not print to stdout."""
    pytest.importorskip("scipy", reason="scipy not installed")
    from skmob2.measures.fitting.mobility_laws import fit_values_to_truncated_powerlaw

    samples = _sample_truncated_powerlaw(1_000, seed=1)
    fit_values_to_truncated_powerlaw(samples, bins=20)
    captured = capsys.readouterr()
    assert captured.out == "", "fit_values_to_truncated_powerlaw must not print"
