"""Correctness tests for fkmob.measures.fitting.mobility_laws — power-law fitting."""

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
    from fkmob.measures.fitting.mobility_laws import log_truncated_powerlaw

    x = np.array([1.0, 10.0, 100.0])
    c, r0, beta, kappa = 1.5, 1.5, 1.75, 400.0

    log_f = log_truncated_powerlaw(x, c, r0, beta, kappa)
    expected = np.log(c) - beta * np.log(x + r0) - x / kappa

    np.testing.assert_allclose(log_f, expected, rtol=1e-12)


def test_log_truncated_powerlaw_scalar_inputs():
    """Works with scalar x as well."""
    from fkmob.measures.fitting.mobility_laws import log_truncated_powerlaw

    val = log_truncated_powerlaw(np.array([5.0]), 1.0, 1.5, 1.75, 400.0)
    expected = np.log(1.0) - 1.75 * np.log(5.0 + 1.5) - 5.0 / 400.0
    assert abs(val[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests for fit_values_to_truncated_powerlaw
# ---------------------------------------------------------------------------


def test_fit_truncated_powerlaw_recovers_beta():
    """Fit on synthetic Gonzalez sample recovers beta within ±0.05."""
    pytest.importorskip("scipy", reason="scipy not installed")
    from fkmob.measures.fitting.mobility_laws import fit_values_to_truncated_powerlaw

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
    from fkmob.measures.fitting.mobility_laws import fit_values_to_truncated_powerlaw

    samples = _sample_truncated_powerlaw(5_000, seed=0)
    popt, x_data, y_data = fit_values_to_truncated_powerlaw(samples, bins=50)

    assert len(popt) == 4, "popt must have 4 parameters: c, r0, beta, kappa"
    assert len(x_data) == len(y_data), "x_data and y_data must have equal length"
    assert len(x_data) > 0, "x_data must be non-empty"
    assert np.all(x_data > 0), "all bin centers must be positive"
    assert np.all(y_data > 0), "all density values must be positive"


def test_fit_truncated_powerlaw_raises_without_scipy(monkeypatch):
    """fit_values_to_truncated_powerlaw raises ImportError when scipy is absent."""
    import fkmob.measures.fitting.mobility_laws as mod

    original = mod._scipy_curve_fit
    monkeypatch.setattr(mod, "_scipy_curve_fit", None)
    with pytest.raises(ImportError, match="scipy"):
        mod.fit_values_to_truncated_powerlaw([1.0, 2.0, 3.0])
    monkeypatch.setattr(mod, "_scipy_curve_fit", original)


def test_fit_truncated_powerlaw_no_print(capsys):
    """fit_values_to_truncated_powerlaw must not print to stdout."""
    pytest.importorskip("scipy", reason="scipy not installed")
    from fkmob.measures.fitting.mobility_laws import fit_values_to_truncated_powerlaw

    samples = _sample_truncated_powerlaw(1_000, seed=1)
    fit_values_to_truncated_powerlaw(samples, bins=20)
    captured = capsys.readouterr()
    assert captured.out == "", "fit_values_to_truncated_powerlaw must not print"


# ---------------------------------------------------------------------------
# Tests for universal visitation law utilities
# ---------------------------------------------------------------------------


def test_compute_visitation_law_data_direct_coordinates():
    import pandas as pd
    from fkmob.measures.fitting.mobility_laws import compute_visitation_law_data

    visits = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "location_id": ["home", "work", "work"],
            "timestamp": pd.to_datetime(["2020-01-01 08:00", "2020-01-01 09:00", "2020-01-02 09:00"]),
            "purpose": ["HOME", "WORK", "WORK"],
            "lat": [0.0, 0.0, 0.0],
            "lng": [0.0, 1.0, 1.0],
        }
    )

    result = compute_visitation_law_data(visits)

    assert list(result.columns) == ["user_id", "location_id", "r_km", "f", "rf", "n_visits"]
    home = result[result["location_id"] == "home"].iloc[0]
    work = result[result["location_id"] == "work"].iloc[0]
    assert home["r_km"] == pytest.approx(0.0)
    assert home["f"] == pytest.approx(1.0)
    assert home["n_visits"] == 1
    assert work["r_km"] == pytest.approx(111.195, rel=1e-3)
    assert work["f"] == pytest.approx(2.0)
    assert work["rf"] == pytest.approx(work["r_km"] * 2.0)
    assert work["n_visits"] == 2


def test_compute_visitation_law_data_locations_lookup_and_home_fallback():
    import pandas as pd
    from fkmob.measures.fitting.mobility_laws import compute_visitation_law_data

    visits = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u2"],
            "location_id": ["A", "B", "X", "X", "Y"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 08:00",
                    "2020-01-01 09:00",
                    "2020-01-01 08:00",
                    "2020-01-02 08:00",
                    "2020-01-02 09:00",
                ]
            ),
            "purpose": ["HOME", "WORK", "OTHER", "OTHER", "OTHER"],
        }
    )
    locations = pd.DataFrame(
        {
            "location_id": ["A", "B", "X", "Y"],
            "lat": [0.0, 0.0, 10.0, 10.0],
            "lng": [0.0, 1.0, 0.0, 2.0],
        }
    )

    result = compute_visitation_law_data(visits, locations_df=locations)

    u2_x = result[(result["user_id"] == "u2") & (result["location_id"] == "X")].iloc[0]
    u2_y = result[(result["user_id"] == "u2") & (result["location_id"] == "Y")].iloc[0]
    assert u2_x["r_km"] == pytest.approx(0.0)
    assert u2_x["f"] == pytest.approx(2.0)
    assert u2_x["n_visits"] == 2
    assert u2_y["r_km"] > 200.0
    assert u2_y["f"] == pytest.approx(1.0)


def test_compute_visitation_law_data_polars_parity():
    import pandas as pd

    pl = pytest.importorskip("polars")
    from fkmob.measures.fitting.mobility_laws import compute_visitation_law_data

    visits_pd = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "location_id": ["home", "work", "work", "home", "gym"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 08:00",
                    "2020-01-01 09:00",
                    "2020-01-02 09:00",
                    "2020-01-01 08:00",
                    "2020-01-01 10:00",
                ]
            ),
            "purpose": ["HOME", "WORK", "WORK", "HOME", "OTHER"],
            "lat": [0.0, 0.0, 0.0, 10.0, 10.0],
            "lng": [0.0, 1.0, 1.0, 0.0, 2.0],
        }
    )
    visits_pl = pl.from_pandas(visits_pd)

    result_pd = compute_visitation_law_data(visits_pd)
    result_pl = compute_visitation_law_data(visits_pl)

    assert isinstance(result_pl, pl.DataFrame)
    result_pl_pd = result_pl.to_pandas()
    result_pd = result_pd.sort_values(["user_id", "location_id"]).reset_index(drop=True)
    result_pl_pd = result_pl_pd.sort_values(["user_id", "location_id"]).reset_index(drop=True)
    assert result_pl_pd[["user_id", "location_id", "f", "n_visits"]].equals(
        result_pd[["user_id", "location_id", "f", "n_visits"]]
    )
    np.testing.assert_allclose(result_pl_pd["r_km"], result_pd["r_km"], rtol=1e-12)
    np.testing.assert_allclose(result_pl_pd["rf"], result_pd["rf"], rtol=1e-12)


def test_bin_visitation_law_data_filters_zero_distance_and_computes_density():
    import pandas as pd
    from fkmob.measures.fitting.mobility_laws import bin_visitation_law_data

    vl_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "location_id": ["A", "A", "home"],
            "r_km": [1.2, 1.2, 0.0],
            "f": [2.0, 2.0, 10.0],
            "rf": [2.4, 2.4, 0.0],
            "n_visits": [2, 2, 10],
        }
    )

    rf, rho, label = bin_visitation_law_data(vl_df, n_bins=4, distance_bin_width_km=1.0)

    assert label == r"$\rho_i(r,f)$ (visitors km$^{-2}$)"
    np.testing.assert_allclose(rf, np.array([3.0]))
    np.testing.assert_allclose(rho, np.array([2.0 / (2.0 * np.pi * 1.5)]))


def test_fit_visitation_law_recovers_known_exponent_and_curve():
    from fkmob.measures.fitting.mobility_laws import fit_visitation_law

    rf = np.logspace(0, 3, 50)
    rho = 7.0 * np.power(rf, -1.4)

    eta, mu, r2, rf_fit, rho_fit = fit_visitation_law(rf, rho, return_curve=True, n_curve_points=25)

    assert eta == pytest.approx(1.4)
    assert mu == pytest.approx(7.0)
    assert r2 == pytest.approx(1.0)
    assert len(rf_fit) == 25
    assert len(rho_fit) == 25
    assert np.all(rf_fit > 0)
    assert np.all(rho_fit > 0)


def test_visitation_law_invalid_inputs_raise():
    from fkmob.measures.fitting.mobility_laws import (
        bin_visitation_law_data,
        fit_visitation_law,
        visitation_law_curve,
    )

    with pytest.raises(ValueError, match="same shape"):
        fit_visitation_law(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="At least two"):
        fit_visitation_law(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="min_rf"):
        fit_visitation_law(np.array([1.0, 2.0]), np.array([1.0, 0.5]), min_rf=0)
    with pytest.raises(ValueError, match="positive finite"):
        visitation_law_curve(np.array([0.0, np.nan]), eta=1.0, mu=1.0)
    with pytest.raises(ValueError, match="n_points"):
        visitation_law_curve(np.array([1.0]), eta=1.0, mu=1.0, n_points=1)

    import pandas as pd

    vl_df = pd.DataFrame({"user_id": ["u1"], "location_id": ["A"], "r_km": [1.0], "f": [1.0], "rf": [1.0]})
    with pytest.raises(ValueError, match="distance_bin_width"):
        bin_visitation_law_data(vl_df, distance_bin_width_km=0)


def test_visitation_distance_core_helpers_match():
    pl = pytest.importorskip("polars")
    from fkmob._core import visitation_distances_arrow, visitation_distances_km, visitation_distances_numpy

    home_lats = np.array([0.0, 10.0], dtype=np.float64)
    home_lngs = np.array([0.0, 0.0], dtype=np.float64)
    loc_lats = np.array([0.0, 10.0], dtype=np.float64)
    loc_lngs = np.array([1.0, 2.0], dtype=np.float64)

    expected = visitation_distances_km(home_lats.tolist(), home_lngs.tolist(), loc_lats.tolist(), loc_lngs.tolist())
    result_numpy = visitation_distances_numpy(home_lats, home_lngs, loc_lats, loc_lngs)
    result_arrow = visitation_distances_arrow(
        pl.Series(home_lats).to_arrow(),
        pl.Series(home_lngs).to_arrow(),
        pl.Series(loc_lats).to_arrow(),
        pl.Series(loc_lngs).to_arrow(),
    )

    np.testing.assert_allclose(result_numpy, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result_arrow, expected, rtol=0.0, atol=1e-12)


def test_visitation_distance_core_helper_validation_errors():
    from fkmob._core import visitation_distances_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        visitation_distances_numpy(arr, arr, arr[:1], arr)
