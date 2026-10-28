"""Correctness tests for the maximum-likelihood distance-distribution fits."""

from __future__ import annotations

import numpy as np
import pytest
from fastmob.measures.fitting import (
    DistributionFit,
    compare_distance_distributions,
    fit_distance_distribution,
)


def _truncated_powerlaw(n, r0=1.5, beta=1.75, kappa=400.0, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.geomspace(1e-2, 3000, 20000)
    pdf = (grid + r0) ** -beta * np.exp(-grid / kappa)
    cdf = np.cumsum(pdf * np.gradient(grid))
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)


def test_truncated_powerlaw_recovers_parameters_inside_the_confidence_interval():
    fit = fit_distance_distribution(_truncated_powerlaw(50_000), "truncated_powerlaw", n_bootstrap=40)

    assert isinstance(fit, DistributionFit)
    assert fit.at_bound == ()
    for name, truth in {"r0": 1.5, "beta": 1.75, "kappa": 400.0}.items():
        assert fit.ci_low[name] <= truth <= fit.ci_high[name], name
        assert fit.std_errors[name] > 0


def test_lognormal_recovers_parameters():
    values = np.random.default_rng(1).lognormal(1.0, 0.8, 5000)
    fit = fit_distance_distribution(values, "lognormal", n_bootstrap=20)

    assert fit.parameters["mu"] == pytest.approx(1.0, abs=0.05)
    assert fit.parameters["sigma"] == pytest.approx(0.8, abs=0.05)


def test_pure_powerlaw_data_put_the_cutoff_at_its_bound():
    values = np.random.default_rng(2).pareto(1.2, 50_000) + 1.0
    fit = fit_distance_distribution(values, "truncated_powerlaw", n_bootstrap=0)

    assert "kappa" in fit.at_bound
    assert fit.parameters["beta"] == pytest.approx(2.2, abs=0.15)


def test_comparison_prefers_the_generating_family_and_weights_sum_to_one():
    values = _truncated_powerlaw(30_000, seed=3)
    table = compare_distance_distributions(values)

    assert table.column("family").to_pylist()[0] == "truncated_powerlaw"
    assert sum(table.column("akaike_weight").to_pylist()) == pytest.approx(1.0)
    assert table.column("delta_aic").to_pylist()[0] == 0.0
    aic = table.column("aic").to_pylist()
    assert aic == sorted(aic)


def test_goodness_of_fit_accepts_the_true_model_and_rejects_a_wrong_one():
    values = np.random.default_rng(4).lognormal(1.0, 0.8, 3000)

    right = fit_distance_distribution(values, "lognormal", n_bootstrap=0, n_gof=30)
    wrong = fit_distance_distribution(values, "exponential", n_bootstrap=0, n_gof=30)

    assert right.gof_p_value > 0.05
    assert wrong.gof_p_value < 0.05


def test_range_excluded_values_and_cdf_behaviour():
    values = np.concatenate([np.zeros(100), np.random.default_rng(5).lognormal(1.0, 0.8, 2000)])
    fit = fit_distance_distribution(values, "lognormal", xmin=0.5, xmax=50.0, n_bootstrap=0)

    assert fit.n + fit.n_excluded == len(values)
    assert fit.n_excluded >= 100
    cdf = fit.cdf([0.5, 5.0, 50.0])
    assert cdf[0] == pytest.approx(0.0)
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) > 0)
    assert fit.pdf([0.1, 100.0]).tolist() == [0.0, 0.0]
    assert fit.summary().column("parameter").to_pylist() == ["mu", "sigma"]


def test_invalid_input_is_rejected():
    values = np.random.default_rng(6).lognormal(1.0, 0.8, 100)
    with pytest.raises(ValueError, match="Unknown family"):
        fit_distance_distribution(values, "gamma")
    with pytest.raises(ValueError, match="At least 20"):
        fit_distance_distribution(values[:5])
    with pytest.raises(ValueError, match="xmin"):
        fit_distance_distribution(values, xmin=5.0, xmax=1.0)
    with pytest.raises(ValueError, match="confidence"):
        fit_distance_distribution(values, confidence=1.5)


def test_fit_is_reproducible_with_a_seed():
    values = np.random.default_rng(7).lognormal(1.0, 0.8, 2000)
    a = fit_distance_distribution(values, "lognormal", n_bootstrap=10, random_state=3)
    b = fit_distance_distribution(values, "lognormal", n_bootstrap=10, random_state=3)

    assert a.parameters == b.parameters
    assert a.ci_low == b.ci_low
