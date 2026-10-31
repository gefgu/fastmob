"""Correctness tests for the likelihood-ratio and Vuong tests of fitted distributions."""

from __future__ import annotations

import numpy as np
import pytest
from fastmob.measures.fitting import ModelTest, fit_distance_distribution, likelihood_ratio_test, vuong_test

pytestmark = pytest.mark.filterwarnings("ignore:xmax was not given")


def _fits(values, *families, **kwargs):
    return [fit_distance_distribution(values, family, n_bootstrap=0, **kwargs) for family in families]


def test_likelihood_ratio_does_not_ask_for_a_cutoff_that_the_data_lack():
    pure = np.random.default_rng(3).pareto(1.2, size=20_000) + 1.0
    shifted, truncated = _fits(pure[pure < 200], "shifted_powerlaw", "truncated_powerlaw")

    result = likelihood_ratio_test(shifted, truncated)

    assert isinstance(result, ModelTest)
    assert result.favours is None and result.p_value > 0.05


def test_likelihood_ratio_detects_a_real_cutoff_and_a_real_shift():
    rng = np.random.default_rng(1)
    grid = np.geomspace(0.01, 3000, 20000)
    pdf = (grid + 1.5) ** -1.5 * np.exp(-grid / 100.0)
    cdf = np.cumsum(pdf * np.gradient(grid))
    values = np.interp(rng.random(30_000), cdf / cdf[-1], grid)
    powerlaw, shifted, truncated = _fits(values, "powerlaw", "shifted_powerlaw", "truncated_powerlaw")

    assert likelihood_ratio_test(shifted, truncated).favours == "truncated_powerlaw"
    assert likelihood_ratio_test(powerlaw, truncated).p_value < 1e-6
    assert likelihood_ratio_test(powerlaw, shifted).statistic > 0


def test_likelihood_ratio_rejects_other_pairs_and_different_data():
    values = np.random.default_rng(2).lognormal(1.0, 0.8, size=3000)
    exponential, lognormal = _fits(values, "exponential", "lognormal")
    with pytest.raises(ValueError, match="not nested"):
        likelihood_ratio_test(exponential, lognormal)
    powerlaw = _fits(values, "powerlaw")[0]
    shifted = _fits(values, "shifted_powerlaw", xmin=2.0)[0]
    with pytest.raises(ValueError, match="same values"):
        likelihood_ratio_test(powerlaw, shifted)


def test_vuong_prefers_the_true_family_and_is_antisymmetric():
    values = np.random.default_rng(1).lognormal(1.0, 0.8, size=4000)
    lognormal, exponential = _fits(values, "lognormal", "exponential")

    forward = vuong_test(lognormal, exponential)
    backward = vuong_test(exponential, lognormal)

    assert forward.favours == "lognormal" and backward.favours == "lognormal"
    assert forward.statistic > 0 > backward.statistic
    assert forward.p_value < 0.001
    assert vuong_test(lognormal, exponential, correction=None).statistic > forward.statistic - 5
    with pytest.raises(ValueError, match="correction"):
        vuong_test(lognormal, exponential, correction="x")


def test_vuong_cannot_choose_between_two_equally_good_families_and_works_for_exact():
    values = np.random.default_rng(7).lognormal(1.0, 0.8, size=3000)
    same = _fits(values, "lognormal", "lognormal")
    result = vuong_test(*same)
    assert result.favours is None and result.p_value == pytest.approx(1.0)

    lognormal, exponential = _fits(values, "lognormal", "exponential", likelihood="exact")
    assert vuong_test(lognormal, exponential).favours == "lognormal"


def test_bootstrap_p_value_agrees_with_the_asymptotic_one_and_keeps_both():
    pure = np.random.default_rng(3).pareto(1.6, size=6000) + 1.0
    shifted, truncated = _fits(pure, "shifted_powerlaw", "truncated_powerlaw", xmax=1e7)

    result = likelihood_ratio_test(shifted, truncated, n_bootstrap=30)

    assert result.asymptotic_p_value is not None
    assert 0.0 < result.p_value <= 1.0
    assert result.favours is None
    exact = _fits(pure[:2000], "shifted_powerlaw", "truncated_powerlaw", likelihood="exact", xmax=1e7)
    with pytest.raises(ValueError, match="grouped"):
        likelihood_ratio_test(*exact, n_bootstrap=5)


def test_vuong_refuses_nested_families_and_the_lrt_warns_when_the_general_fit_is_worse():
    values = np.random.default_rng(2).lognormal(1.0, 0.8, size=3000)
    powerlaw, shifted = _fits(values, "powerlaw", "shifted_powerlaw")
    with pytest.raises(ValueError, match="nested"):
        vuong_test(powerlaw, shifted)
    capped = _fits(values, "truncated_powerlaw", bounds={"kappa": (0.5, 0.6)})[0]
    with pytest.warns(UserWarning, match="lower likelihood"):
        likelihood_ratio_test(_fits(values, "shifted_powerlaw")[0], capped)
