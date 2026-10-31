"""Tests that compare two fitted distance distributions: likelihood ratio (nested) and Vuong (non-nested)."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from fastmob.measures.fitting.distance_distribution import (
    DistributionFit,
    _fit_counts,
    _Grid,
    _starts,
    _to_unconstrained,
)

# restricted -> general: which parameters of the general family sit on the boundary when it reduces to the restricted one
_NESTED = {
    ("powerlaw", "shifted_powerlaw"): 1,  # r0 -> 0
    ("shifted_powerlaw", "truncated_powerlaw"): 1,  # kappa -> infinity
    ("powerlaw", "truncated_powerlaw"): 2,  # r0 -> 0 and kappa -> infinity
}


@dataclass(frozen=True)
class ModelTest:
    """Result of `likelihood_ratio_test` or `vuong_test`.

    Parameters
    ----------
    test : str
        ``"likelihood ratio"`` or ``"vuong"``.
    statistic : float
        The likelihood-ratio statistic ``2 (log L_general - log L_restricted)`` or the Vuong z statistic
        (positive when the first family fits better).
    p_value : float
        The p-value.
    favours : str or None
        The family the test supports at the 5% level, or None when it cannot choose.
    note : str
        How to read the result.
    asymptotic_p_value : float or None
        For `likelihood_ratio_test` with ``n_bootstrap`` above 0, the p-value of the boundary chi-squared mixture;
        ``p_value`` is then the parametric-bootstrap one. None otherwise.
    """

    test: str
    statistic: float
    p_value: float
    favours: str | None
    note: str
    asymptotic_p_value: float | None = None


def _same_data(a: DistributionFit, b: DistributionFit) -> None:
    if (
        a.xmin != b.xmin
        or a.xmax != b.xmax
        or len(a.counts) != len(b.counts)
        or a.likelihood != b.likelihood
        or not np.array_equal(a.counts, b.counts)
    ):
        raise ValueError("The fits must be of the same values on the same range, bins and likelihood")


def likelihood_ratio_test(
    restricted: DistributionFit, general: DistributionFit, n_bootstrap: int = 0, random_state: int | None = 0
) -> ModelTest:
    """Test a family against the family that extends it.

    The statistic is ``2 (log L_general - log L_restricted)``. The restricted family is a limit of the general
    one with parameters on the boundary of their range (``r0 -> 0``, or the cutoff ``kappa -> infinity``), so
    the null distribution is not chi-squared: it is a mixture of chi-squared distributions (Chernoff, 1954), which is used here
    (for the pair with two boundary parameters the mixture assumes that they are orthogonal, Self and Liang, 1987).
    That asymptotic p-value needs a range that does not clip the data: when ``xmax`` is the sample maximum the
    cutoff can hide behind the end of the data and the test has little power, and its size is wrong. With
    ``n_bootstrap`` above 0 the p-value comes instead from a parametric bootstrap: data are simulated from the
    restricted fit, both families are refitted, and the p-value is the share of statistics at least as large as
    the observed one. It is slower but does not rely on the asymptotic theory.
    The pairs that can be tested are ``powerlaw`` against ``shifted_powerlaw``, ``shifted_powerlaw`` against
    ``truncated_powerlaw`` (does the data support a cutoff at all?) and ``powerlaw`` against
    ``truncated_powerlaw``. The test treats the values as independent, so it overstates the evidence for
    clustered values.

    Parameters
    ----------
    restricted : DistributionFit
        The simpler family.
    general : DistributionFit
        The family that extends it. Both must be fitted to the same values with the same ``xmin``, ``xmax``,
        ``bins`` and ``likelihood``.
    n_bootstrap : int, optional
        Parametric-bootstrap replicates for the p-value; 0 (default) uses the asymptotic mixture. Needs
        ``likelihood="grouped"``.
    random_state : int, optional
        Seed of the bootstrap. Default 0.

    Returns
    -------
    ModelTest
        The statistic, p-value and which family is supported.

    Raises
    ------
    ValueError
        If the two families are not a nested pair of the list above, were not fitted to the same data, or
        ``n_bootstrap`` is positive for an exact likelihood.

    Warns
    -----
    UserWarning
        If the general family has a lower likelihood than the restricted one, which can only be a search
        limit (for example the cap on ``kappa``) or a failed optimisation; the statistic is then 0.

    See Also
    --------
    vuong_test : For families that are not nested.
    compare_distance_distributions : Rank families by AIC.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.measures.fitting import fit_distance_distribution, likelihood_ratio_test
    >>> x = np.random.default_rng(3).pareto(1.5, size=4000) + 1.0
    >>> a = fit_distance_distribution(x, "powerlaw", n_bootstrap=0)
    >>> b = fit_distance_distribution(x, "shifted_powerlaw", n_bootstrap=0)
    >>> likelihood_ratio_test(a, b).test
    'likelihood ratio'
    """
    key = (restricted.family, general.family)
    if key not in _NESTED:
        raise ValueError(f"{key[0]!r} is not nested in {key[1]!r}; nested pairs are {sorted(_NESTED)}")
    _same_data(restricted, general)
    if n_bootstrap > 0 and general.values is not None:
        raise ValueError('n_bootstrap needs likelihood="grouped"')
    raw = 2.0 * (general.log_likelihood - restricted.log_likelihood)
    if raw < -2.0:
        warnings.warn(
            f"The {general.family} fit has a lower likelihood than the {restricted.family} fit (by {-raw / 2:.2f}), "
            "which its search limits or the optimiser cause; the statistic is set to 0",
            UserWarning,
            stacklevel=2,
        )
    stat = max(raw, 0.0)
    sf1 = math.erfc(math.sqrt(stat / 2.0))  # P(chi2_1 > stat)
    if _NESTED[key] == 1:
        p = 1.0 if stat == 0.0 else 0.5 * sf1
    else:
        p = 1.0 if stat == 0.0 else 0.5 * sf1 + 0.25 * math.exp(-stat / 2.0)
    asymptotic = None
    if n_bootstrap > 0:
        asymptotic = p
        rng = np.random.default_rng(random_state)
        grid = _Grid(restricted.xmin, restricted.xmax, len(restricted.counts))
        r_specs, g_specs = restricted._specs(), general._specs()
        r_params = np.array(list(restricted.parameters.values()))
        mass = grid.masses(restricted.family, r_params)
        n = int(restricted.counts.sum())
        r_start = _to_unconstrained(r_params, r_specs)
        g_start = _to_unconstrained(np.array(list(general.parameters.values())), g_specs)
        exceed = 0
        for _ in range(n_bootstrap):
            simulated = rng.multinomial(n, mass).astype(float)
            _, r_value = _fit_counts(restricted.family, r_specs, grid, simulated, [r_start], step=0.1, max_iter=200)
            starts = [g_start, *_starts(g_specs, rng, 2)]
            _, g_value = _fit_counts(general.family, g_specs, grid, simulated, starts, step=0.1, max_iter=200)
            exceed += max(2.0 * (r_value - g_value), 0.0) >= stat - 1e-9
        p = (exceed + 1) / (n_bootstrap + 1)
    favours = general.family if p < 0.05 else None
    note = (
        f"{general.family} fits significantly better than {restricted.family}"
        if favours
        else f"the data do not need the extra terms of {general.family}; {restricted.family} is enough"
    )
    return ModelTest("likelihood ratio", float(stat), float(p), favours, note, asymptotic)


def vuong_test(first: DistributionFit, second: DistributionFit, correction: str | None = "aic") -> ModelTest:
    """Vuong test of two families that are not nested.

    The test asks whether the two fitted families have the same expected log-likelihood per value. Under
    that null the standardised difference of the log-likelihoods is approximately standard normal (Vuong,
    1989). A significant positive statistic favours ``first``, a negative one ``second``; a small statistic
    means that the data cannot choose, however large the AIC difference looks. The values are treated as
    independent, so the evidence is overstated for clustered values.

    Parameters
    ----------
    first : DistributionFit
        One family.
    second : DistributionFit
        The other. Both must be fitted to the same values with the same ``xmin``, ``xmax``, ``bins`` and
        ``likelihood``.
    correction : {"aic", "bic", None}, optional
        Penalty for the number of parameters subtracted from the log-likelihood ratio (``k1 - k2`` for AIC,
        ``(k1 - k2) log(n) / 2`` for BIC). Default ``"aic"``.

    Returns
    -------
    ModelTest
        The statistic, two-sided p-value and which family is supported at the 5% level.

    Raises
    ------
    ValueError
        If the fits are not of the same data, the families are nested (use `likelihood_ratio_test`), or
        ``correction`` is invalid.

    See Also
    --------
    likelihood_ratio_test : For nested families.
    compare_distance_distributions : Rank families by AIC.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.measures.fitting import fit_distance_distribution, vuong_test
    >>> x = np.random.default_rng(1).lognormal(1.0, 0.8, size=4000)
    >>> a = fit_distance_distribution(x, "lognormal", n_bootstrap=0)
    >>> b = fit_distance_distribution(x, "exponential", n_bootstrap=0)
    >>> vuong_test(a, b).favours
    'lognormal'
    """
    if correction not in ("aic", "bic", None):
        raise ValueError('correction must be "aic", "bic" or None')
    if (first.family, second.family) in _NESTED or (second.family, first.family) in _NESTED:
        raise ValueError("The Vuong test is not valid for nested families; use likelihood_ratio_test")
    _same_data(first, second)
    t1, w = first._loglik_terms()
    t2, _ = second._loglik_terms()
    n = float(w.sum())
    d = t1 - t2
    total = float((w * d).sum())
    mean = total / n
    var = float((w * (d - mean) ** 2).sum() / n)
    penalty = 0.0
    if correction == "aic":
        penalty = first.n_parameters - second.n_parameters
    elif correction == "bic":
        penalty = (first.n_parameters - second.n_parameters) * math.log(n) / 2.0
    z = 0.0 if var <= 0 else (total - penalty) / (math.sqrt(n) * math.sqrt(var))
    p = math.erfc(abs(z) / math.sqrt(2.0))
    favours = (first.family if z > 0 else second.family) if p < 0.05 else None
    note = f"{favours} is supported" if favours else "the data cannot choose between the two families"
    return ModelTest("vuong", float(z), float(p), favours, note)
