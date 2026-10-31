"""Tests that compare two fitted distance distributions: likelihood ratio (nested) and Vuong (non-nested)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fastmob.measures.fitting.distance_distribution import DistributionFit

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
    """

    test: str
    statistic: float
    p_value: float
    favours: str | None
    note: str


def _same_data(a: DistributionFit, b: DistributionFit) -> None:
    if (
        a.xmin != b.xmin
        or a.xmax != b.xmax
        or len(a.counts) != len(b.counts)
        or a.likelihood != b.likelihood
        or not np.array_equal(a.counts, b.counts)
    ):
        raise ValueError("The fits must be of the same values on the same range, bins and likelihood")


def likelihood_ratio_test(restricted: DistributionFit, general: DistributionFit) -> ModelTest:
    """Test a family against the family that extends it.

    The statistic is ``2 (log L_general - log L_restricted)``. The restricted family is a limit of the general
    one with parameters on the boundary of their range (``r0 -> 0``, or the cutoff ``kappa -> infinity``), so
    the null distribution is not chi-squared: it is a mixture of chi-squared distributions (Chernoff, 1954), which is used here.
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

    Returns
    -------
    ModelTest
        The statistic, p-value and which family is supported.

    Raises
    ------
    ValueError
        If the two families are not a nested pair of the list above, or were not fitted to the same data.

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
    stat = max(2.0 * (general.log_likelihood - restricted.log_likelihood), 0.0)
    sf1 = math.erfc(math.sqrt(stat / 2.0))  # P(chi2_1 > stat)
    if _NESTED[key] == 1:
        p = 1.0 if stat == 0.0 else 0.5 * sf1
    else:
        p = 1.0 if stat == 0.0 else 0.5 * sf1 + 0.25 * math.exp(-stat / 2.0)
    favours = general.family if p < 0.05 else None
    note = (
        f"{general.family} fits significantly better than {restricted.family}"
        if favours
        else f"the data do not need the extra terms of {general.family}; {restricted.family} is enough"
    )
    return ModelTest("likelihood ratio", float(stat), float(p), favours, note)


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
        If the fits are not of the same data, or ``correction`` is invalid.

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
