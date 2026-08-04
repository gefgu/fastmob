"""Correlation metrics for comparing predicted vs. true mobility flows.

Both functions require scipy.stats; if scipy is absent the module still
imports cleanly and errors are raised only when the functions are called.
"""

from __future__ import annotations

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


def pearson_correlation(true, pred) -> tuple[float, float]:
    """Return the Pearson correlation coefficient and its two-tailed p-value.

    Parameters
    ----------
    true:
        Ground truth values.
    pred:
        Predicted values.

    Returns
    -------
    tuple[float, float]
        (Pearson r, two-tailed p-value).

    Raises
    ------
    ImportError
        When scipy is not installed.

    Examples
    --------
    >>> from fastmob import pearson_correlation
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> r, p_value = pearson_correlation(observed, predicted)
    >>> print(round(r, 3), round(p_value, 3))
    0.975 0.025
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for pearson_correlation: pip install fastmob[fitting]")
    result = _scipy_stats.pearsonr(true, pred)
    return (float(result[0]), float(result[1]))


def spearman_correlation(true, pred) -> tuple[float, float]:
    """Return the Spearman rank-order correlation coefficient and its p-value.

    Parameters
    ----------
    true:
        Ground truth values.
    pred:
        Predicted values.

    Returns
    -------
    tuple[float, float]
        (Spearman rho, two-tailed p-value).

    Raises
    ------
    ImportError
        When scipy is not installed.

    Examples
    --------
    >>> from fastmob import spearman_correlation
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> rho, p_value = spearman_correlation(observed, predicted)
    >>> print(round(rho, 3), round(p_value, 3))
    1.0 0.0
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for spearman_correlation: pip install fastmob[fitting]")
    result = _scipy_stats.spearmanr(true, pred)
    return (float(result[0]), float(result[1]))
