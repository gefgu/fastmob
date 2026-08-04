"""Information-theoretic metrics for comparing predicted vs. true mobility flows.

``information_gain`` uses only numpy. ``kullback_leibler_divergence`` requires
scipy.stats; if scipy is absent the module still imports cleanly and an error
is raised only when that function is called.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


def information_gain(true, pred) -> float:
    """Return the information gain (KL divergence variant) of true over pred.

    \\[
    \\mathrm{IG} = \\sum_i \\frac{y_i}{N} \\log\\left(\\frac{y_i}{\\hat{y}_i}\\right)
    \\]

    where \\(N = \\sum_i y_i\\).

    Parameters
    ----------
    true:
        Ground truth target values (must be positive).
    pred:
        Estimated target values (must be positive).

    Returns
    -------
    float
        Information gain value; 0.0 when the distributions are identical.

    Examples
    --------
    >>> from fastmob import information_gain
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> print(round(information_gain(observed, predicted), 3))
    0.005
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    n = float(np.sum(t))
    if n == 0.0:
        return 0.0
    return float(np.sum((t / n) * np.log(t / p)))


def kullback_leibler_divergence(true, pred) -> float:
    """Return the Kullback–Leibler divergence \\(D_\\mathrm{KL}(\\mathrm{true}\\,\\|\\,\\mathrm{pred})\\).

    \\[
    D_\\mathrm{KL}(P \\,\\|\\, Q) = \\sum_k p_k \\log\\left(\\frac{p_k}{q_k}\\right)
    \\]

    Computed via scipy.stats.entropy.

    Parameters
    ----------
    true:
        Probability distribution P (reference).
    pred:
        Probability distribution Q (approximation).

    Returns
    -------
    float
        KL divergence; 0.0 when the distributions are identical.

    Raises
    ------
    ImportError
        When scipy is not installed.

    Examples
    --------
    >>> from fastmob import kullback_leibler_divergence
    >>> observed = [10, 20, 30, 40]
    >>> predicted = [12, 18, 33, 37]
    >>> print(round(kullback_leibler_divergence(observed, predicted), 3))
    0.005
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for kullback_leibler_divergence: pip install fastmob[fitting]")
    return float(_scipy_stats.entropy(true, pred))
