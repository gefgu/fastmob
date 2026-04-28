"""Evaluation functions for comparing predicted vs. true mobility flows.

All functions in this module operate on plain Python sequences or numpy arrays.
They do not accept DataFrame inputs and do not use the Narwhals API.

scipy is an optional dependency:
  - ``pearson_correlation`` and ``spearman_correlation`` require scipy.stats.
  - ``kullback_leibler_divergence`` requires scipy.stats.
  - All other functions use only numpy.

If scipy is absent the module still imports cleanly; errors are raised only
when the affected functions are called.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------


def common_part_of_commuters(values1, values2) -> float:
    """Return the common part of commuters (CPC) between two flow arrays.

    CPC = 2 * sum(min(v1_i, v2_i)) / (sum(v1) + sum(v2)).

    Parameters
    ----------
    values1:
        First array of flow values (e.g. observed commuter counts).
    values2:
        Second array of flow values (e.g. predicted commuter counts).

    Returns
    -------
    float
        CPC in the range [0, 1] where 1 indicates perfect agreement.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    v1 = np.asarray(values1, dtype=float)
    v2 = np.asarray(values2, dtype=float)
    denom = np.sum(v1) + np.sum(v2)
    if denom == 0.0:
        return 0.0
    return float(2.0 * np.sum(np.minimum(v1, v2)) / denom)


def common_part_of_links(values1, values2) -> float:
    """Return the common part of links (CPL) between two flow arrays.

    CPL counts links that are active (> 0) in both arrays and normalises by
    the total number of active links across both arrays:
    CPL = 2 * |{i : v1_i > 0 and v2_i > 0}| / (|{i : v1_i > 0}| + |{i : v2_i > 0}|).

    Parameters
    ----------
    values1:
        First array of flow values.
    values2:
        Second array of flow values.

    Returns
    -------
    float
        CPL in the range [0, 1] where 1 means identical link sets.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    v1 = np.asarray(values1, dtype=float)
    v2 = np.asarray(values2, dtype=float)
    active1 = (v1 > 0).astype(float)
    active2 = (v2 > 0).astype(float)
    denom = float(np.sum(active1) + np.sum(active2))
    if denom == 0.0:
        return 0.0
    numerator = float(np.sum(active1 * active2))
    return 2.0 * numerator / denom


def common_part_of_commuters_distance(values1, values2) -> float:
    """Return the common part of commuters by distance (CPCD).

    Bins both distance arrays into histograms with bin width 2 and returns
    the fraction of values1 that overlaps with values2 bin-by-bin:
    CPCD = sum_k(min(hist1_k, hist2_k)) / sum(values1).

    Parameters
    ----------
    values1:
        First array of distance values (e.g. observed commuting distances in km).
    values2:
        Second array of distance values (e.g. predicted commuting distances in km).

    Returns
    -------
    float
        CPCD value; 0.0 when values1 sums to zero or there is no bin overlap.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    v1 = np.asarray(values1, dtype=float)
    v2 = np.asarray(values2, dtype=float)
    n = float(v1.sum())
    if n == 0.0:
        return 0.0
    max_val = max(float(v1.max()), float(v2.max()))
    bins = np.arange(0, max_val, 2)
    hist1, _ = np.histogram(v1, bins)
    hist2, _ = np.histogram(v2, bins)
    return float(np.sum(np.minimum(hist1, hist2))) / n


# ---------------------------------------------------------------------------
# Regression loss functions (pure numpy)
# ---------------------------------------------------------------------------


def r_squared(true, pred) -> float:
    """Return the coefficient of determination R².

    R² = 1 - SS_res / SS_tot where SS_tot = sum((true - mean(true))^2).

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        R² score. Best possible value is 1.0; can be negative.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def mse(true, pred) -> float:
    """Return the mean squared error between true and predicted values.

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Non-negative MSE; 0.0 is the best possible value.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return float(np.mean((t - p) ** 2))


def rmse(true, pred) -> float:
    """Return the root mean squared error between true and predicted values.

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Non-negative RMSE; 0.0 is the best possible value.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    return float(np.sqrt(mse(true, pred)))


def nrmse(true, pred) -> float:
    """Return the normalized root mean squared error (RMSE / sum(true)).

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Non-negative NRMSE; 0.0 is the best possible value.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    t = np.asarray(true, dtype=float)
    total = float(np.sum(t))
    if total == 0.0:
        return 0.0
    return rmse(true, pred) / total


def max_error(true, pred) -> float:
    """Return the maximum signed error max(true_i - pred_i).

    Parameters
    ----------
    true:
        Ground truth target values.
    pred:
        Estimated target values.

    Returns
    -------
    float
        Maximum element-wise difference true - pred.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return float(np.max(t - p))


# ---------------------------------------------------------------------------
# Information-theoretic metrics
# ---------------------------------------------------------------------------


def information_gain(true, pred) -> float:
    """Return the information gain (KL divergence variant) of true over pred.

    IG = sum((true_i / N) * log(true_i / pred_i)) where N = sum(true).

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

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    n = float(np.sum(t))
    if n == 0.0:
        return 0.0
    return float(np.sum((t / n) * np.log(t / p)))


def kullback_leibler_divergence(true, pred) -> float:
    """Return the Kullback–Leibler divergence D_KL(true || pred).

    D_KL = sum(pk * log(pk / qk)) computed via scipy.stats.entropy.

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

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for kullback_leibler_divergence: pip install skmob2[fitting]")
    return float(_scipy_stats.entropy(true, pred))


# ---------------------------------------------------------------------------
# Correlation metrics (require scipy)
# ---------------------------------------------------------------------------


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

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for pearson_correlation: pip install skmob2[fitting]")
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

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for spearman_correlation: pip install skmob2[fitting]")
    result = _scipy_stats.spearmanr(true, pred)
    return (float(result[0]), float(result[1]))
