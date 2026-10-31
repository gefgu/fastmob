"""Choose the lower end of the fitted range by the Kolmogorov-Smirnov distance (Clauset et al., 2009)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from fastmob.measures.fitting.distance_distribution import fit_distance_distribution


def select_xmin(
    values: Any,
    family: str = "truncated_powerlaw",
    *,
    candidates: Any = None,
    n_candidates: int = 30,
    min_values: int = 100,
    **kwargs: Any,
) -> pa.Table:
    """Pick ``xmin`` as the lower end of the range that makes the fit closest to the data.

    For each candidate ``xmin`` the family is fitted on ``[xmin, xmax]`` and the Kolmogorov-Smirnov distance
    between the fitted and the empirical distribution is computed; the candidate with the smallest distance
    is selected (Clauset, Shalizi and Newman, 2009). Fit ``fit_distance_distribution`` again with the selected
    ``xmin`` to get the estimates and intervals. The selection uses the data, so the intervals of that
    final fit do not include the uncertainty about ``xmin``: report the selected value and how the estimate moves over
    the candidates (the table). The KS distance is computed on the bin edges of each fit, and a lower
    ``xmin`` leaves more values, so a family that does not describe the small values gets a high ``xmin``.

    Parameters
    ----------
    values : array-like
        Positive values, as for `fit_distance_distribution`.
    family : str, optional
        Family to fit. Default ``"truncated_powerlaw"``.
    candidates : array-like, optional
        The ``xmin`` values to try. Default: ``n_candidates`` points spaced geometrically from the smallest
        positive value to the value above which ``min_values`` observations remain.
    n_candidates : int, optional
        Number of default candidates. Default 30.
    min_values : int, optional
        Fewest values a candidate range may leave (at least 20). Default 100.
    **kwargs
        Passed to `fit_distance_distribution` (``xmax``, ``bins``, ``max_cutoff``, ``bounds``, ``likelihood``,
        ``random_state``). ``n_bootstrap`` is 0.

    Returns
    -------
    pyarrow.Table
        One row per candidate with ``xmin``, ``n`` (values in range), ``ks_statistic``, ``log_likelihood``,
        one column per parameter and ``selected`` (True on the row with the smallest KS distance).

    Raises
    ------
    ValueError
        If ``min_values`` is below 20 or too few values are positive, or no candidate leaves ``min_values`` values.

    See Also
    --------
    fit_distance_distribution : Fit with the selected ``xmin``.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.measures.fitting import select_xmin
    >>> x = np.random.default_rng(2).pareto(1.5, size=3000) + 1.0
    >>> table = select_xmin(x, "powerlaw", n_candidates=8)
    >>> bool(table.column("selected").to_pylist().count(True) == 1)
    True
    """
    if min_values < 20:
        raise ValueError("min_values must be at least 20")
    kwargs.pop("n_bootstrap", None)
    raw = np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values, dtype=float).ravel()
    positive = np.sort(raw[np.isfinite(raw) & (raw > 0)])
    if positive.size < max(min_values, 20):
        raise ValueError("Too few positive values to select xmin")
    if candidates is None:
        upper = positive[-min_values]
        if upper <= positive[0]:
            raise ValueError("No candidate xmin leaves min_values values")
        grid = np.geomspace(positive[0], upper, n_candidates)
    else:
        grid = np.sort(np.asarray(candidates, dtype=float))
        grid = grid[grid > 0]
    fits = []
    for lo in grid:
        if (positive >= lo).sum() < min_values:
            continue
        fits.append(fit_distance_distribution(positive, family, xmin=float(lo), n_bootstrap=0, **kwargs))
    if not fits:
        raise ValueError("No candidate xmin leaves min_values values")
    ks = np.array([f.ks_statistic for f in fits])
    best = int(np.argmin(ks))
    table = {
        "xmin": [f.xmin for f in fits],
        "n": [f.n for f in fits],
        "ks_statistic": ks.tolist(),
        "log_likelihood": [f.log_likelihood for f in fits],
    }
    for name in fits[0].parameters:
        table[name] = [f.parameters[name] for f in fits]
    table["selected"] = [i == best for i in range(len(fits))]
    return pa.table(table)
