"""Maximum-likelihood fits of positive distance-like values, with uncertainty and model comparison.

The values (jump lengths, trip lengths, radii of gyration, ...) are grouped into geometric bins between
``xmin`` and ``xmax``. Each family is fitted by maximising the grouped-data (multinomial) log-likelihood

.. math:: \\ell(\\theta) = \\sum_j n_j \\log P_j(\\theta), \\qquad P_j(\\theta) = \\frac{\\int_{\\text{bin } j} f(x;\\theta)\\,dx}{\\int_{x_{min}}^{x_{max}} f(x;\\theta)\\,dx},

so every family is normalised on the same range and their likelihoods, AIC and BIC are comparable. Uncertainty
comes from a bootstrap of the bin counts, and the goodness of fit from a parametric bootstrap of the
Kolmogorov-Smirnov distance. Only NumPy is needed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pyarrow as pa

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)
_BOUND_MARGIN = 1e-3


@dataclass(frozen=True)
class _Param:
    name: str
    low: float
    high: float
    log: bool = False


def _default_specs(family: str, xmin: float, xmax: float, max_cutoff: float) -> list[_Param]:
    if family == "powerlaw":
        return [_Param("beta", 0.01, 10.0)]
    if family == "exponential":
        return [_Param("kappa", xmin * 0.01, max_cutoff, log=True)]
    if family == "lognormal":
        return [_Param("mu", math.log(xmin) - 5.0, math.log(xmax) + 5.0), _Param("sigma", 0.02, 10.0, log=True)]
    if family == "shifted_powerlaw":
        return [_Param("r0", xmin * 0.01, xmax, log=True), _Param("beta", 0.01, 6.0)]
    if family == "truncated_powerlaw":
        return [
            _Param("r0", xmin * 0.01, xmax, log=True),
            _Param("beta", 0.01, 6.0),
            _Param("kappa", xmin, max_cutoff, log=True),
        ]
    raise ValueError(f"Unknown family {family!r}; choose from {sorted(FAMILIES)}")


def _specs(
    family: str,
    xmin: float,
    xmax: float,
    max_cutoff: float,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[_Param]:
    specs = _default_specs(family, xmin, xmax, max_cutoff)
    if not bounds:
        return specs
    names = [sp.name for sp in specs]
    unknown = sorted(set(bounds) - set(names))
    if unknown:
        raise ValueError(f"bounds names {unknown} are not parameters of {family!r}; its parameters are {names}")
    out = []
    for sp in specs:
        if sp.name in bounds:
            low, high = (float(b) for b in bounds[sp.name])
            if not low < high or (sp.log and low <= 0):
                raise ValueError(f"bounds for {sp.name!r} must satisfy low < high" + (" and low > 0" if sp.log else ""))
            sp = _Param(sp.name, low, high, sp.log)
        out.append(sp)
    return out


def _log_density(family: str, x: np.ndarray, p: Sequence[float]) -> np.ndarray:
    if family == "powerlaw":
        return -p[0] * np.log(x)
    if family == "exponential":
        return -x / p[0]
    if family == "lognormal":
        z = np.log(x)
        return -z - (z - p[0]) ** 2 / (2.0 * p[1] ** 2)
    if family == "shifted_powerlaw":
        return -p[1] * np.log(x + p[0])
    return -p[1] * np.log(x + p[0]) - x / p[2]  # truncated_powerlaw: r0, beta, kappa


FAMILIES = ("powerlaw", "shifted_powerlaw", "exponential", "lognormal", "truncated_powerlaw")


def _to_params(u: np.ndarray, specs: list[_Param]) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-np.clip(u, -30.0, 30.0)))
    out = np.empty(len(specs))
    for i, sp in enumerate(specs):
        if sp.log:
            out[i] = math.exp(math.log(sp.low) + (math.log(sp.high) - math.log(sp.low)) * s[i])
        else:
            out[i] = sp.low + (sp.high - sp.low) * s[i]
    return out


def _to_unconstrained(p: Sequence[float], specs: list[_Param]) -> np.ndarray:
    u = np.empty(len(specs))
    for i, sp in enumerate(specs):
        if sp.log:
            s = (math.log(p[i]) - math.log(sp.low)) / (math.log(sp.high) - math.log(sp.low))
        else:
            s = (p[i] - sp.low) / (sp.high - sp.low)
        s = min(max(s, 1e-9), 1 - 1e-9)
        u[i] = math.log(s / (1.0 - s))
    return u


def _position(p: float, sp: _Param) -> float:
    if sp.log:
        return (math.log(p) - math.log(sp.low)) / (math.log(sp.high) - math.log(sp.low))
    return (p - sp.low) / (sp.high - sp.low)


class _Grid:
    """Geometric bins and the quadrature nodes inside them."""

    def __init__(self, xmin: float, xmax: float, bins: int) -> None:
        self.edges = np.geomspace(xmin, xmax, bins + 1)
        half = (self.edges[1:] - self.edges[:-1]) / 2.0
        mid = (self.edges[1:] + self.edges[:-1]) / 2.0
        self.x = mid[:, None] + half[:, None] * _NODES[None, :]
        self.w = half[:, None] * _WEIGHTS[None, :]

    def masses(self, family: str, p: Sequence[float]) -> np.ndarray:
        logf = _log_density(family, self.x, p)
        logf -= logf.max()
        mass = (self.w * np.exp(logf)).sum(axis=1)
        return mass / mass.sum()


def _neg_ll(family: str, grid: _Grid, counts: np.ndarray, values: np.ndarray | None, p: Sequence[float]) -> float:
    """Negative log-likelihood: grouped (from the bin counts) or, when ``values`` is given, exact."""
    with np.errstate(all="ignore"):
        if values is None:
            mass = grid.masses(family, p)
            if not np.all(np.isfinite(mass)):
                return 1e300
            return float(-(counts * np.log(np.clip(mass, 1e-300, None))).sum())
        logf = _log_density(family, grid.x, p)
        shift = logf.max()
        total = (grid.w * np.exp(logf - shift)).sum()
        ll = _log_density(family, values, p).sum() - len(values) * (shift + np.log(total))
    return float(-ll) if np.isfinite(ll) else 1e300


def _neg_loglik(
    u: np.ndarray,
    family: str,
    specs: list[_Param],
    grid: _Grid,
    counts: np.ndarray,
    values: np.ndarray | None = None,
) -> float:
    return _neg_ll(family, grid, counts, values, _to_params(u, specs))


def _nelder_mead(
    f,
    x0: np.ndarray,
    step: float = 0.5,
    max_iter: int = 400,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float]:
    n = len(x0)
    simplex = np.vstack([x0] + [x0 + step * np.eye(n)[i] for i in range(n)])
    values = np.array([f(v) for v in simplex])
    for _ in range(max_iter):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if abs(values[-1] - values[0]) < tol * (1.0 + abs(values[0])) and np.ptp(simplex, axis=0).max() < 1e-6:
            break
        centroid = simplex[:-1].mean(axis=0)
        reflected = centroid + (centroid - simplex[-1])
        fr = f(reflected)
        if fr < values[0]:
            expanded = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = centroid + 0.5 * ((reflected if fr < values[-1] else simplex[-1]) - centroid)
            fc = f(contracted)
            if fc < min(fr, values[-1]):
                simplex[-1], values[-1] = contracted, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                values[1:] = [f(v) for v in simplex[1:]]
    best = int(np.argmin(values))
    return simplex[best], float(values[best])


def _fit_counts(
    family: str,
    specs: list[_Param],
    grid: _Grid,
    counts: np.ndarray,
    starts: list[np.ndarray],
    *,
    step: float = 0.5,
    max_iter: int = 400,
    values: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    def objective(u):
        return _neg_loglik(u, family, specs, grid, counts, values)

    best_u, best_f = None, math.inf
    for start in starts:
        u, value = _nelder_mead(objective, start, step=step, max_iter=max_iter)
        if value < best_f:
            best_u, best_f = u, value
    u, value = _nelder_mead(objective, best_u, step=0.05, max_iter=max_iter)  # polish
    return (u, value) if value < best_f else (best_u, best_f)


def _ks(counts: np.ndarray, mass: np.ndarray) -> float:
    return float(np.abs(np.cumsum(counts) / counts.sum() - np.cumsum(mass)).max())


@dataclass(frozen=True)
class ProfileInterval:
    """Profile-likelihood interval of one parameter, from `DistributionFit.profile`.

    Parameters
    ----------
    parameter : str
        The parameter name.
    estimate : float
        The maximum-likelihood estimate.
    low : float
        Lower end of the interval. When ``low_open`` this is the lower edge of the search range.
    high : float
        Upper end of the interval. When ``high_open`` this is the upper edge of the search range.
    low_open : bool
        True when the likelihood is still within the interval at the lower edge of the search range, so the
        data give no lower limit.
    high_open : bool
        True when the data give no upper limit (for a cutoff: no evidence of a cutoff below ``high``).
    confidence : float
        Confidence level.
    table : pyarrow.Table
        The profile curve: ``value``, ``delta_deviance`` and ``inside`` on the grid.
    """

    parameter: str
    estimate: float
    low: float
    high: float
    low_open: bool
    high_open: bool
    confidence: float
    table: pa.Table = field(repr=False)

    def __str__(self) -> str:
        level = f"{100 * self.confidence:g}%"
        if self.low_open and self.high_open:
            return f"{self.parameter}: not identified (the whole search range is within the {level} interval)"
        if self.high_open:
            return f"{self.parameter} >= {self.low:.4g} ({level}; no upper limit), estimate {self.estimate:.4g}"
        if self.low_open:
            return f"{self.parameter} <= {self.high:.4g} ({level}; no lower limit), estimate {self.estimate:.4g}"
        return f"{self.parameter} in [{self.low:.4g}, {self.high:.4g}] ({level}), estimate {self.estimate:.4g}"


@dataclass(frozen=True)
class DistributionFit:
    """A fitted distribution of positive values, with its uncertainty and fit quality.

    Parameters
    ----------
    family : str
        ``"powerlaw"``, ``"shifted_powerlaw"``, ``"exponential"``, ``"lognormal"`` or ``"truncated_powerlaw"``.
    parameters : dict
        Maximum-likelihood estimates by name.
    std_errors : dict
        Bootstrap standard errors (NaN when ``n_bootstrap=0`` and for parameters at a bound).
    ci_low : dict
        Lower bounds of the percentile bootstrap confidence intervals (NaN as for ``std_errors``).
    ci_high : dict
        Upper bounds of the percentile bootstrap confidence intervals.
    confidence : float
        Confidence level of the intervals.
    n : int
        Number of values inside ``[xmin, xmax]`` that were fitted.
    n_excluded : int
        Number of values left out because they are not positive or fall outside the range.
    xmin : float
        Lower end of the fitted range, in the units of the values.
    xmax : float
        Upper end of the fitted range.
    log_likelihood : float
        Grouped-data log-likelihood. Compare it between families fitted on the same range and bins;
        its absolute value is not meaningful.
    n_parameters : int
        Number of fitted parameters.
    aic : float
        Akaike information criterion, ``2k - 2 log L``.
    bic : float
        Bayesian information criterion, ``k log(n) - 2 log L``.
    ks_statistic : float
        Largest distance between the empirical and the fitted cumulative distribution at the bin edges.
    gof_p_value : float or None
        Parametric-bootstrap p-value of the KS distance, or None when ``n_gof=0``. With many observations
        every simple model is rejected, so read it together with the KS distance.
    at_bound : tuple of str
        Parameters that ended on the edge of their search range, so are not identified by the data (for
        example a cutoff ``kappa`` at its upper bound means that the data show no cutoff).
    edges : numpy.ndarray
        The bin edges.
    counts : numpy.ndarray
        The observed counts per bin.
    likelihood : str
        ``"grouped"`` (the likelihood of the bin counts) or ``"exact"`` (the likelihood of the individual values).
    bounds : dict
        The search range ``(low, high)`` of each parameter. A parameter that ends on one of these edges is in
        ``at_bound``.
    values : numpy.ndarray or None
        The fitted values inside the range, kept only for ``likelihood="exact"``.
    """

    family: str
    parameters: dict[str, float]
    std_errors: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    confidence: float
    n: int
    n_excluded: int
    xmin: float
    xmax: float
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    ks_statistic: float
    gof_p_value: float | None
    at_bound: tuple[str, ...]
    edges: np.ndarray = field(repr=False)
    counts: np.ndarray = field(repr=False)
    likelihood: str = "grouped"
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    values: np.ndarray | None = field(default=None, repr=False)

    def _specs(self) -> list[_Param]:
        specs = _default_specs(self.family, self.xmin, self.xmax, 10.0 * self.xmax)
        return [_Param(sp.name, *self.bounds.get(sp.name, (sp.low, sp.high)), sp.log) for sp in specs]

    def _loglik_terms(self) -> tuple[np.ndarray, np.ndarray]:
        """Log-likelihood contribution of each bin (grouped) or value (exact), and how many values it stands for."""
        grid, p = self._grid()
        if self.values is None:
            mass = grid.masses(self.family, p)
            seen = self.counts > 0
            return np.log(mass[seen]), self.counts[seen]
        logf = _log_density(self.family, grid.x, p)
        shift = logf.max()
        log_total = shift + np.log((grid.w * np.exp(logf - shift)).sum())
        return _log_density(self.family, self.values, p) - log_total, np.ones(len(self.values))

    def _grid(self) -> tuple[_Grid, np.ndarray]:
        grid = _Grid(self.xmin, self.xmax, len(self.counts))
        return grid, np.array(list(self.parameters.values()))

    def pdf(self, x: Any) -> np.ndarray:
        """Probability density at ``x`` of the fitted distribution, normalised on ``[xmin, xmax]``.

        Parameters
        ----------
        x : array-like
            Positive points, in the units of the fitted values.

        Returns
        -------
        numpy.ndarray
            The density at each point; 0 outside ``[xmin, xmax]``.
        """
        grid, p = self._grid()
        x = np.asarray(x, dtype=float)
        logf = _log_density(self.family, grid.x, p)
        shift = logf.max()
        total = (grid.w * np.exp(logf - shift)).sum()
        inside = (x >= self.xmin) & (x <= self.xmax)
        with np.errstate(all="ignore"):
            values = np.exp(_log_density(self.family, np.where(inside, x, self.xmin), p) - shift) / total
        return np.where(inside, values, 0.0)

    def cdf(self, x: Any) -> np.ndarray:
        """Cumulative distribution function at ``x`` of the fitted distribution.

        Parameters
        ----------
        x : array-like
            Positive points, in the units of the fitted values.

        Returns
        -------
        numpy.ndarray
            The probability of a value at most ``x``, interpolated between the bin edges.
        """
        grid, p = self._grid()
        cumulative = np.concatenate([[0.0], np.cumsum(grid.masses(self.family, p))])
        x = np.asarray(x, dtype=float)
        return np.interp(np.log(np.clip(x, self.xmin, self.xmax)), np.log(self.edges), cumulative)

    def profile(self, parameter: str, confidence: float | None = None, n_points: int = 40) -> ProfileInterval:
        """Profile-likelihood interval of one parameter.

        The parameter is fixed on a grid across its search range, the other parameters are refitted at each
        point, and the interval is where the likelihood-ratio statistic ``2 (log L_max - log L_profile)`` stays
        below the chi-squared quantile with one degree of freedom. Unlike the percentile bootstrap it is
        defined for a parameter that the data cannot pin down: when the curve is still inside the interval at
        the edge of the search range, that end is reported as open (for a cutoff ``kappa`` it reads
        "kappa is at least X km"). The interval uses the fitted likelihood, so it treats the values as
        independent, and its ends are interpolated between grid points.

        Parameters
        ----------
        parameter : str
            Name of a parameter of the fitted family, as in ``parameters``.
        confidence : float, optional
            Confidence level. Default: the level of the fit.
        n_points : int, optional
            Grid points across the search range, at least 5. Default 40.

        Returns
        -------
        ProfileInterval
            The interval, which ends are open, and the profile curve as a table.

        Raises
        ------
        ValueError
            If ``parameter`` is not a parameter of the family, or ``confidence`` or ``n_points`` is invalid.
        """
        confidence = self.confidence if confidence is None else confidence
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if n_points < 5:
            raise ValueError("n_points must be at least 5")
        specs = self._specs()
        names = [sp.name for sp in specs]
        if parameter not in names:
            raise ValueError(f"{parameter!r} is not a parameter of {self.family!r}; its parameters are {names}")
        index = names.index(parameter)
        spec = specs[index]
        grid = _Grid(self.xmin, self.xmax, len(self.counts))
        mle = np.array([self.parameters[n] for n in names])
        free = [sp for i, sp in enumerate(specs) if i != index]
        if spec.log:
            points = np.geomspace(spec.low, spec.high, n_points)
        else:
            points = np.linspace(spec.low, spec.high, n_points)
        points = np.unique(np.concatenate([points, [mle[index]]]))
        neg_ll_max = -self.log_likelihood
        deltas = np.empty(len(points))
        previous = None
        for j, value in enumerate(points):
            if not free:
                best = _neg_ll(self.family, grid, self.counts, self.values, np.array([value]))
            else:

                def objective(u, value=value):
                    p = np.insert(_to_params(u, free), index, value)
                    return _neg_ll(self.family, grid, self.counts, self.values, p)

                starts = [_to_unconstrained(np.delete(mle, index), free)]
                if previous is not None:
                    starts.append(previous)
                best = math.inf
                for start in starts:
                    u, f = _nelder_mead(objective, start, step=0.3, max_iter=300)
                    if f < best:
                        best, previous = f, u
            deltas[j] = max(2.0 * (best - neg_ll_max), 0.0)
        critical = NormalDist().inv_cdf(0.5 + confidence / 2.0) ** 2
        inside = deltas <= critical
        at = int(np.argmin(np.abs(points - mle[index])))
        lo_i = at
        while lo_i > 0 and inside[lo_i - 1]:
            lo_i -= 1
        hi_i = at
        while hi_i < len(points) - 1 and inside[hi_i + 1]:
            hi_i += 1

        def edge(i_in: int, i_out: int) -> float:  # interpolate where the curve crosses the critical value
            a, b = points[i_in], points[i_out]
            fa, fb = deltas[i_in], deltas[i_out]
            t = 0.0 if fb == fa else (critical - fa) / (fb - fa)
            if spec.log:
                return float(math.exp(math.log(a) + t * (math.log(b) - math.log(a))))
            return float(a + t * (b - a))

        low_open, high_open = lo_i == 0, hi_i == len(points) - 1
        low = float(points[0]) if low_open else edge(lo_i, lo_i - 1)
        high = float(points[-1]) if high_open else edge(hi_i, hi_i + 1)
        table = pa.table({"value": points.tolist(), "delta_deviance": deltas.tolist(), "inside": inside.tolist()})
        return ProfileInterval(
            parameter=parameter,
            estimate=float(mle[index]),
            low=low,
            high=high,
            low_open=low_open,
            high_open=high_open,
            confidence=confidence,
            table=table,
        )

    def summary(self) -> pa.Table:
        """The estimates as a table.

        Returns
        -------
        pyarrow.Table
            One row per parameter with ``parameter``, ``estimate``, ``std_error``, ``ci_low``, ``ci_high`` and
            ``at_bound`` (True when the estimate is on the edge of its search range).
        """
        names = list(self.parameters)
        return pa.table(
            {
                "parameter": names,
                "estimate": [self.parameters[n] for n in names],
                "std_error": [self.std_errors[n] for n in names],
                "ci_low": [self.ci_low[n] for n in names],
                "ci_high": [self.ci_high[n] for n in names],
                "at_bound": [n in self.at_bound for n in names],
            }
        )


def _prepare(
    values: Any, xmin: float | None, xmax: float | None, bins: int, groups: Any = None
) -> tuple[np.ndarray, int, float, float, np.ndarray | None, np.ndarray, np.ndarray | None]:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    raw = np.asarray(values, dtype=float).ravel()
    label = None
    if groups is not None:
        label = np.asarray(groups.to_numpy() if hasattr(groups, "to_numpy") else groups).ravel()
        if len(label) != len(raw):
            raise ValueError("groups must have one entry per value")
    finite = np.isfinite(raw)
    array = raw[finite]
    label = label[finite] if label is not None else None
    keep = array > 0
    positive = array[keep]
    label = label[keep] if label is not None else None
    if positive.size < 20:
        raise ValueError("At least 20 positive finite values are needed to fit a distribution")
    lo = float(positive.min()) if xmin is None else float(xmin)
    hi = float(positive.max()) if xmax is None else float(xmax)
    if not 0 < lo < hi:
        raise ValueError("xmin and xmax must satisfy 0 < xmin < xmax")
    if bins < 10:
        raise ValueError("bins must be at least 10")
    in_range = (positive >= lo) & (positive <= hi)
    inside = positive[in_range]
    if inside.size < 20:
        raise ValueError("Fewer than 20 values lie inside [xmin, xmax]")
    edges = np.geomspace(lo, hi, bins + 1)
    counts, _ = np.histogram(inside, bins=edges)
    per_group = None
    codes = None
    if label is not None:
        codes = np.unique(label[in_range], return_inverse=True)[1]
        which = np.clip(np.searchsorted(edges, inside, side="right") - 1, 0, bins - 1)
        per_group = np.zeros((codes.max() + 1, bins))
        np.add.at(per_group, (codes, which), 1.0)
    return counts.astype(float), int(array.size - inside.size), lo, hi, per_group, inside, codes


def _starts(specs: list[_Param], rng: np.random.Generator, n: int = 6) -> list[np.ndarray]:
    return [rng.normal(0.0, 1.5, size=len(specs)) for _ in range(n)]


def fit_distance_distribution(
    values: Any,
    family: str = "truncated_powerlaw",
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    bins: int = 200,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    n_gof: int = 0,
    groups: Any = None,
    max_cutoff: float | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    likelihood: str = "grouped",
    random_state: int | None = 0,
) -> DistributionFit:
    """Fit a distribution to positive values by maximum likelihood, with confidence intervals.

    The values are grouped into geometric bins on ``[xmin, xmax]`` and the family is fitted by maximising the
    multinomial log-likelihood of the bin counts. The confidence intervals are percentile intervals of a
    bootstrap of the bin counts, which treats the values as independent: when they come from a limited number
    of people pass ``groups`` so that whole people are resampled. When ``n_gof`` is positive, a parametric
    bootstrap gives a p-value for the Kolmogorov-Smirnov distance. Values that are zero, negative, not finite or outside the range are left
    out and counted in ``n_excluded``, so state the range when you report a fit: the estimates depend on it.

    The families are ``f(x) ∝ x^-beta`` (``"powerlaw"``), ``exp(-x / kappa)`` (``"exponential"``),
    a log-normal with parameters ``mu`` and ``sigma`` (``"lognormal"``), ``(x + r0)^-beta``
    (``"shifted_powerlaw"``) and ``(x + r0)^-beta exp(-x / kappa)`` (``"truncated_powerlaw"``, as in
    González et al., 2008). Comparing the last two shows whether the cutoff adds anything beyond the shift
    ``r0``. A parameter that
    ends on the edge of its search range is listed in ``at_bound``; for the truncated power law this
    typically means that the data have no cutoff.

    Parameters
    ----------
    values : array-like
        Positive values such as jump lengths in kilometres. Pass a flat array; use ``numpy.concatenate`` on
        the lists returned by `jump_lengths`.
    family : str, optional
        One of ``"truncated_powerlaw"`` (default), ``"shifted_powerlaw"``, ``"powerlaw"``, ``"exponential"`` or
        ``"lognormal"``.
    xmin : float, optional
        Lower end of the fitted range. Default: the smallest positive value. Raise it to ignore GPS noise.
    xmax : float, optional
        Upper end of the fitted range. Default: the largest value.
    bins : int, optional
        Number of geometric bins, at least 10. Default 200.
    n_bootstrap : int, optional
        Bootstrap replicates for the standard errors and intervals; 0 skips them. Default 100.
    confidence : float, optional
        Confidence level of the intervals, between 0 and 1. Default 0.95.
    n_gof : int, optional
        Parametric-bootstrap replicates for the goodness-of-fit p-value; 0 skips it. Default 0. The test treats
        the values as independent, so with clustered values (many jumps per user) it is too optimistic.
    groups : array-like, optional
        One label per value (for example the user id of each jump). When given, the bootstrap for the standard
        errors and intervals resamples whole groups instead of single values, which widens the intervals when
        the values of a group are alike. Use it whenever the values come from a limited number of people.
    max_cutoff : float, optional
        Largest value that ``kappa`` can take. Default ten times ``xmax``.
    bounds : dict, optional
        Search range ``{parameter: (low, high)}`` of any parameter, in the units of the values for ``r0`` and
        ``kappa``. It overrides the defaults (and ``max_cutoff``). Use it to check that a conclusion does not
        depend on the range; the defaults are ``beta`` 0.01 to 10 (6 for the shifted and truncated power laws),
        ``r0`` ``0.01 xmin`` to ``xmax``, ``kappa`` ``0.01 xmin`` (``xmin`` for the truncated power law) to
        ``10 xmax``, ``mu`` ``log(xmin) - 5`` to ``log(xmax) + 5`` and ``sigma`` 0.02 to 10. ``r0``, ``kappa`` and
        ``sigma`` are searched on a log scale, so their lower bound must be positive.
    likelihood : {"grouped", "exact"}, optional
        ``"grouped"`` (default) maximises the likelihood of the bin counts, whose cost does not grow with the
        number of values. ``"exact"`` maximises the likelihood of the individual values (the density
        normalised on ``[xmin, xmax]``); it removes the approximation of the bins but every evaluation
        passes over all values, so it is slow beyond about 10^5 values, and ``n_gof`` is not available. The
        two likelihoods are not comparable with each other.
    random_state : int, optional
        Seed of the bootstrap. Default 0.

    Returns
    -------
    DistributionFit
        The estimates, intervals, likelihood, AIC, BIC, KS distance and the parameters at their bounds.

    Raises
    ------
    ValueError
        If ``family`` is unknown, fewer than 20 positive values (or 20 inside the range) are given, or the
        range, ``bins``, ``confidence``, ``likelihood`` or ``bounds`` is invalid, or ``n_gof`` is positive with
        ``likelihood="exact"``.

    See Also
    --------
    DistributionFit.profile : Profile-likelihood interval, also for a parameter at its bound.
    compare_distance_distributions : Fit several families and rank them by AIC.
    fit_values_to_truncated_powerlaw : The earlier histogram grid search.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.measures.fitting import fit_distance_distribution
    >>> rng = np.random.default_rng(1)
    >>> jumps = rng.lognormal(mean=1.0, sigma=0.8, size=5000)
    >>> fit = fit_distance_distribution(jumps, "lognormal", n_bootstrap=20)
    >>> round(fit.parameters["mu"], 1), round(fit.parameters["sigma"], 1)
    (1.0, 0.8)
    >>> fit.at_bound
    ()
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if likelihood not in ("grouped", "exact"):
        raise ValueError('likelihood must be "grouped" or "exact"')
    if likelihood == "exact" and n_gof > 0:
        raise ValueError('n_gof needs likelihood="grouped"')
    counts, excluded, lo, hi, per_group, inside, codes = _prepare(values, xmin, xmax, bins, groups)
    cutoff = float(max_cutoff) if max_cutoff is not None else 10.0 * hi
    specs = _specs(family, lo, hi, cutoff, bounds)
    grid = _Grid(lo, hi, bins)
    rng = np.random.default_rng(random_state)
    exact = inside if likelihood == "exact" else None
    u, neg_ll = _fit_counts(family, specs, grid, counts, _starts(specs, rng), values=exact)
    params = _to_params(u, specs)
    mass = grid.masses(family, params)
    n = int(counts.sum())
    k = len(specs)
    ks = _ks(counts, mass)

    ci_low = {s.name: math.nan for s in specs}
    ci_high = dict(ci_low)
    std_errors = dict(ci_low)
    if n_bootstrap > 0:
        draws = np.empty((n_bootstrap, k))
        if exact is not None and codes is not None:
            order = np.argsort(codes, kind="stable")
            by_group = np.split(exact[order], np.cumsum(np.bincount(codes))[:-1])
        for b in range(n_bootstrap):
            resampled_values = None
            if exact is not None:
                if codes is None:
                    resampled_values = rng.choice(exact, size=n)
                else:
                    chosen = rng.integers(0, len(by_group), len(by_group))
                    resampled_values = np.concatenate([by_group[i] for i in chosen])
                resampled = counts
            elif per_group is None:
                resampled = rng.multinomial(n, counts / n).astype(float)
            else:  # resample whole groups (for example users), as their values are not independent
                resampled = per_group[rng.integers(0, len(per_group), len(per_group))].sum(axis=0)
            ub, _ = _fit_counts(family, specs, grid, resampled, [u], step=0.1, max_iter=200, values=resampled_values)
            draws[b] = _to_params(ub, specs)
        alpha = (1.0 - confidence) / 2.0
        for i, s in enumerate(specs):
            std_errors[s.name] = float(draws[:, i].std(ddof=1)) if n_bootstrap > 1 else math.nan
            ci_low[s.name] = float(np.quantile(draws[:, i], alpha))
            ci_high[s.name] = float(np.quantile(draws[:, i], 1.0 - alpha))

    p_value = None
    if n_gof > 0:
        exceed = 0
        for _ in range(n_gof):
            simulated = rng.multinomial(n, mass).astype(float)
            us, _ = _fit_counts(family, specs, grid, simulated, [u], step=0.1, max_iter=200)
            exceed += _ks(simulated, grid.masses(family, _to_params(us, specs))) >= ks
        p_value = (exceed + 1) / (n_gof + 1)

    positions = {s.name: _position(params[i], s) for i, s in enumerate(specs)}
    at_bound = tuple(name for name, pos in positions.items() if pos < _BOUND_MARGIN or pos > 1 - _BOUND_MARGIN)
    for name in at_bound:  # a parameter on the edge of its range has no meaningful error or interval
        std_errors[name] = ci_low[name] = ci_high[name] = math.nan
    return DistributionFit(
        family=family,
        parameters={s.name: float(params[i]) for i, s in enumerate(specs)},
        std_errors=std_errors,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n=n,
        n_excluded=excluded,
        xmin=lo,
        xmax=hi,
        log_likelihood=-neg_ll,
        n_parameters=k,
        aic=2.0 * k + 2.0 * neg_ll,
        bic=k * math.log(n) + 2.0 * neg_ll,
        ks_statistic=ks,
        gof_p_value=p_value,
        at_bound=at_bound,
        edges=grid.edges,
        counts=counts,
        likelihood=likelihood,
        bounds={sp.name: (sp.low, sp.high) for sp in specs},
        values=exact,
    )


def compare_distance_distributions(
    values: Any,
    families: Sequence[str] = FAMILIES,
    *,
    degenerate_last: bool = True,
    **kwargs: Any,
) -> pa.Table:
    """Fit several families to the same values and rank them by AIC.

    Every family is fitted on the same range and bins, so their likelihoods, AIC and BIC are comparable.
    A small AIC difference (under about 2) means the data cannot tell the families apart.

    A family with a parameter on the edge of its search range (``at_bound``) is a degenerate fit: the shape it
    reached is not the one the family is meant to describe (a shifted power law with ``beta`` at its upper
    limit is an exponential in disguise), so its AIC says little. With ``degenerate_last=True`` (the default)
    those families are listed after the others, ``delta_aic`` is measured from the best family that is not
    degenerate, and ``akaike_weight`` is computed among the families that are not degenerate only (it is
    missing for the others). A negative ``delta_aic`` on a degenerate row means that its likelihood is
    higher, which is what the degeneracy buys; do not read it as support.

    Parameters
    ----------
    values : array-like
        Positive values, as for `fit_distance_distribution`.
    families : sequence of str, optional
        Families to compare. Default: all five.
    degenerate_last : bool, optional
        Rank families with a parameter at its bound after the others and leave them out of the Akaike
        weights. With False every family is ranked by AIC alone. Default True.
    **kwargs
        Passed to `fit_distance_distribution` (``xmin``, ``xmax``, ``bins``, ``n_bootstrap``, ``n_gof``,
        ``max_cutoff``, ``bounds``, ``likelihood``, ``random_state``). ``n_bootstrap`` defaults to 0 here,
        because only the criteria are needed. ``bounds`` may name parameters that some families do not have;
        each family uses the entries that apply to it.

    Returns
    -------
    pyarrow.Table
        One row per family with ``family``, ``n_parameters``, ``log_likelihood``, ``aic``, ``bic``,
        ``delta_aic``, ``akaike_weight``, ``ks_statistic``, ``gof_p_value``, ``at_bound`` (the parameters on
        the edge of their range) and ``degenerate`` (True when there is any). The AIC and the Akaike weights
        treat the values as independent, so they overstate the evidence when many values come from the same people.

    See Also
    --------
    fit_distance_distribution : Fit one family with confidence intervals.
    likelihood_ratio_test : Test a family against the one that extends it.
    vuong_test : Test two families that are not nested.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.measures.fitting import compare_distance_distributions
    >>> jumps = np.random.default_rng(1).lognormal(1.0, 0.8, size=5000)
    >>> table = compare_distance_distributions(jumps, ["exponential", "lognormal"])
    >>> table.column("family").to_pylist()[0]
    'lognormal'
    """
    kwargs.setdefault("n_bootstrap", 0)
    bounds = kwargs.pop("bounds", None)
    fits = []
    for family in families:
        names = {sp.name for sp in _default_specs(family, 1.0, 2.0, 20.0)}
        own = {k: v for k, v in bounds.items() if k in names} if bounds else None
        fits.append(fit_distance_distribution(values, family, bounds=own, **kwargs))
    fits.sort(key=lambda f: f.aic)
    clean = [f for f in fits if not f.at_bound]
    if degenerate_last and clean:
        fits = clean + [f for f in fits if f.at_bound]
        reference = clean
    else:
        reference = fits
    aic = np.array([f.aic for f in fits])
    delta = aic - min(f.aic for f in reference)
    support = np.exp(-(np.array([f.aic for f in reference]) - min(f.aic for f in reference)) / 2.0)
    support /= support.sum()
    by_fit = {id(f): float(w) for f, w in zip(reference, support)}
    weights = [by_fit.get(id(f), math.nan) for f in fits]
    return pa.table(
        {
            "family": [f.family for f in fits],
            "n_parameters": [f.n_parameters for f in fits],
            "log_likelihood": [f.log_likelihood for f in fits],
            "aic": aic.tolist(),
            "bic": [f.bic for f in fits],
            "delta_aic": delta.tolist(),
            "akaike_weight": weights,
            "ks_statistic": [f.ks_statistic for f in fits],
            "gof_p_value": [f.gof_p_value for f in fits],
            "at_bound": [", ".join(f.at_bound) for f in fits],
            "degenerate": [bool(f.at_bound) for f in fits],
        }
    )
