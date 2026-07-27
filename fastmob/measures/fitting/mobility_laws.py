"""Mobility law measures: power-law fitting and related utilities."""

from __future__ import annotations

import math
from typing import Any, Literal

import narwhals as nw
import numpy as np

from fastmob._core import visitation_distances
from fastmob.utils._common import (
    LAT_CANDIDATES,
    LNG_CANDIDATES,
    LOCATION_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _is_polars_backed,
    _pick_existing_column,
)

try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
except ImportError:
    _scipy_curve_fit = None


_VISITATION_YLABEL = r"$\rho_i(r,f)$ (visitors km$^{-2}$)"


def _detect_required_column(
    df: nw.DataFrame,
    explicit: str | None,
    candidates: list[str],
    role: str,
) -> str:
    if explicit is None:
        explicit = _pick_existing_column(df.columns, candidates)
    if explicit is None:
        raise ValueError(
            f"Could not detect required {role} column. "
            f"Tried: {candidates}. Available columns: {df.columns}. "
            f"Pass the column name explicitly."
        )
    if explicit not in df.columns:
        raise ValueError(f"Column {explicit!r} ({role}) not found. Available columns: {df.columns}.")
    return explicit


def _location_coordinates(
    visits_df: nw.DataFrame,
    *,
    locations_df: Any | None,
    location_id_col: str,
    lat_col: str | None,
    lng_col: str | None,
) -> nw.DataFrame:
    source = nw.from_native(locations_df, eager_only=True) if locations_df is not None else visits_df
    source_location_col = (
        location_id_col
        if location_id_col in source.columns
        else _pick_existing_column(
            source.columns,
            LOCATION_CANDIDATES,
        )
    )
    if source_location_col is None:
        raise ValueError(
            f"Could not detect a location column in locations_df. "
            f"Tried: {LOCATION_CANDIDATES}. Available columns: {source.columns}."
        )
    source_lat_col = _detect_required_column(source, lat_col, LAT_CANDIDATES, "latitude")
    source_lng_col = _detect_required_column(source, lng_col, LNG_CANDIDATES, "longitude")

    coords = (
        source.select([source_location_col, source_lat_col, source_lng_col])
        .drop_nulls(subset=[source_location_col, source_lat_col, source_lng_col])
        .with_columns(
            nw.col(source_lat_col).cast(nw.Float64).alias("loc_lat"),
            nw.col(source_lng_col).cast(nw.Float64).alias("loc_lng"),
        )
        .rename({source_location_col: location_id_col})
        .select([location_id_col, "loc_lat", "loc_lng"])
        .group_by(location_id_col)
        .agg(
            nw.col("loc_lat").first().alias("loc_lat"),
            nw.col("loc_lng").first().alias("loc_lng"),
        )
    )
    if len(coords) == 0:
        raise ValueError("No valid location coordinates found.")
    return coords


def _route_visitation_distances(df: nw.DataFrame) -> list[float]:
    use_arrow = _is_polars_backed(df)
    home_lats = df.get_column("home_lat")
    home_lngs = df.get_column("home_lng")
    loc_lats = df.get_column("loc_lat")
    loc_lngs = df.get_column("loc_lng")
    if use_arrow:
        return visitation_distances(
            home_lats.to_arrow(),
            home_lngs.to_arrow(),
            loc_lats.to_arrow(),
            loc_lngs.to_arrow(),
        )
    return visitation_distances(
        home_lats.to_numpy(),
        home_lngs.to_numpy(),
        loc_lats.to_numpy(),
        loc_lngs.to_numpy(),
    )


def compute_visitation_law_data(
    visits: Any,
    *,
    locations_df: Any | None = None,
    observation_period_days: float | None = None,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    timestamp_col: str | None = None,
    purpose_col: str | None = "purpose",
    home_purpose: str = "HOME",
    lat_col: str | None = None,
    lng_col: str | None = None,
) -> Any:
    """Compute per-user, per-location inputs for the universal visitation law.

    Parameters
    ----------
    visits:
        Visit dataframe; any Narwhals-compatible eager backend. Required
        columns identify users, locations, and visit timestamps. Location
        centroid coordinates may be present on this dataframe or supplied via
        ``locations_df``.
    locations_df:
        Optional location lookup dataframe with one row per location and
        latitude/longitude centroid columns.
    observation_period_days:
        Accepted for compatibility. The visitation frequency ``f`` is a count
        of distinct visiting days and this value is not used.
    user_id_col:
        User identifier column. Auto-detected when ``None``.
    location_id_col:
        Location identifier column. Auto-detected when ``None``.
    timestamp_col:
        Visit timestamp column. Auto-detected when ``None``.
    purpose_col:
        Purpose/activity column used for home inference. When the column is
        missing, home falls back to each user's most visited location.
    home_purpose:
        Purpose value treated as home. Default ``"HOME"``.
    lat_col, lng_col:
        Latitude and longitude centroid columns. Auto-detected when ``None``.

    Returns
    -------
    DataFrame
        Same backend as ``visits``. Columns are ``[user_id_col,
        location_id_col, "r_km", "f", "rf", "n_visits"]``. ``r_km`` is the
        Haversine distance from the inferred home location to the visited
        location, ``f`` is the number of distinct visit days, and ``rf`` is
        ``r_km * f``.

    Raises
    ------
    ValueError
        If required columns or valid location coordinates cannot be found.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.fitting.mobility_laws import compute_visitation_law_data
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1"],
    ...         "location_id": ["home", "work", "work"],
    ...         "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
    ...         "purpose": ["HOME", "WORK", "WORK"],
    ...         "lat": [0.0, 0.0, 0.0],
    ...         "lng": [0.0, 1.0, 1.0],
    ...     }
    ... )
    >>> result = compute_visitation_law_data(visits, timestamp_col="timestamp")
    >>> print(result[["user_id", "location_id", "f", "n_visits"]].to_string(index=False))
    user_id location_id  f  n_visits
         u1        home  1         1
         u1        work  2         2
    """
    del observation_period_days

    df = nw.from_native(visits, eager_only=True)
    backend = df.implementation
    user_id_col = _detect_required_column(df, user_id_col, USER_ID_CANDIDATES, "user_id")
    location_id_col = _detect_required_column(df, location_id_col, LOCATION_CANDIDATES, "location")
    timestamp_col = _detect_required_column(df, timestamp_col, TIMESTAMP_CANDIDATES, "timestamp")

    coords = _location_coordinates(
        df,
        locations_df=locations_df,
        location_id_col=location_id_col,
        lat_col=lat_col,
        lng_col=lng_col,
    )

    df = (
        df.drop_nulls(subset=[user_id_col, location_id_col, timestamp_col])
        .with_columns(nw.col(timestamp_col).dt.truncate("1d").alias("__visit_day__"))
        .drop_nulls(subset=["__visit_day__"])
    )
    if len(df) == 0:
        return nw.from_dict(
            {user_id_col: [], location_id_col: [], "r_km": [], "f": [], "rf": [], "n_visits": []},
            backend=backend,
        ).to_native()

    visit_counts = (
        df.group_by([user_id_col, location_id_col])
        .agg(
            nw.len().alias("n_visits"),
            nw.col("__visit_day__").n_unique().alias("f"),
        )
        .sort([user_id_col, location_id_col])
    )

    fallback_home = (
        visit_counts.sort([user_id_col, "n_visits", location_id_col], descending=[False, True, False])
        .group_by(user_id_col)
        .agg(nw.col(location_id_col).first().alias("_fallback_home_location"))
    )

    if purpose_col is None:
        home_locations = fallback_home.rename({"_fallback_home_location": "_home_location"})
    elif purpose_col in df.columns:
        purpose_home_counts = (
            df.filter(nw.col(purpose_col) == home_purpose)
            .group_by([user_id_col, location_id_col])
            .agg(nw.len().alias("_home_count"))
        )
        if len(purpose_home_counts) == 0:
            home_locations = fallback_home.rename({"_fallback_home_location": "_home_location"})
        else:
            purpose_home = (
                purpose_home_counts.sort([user_id_col, "_home_count", location_id_col], descending=[False, True, False])
                .group_by(user_id_col)
                .agg(nw.col(location_id_col).first().alias("_purpose_home_location"))
            )
            purpose_users = set(purpose_home.get_column(user_id_col).to_list())
            fallback_missing = fallback_home.filter(~nw.col(user_id_col).is_in(list(purpose_users))).rename(
                {"_fallback_home_location": "_home_location"}
            )
            home_locations = nw.concat(
                [
                    purpose_home.rename({"_purpose_home_location": "_home_location"}),
                    fallback_missing,
                ]
            )
    else:
        home_locations = fallback_home.rename({"_fallback_home_location": "_home_location"})

    home_coords = coords.rename(
        {
            location_id_col: "_home_location",
            "loc_lat": "home_lat",
            "loc_lng": "home_lng",
        }
    )
    merged = (
        visit_counts.join(home_locations, on=user_id_col, how="inner")
        .join(home_coords, on="_home_location", how="inner")
        .join(coords, on=location_id_col, how="inner")
        .with_columns(
            nw.col("home_lat").cast(nw.Float64),
            nw.col("home_lng").cast(nw.Float64),
            nw.col("loc_lat").cast(nw.Float64),
            nw.col("loc_lng").cast(nw.Float64),
            nw.col("f").cast(nw.Float64),
            nw.col("n_visits").cast(nw.Int64),
        )
    )
    if len(merged) == 0:
        return nw.from_dict(
            {user_id_col: [], location_id_col: [], "r_km": [], "f": [], "rf": [], "n_visits": []},
            backend=backend,
        ).to_native()

    r_km = _route_visitation_distances(merged)
    result = (
        merged.with_columns(nw.new_series("r_km", r_km, backend=backend))
        .with_columns((nw.col("r_km") * nw.col("f")).alias("rf"))
        .select([user_id_col, location_id_col, "r_km", "f", "rf", "n_visits"])
        .sort([user_id_col, location_id_col])
    )
    return result.to_native()


def bin_visitation_law_data(
    vl_df: Any,
    *,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_area_km2: float | None = None,
    n_bins: int = 30,
    distance_bin_width_km: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Aggregate visitation-law rows into binned ``(rf, rho)`` values.

    Parameters
    ----------
    vl_df:
        Dataframe returned by :func:`compute_visitation_law_data`.
    user_id_col:
        User identifier column. Auto-detected when ``None``.
    location_id_col:
        Location identifier column. Auto-detected when ``None``.
    location_area_km2:
        Accepted for compatibility and not used in the Schläpfer density
        normalization.
    n_bins:
        Number of logarithmic ``rf`` bins.
    distance_bin_width_km:
        Width of radial distance bins in kilometres. Must be positive.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, str]
        Log-bin centers for ``r*f``, mean density values, and the y-axis label.
    """
    del location_area_km2

    if distance_bin_width_km <= 0:
        raise ValueError("distance_bin_width_km must be positive.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    df = nw.from_native(vl_df, eager_only=True)
    user_id_col = _detect_required_column(df, user_id_col, USER_ID_CANDIDATES, "user_id")
    location_id_col = _detect_required_column(df, location_id_col, LOCATION_CANDIDATES, "location")

    for col in ["r_km", "f", "rf"]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} is required.")

    rows = df.select([user_id_col, location_id_col, "r_km", "f", "rf"]).rows(named=True)
    valid_rows = [
        row
        for row in rows
        if np.isfinite(float(row["r_km"]))
        and np.isfinite(float(row["f"]))
        and np.isfinite(float(row["rf"]))
        and float(row["r_km"]) > 0
        and float(row["f"]) > 0
        and float(row["rf"]) > 0
    ]
    if not valid_rows:
        return np.array([]), np.array([]), _VISITATION_YLABEL

    spectrum_counts: dict[tuple[Any, float, float], set[Any]] = {}
    for row in valid_rows:
        r_km = float(row["r_km"])
        f = float(row["f"])
        r_center = np.floor(r_km / distance_bin_width_km) * distance_bin_width_km + distance_bin_width_km / 2.0
        key = (row[location_id_col], r_center, f)
        spectrum_counts.setdefault(key, set()).add(row[user_id_col])

    rf_values = []
    rho_values = []
    for (_location, r_center, f), users in spectrum_counts.items():
        annulus_area = 2.0 * np.pi * r_center * distance_bin_width_km
        if annulus_area <= 0:
            continue
        rho = len(users) / annulus_area
        rf = r_center * f
        if rf > 0 and rho > 0:
            rf_values.append(rf)
            rho_values.append(rho)

    if not rf_values:
        return np.array([]), np.array([]), _VISITATION_YLABEL

    spectrum_rf = np.asarray(rf_values, dtype=float)
    spectrum_rho = np.asarray(rho_values, dtype=float)
    rf_min = float(spectrum_rf.min())
    rf_max = float(spectrum_rf.max())
    if rf_min == rf_max:
        return np.array([rf_min]), np.array([float(spectrum_rho.mean())]), _VISITATION_YLABEL

    bin_edges = np.logspace(np.log10(rf_min), np.log10(rf_max), n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    rf_centers = []
    rho_binned = []
    for idx, (lo, hi, center) in enumerate(zip(bin_edges[:-1], bin_edges[1:], bin_centers)):
        if idx == len(bin_centers) - 1:
            mask = (spectrum_rf >= lo) & (spectrum_rf <= hi)
        else:
            mask = (spectrum_rf >= lo) & (spectrum_rf < hi)
        if mask.any():
            rf_centers.append(center)
            rho_binned.append(float(spectrum_rho[mask].mean()))

    return np.asarray(rf_centers), np.asarray(rho_binned), _VISITATION_YLABEL


def visitation_law_curve(
    rf_values: np.ndarray,
    eta: float,
    mu: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth curve for ``rho(r, f) = mu * (r*f)^(-eta)``.

    Parameters
    ----------
    rf_values:
        Positive ``r*f`` values defining the curve range.
    eta:
        Scaling exponent.
    mu:
        Attractiveness prefactor.
    n_points:
        Number of curve points. Must be at least 2 unless all ``rf`` values are
        equal, in which case one point is returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``rf`` and ``rho`` arrays for plotting.
    """
    rf_values = np.asarray(rf_values, dtype=float)
    mask = np.isfinite(rf_values) & (rf_values > 0)
    if not mask.any():
        raise ValueError("rf_values must contain at least one positive finite value.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    rf_min = float(rf_values[mask].min())
    rf_max = float(rf_values[mask].max())
    if rf_min == rf_max:
        rf_fit = np.array([rf_min], dtype=float)
    else:
        rf_fit = np.logspace(np.log10(rf_min), np.log10(rf_max), n_points)
    rho_fit = float(mu) * np.power(rf_fit, -float(eta))
    return rf_fit, rho_fit


def fit_visitation_law(
    rf_values: np.ndarray,
    rho_values: np.ndarray,
    *,
    min_rf: float | None = None,
    max_rf: float | None = None,
    return_curve: bool = False,
    n_curve_points: int = 200,
) -> tuple[float, float, float] | tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit the universal visitation law to binned ``(rf, rho)`` values.

    The fitted model is ``rho(r, f) = mu * (r*f)^(-eta)``. Fitting uses
    ordinary least squares in log-log space.

    Parameters
    ----------
    rf_values, rho_values:
        Positive binned values, typically returned by
        :func:`bin_visitation_law_data`.
    min_rf, max_rf:
        Optional positive inclusive range for fitting.
    return_curve:
        When ``True``, also return smooth fitted curve arrays.
    n_curve_points:
        Number of points in the fitted curve.

    Returns
    -------
    tuple
        ``(eta, mu, r2)`` or ``(eta, mu, r2, rf_fit, rho_fit)`` when
        ``return_curve=True``. ``r2`` is computed in log-log space.
    """
    rf_values = np.asarray(rf_values, dtype=float)
    rho_values = np.asarray(rho_values, dtype=float)
    if rf_values.shape != rho_values.shape:
        raise ValueError("rf_values and rho_values must have the same shape.")

    mask = np.isfinite(rf_values) & np.isfinite(rho_values) & (rf_values > 0) & (rho_values > 0)
    if min_rf is not None:
        if min_rf <= 0:
            raise ValueError("min_rf must be positive when provided.")
        mask &= rf_values >= min_rf
    if max_rf is not None:
        if max_rf <= 0:
            raise ValueError("max_rf must be positive when provided.")
        mask &= rf_values <= max_rf

    x = np.log(rf_values[mask])
    y = np.log(rho_values[mask])
    if len(x) < 2:
        raise ValueError("At least two positive finite data points are required to fit.")

    slope, intercept = np.polyfit(x, y, 1)
    eta = float(-slope)
    mu = float(np.exp(intercept))

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    if return_curve:
        rf_fit, rho_fit = visitation_law_curve(
            rf_values[mask],
            eta=eta,
            mu=mu,
            n_points=n_curve_points,
        )
        return eta, mu, r2, rf_fit, rho_fit
    return eta, mu, r2


def daily_location_lognormal_fit(
    visits: Any,
    *,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    timestamp_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a lognormal distribution to daily distinct-location counts.

    For each user and calendar day, counts the number of distinct locations
    visited, then fits ``mu``/``sigma`` of the lognormal distribution to the
    (natural) log of those per-user-per-day counts.

    Parameters
    ----------
    visits:
        Visits/stays dataframe; any Narwhals-compatible eager backend.
    user_id_col, location_id_col, timestamp_col:
        Explicit column name overrides; auto-detected when None.

    Returns
    -------
    x_points, y_points, mu, sigma:
        ``x_points`` are the distinct daily-location-count values observed
        (sorted ascending), ``y_points`` their empirical frequencies (summing
        to 1), and ``mu``/``sigma`` the fitted lognormal parameters (fit on
        the log of every individual count, not on the binned points).

    Raises
    ------
    ValueError
        If fewer than two daily location counts are available, or if the
        fitted log-variance is degenerate (<= 0).

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.fitting.mobility_laws import daily_location_lognormal_fit
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u2"],
    ...         "location_id": ["home", "work", "home"],
    ...         "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
    ...     }
    ... )
    >>> x_points, y_points, mu, sigma = daily_location_lognormal_fit(visits)
    >>> x_points
    array([1., 2.])
    """
    df = nw.from_native(visits, eager_only=True)
    user_id_col = _detect_required_column(df, user_id_col, USER_ID_CANDIDATES, "user_id")
    location_id_col = _detect_required_column(df, location_id_col, LOCATION_CANDIDATES, "location")
    timestamp_col = _detect_required_column(df, timestamp_col, TIMESTAMP_CANDIDATES, "timestamp")

    daily = (
        df.with_columns(nw.col(timestamp_col).dt.truncate("1d").alias("__day__"))
        .group_by([user_id_col, "__day__"])
        .agg(nw.col(location_id_col).n_unique().alias("__count__"))
    )

    values = daily.get_column("__count__").to_numpy().astype(float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        raise ValueError("At least two daily location counts are required to fit.")

    log_values = np.log(values)
    mu = float(log_values.mean())
    sigma = float(log_values.std())
    if not np.isfinite(sigma) or sigma <= 1e-12:
        raise ValueError("Daily location counts must have positive log variance.")

    x_points, counts = np.unique(values, return_counts=True)
    y_points = counts / counts.sum()
    return x_points, y_points, mu, sigma


def log_truncated_powerlaw(
    x: np.ndarray,
    c: float,
    r0: float,
    beta: float,
    kappa: float,
) -> np.ndarray:
    """Log of a truncated power-law (Gonzalez et al. 2008).

    \\[
    f(x) = c (x + r_0)^{-\\beta} \\exp\\left(-\\frac{x}{\\kappa}\\right)
    \\]

    \\[
    \\log f(x) = \\log(c) - \\beta \\log(x + r_0) - \\frac{x}{\\kappa}
    \\]

    Parameters
    ----------
    x:
        Input values (must be positive for meaningful results).
    c:
        Scale parameter.
    r0:
        Offset parameter (shifts the origin to avoid singularity at x=0).
    beta:
        Power-law exponent.
    kappa:
        Exponential cutoff scale.

    Returns
    -------
    np.ndarray
        \\(\\log f(x)\\) evaluated at each element of x.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.measures.fitting.mobility_laws import log_truncated_powerlaw
    >>> x = np.array([1.0, 2.0, 5.0, 10.0])
    >>> values = log_truncated_powerlaw(x, c=2.0, r0=1.0, beta=1.5, kappa=20.0)
    >>> print(np.round(values, 3))
    [-0.397 -1.055 -2.244 -3.404]
    """
    return np.log(c) - beta * np.log(x + r0) - (x / kappa)


def _geom_grid(lo: float, hi: float, n: int) -> np.ndarray:
    log_lo, log_hi = math.log(lo), math.log(hi)
    if n <= 1:
        return np.array([math.exp(log_lo)])
    return np.exp(log_lo + (log_hi - log_lo) * np.arange(n) / (n - 1))


def _lin_grid(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([lo])
    return lo + (hi - lo) * np.arange(n) / (n - 1)


def _grid_fit_candidates(
    x_data: np.ndarray, log_y_data: np.ndarray, r0_values: np.ndarray, beta_values: np.ndarray, kappa_values: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Evaluate every `(r0, beta, kappa)` candidate on a grid, solving the
    optimal `c` in closed form (log-space OLS intercept) for each -- returns
    the best `(sse, c, r0, beta, kappa)`.
    """
    best: tuple[float, float, float, float, float] = (math.inf, 1.0, 1.0, 1.75, 400.0)
    for r0 in r0_values:
        log_x_r0 = np.log(x_data + r0)
        for beta in beta_values:
            for kappa in kappa_values:
                shape_log = -beta * log_x_r0 - x_data / kappa
                log_c = float(np.mean(log_y_data - shape_log))
                sse = float(np.sum((log_y_data - (log_c + shape_log)) ** 2))
                if sse < best[0]:
                    best = (sse, math.exp(log_c), float(r0), float(beta), float(kappa))
    return best


def _fit_truncated_powerlaw_grid(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """Dependency-free coarse-to-fine grid search fit, ported from
    citybehavex-web's Rust `truncated_powerlaw_dataset` (written there
    specifically to avoid a scipy dependency in a Python-free web backend).

    This is an *approximation*, not a bit-identical match to scipy's
    Trust-Region-Reflective solver -- "close enough for rendered reference
    curves" per the original Rust source's own framing, not a claim of
    numerical equivalence.
    """
    log_y_data = np.log(y_data)
    max_x = float(x_data.max())

    _sse0, _c0, r00, beta0, kappa0 = _grid_fit_candidates(
        x_data,
        log_y_data,
        _geom_grid(0.01, max(max_x, 1.0), 10),
        _lin_grid(0.2, 4.0, 16),
        _geom_grid(1.0, max(max_x * 20.0, 10.0), 14),
    )
    r0_lo, r0_hi = max(r00 / 3.0, 0.001), max(r00 * 3.0, max(r00 / 3.0, 0.001) * 1.01)
    beta_lo, beta_hi = max(beta0 - 0.6, 0.01), beta0 + 0.6
    kappa_lo, kappa_hi = max(kappa0 / 3.0, 0.1), max(kappa0 * 3.0, max(kappa0 / 3.0, 0.1) * 1.01)

    _sse, c, r0, beta, kappa = _grid_fit_candidates(
        x_data,
        log_y_data,
        _geom_grid(r0_lo, r0_hi, 12),
        _lin_grid(beta_lo, beta_hi, 14),
        _geom_grid(kappa_lo, kappa_hi, 12),
    )
    return np.array([c, r0, beta, kappa])


def fit_values_to_truncated_powerlaw(
    values: list[float] | np.ndarray,
    bins: int = 100,
    *,
    method: Literal["scipy", "grid"] = "scipy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a truncated power-law to a 1-D array of positive values.

    Builds a log-spaced histogram of ``values``, then fits
    :func:`log_truncated_powerlaw` to the log-density.

    Parameters
    ----------
    values:
        1-D array of positive float values (e.g. jump lengths in km).
    bins:
        Number of log-spaced histogram bins. Default 100.
    method:
        ``"scipy"`` (default): nonlinear least-squares via
        ``scipy.optimize.curve_fit`` (Trust-Region-Reflective). ``"grid"``:
        a dependency-free coarse-to-fine grid search, ported from
        citybehavex-web's Rust implementation (written there to avoid a
        scipy dependency in a Python-free web backend) -- an
        *approximation*, not bit-identical to the scipy fit; use it only
        when scipy is unavailable or a rougher fit is acceptable.

    Returns
    -------
    popt : np.ndarray
        Fitted parameters ``[c, r0, beta, kappa]``.
    x_data : np.ndarray
        Geometric bin centers used in the fit (all positive).
    y_data : np.ndarray
        Density values at bin centers (all positive).

    Raises
    ------
    ImportError
        When ``method="scipy"`` and scipy is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.measures.fitting.mobility_laws import fit_values_to_truncated_powerlaw
    >>> values = np.array([1, 1.2, 1.5, 2, 2.5, 3, 4, 6, 8, 13, 21, 34])
    >>> popt, x_data, y_data = fit_values_to_truncated_powerlaw(values, bins=6)
    >>> print(popt.shape)
    (4,)
    >>> print(np.round(x_data[:3], 3))
    [1.423 2.88  5.831]
    >>> print(np.round(y_data[:3], 3))
    [0.325 0.121 0.04 ]
    """
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[values_array > 0]

    bins_edges = np.logspace(np.log10(values_array.min()), np.log10(values_array.max()), num=bins)
    hist, bin_edges = np.histogram(values_array, bins=bins_edges, density=True)

    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    valid = hist > 0
    x_data = bin_centers[valid]
    y_data = hist[valid]

    if method == "grid":
        popt = _fit_truncated_powerlaw_grid(x_data, y_data)
        return popt, x_data, y_data

    if method != "scipy":
        raise ValueError(f"Unknown method {method!r}; expected 'scipy' or 'grid'.")
    if _scipy_curve_fit is None:
        raise ImportError("scipy is required for power-law fitting: pip install fastmob[fitting]")

    log_y_data = np.log(y_data)

    initial_guess = [1.0, 1.0, 1.75, 400.0]
    bounds = ([1e-5, 1e-5, 0, 1e-5], [np.inf, np.inf, np.inf, np.inf])

    popt, _pcov = _scipy_curve_fit(
        log_truncated_powerlaw,
        x_data,
        log_y_data,
        p0=initial_guess,
        bounds=bounds,
    )

    return popt, x_data, y_data
