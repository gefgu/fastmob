"""Spatial comparisons: spatio-temporal Wasserstein distance (STVD-EMD)."""

from __future__ import annotations

from typing import Any

import narwhals as nw

try:
    from fastmob._core import stvd_emd as _stvd_emd
except ImportError:  # Built without the optional SIMD/wass backend.
    _stvd_emd = None

_TIME_CANDIDATES = ["time_bin", "time", "hour", "timestamp"]
_WEIGHT_CANDIDATES = ["mean_volume", "volume", "weight", "count", "density"]
_LAT_CANDIDATES = ["center_lat", "lat", "latitude"]
_LNG_CANDIDATES = ["center_lng", "lng", "lon", "longitude"]


def _detect_column(columns: list[str], candidates: list[str], role: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Could not find a {role!r} column. Tried: {candidates}. Available: {columns}")


def _route_and_call(
    arrays_a: tuple[nw.Series, nw.Series, nw.Series, nw.Series],
    arrays_b: tuple[nw.Series, nw.Series, nw.Series, nw.Series],
    alpha: float,
    cyclical_period: float,
) -> float:
    if _stvd_emd is None:
        raise ImportError("stvd_emd requires fastmob to be built with the optional stvd-emd feature")

    args = [x.to_arrow() for x in (*arrays_a, *arrays_b)]
    return _stvd_emd(*args, alpha, cyclical_period)


def stvd_emd(
    dist_a: Any,
    dist_b: Any,
    alpha: float = 10.0,
    cyclical_period: float = 1440.0,
    *,
    time_col: str | None = None,
    weight_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
) -> float:
    """Compute the spatio-temporal Wasserstein distance between two distributions.

    Each input is a DataFrame representing a spatial-temporal distribution with
    columns for a time bin (HH:MM string), a weight/volume value, and a WGS84
    latitude/longitude centroid — the exact shape produced by
    :func:`fastmob.measures.collective.build_stvd`.

    Distances are computed as an entropy-regularised (Sinkhorn) Earth Mover's
    Distance over an explicit cost matrix combining the great-circle
    (Haversine) distance between centroids with a cyclical-time term: two
    points at times ``t1``/``t2`` contribute ``2 * r * sin(d_theta / 2)``
    metres, where ``r = alpha * cyclical_period / (2*pi)``,
    ``theta = 2*pi*(t / cyclical_period)``, and ``d_theta`` is the shortest
    angular difference between the two times.

    Parameters
    ----------
    dist_a, dist_b:
        Any Narwhals-compatible eager DataFrame (pandas, polars, …).
    alpha:
        Space-time tradeoff: 1 minute equals ``alpha`` metres. Default ``10.0``.
    cyclical_period:
        Wrap-around period in minutes. Default ``1440`` (one day).
    time_col:
        Explicit time column name. Auto-detected when ``None``.
    weight_col:
        Explicit weight column name. Auto-detected when ``None``.
    lat_col, lng_col:
        Explicit latitude/longitude column names. Auto-detected when ``None``.

    Returns
    -------
    float
        Spatio-temporal Wasserstein distance in metres.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.evaluation import stvd_emd
    >>> dist_a = pd.DataFrame(
    ...     {
    ...         "time_bin": ["08:00", "08:10"],
    ...         "mean_volume": [2.0, 1.0],
    ...         "center_lat": [0.0, 0.0],
    ...         "center_lng": [0.0, 0.001],
    ...     }
    ... )
    >>> dist_b = pd.DataFrame(
    ...     {
    ...         "time_bin": ["08:00", "08:10"],
    ...         "mean_volume": [1.0, 2.0],
    ...         "center_lat": [0.0, 0.0],
    ...         "center_lng": [0.0, 0.002],
    ...     }
    ... )
    >>> print(round(stvd_emd(dist_a, dist_b), 3))
    74.156
    """
    nw_a = nw.from_native(dist_a, eager_only=True)
    nw_b = nw.from_native(dist_b, eager_only=True)

    cols_a = nw_a.columns
    cols_b = nw_b.columns

    tc = time_col or _detect_column(cols_a, _TIME_CANDIDATES, "time")
    wc = weight_col or _detect_column(cols_a, _WEIGHT_CANDIDATES, "weight")
    lac = lat_col or _detect_column(cols_a, _LAT_CANDIDATES, "latitude")
    lnc = lng_col or _detect_column(cols_a, _LNG_CANDIDATES, "longitude")

    for col, role in [(tc, "time"), (wc, "weight"), (lac, "latitude"), (lnc, "longitude")]:
        if col not in cols_b:
            raise ValueError(f"Column {col!r} ({role}) not found in dist_b")

    def prepare_arrays(
        df: nw.DataFrame,
        time_col: str,
        lat_col: str,
        lng_col: str,
        weight_col: str,
    ) -> tuple[nw.Series, nw.Series, nw.Series, nw.Series]:
        t_col = df.get_column(time_col)
        times = t_col.str.slice(0, 2).cast(nw.Float64) * 60 + t_col.str.slice(3, 5).cast(nw.Float64)
        lats = df.get_column(lat_col).cast(nw.Float64)
        lngs = df.get_column(lng_col).cast(nw.Float64)
        weights = df.get_column(weight_col).cast(nw.Float64)

        return lats, lngs, times, weights

    arrays_a = prepare_arrays(nw_a, tc, lac, lnc, wc)
    arrays_b = prepare_arrays(nw_b, tc, lac, lnc, wc)

    return _route_and_call(
        arrays_a,
        arrays_b,
        alpha,
        cyclical_period,
    )
