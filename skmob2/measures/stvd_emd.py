"""Spatio-temporal Wasserstein distance between two spatial distributions."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from skmob2._core import stvd_emd_arrow as _stvd_emd_arrow
from skmob2._core import stvd_emd_numpy as _stvd_emd_numpy
from ._common import _is_polars_backed

_TIME_CANDIDATES = ["time_bin", "time", "hour", "timestamp"]
_WEIGHT_CANDIDATES = ["mean_volume", "volume", "weight", "count", "density"]
_CENTROID_CANDIDATES = ["centroid", "geometry", "point"]


def _detect_column(columns: list[str], candidates: list[str], role: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Could not find a {role!r} column. Tried: {candidates}. Available: {columns}")


def _point_xy(val: Any) -> tuple[float, float]:
    """Extract (x, y) from a Shapely-like object or WKT string 'POINT (x y)'."""
    if hasattr(val, "x") and hasattr(val, "y"):
        return float(val.x), float(val.y)
    s = str(val).strip()
    inner = s[s.index("(") + 1 : s.rindex(")")]
    x_str, y_str = inner.split()
    return float(x_str), float(y_str)


def _hhmm_to_minutes(val: Any) -> float:
    h, m = str(val).split(":", 1)
    return float(h) * 60 + float(m)


def _route_and_call(
    arrays_a: tuple[nw.Series, nw.Series, nw.Series, nw.Series],
    arrays_b: tuple[nw.Series, nw.Series, nw.Series, nw.Series],
    alpha: float,
    cyclical_period: float,
    num_projections: int,
    *,
    use_arrow: bool,
) -> float:
    if use_arrow:
        args = [x.to_arrow() for x in (*arrays_a, *arrays_b)]
        return _stvd_emd_arrow(*args, alpha, cyclical_period, num_projections)

    args = [x.to_numpy() for x in (*arrays_a, *arrays_b)]
    return _stvd_emd_numpy(*args, alpha, cyclical_period, num_projections)


def stvd_emd(
    dist_a: Any,
    dist_b: Any,
    alpha: float = 10.0,
    cyclical_period: float = 1440.0,
    num_projections: int = 50,
    *,
    time_col: str | None = None,
    weight_col: str | None = None,
    centroid_col: str | None = None,
) -> float:
    """Compute the spatio-temporal Wasserstein distance between two distributions.

    Each input is a DataFrame representing a spatial-temporal distribution with
    columns for a time bin (HH:MM string), a weight/volume value, and a centroid
    geometry in a projected coordinate system.  Centroids may be WKT strings
    (``'POINT (x y)'``) or objects with ``.x`` / ``.y`` attributes.

    Distances are computed using sliced Wasserstein over a 4D embedding
    ``(x, y, r*cos(theta), r*sin(theta))`` where
    ``r = alpha * cyclical_period / (2*pi)`` and
    ``theta = 2*pi*(t / cyclical_period)``.

    Parameters
    ----------
    dist_a, dist_b:
        Any Narwhals-compatible eager DataFrame (pandas, polars, …).
    alpha:
        Space-time tradeoff: 1 minute equals ``alpha`` metres. Default ``10.0``.
    cyclical_period:
        Wrap-around period in minutes. Default ``1440`` (one day).
    num_projections:
        Number of random projections used by sliced Wasserstein. Must be > 0.
        Larger values improve stability at higher computational cost.
    time_col:
        Explicit time column name. Auto-detected when ``None``.
    weight_col:
        Explicit weight column name. Auto-detected when ``None``.
    centroid_col:
        Explicit centroid column name. Auto-detected when ``None``.

    Returns
    -------
    float
        Spatio-temporal Wasserstein distance in metres.
    """
    if num_projections <= 0:
        raise ValueError(f"num_projections must be positive, got {num_projections}")

    nw_a = nw.from_native(dist_a, eager_only=True)
    nw_b = nw.from_native(dist_b, eager_only=True)

    cols_a = nw_a.columns
    cols_b = nw_b.columns

    tc = time_col or _detect_column(cols_a, _TIME_CANDIDATES, "time")
    wc = weight_col or _detect_column(cols_a, _WEIGHT_CANDIDATES, "weight")
    cc = centroid_col or _detect_column(cols_a, _CENTROID_CANDIDATES, "centroid")

    for col, role in [(tc, "time"), (wc, "weight"), (cc, "centroid")]:
        if col not in cols_b:
            raise ValueError(f"Column {col!r} ({role}) not found in dist_b")

    def prepare_arrays(
        df: nw.DataFrame,
        time_col: str,
        centroid_col: str,
        weight_col: str,
    ) -> tuple[nw.Series, nw.Series, nw.Series, nw.Series]:
        # Parse HH:MM in-vector to reduce Python-level per-row overhead.
        t_col = df.get_column(time_col)
        times = t_col.str.slice(0, 2).cast(nw.Float64) * 60 + t_col.str.slice(3, 5).cast(nw.Float64)

        c_col = df.get_column(centroid_col)
        native_series = c_col.to_native()

        # GeoPandas GeoSeries exposes vectorized x/y in native code.
        if hasattr(native_series, "x") and hasattr(native_series, "y"):
            xs = nw.from_native(native_series.x, series_only=True).cast(nw.Float64)
            ys = nw.from_native(native_series.y, series_only=True).cast(nw.Float64)
        else:
            c_first = c_col[0] if len(c_col) > 0 else None
            if c_first is not None and hasattr(c_first, "x") and hasattr(c_first, "y"):
                # Compatibility fallback for object series of point-like values.
                xy = [_point_xy(v) for v in c_col.to_list()]
                xs = nw.new_series("x", [p[0] for p in xy], backend=df.implementation)
                ys = nw.new_series("y", [p[1] for p in xy], backend=df.implementation)
            else:
                # WKT fallback. Keep parsing explicit because Narwhals string
                # replacement is regex-backed on some pandas Arrow strings.
                xy = [_point_xy(v) for v in c_col.to_list()]
                xs = nw.new_series("x", [p[0] for p in xy], backend=df.implementation)
                ys = nw.new_series("y", [p[1] for p in xy], backend=df.implementation)

        weights = df.get_column(weight_col).cast(nw.Float64)

        return xs, ys, times, weights

    arrays_a = prepare_arrays(nw_a, tc, cc, wc)
    arrays_b = prepare_arrays(nw_b, tc, cc, wc)

    use_arrow = _is_polars_backed(nw_a)

    return _route_and_call(
        arrays_a,
        arrays_b,
        alpha,
        cyclical_period,
        num_projections,
        use_arrow=use_arrow,
    )
