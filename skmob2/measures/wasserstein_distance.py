"""Spatio-temporal Wasserstein distance between two spatial distributions."""
from __future__ import annotations

from typing import Any

import narwhals as nw

from skmob2._core import wasserstein_emd

_TIME_CANDIDATES = ["time_bin", "time", "hour", "timestamp"]
_WEIGHT_CANDIDATES = ["mean_volume", "volume", "weight", "count", "density"]
_CENTROID_CANDIDATES = ["centroid", "geometry", "point"]


def _detect_column(columns: list[str], candidates: list[str], role: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find a {role!r} column. "
        f"Tried: {candidates}. Available: {columns}"
    )


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


def wasserstein_distance(
    dist_a: Any,
    dist_b: Any,
    alpha: float = 10.0,
    cyclical_period: float = 1440.0,
    reg: float = 0.01,
    max_iter: int = 1000,
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

    The cost between two cells is::

        sqrt((dx_m)^2 + (alpha * dt_cyclic_min)^2)

    where ``dx_m`` is the Euclidean distance in metres between centroids and
    ``dt_cyclic_min`` is the cyclical time difference in minutes
    (``min(|t1-t2|, cyclical_period - |t1-t2|)``).

    Parameters
    ----------
    dist_a, dist_b:
        Any Narwhals-compatible eager DataFrame (pandas, polars, …).
    alpha:
        Space-time tradeoff: 1 minute equals ``alpha`` metres. Default ``10.0``.
    cyclical_period:
        Wrap-around period in minutes. Default ``1440`` (one day).
    reg:
        Sinkhorn regularisation parameter (must be > 0). Smaller values give a
        more exact result at the cost of convergence speed. Default ``0.01``.
    max_iter:
        Maximum Sinkhorn iterations. Default ``1000``.
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
    if reg <= 0:
        raise ValueError(f"reg must be positive, got {reg}")

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

    xy_a = [_point_xy(v) for v in nw_a.get_column(cc).to_list()]
    xy_b = [_point_xy(v) for v in nw_b.get_column(cc).to_list()]
    xs_a = [p[0] for p in xy_a]
    ys_a = [p[1] for p in xy_a]
    xs_b = [p[0] for p in xy_b]
    ys_b = [p[1] for p in xy_b]

    times_a = [_hhmm_to_minutes(v) for v in nw_a.get_column(tc).to_list()]
    times_b = [_hhmm_to_minutes(v) for v in nw_b.get_column(tc).to_list()]

    weights_a = nw_a.get_column(wc).cast(nw.Float64).to_list()
    weights_b = nw_b.get_column(wc).cast(nw.Float64).to_list()

    return wasserstein_emd(
        xs_a,
        ys_a,
        times_a,
        weights_a,
        xs_b,
        ys_b,
        times_b,
        weights_b,
        alpha,
        cyclical_period,
        reg,
        max_iter,
    )
