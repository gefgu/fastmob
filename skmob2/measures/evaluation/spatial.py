"""Spatial comparisons: OD matrix, radius of gyration, dwell time, and STVD-EMD."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import narwhals as nw
import numpy as np

from skmob2._core import stvd_emd_arrow as _stvd_emd_arrow
from skmob2._core import stvd_emd_numpy as _stvd_emd_numpy
from skmob2._core import (
    trajectory_common_part_of_commuters_arrow as _trajectory_cpc_arrow,
)
from skmob2._core import (
    trajectory_common_part_of_commuters_numpy as _trajectory_cpc_numpy,
)
from skmob2.measures._common import (
    DURATION_CANDIDATES,
    _as_index_array,
    _build_time_ordered_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_ms,
    _is_polars_backed,
    _pick_existing_column,
    _use_arrow_kernel_path,
    _with_datetime_column,
)

from .distribution import column_distribution_wasserstein_distance
from .metrics import wasserstein_distance

_TIME_CANDIDATES = ["time_bin", "time", "hour", "timestamp"]
_WEIGHT_CANDIDATES = ["mean_volume", "volume", "weight", "count", "density"]
_CENTROID_CANDIDATES = ["centroid", "geometry", "point"]


# ---------------------------------------------------------------------------
# OD matrix and scalar profile comparisons
# ---------------------------------------------------------------------------


def od_matrix_common_part_of_commuters(od1: Any, od2: Any) -> float:
    """Return CPC between two OD matrices, aligning labels when available."""
    if all(hasattr(obj, attr) for obj in (od1, od2) for attr in ("index", "columns", "reindex")):
        origins = od1.index.union(od2.index)
        destinations = od1.columns.union(od2.columns)
        values1 = od1.reindex(index=origins, columns=destinations, fill_value=0).values
        values2 = od2.reindex(index=origins, columns=destinations, fill_value=0).values
    else:
        values1 = np.asarray(od1, dtype=np.float64)
        values2 = np.asarray(od2, dtype=np.float64)
        if values1.shape != values2.shape:
            raise ValueError(
                f"OD matrix shapes must match when labels are unavailable. Got {values1.shape} and {values2.shape}."
            )
    min_flows = float(np.minimum(values1, values2).sum())
    total_flows = float(np.asarray(values1).sum() + np.asarray(values2).sum())
    return 0.0 if total_flows == 0.0 else float(2.0 * min_flows / total_flows)


def _trajectory_input(
    traj: Any,
    *,
    datetime_col: str | None,
    lat_col: str | None,
    lng_col: str | None,
    uid_col: str | None,
) -> tuple[nw.DataFrame, str, str, str, str | None]:
    native = getattr(traj, "df", traj)
    datetime_col = datetime_col or getattr(traj, "datetime_col", None)
    lat_col = lat_col or getattr(traj, "lat_col", None)
    lng_col = lng_col or getattr(traj, "lng_col", None)
    uid_col = uid_col or getattr(traj, "uid_col", None)

    df = nw.from_native(native, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    df = _with_datetime_column(df, datetime_col)
    schema = df.schema
    if schema[lat_col] != nw.Float64 or schema[lng_col] != nw.Float64:
        df = df.with_columns(
            nw.col(lat_col).cast(nw.Float64),
            nw.col(lng_col).cast(nw.Float64),
        )

    required = [datetime_col, lat_col, lng_col]
    if uid_col is not None:
        required.append(uid_col)
    df = df.drop_nulls(subset=required)
    return df, datetime_col, lat_col, lng_col, uid_col


def _trajectory_cpc_inputs(
    traj: Any,
    *,
    datetime_col: str | None,
    lat_col: str | None,
    lng_col: str | None,
    uid_col: str | None,
) -> tuple[nw.DataFrame, str, str, Any, Any, bool]:
    df, datetime_col, lat_col, lng_col, uid_col = _trajectory_input(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    timestamps = _extract_timestamps_ms(df, datetime_col)
    use_arrow = _use_arrow_kernel_path(df)
    timestamp_data = timestamps.to_arrow() if use_arrow else timestamps.to_numpy()
    _, indices, ends = _build_time_ordered_user_ranges(
        df, uid_col, datetime_col, timestamp_data
    )
    return (
        df,
        lat_col,
        lng_col,
        _as_index_array(indices),
        _as_index_array(ends),
        use_arrow,
    )


def trajectory_common_part_of_commuters(
    traj_a: Any,
    traj_b: Any,
    resolution: int = 9,
    *,
    datetime_col_a: str | None = None,
    lat_col_a: str | None = None,
    lng_col_a: str | None = None,
    uid_col_a: str | None = None,
    datetime_col_b: str | None = None,
    lat_col_b: str | None = None,
    lng_col_b: str | None = None,
    uid_col_b: str | None = None,
) -> float:
    """Compute trajectory CPC from sparse H3 OD flows without materialising an OD matrix.

    Rows are ordered by user and datetime. Invalid coordinates, null required
    fields, missing destinations, and self-loops are excluded before comparing
    OD edge counts.
    """
    if not 0 <= int(resolution) <= 15:
        raise ValueError(f"H3 resolution must be between 0 and 15, got {resolution}")

    df_a, lat_a, lng_a, indices_a, ends_a, use_arrow_a = _trajectory_cpc_inputs(
        traj_a,
        datetime_col=datetime_col_a,
        lat_col=lat_col_a,
        lng_col=lng_col_a,
        uid_col=uid_col_a,
    )
    df_b, lat_b, lng_b, indices_b, ends_b, use_arrow_b = _trajectory_cpc_inputs(
        traj_b,
        datetime_col=datetime_col_b,
        lat_col=lat_col_b,
        lng_col=lng_col_b,
        uid_col=uid_col_b,
    )

    if use_arrow_a and use_arrow_b:
        return float(
            _trajectory_cpc_arrow(
                df_a.get_column(lat_a).to_arrow(),
                df_a.get_column(lng_a).to_arrow(),
                indices_a,
                ends_a,
                df_b.get_column(lat_b).to_arrow(),
                df_b.get_column(lng_b).to_arrow(),
                indices_b,
                ends_b,
                int(resolution),
            )
        )

    return float(
        _trajectory_cpc_numpy(
            df_a.get_column(lat_a).to_numpy(),
            df_a.get_column(lng_a).to_numpy(),
            indices_a,
            ends_a,
            df_b.get_column(lat_b).to_numpy(),
            df_b.get_column(lng_b).to_numpy(),
            indices_b,
            ends_b,
            int(resolution),
        )
    )


def trajectory_common_part_of_commuters_multi(
    traj_a: Any,
    traj_b: Any,
    resolutions: Sequence[int] = (7, 8, 9),
    *,
    datetime_col_a: str | None = None,
    lat_col_a: str | None = None,
    lng_col_a: str | None = None,
    uid_col_a: str | None = None,
    datetime_col_b: str | None = None,
    lat_col_b: str | None = None,
    lng_col_b: str | None = None,
    uid_col_b: str | None = None,
) -> list[tuple[int, float]]:
    """Compute trajectory CPC at multiple H3 resolutions, preparing each
    trajectory's inputs only once instead of once per resolution.

    Equivalent to calling :func:`trajectory_common_part_of_commuters` once per
    resolution, but the dataframe conversion, column detection, datetime
    casting, null-dropping, and time-ordered user-range construction done by
    ``_trajectory_cpc_inputs`` are resolution-independent and were otherwise
    being redone for every resolution -- tripling cost (or more) on large
    trajectories for no benefit, since only the final H3-binning kernel call
    actually depends on ``resolution``.
    """
    for resolution in resolutions:
        if not 0 <= int(resolution) <= 15:
            raise ValueError(f"H3 resolution must be between 0 and 15, got {resolution}")

    df_a, lat_a, lng_a, indices_a, ends_a, use_arrow_a = _trajectory_cpc_inputs(
        traj_a,
        datetime_col=datetime_col_a,
        lat_col=lat_col_a,
        lng_col=lng_col_a,
        uid_col=uid_col_a,
    )
    df_b, lat_b, lng_b, indices_b, ends_b, use_arrow_b = _trajectory_cpc_inputs(
        traj_b,
        datetime_col=datetime_col_b,
        lat_col=lat_col_b,
        lng_col=lng_col_b,
        uid_col=uid_col_b,
    )

    use_arrow = use_arrow_a and use_arrow_b
    if use_arrow:
        lats_a = df_a.get_column(lat_a).to_arrow()
        lngs_a = df_a.get_column(lng_a).to_arrow()
        lats_b = df_b.get_column(lat_b).to_arrow()
        lngs_b = df_b.get_column(lng_b).to_arrow()
        kernel = _trajectory_cpc_arrow
    else:
        lats_a = df_a.get_column(lat_a).to_numpy()
        lngs_a = df_a.get_column(lng_a).to_numpy()
        lats_b = df_b.get_column(lat_b).to_numpy()
        lngs_b = df_b.get_column(lng_b).to_numpy()
        kernel = _trajectory_cpc_numpy

    return [
        (
            int(resolution),
            float(
                kernel(
                    lats_a, lngs_a, indices_a, ends_a,
                    lats_b, lngs_b, indices_b, ends_b,
                    int(resolution),
                )
            ),
        )
        for resolution in resolutions
    ]


def profile_metric_wasserstein_distance(df1: Any, df2: Any, metric_col: str) -> float:
    """Compare a scalar profile metric column with Rust-backed Wasserstein distance."""
    n1 = nw.from_native(df1, eager_only=True)
    n2 = nw.from_native(df2, eager_only=True)
    if metric_col not in n1.columns or metric_col not in n2.columns:
        raise ValueError(f"Column {metric_col!r} must be present in both dataframes.")
    return wasserstein_distance(n1.get_column(metric_col).to_numpy(), n2.get_column(metric_col).to_numpy())


def radius_of_gyration_wasserstein_distance(
    df1: Any,
    df2: Any,
    radius_col: str = "radius_of_gyration_km",
    grouping_col: str | None = None,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare radius-of-gyration distributions, optionally grouped by a column."""
    n1 = nw.from_native(df1, eager_only=True)
    if grouping_col is None or grouping_col not in n1.columns:
        value = profile_metric_wasserstein_distance(df1, df2, radius_col)
        return value, [("Overall", value)] if not np.isnan(value) else []
    return column_distribution_wasserstein_distance(df1, df2, radius_col, hue="purpose", purpose_col=grouping_col)


def dwell_time_wasserstein_distance(
    df1: Any,
    df2: Any,
    duration_col: str | None = None,
    *,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare dwell-time distributions in hours."""
    n1 = nw.from_native(df1, eager_only=True)
    duration_col = duration_col or _pick_existing_column(n1.columns, DURATION_CANDIDATES) or "duration_minutes"

    def add_hours(data: Any) -> Any:
        native = nw.from_native(data, eager_only=True)
        return native.with_columns((nw.col(duration_col) / 60.0).alias("__skmob2_dwell_hours__")).to_native()

    return column_distribution_wasserstein_distance(
        add_hours(df1),
        add_hours(df2),
        "__skmob2_dwell_hours__",
        hue=hue,
        day_col1=day_col1,
        day_col2=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=True,
    )


# ---------------------------------------------------------------------------
# Spatio-temporal Wasserstein distance (STVD-EMD)
# ---------------------------------------------------------------------------


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

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.evaluation import stvd_emd
    >>> dist_a = pd.DataFrame(
    ...     {
    ...         "time_bin": ["08:00", "08:10"],
    ...         "mean_volume": [2.0, 1.0],
    ...         "centroid": ["POINT (0 0)", "POINT (100 0)"],
    ...     }
    ... )
    >>> dist_b = pd.DataFrame(
    ...     {
    ...         "time_bin": ["08:00", "08:10"],
    ...         "mean_volume": [1.0, 2.0],
    ...         "centroid": ["POINT (0 0)", "POINT (200 0)"],
    ...     }
    ... )
    >>> print(round(stvd_emd(dist_a, dist_b, num_projections=10), 3))
    19.052
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
        t_col = df.get_column(time_col)
        times = t_col.str.slice(0, 2).cast(nw.Float64) * 60 + t_col.str.slice(3, 5).cast(nw.Float64)

        c_col = df.get_column(centroid_col)
        native_series = c_col.to_native()

        if hasattr(native_series, "x") and hasattr(native_series, "y"):
            xs = nw.from_native(native_series.x, series_only=True).cast(nw.Float64)
            ys = nw.from_native(native_series.y, series_only=True).cast(nw.Float64)
        else:
            c_first = c_col[0] if len(c_col) > 0 else None
            if c_first is not None and hasattr(c_first, "x") and hasattr(c_first, "y"):
                xy = [_point_xy(v) for v in c_col.to_list()]
                xs = nw.new_series("x", [p[0] for p in xy], backend=df.implementation)
                ys = nw.new_series("y", [p[1] for p in xy], backend=df.implementation)
            else:
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
