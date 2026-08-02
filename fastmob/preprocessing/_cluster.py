from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import narwhals as nw

from fastmob._core import h3_cluster_labels_arrow as _h3_cluster_labels
from fastmob.utils._common import _as_arrow, _build_presorted_user_ends, _detect_trajectory_columns, _prepare_trajectory

# Calibrated around H3's average cell edge lengths.  The automatic path keeps
# the long-standing kilometre-radius API useful; callers needing exact grid
# control can supply ``h3_resolution`` directly.
_RADIUS_RESOLUTION = (
    (0.025, 12),
    (0.05, 11),
    (0.1, 10),
    (0.2, 9),
    (0.5, 8),
    (1.0, 7),
)


def _auto_resolution(cluster_radius_km: float) -> int:
    if not math.isfinite(cluster_radius_km) or cluster_radius_km <= 0:
        raise ValueError("cluster_radius_km must be a finite positive number")
    log_radius = math.log(cluster_radius_km)
    # Prefer the finer grid for an exact midpoint tie: accidental merging is
    # more surprising than splitting a boundary-adjacent location.
    return min(_RADIUS_RESOLUTION, key=lambda item: (abs(math.log(item[0]) - log_radius), -item[1]))[1]


def _resolve_resolution(cluster_radius_km: float, h3_resolution: int | None) -> int:
    if h3_resolution is None:
        return _auto_resolution(cluster_radius_km)
    if isinstance(h3_resolution, bool) or not isinstance(h3_resolution, Integral):
        raise ValueError("h3_resolution must be an integer between 0 and 15")
    if not 0 <= int(h3_resolution) <= 15:
        raise ValueError("h3_resolution must be an integer between 0 and 15")
    return int(h3_resolution)


def cluster(
    traj: Any,
    cluster_radius_km: float = 0.1,
    min_samples: int = 1,
    *,
    h3_resolution: int | None = None,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted=False,
    n_jobs: int | None = None,
) -> Any:
    """Cluster stop locations with H3 cells and connected components.

    Points are assigned to an H3 cell. Cells containing fewer than
    ``min_samples`` points are noise; active cells touching at an edge are
    joined into a cluster. ``cluster_radius_km`` selects a calibrated grid
    resolution for compatibility with the previous DBSCAN API. Pass
    ``h3_resolution`` (0--15) to control the grid explicitly; it takes
    precedence over ``cluster_radius_km``. ``n_jobs`` is accepted for API
    compatibility and has no effect.
    """
    if isinstance(min_samples, bool) or not isinstance(min_samples, Integral) or min_samples < 1:
        raise ValueError("min_samples must be an integer of at least 1")
    resolution = _resolve_resolution(cluster_radius_km, h3_resolution)

    df = nw.from_native(traj, eager_only=True)
    df, datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        cast_float_coordinates=True,
    )
    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=not presorted,
    )

    _, group_ends = _build_presorted_user_ends(df, uid_col)
    all_labels = _as_arrow(
        _h3_cluster_labels(
            df.get_column(lat_col).to_arrow(),
            df.get_column(lng_col).to_arrow(),
            group_ends,
            resolution,
            int(min_samples),
        )
    )

    try:
        cluster_series = nw.new_series(name="cluster", values=all_labels, backend=df.implementation)
    except Exception:  # noqa: BLE001
        cluster_series = nw.from_dict({"cluster": all_labels}, backend=df.implementation).get_column("cluster")
    return df.with_columns(cluster_series).to_native()


cluster.__module__ = "fastmob.preprocessing"
