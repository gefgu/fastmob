from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import filter_trajectory_batch as _filter_trajectory_batch

from ..measures._common import _build_user_ranges, _prepare_trajectory


def filter(
    traj: Any,
    max_speed_kmh: float = 500.0,
    include_loops: bool = False,
    speed_kmh: float = 5.0,
    max_loop: int = 6,
    ratio_max: float = 0.25,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Filter trajectory noise by removing high-speed outlier points.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    max_speed_kmh:
        Remove points where speed from the previous point exceeds this threshold.
    include_loops:
        If True, also remove points forming short fast return loops.
    speed_kmh:
        Minimum loop speed threshold (km/h); only used when include_loops=True.
    max_loop:
        Maximum number of points to look ahead for loop detection.
    ratio_max:
        Distance ratio threshold for loop detection.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.

    Returns
    -------
    DataFrame
        Filtered trajectory in the same backend as input.

    References
    ----------
    - [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), https://dl.acm.org/citation.cfm?id=2743025

    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    timestamps_s: list[float] = (
        df.with_columns((nw.col(datetime_col).dt.timestamp("ms") / 1000.0).alias("__ts_s__"))
        .get_column("__ts_s__")
        .to_list()
    )
    lats: list[float] = df.get_column(lat_col).to_list()
    lngs: list[float] = df.get_column(lng_col).to_list()

    _, ranges = _build_user_ranges(df, uid_col)

    mask: list[bool] = _filter_trajectory_batch(
        lats,
        lngs,
        timestamps_s,
        ranges,
        max_speed_kmh,
        include_loops,
        speed_kmh,
        max_loop,
        ratio_max,
    )

    keep_indices = [i for i, k in enumerate(mask) if k]
    return df[keep_indices].to_native()
