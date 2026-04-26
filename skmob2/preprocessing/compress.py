from __future__ import annotations

from typing import Any

import numpy as np
import narwhals as nw
from skmob2._core import compress_trajectory_batch as _compress_trajectory_batch

from ..measures._common import _build_user_ranges, _prepare_trajectory


def compress(
    traj: Any,
    spatial_radius_km: float = 0.2,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Compress trajectory by collapsing nearby points into single representative points.

    All points within spatial_radius_km from an initial point are collapsed into
    one point with median lat/lng and the initial point's timestamp.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    spatial_radius_km:
        Minimum distance (km) between consecutive output points.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.

    Returns
    -------
    DataFrame
        Compressed trajectory in the same backend as input.
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    lats: list[float] = df.get_column(lat_col).to_list()
    lngs: list[float] = df.get_column(lng_col).to_list()

    _, ranges = _build_user_ranges(df, uid_col)

    groups: list[tuple[int, int]] = _compress_trajectory_batch(lats, lngs, ranges, spatial_radius_km)

    # Build output rows: for each group (start, end), compute median lat/lng
    # and take datetime + extra columns from the first row of the group.
    all_columns = df.columns
    extra_cols = [c for c in all_columns if c not in (lat_col, lng_col, datetime_col)]

    out_rows: dict[str, list] = {lat_col: [], lng_col: [], datetime_col: []}
    for col in extra_cols:
        out_rows[col] = []

    # Pre-extract all columns as Python lists for fast access
    col_lists: dict[str, list] = {col: df.get_column(col).to_list() for col in all_columns}

    for s, e in groups:
        group_lats = lats[s:e]
        group_lngs = lngs[s:e]
        out_rows[lat_col].append(float(np.median(group_lats)))
        out_rows[lng_col].append(float(np.median(group_lngs)))
        out_rows[datetime_col].append(col_lists[datetime_col][s])
        for col in extra_cols:
            out_rows[col].append(col_lists[col][s])

    result = nw.from_dict(out_rows, backend=df.implementation)
    # Preserve original column order
    result = result.select(all_columns)
    return result.to_native()
