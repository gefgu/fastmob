from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
import numpy as np

from fastmob.utils._common import _build_presorted_user_ends, _detect_trajectory_columns, _prepare_trajectory

from ..measures.collective.od import od_matrix
from ._h3 import latlng_to_h3

_ORIGIN_COL = "__od_origin_cell__"
_DESTINATION_COL = "__od_destination_cell__"


def trajectory_to_od(
    traj: Any,
    resolution: int = 9,
    *,
    uid_col: str | None = None,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    drop_self_loops: bool = True,
) -> Any:
    """Build a long-format Origin-Destination matrix directly from a raw trajectory.

    Tessellates each point into an H3 cell (via :func:`fastmob.preprocessing.latlng_to_h3`),
    sorts each user chronologically, and pairs each point's cell with the next
    point's cell (per user) to derive one trip per consecutive fix. Delegates
    the final counting to :func:`fastmob.measures.collective.od.od_matrix`.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    resolution:
        H3 resolution, 0-15.
    uid_col, datetime_col, lat_col, lng_col:
        Explicit column name overrides; auto-detected when None. When
        ``uid_col`` cannot be found, the whole frame is treated as one user.
    drop_self_loops:
        When True (default), consecutive fixes that land in the same cell
        are not counted as a trip -- matches how "did the user actually move
        somewhere" trip derivation is normally defined. Set False to also
        count same-cell "stayed put" pairs (as `MoveInside` traffic).

    Returns
    -------
    DataFrame
        Long-format OD counts with columns ``["origin", "destination",
        "count"]``, matching :func:`od_matrix`'s contract. Callers that want
        the wide pivoted form used by some report code can
        ``.pivot(on="destination", index="origin", values="count")`` this
        themselves.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.preprocessing import trajectory_to_od
    >>> traj = pd.DataFrame(
    ...     {
    ...         "uid": ["u1", "u1", "u1"],
    ...         "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]),
    ...         "lat": [37.7793, 37.6007, 37.6007],
    ...         "lng": [-122.4194, -122.3824, -122.3824],
    ...     }
    ... )
    >>> result = trajectory_to_od(traj, resolution=7)
    >>> len(result)
    1
    """
    df = nw.from_native(unwrap_native(traj), eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=True,
        drop_nulls=True,
    )
    df = df.filter(nw.col(lat_col).is_between(-90.0, 90.0) & nw.col(lng_col).is_between(-180.0, 180.0))
    if len(df) == 0:
        empty = nw.from_dict({"origin": [], "destination": [], "count": []}, backend=df.implementation)
        return empty.to_native()

    with_cells = latlng_to_h3(df.to_native(), resolution, lat_col=lat_col, lng_col=lng_col, output_col=_ORIGIN_COL)
    df = nw.from_native(with_cells, eager_only=True).drop_nulls(subset=[_ORIGIN_COL])

    # Deriving destination via a plain `.shift(-1)` risks a real pandas
    # precision bug: shifting introduces a NaN for each group's last row,
    # which upcasts a uint64 column to float64 -- but H3 cell indices are
    # ~60-bit values, well past float64's 53-bit exact-integer range, so the
    # shifted destination values silently corrupt (observed: off by 1 on a
    # real cell index). Building the origin/destination pairing from raw
    # per-user row ranges instead keeps everything in integer arithmetic.
    cells = df.get_column(_ORIGIN_COL).to_numpy().astype(np.uint64)
    _, group_ends = _build_presorted_user_ends(df, uid_col)
    ends = np.asarray(group_ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    next_idx = np.full(len(cells), -1, dtype=np.int64)
    for start, end in zip(starts, ends):
        if end - start > 1:
            next_idx[start : end - 1] = np.arange(start + 1, end)

    has_next = next_idx >= 0
    origins = cells[has_next]
    destinations = cells[next_idx[has_next]]

    if drop_self_loops:
        keep = origins != destinations
        origins = origins[keep]
        destinations = destinations[keep]

    if origins.size == 0:
        empty = nw.from_dict({"origin": [], "destination": [], "count": []}, backend=df.implementation)
        return empty.to_native()

    trips_native = nw.from_dict(
        {_ORIGIN_COL: origins, _DESTINATION_COL: destinations}, backend=df.implementation
    ).to_native()
    result = od_matrix(trips_native, origin_col=_ORIGIN_COL, destination_col=_DESTINATION_COL)
    result_nw = nw.from_native(result, eager_only=True).rename({_ORIGIN_COL: "origin", _DESTINATION_COL: "destination"})
    return result_nw.to_native()


trajectory_to_od.__module__ = "fastmob.preprocessing"
