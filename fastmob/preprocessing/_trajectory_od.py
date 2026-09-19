from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from fastmob.core.base import unwrap_native
from fastmob.utils._common import (
    _as_arrow,
    _build_presorted_user_ends,
    _detect_trajectory_columns,
    _prepare_trajectory,
    _to_native,
)

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
    presorted: bool = False,
) -> Any:
    """Build a long-format Origin-Destination matrix directly from a raw trajectory.

    Tessellates each point into an H3 cell (via :func:`fastmob.preprocessing.latlng_to_h3`),
    sorts each user chronologically, and pairs each point's cell with the next
    point's cell (per user) to derive one trip per consecutive fix. Pairing and
    counting are fused into a single Rust pass (see ``od_edge_counts_presorted``)
    rather than materializing every paired trip before counting.

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
    presorted:
        When True, skip the internal chronological sort and trust that rows
        are already ordered by ``[uid_col, datetime_col]`` (or just
        ``datetime_col`` when there is no ``uid_col``). Rows are still
        cleaned (nulls dropped, coordinates cast) but not physically
        reordered. Passing already-interleaved-by-user data with this flag
        set silently produces wrong pairings -- only set it when the caller
        can guarantee the ordering.

    Returns
    -------
    DataFrame
        Long-format OD counts with columns ``["origin", "destination",
        "count"]``, sorted by ``["origin", "destination"]`` -- the same shape
        :func:`fastmob.measures.collective.od.od_matrix` returns. Callers that
        want the wide pivoted form used by some report code can
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
        sort=not presorted,
        drop_nulls=True,
    )
    df = df.filter(nw.col(lat_col).is_between(-90.0, 90.0) & nw.col(lng_col).is_between(-180.0, 180.0))
    if len(df) == 0:
        empty = nw.from_dict({"origin": [], "destination": [], "count": []}, backend=df.implementation)
        return empty.to_native()

    with_cells = latlng_to_h3(df.to_native(), resolution, lat_col=lat_col, lng_col=lng_col, output_col=_ORIGIN_COL)
    df = nw.from_native(with_cells, eager_only=True).drop_nulls(subset=[_ORIGIN_COL])

    origins, destinations, counts = _paired_od_edges(df, uid_col, drop_self_loops)
    if len(origins) == 0:
        empty = nw.from_dict({"origin": [], "destination": [], "count": []}, backend=df.implementation)
        return empty.to_native()

    return _to_native({"origin": origins, "destination": destinations, "count": counts}, df)


def _paired_od_edges(df: nw.DataFrame, uid_col: str | None, drop_self_loops: bool):
    """Pair each user's consecutive H3 cells and count unique OD edges in Rust.

    Fuses what used to be a NumPy pairing pass (see :func:`_pair_locations`)
    followed by a separate Narwhals ``group_by`` in :func:`od_matrix` into one
    pass over data already sorted and grouped by user.
    """
    from fastmob._core import od_edge_counts_presorted

    cells = df.get_column(_ORIGIN_COL).to_arrow()
    _, ends = _build_presorted_user_ends(df, uid_col)
    origins, destinations, counts = od_edge_counts_presorted(cells, ends, drop_self_loops)
    return _as_arrow(origins), _as_arrow(destinations), _as_arrow(counts)


def _pair_locations(df: nw.DataFrame, uid_col: str | None, drop_self_loops: bool):
    """Return paired H3 IDs and source/destination row indices."""
    cells = df.get_column(_ORIGIN_COL).to_numpy().astype(np.uint64)
    _, group_ends = _build_presorted_user_ends(df, uid_col)
    ends = np.asarray(group_ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    next_idx = np.full(len(cells), -1, dtype=np.int64)
    for start, end in zip(starts, ends):
        if end - start > 1:
            next_idx[start : end - 1] = np.arange(start + 1, end)

    source_idx = np.flatnonzero(next_idx >= 0)
    destination_idx = next_idx[source_idx]
    origins = cells[source_idx]
    destinations = cells[destination_idx]
    if drop_self_loops:
        keep = origins != destinations
        source_idx = source_idx[keep]
        destination_idx = destination_idx[keep]
        origins = origins[keep]
        destinations = destinations[keep]
    return origins, destinations, source_idx, destination_idx


def trajectory_to_trips(
    traj: Any,
    resolution: int = 9,
    *,
    uid_col: str | None = None,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    drop_self_loops: bool = True,
    presorted: bool = False,
) -> Any:
    """Build a :class:`fastmob.Trips` row for each consecutive ping pair.

    Each user's fixes are sorted chronologically, tessellated to H3, then
    paired with that user's next fix. Same-cell pairs are dropped by default,
    matching :func:`trajectory_to_od`. The returned Trips contain
    ``origin_location_id`` and ``destination_location_id`` for direct sparse
    CPC comparisons, alongside trip IDs and the pair's start/finish times.

    This represents a consecutive-fix movement pair, not the activity-to-
    activity trips built from triplegs and staypoints.

    ``presorted=True`` skips the internal chronological sort, trusting that
    rows are already ordered by ``[uid_col, datetime_col]`` (or just
    ``datetime_col`` when there is no ``uid_col``) -- see
    :func:`trajectory_to_od` for the full caveat.
    """
    from fastmob.core.trips_dataframe import Trips

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
        sort=not presorted,
        drop_nulls=True,
    )
    df = df.filter(nw.col(lat_col).is_between(-90.0, 90.0) & nw.col(lng_col).is_between(-180.0, 180.0))
    if len(df) == 0:
        return _empty_trajectory_trips(df)

    with_cells = latlng_to_h3(df.to_native(), resolution, lat_col=lat_col, lng_col=lng_col, output_col=_ORIGIN_COL)
    df = nw.from_native(with_cells, eager_only=True).drop_nulls(subset=[_ORIGIN_COL])
    origins, destinations, source_idx, destination_idx = _pair_locations(df, uid_col, drop_self_loops)
    if origins.size == 0:
        return _empty_trajectory_trips(df)

    timestamps = df.get_column(datetime_col).to_numpy()
    columns = {
        "trip_id": np.arange(origins.size, dtype=np.int64),
        "started_at": timestamps[source_idx],
        "finished_at": timestamps[destination_idx],
        "origin_location_id": origins,
        "destination_location_id": destinations,
    }
    trips = nw.from_dict(columns, backend=df.implementation).to_native()
    return Trips(trips)


def _empty_trajectory_trips(df: nw.DataFrame):
    from fastmob.core.trips_dataframe import Trips

    columns = {
        "trip_id": np.array([], dtype=np.int64),
        "started_at": np.array([], dtype="datetime64[us]"),
        "finished_at": np.array([], dtype="datetime64[us]"),
        "origin_location_id": np.array([], dtype=np.uint64),
        "destination_location_id": np.array([], dtype=np.uint64),
    }
    return Trips(nw.from_dict(columns, backend=df.implementation).to_native())


trajectory_to_od.__module__ = "fastmob.preprocessing"
trajectory_to_trips.__module__ = "fastmob.preprocessing"
