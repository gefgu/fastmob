"""PyMove-style trajectory/event spatiotemporal join.

Mirrors ``pymove.utils.integration.join_with_events``: augments a
trajectory dataframe with columns describing the nearest event within a
time window of each trajectory point, rather than aggregating the
trajectory into a per-user summary like the measures under
``fastmob.measures``.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from fastmob.utils._common import (
    DATETIME_CANDIDATES,
    LAT_CANDIDATES,
    LNG_CANDIDATES,
    _extract_timestamps_s,
    _pick_existing_column,
)


def _detect_lat_lng_datetime(
    df: nw.DataFrame, lat_col: str | None, lng_col: str | None, datetime_col: str | None
) -> tuple[str, str, str]:
    if lat_col is None:
        lat_col = _pick_existing_column(df.columns, LAT_CANDIDATES)
    if lng_col is None:
        lng_col = _pick_existing_column(df.columns, LNG_CANDIDATES)
    if datetime_col is None:
        datetime_col = _pick_existing_column(df.columns, DATETIME_CANDIDATES)
    missing = [
        name for name, col in [("latitude", lat_col), ("longitude", lng_col), ("datetime", datetime_col)] if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not detect required column(s): {missing}. Available columns: {df.columns}. "
            "Pass the column name(s) explicitly."
        )
    return lat_col, lng_col, datetime_col


def join_with_events(
    traj: Any,
    events_df: Any,
    *,
    lat_col: str | None = None,
    lng_col: str | None = None,
    datetime_col: str | None = None,
    event_lat_col: str = "lat",
    event_lng_col: str = "lng",
    event_datetime_col: str = "datetime",
    event_id_col: str = "event_id",
    event_type_col: str = "event_type",
    time_window_s: float = 900.0,
) -> Any:
    """Join each trajectory point with the nearest event within a time window.

    Mirrors PyMove's ``join_with_events``: among events whose timestamp
    falls within ``[t - time_window_s, t + time_window_s]`` of a
    trajectory point's own timestamp ``t``, picks the *spatially* nearest
    one (not the temporally nearest one) and adds ``event_id``,
    ``event_type``, ``dist_event`` columns.

    A trajectory point with no event in its window gets a real null for
    ``event_id``/``event_type`` and ``inf`` for ``dist_event`` (fastmob's
    own null-handling convention, unlike PyMove's untyped `NaN`/`inf`
    sentinel columns).

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    events_df:
        Events; any Narwhals-compatible eager backend, with
        `event_lat_col`/`event_lng_col`/`event_datetime_col` columns and,
        unless overridden, `event_id_col`/`event_type_col` columns.
    lat_col, lng_col, datetime_col:
        Explicit trajectory column overrides; auto-detected when None.
    time_window_s:
        Symmetric time window (seconds) around each trajectory point's
        timestamp to search for a matching event.

    Returns
    -------
    DataFrame
        `traj`'s columns plus ``event_id``, ``event_type``, ``dist_event``,
        in the same backend as `traj`.
    """
    from fastmob._core import nearest_event_within_window

    df = nw.from_native(traj, eager_only=True)
    lat_col, lng_col, datetime_col = _detect_lat_lng_datetime(df, lat_col, lng_col, datetime_col)

    query_lat = df.get_column(lat_col).to_numpy().astype(np.float64)
    query_lng = df.get_column(lng_col).to_numpy().astype(np.float64)
    query_time_i64 = np.rint(_extract_timestamps_s(df, datetime_col).to_numpy()).astype(np.int64)

    events = nw.from_native(events_df, eager_only=True)
    event_lat = events.get_column(event_lat_col).to_numpy().astype(np.float64)
    event_lng = events.get_column(event_lng_col).to_numpy().astype(np.float64)
    event_id = events.get_column(event_id_col).to_numpy()
    event_type = events.get_column(event_type_col).to_numpy()

    n_events = len(event_lat)
    n_query = len(query_lat)
    if n_events == 0:
        nearest_idx = np.full(n_query, -1, dtype=np.int64)
        dist_m = np.full(n_query, np.inf, dtype=np.float64)
    else:
        event_time_i64 = np.rint(_extract_timestamps_s(events, event_datetime_col).to_numpy()).astype(np.int64)
        sort_order = np.argsort(event_time_i64, kind="stable")
        nearest_idx, dist_m = nearest_event_within_window(
            query_lat,
            query_lng,
            query_time_i64,
            event_lat[sort_order],
            event_lng[sort_order],
            event_time_i64[sort_order],
            sort_order.astype(np.int64),
            round(time_window_s),
        )
        nearest_idx = np.asarray(nearest_idx)
        dist_m = np.asarray(dist_m, dtype=np.float64)

    matched = nearest_idx >= 0
    safe_idx = np.clip(nearest_idx, 0, None)
    if n_events == 0:
        out_event_id = np.full(n_query, None, dtype=object)
        out_event_type = np.full(n_query, None, dtype=object)
    else:
        out_event_id = np.where(matched, event_id[safe_idx], None)
        out_event_type = np.where(matched, event_type[safe_idx], None)

    result = df.with_columns(
        nw.new_series("event_id", out_event_id, backend=df.implementation),
        nw.new_series("event_type", out_event_type, backend=df.implementation),
        nw.new_series("dist_event", dist_m, backend=df.implementation),
    )
    return result.to_native()


join_with_events.__module__ = "fastmob.integration"
