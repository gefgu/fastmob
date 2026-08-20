"""Work-location inference from weekday daytime observations."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import home_location_indexed, home_location_presorted
from fastmob.utils._common import (
    _as_arrow,
    _build_user_ranges_auto,
    _detect_trajectory_columns,
    _extract_hours,
    _to_native,
)


def work_location(
    traj: Any,
    *,
    start_work: int = 8,
    end_work: int = 18,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return each user's most-visited weekday daytime location.

    Uses Monday--Friday observations whose local hour lies in
    ``[start_work, end_work)``. Users without an observation in that window
    are omitted: unlike home inference, treating arbitrary visits as work is
    not a useful fallback.
    """
    if not (0 <= start_work <= 23 and 0 <= end_work <= 24 and start_work < end_work):
        raise ValueError("work-hour bounds must satisfy 0 <= start_work < end_work <= 24")

    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = df.with_columns(nw.col(lat_col).cast(nw.Float64), nw.col(lng_col).cast(nw.Float64))
    df, hours = _extract_hours(df, datetime_col)
    weekday = nw.col(datetime_col).dt.weekday()
    work = df.filter((weekday < 5) & (hours >= start_work) & (hours < end_work))
    if len(work) == 0:
        values = {lat_col: _as_arrow([]), lng_col: _as_arrow([])}
        if uid_col is not None:
            values = {uid_col: _as_arrow([]), **values}
        return _to_native(values, df)

    work, work_hours = _extract_hours(work, datetime_col)
    lats = work.get_column(lat_col).to_arrow()
    lngs = work.get_column(lng_col).to_arrow()
    hours = work_hours.to_arrow()
    # Reuse the Rust time-window mode finder with a non-wrapping daytime
    # interval. All rows are already weekday/daytime, so this also gives the
    # expected deterministic tie-breaking and Arrow fast path.
    uids, indices, ends = _build_user_ranges_auto(work, uid_col, coordinates=(lats, lngs))
    if indices is None:
        out_lat, out_lng = home_location_presorted(lats, lngs, hours, ends, float(start_work), float(end_work))
    else:
        out_lat, out_lng = home_location_indexed(lats, lngs, hours, indices, ends, float(start_work), float(end_work))
    values = {lat_col: _as_arrow(out_lat), lng_col: _as_arrow(out_lng)}
    if uid_col is not None:
        values = {uid_col: uids, **values}
    return _to_native(values, work)


work_location.__module__ = "fastmob.measures.individual"
