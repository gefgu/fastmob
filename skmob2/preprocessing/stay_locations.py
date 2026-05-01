from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import narwhals as nw
from skmob2._core import detect_stay_locations_batch as _detect_stay_locations_batch

from ..measures._common import _build_user_ranges, _prepare_trajectory


def stay_locations(
    traj: Any,
    stop_radius_factor: float = 0.5,
    minutes_for_a_stop: float = 20.0,
    spatial_radius_km: float = 0.2,
    leaving_time: bool = True,
    no_data_for_minutes: float = 1e12,
    min_speed_kmh: float | None = None,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Detect stay locations (stops) in trajectory data.

    A stop is detected when the individual stays within spatial_radius_km for
    at least minutes_for_a_stop minutes. Stop coordinates are median lat/lng.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    stop_radius_factor:
        Accepted for skmob API compatibility; not used by this implementation.
    minutes_for_a_stop:
        Minimum duration (minutes) to qualify as a stop.
    spatial_radius_km:
        Radius (km) within which points are grouped into a stop.
    leaving_time:
        If True, add a 'leaving_datetime' column with the departure time.
    no_data_for_minutes:
        Gap threshold (minutes) above which data is treated as missing.
    min_speed_kmh:
        If set, trim trailing high-speed points from the end of each stop.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.

    Returns
    -------
    DataFrame
        Stop locations in the same backend as input.
        Schema: [uid_col, lat_col, lng_col, datetime_col, (leaving_datetime)]

    References
    ----------
    - [RT2004] Ramaswamy, H. & Toyama, K. (2004) Project Lachesis: parsing and modeling location histories. In International Conference on Geographic Information Science, 106-124, http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf
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

    uid_values, ranges = _build_user_ranges(df, uid_col)

    effective_min_speed = min_speed_kmh if min_speed_kmh is not None else math.inf

    out_lats, out_lngs, entry_times_s, leaving_times_s, user_range_indices = _detect_stay_locations_batch(
        lats,
        lngs,
        timestamps_s,
        ranges,
        spatial_radius_km,
        minutes_for_a_stop,
        no_data_for_minutes,
        effective_min_speed,
    )

    if len(out_lats) == 0:
        out_dict: dict[str, list] = {lat_col: [], lng_col: [], datetime_col: []}
        if uid_col is not None:
            out_dict[uid_col] = []
        if leaving_time:
            out_dict["leaving_datetime"] = []
        return nw.from_dict(out_dict, backend=df.implementation).to_native()

    # Map user_range_idx → uid value
    stop_uids = [uid_values[idx] for idx in user_range_indices]

    # Convert Unix seconds back to naive UTC datetime objects
    def _ts_to_dt(ts: float) -> datetime:
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)

    entry_datetimes = [_ts_to_dt(ts) for ts in entry_times_s]

    out_dict = {
        lat_col: out_lats,
        lng_col: out_lngs,
        datetime_col: entry_datetimes,
    }
    if uid_col is not None:
        out_dict[uid_col] = stop_uids

    if leaving_time:
        leaving_datetimes = [_ts_to_dt(ts) for ts in leaving_times_s]
        out_dict["leaving_datetime"] = leaving_datetimes

    result = nw.from_dict(out_dict, backend=df.implementation)
    return result.to_native()
