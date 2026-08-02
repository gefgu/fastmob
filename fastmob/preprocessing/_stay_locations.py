from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from fastmob._core import detect_stay_locations_batch_indexed
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _as_arrow,
    _build_indexed_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps,
    _take_uid_values,
    _timestamps_ms_to_datetime_ns,
    _to_native,
)


def _unwrap_stay(_l: Any, _g: Any, _e: Any, _lv: Any, _r: Any) -> tuple:
    return (
        _as_arrow(_l),
        _as_arrow(_g),
        _as_arrow(_e),
        _as_arrow(_lv),
        _r,
    )


# Bare extractor used only for `.get_ops(df)["extract_data"]` on the
# timestamps column, which also feeds `_build_indexed_user_ranges` below
# (Rule 1: never inline backend branching). lat/lng go to Arrow unconditionally.
_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def stay_locations(
    traj: Any,
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
    presorted=False,
) -> Any:
    """Detect stay locations (stops) in trajectory data.

    A stop is detected when the individual stays within spatial_radius_km for
    at least minutes_for_a_stop minutes. Stop coordinates are median lat/lng.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
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
    presorted:
        Whether the trajectory is already sorted by user and time.

    Returns
    -------
    DataFrame
        Stop locations in the same backend as input.
        Schema: [uid_col, lat_col, lng_col, datetime_col, (leaving_datetime)]


    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.utils.constants.BRIGHTKITE_SAMPLE
    >>> df = pd.read_csv(
    ...     url,
    ...     sep="\\t",
    ...     header=0,
    ...     nrows=5000,
    ...     names=["uid", "datetime", "lat", "lng", "location id"],
    ... )
    >>> df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    >>> df["location_id"] = df["location id"].astype("string")
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng", "location_id"]
    ... ]
    >>> print(df.head().to_string(index=False))
     uid                  datetime       lat         lng                              location_id
       0 2010-10-16 06:02:04+00:00 39.891383 -105.070814         7a0f88982aa015062b95e3b4843f9ca2
       0 2010-10-16 03:48:54+00:00 39.891077 -105.068532         dd7cd3d264c2d063832db506fba8bf79
       0 2010-10-14 18:25:51+00:00 39.750469 -104.999073 9848afcc62e500a01cf6fbf24b797732f8963683
       0 2010-10-14 00:21:47+00:00 39.752713 -104.996337         2ef143e12038c870038df53e0478cefc
       0 2010-10-13 23:31:51+00:00 39.752508 -104.996637         424eb3dd143292f9e013efa00486c907
    >>> from fastmob.preprocessing import stay_locations
    >>> stops = stay_locations(df, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    >>> print(len(stops))
    3029
    >>> print(stops.head().to_string(index=False))
          lat         lng            datetime  uid    leaving_datetime
    37.774929 -122.419415 2009-05-25 20:56:10    0 2009-05-25 21:35:28
    37.600747 -122.382376 2009-05-25 21:35:28    0 2009-05-25 22:13:23
    37.615223 -122.389979 2009-05-25 22:13:23    0 2009-05-26 02:21:12
    39.878664 -104.682105 2009-05-26 02:21:12    0 2009-05-26 04:59:44
    39.739154 -104.984703 2009-05-26 04:59:44    0 2009-05-26 16:43:59

    References
    ----------
    - [RT2004] Ramaswamy, H. & Toyama, K. (2004) Project Lachesis: parsing and modeling location histories. In International Conference on Geographic Information Science, 106-124, <a href="http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf">http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf</a>
    - [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), <a href="https://dl.acm.org/citation.cfm?id=2743025">https://dl.acm.org/citation.cfm?id=2743025</a>

    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = df.with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )

    timestamps = _extract_timestamps(df, datetime_col)
    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()
    timestamps_data = _TIMESTAMP_EXTRACTOR.get_ops(df)["extract_data"](timestamps)

    effective_min_speed = min_speed_kmh if min_speed_kmh is not None else math.inf

    if presorted:
        uid_values, sorted_indices, ends = _build_indexed_user_ranges(df, uid_col)
        _result = detect_stay_locations_batch_indexed(
            lats_data,
            lngs_data,
            timestamps_data,
            sorted_indices,
            ends,
            spatial_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            effective_min_speed,
        )
        out_lats, out_lngs, entry_times_ms, leaving_times_ms, user_range_indices = _unwrap_stay(*_result)
    else:
        uid_values, sorted_indices, ends = _build_indexed_user_ranges(
            df,
            uid_col,
            timestamps=timestamps,
        )
        _result = detect_stay_locations_batch_indexed(
            lats_data,
            lngs_data,
            timestamps_data,
            sorted_indices,
            ends,
            spatial_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            effective_min_speed,
        )
        out_lats, out_lngs, entry_times_ms, leaving_times_ms, user_range_indices = _unwrap_stay(*_result)

    if len(out_lats) == 0:
        out_dict: dict[str, list] = {lat_col: [], lng_col: [], datetime_col: []}
        if uid_col is not None:
            out_dict[uid_col] = []
        if leaving_time:
            out_dict["leaving_datetime"] = []
        return nw.from_dict(out_dict, backend=df.implementation).to_native()

    entry_datetimes = _timestamps_ms_to_datetime_ns(entry_times_ms)

    out_dict = {
        lat_col: out_lats,
        lng_col: out_lngs,
        datetime_col: entry_datetimes,
    }
    if uid_col is not None:
        out_dict[uid_col] = _take_uid_values(uid_values, user_range_indices)

    if leaving_time:
        leaving_datetimes = _timestamps_ms_to_datetime_ns(leaving_times_ms)
        out_dict["leaving_datetime"] = leaving_datetimes

    return _to_native(out_dict, df)


stay_locations.__module__ = "fastmob.preprocessing"
