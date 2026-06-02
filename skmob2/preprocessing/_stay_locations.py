from __future__ import annotations

import math
from typing import Any

import narwhals as nw
import numpy as np
from skmob2._core import (
    detect_stay_locations_batch_arrow as _stay_arrow,
    detect_stay_locations_batch_indexed_arrow as _stay_indexed_arrow,
    detect_stay_locations_batch_indexed_numpy as _stay_indexed_numpy,
    detect_stay_locations_batch_numpy as _stay_numpy,
)

from ..measures._common import (
    _arrow_result_values,
    _build_time_ordered_user_ranges,
    _build_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_s,
    _is_polars_backed,
    _prepare_trajectory,
)


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
    sorted=False,
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
    sorted:
        Whether the trajectory is already sorted by user and time.

    Returns
    -------
    DataFrame
        Stop locations in the same backend as input.
        Schema: [uid_col, lat_col, lng_col, datetime_col, (leaving_datetime)]


    Examples
    --------
    >>> import pandas as pd
    >>> import skmob2
    >>> url = skmob2.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from skmob2.preprocessing import stay_locations
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
    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
    )

    timestamps_s = _extract_timestamps_s(df, datetime_col)
    lats = df.get_column(lat_col)
    lngs = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    effective_min_speed = min_speed_kmh if min_speed_kmh is not None else math.inf

    if sorted:
        uid_values, ranges = _build_user_ranges(df, uid_col)

        if use_arrow:
            _l, _g, _e, _lv, _r = _stay_arrow(
                lats.to_arrow(),
                lngs.to_arrow(),
                timestamps_s.to_arrow(),
                ranges,
                spatial_radius_km,
                minutes_for_a_stop,
                no_data_for_minutes,
                effective_min_speed,
            )
            out_lats = np.asarray(_arrow_result_values(_l))
            out_lngs = np.asarray(_arrow_result_values(_g))
            entry_times_s = np.asarray(_arrow_result_values(_e))
            leaving_times_s = np.asarray(_arrow_result_values(_lv))
            user_range_indices = np.asarray(_arrow_result_values(_r), dtype=np.uintp)
        else:
            out_lats, out_lngs, entry_times_s, leaving_times_s, user_range_indices = _stay_numpy(
                lats.to_numpy(),
                lngs.to_numpy(),
                timestamps_s.to_numpy(),
                ranges,
                spatial_radius_km,
                minutes_for_a_stop,
                no_data_for_minutes,
                effective_min_speed,
            )
    else:
        uid_values, sorted_indices, starts, ends = _build_time_ordered_user_ranges(
            df,
            uid_col,
            datetime_col=datetime_col,
            timestamps=timestamps_s,
            use_arrow=use_arrow,
        )

        if use_arrow:
            _l, _g, _e, _lv, _r = _stay_indexed_arrow(
                lats.to_arrow(),
                lngs.to_arrow(),
                timestamps_s.to_arrow(),
                sorted_indices,
                starts,
                ends,
                spatial_radius_km,
                minutes_for_a_stop,
                no_data_for_minutes,
                effective_min_speed,
            )
            out_lats = np.asarray(_arrow_result_values(_l))
            out_lngs = np.asarray(_arrow_result_values(_g))
            entry_times_s = np.asarray(_arrow_result_values(_e))
            leaving_times_s = np.asarray(_arrow_result_values(_lv))
            user_range_indices = np.asarray(_arrow_result_values(_r), dtype=np.uintp)
        else:
            out_lats, out_lngs, entry_times_s, leaving_times_s, user_range_indices = _stay_indexed_numpy(
                lats.to_numpy(),
                lngs.to_numpy(),
                timestamps_s.to_numpy(),
                sorted_indices,
                starts,
                ends,
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

    stop_uids = [uid_values[idx] for idx in user_range_indices]

    entry_datetimes = _seconds_to_naive_utc(entry_times_s)

    out_dict = {
        lat_col: out_lats,
        lng_col: out_lngs,
        datetime_col: entry_datetimes,
    }
    if uid_col is not None:
        out_dict[uid_col] = stop_uids

    if leaving_time:
        leaving_datetimes = _seconds_to_naive_utc(leaving_times_s)
        out_dict["leaving_datetime"] = leaving_datetimes

    result = nw.from_dict(out_dict, backend=df.implementation)
    return result.to_native()


stay_locations.__module__ = "skmob2.preprocessing"


def _seconds_to_naive_utc(seconds) -> np.ndarray:
    return (np.asarray(seconds, dtype="float64") * 1e9).astype("int64").view("datetime64[ns]")
