from __future__ import annotations

import math
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
from skmob2._core import detect_stay_locations_batch as _detect_stay_locations_batch

try:
    from skmob2._core import detect_stay_locations_batch_numpy as _detect_stay_locations_batch_numpy
except ImportError:  # pragma: no cover - fallback for older extension builds
    _detect_stay_locations_batch_numpy = None

from ..measures._common import _build_user_ranges, _detect_trajectory_columns, _prepare_trajectory


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
    - [RT2004] Ramaswamy, H. & Toyama, K. (2004) Project Lachesis: parsing and modeling location histories. In International Conference on Geographic Information Science, 106-124, http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf
    - [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), https://dl.acm.org/citation.cfm?id=2743025

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
    )

    timestamps_s = (
        df.with_columns((nw.col(datetime_col).dt.timestamp("ms") / 1000.0).alias("__ts_s__"))
        .get_column("__ts_s__")
        .to_numpy()
    )
    lats = df.get_column(lat_col).to_numpy()
    lngs = df.get_column(lng_col).to_numpy()

    uid_values, ranges = _build_user_ranges(df, uid_col)

    effective_min_speed = min_speed_kmh if min_speed_kmh is not None else math.inf

    batch_func = _detect_stay_locations_batch_numpy or _detect_stay_locations_batch
    out_lats, out_lngs, entry_times_s, leaving_times_s, user_range_indices = batch_func(
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


def _seconds_to_naive_utc(seconds: list[float]) -> np.ndarray:
    """Convert Unix seconds to naive UTC datetimes without a per-row Python loop."""
    return (
        pd.to_datetime(np.asarray(seconds, dtype="float64"), unit="s", utc=True)
        .tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )
