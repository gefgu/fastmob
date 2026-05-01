from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import narwhals as nw
import numpy as np
from skmob2._core import cdr_approx_travel_minutes as _cdr_approx_travel_minutes
from skmob2._core import cdr_trip_indices as _cdr_trip_indices
from skmob2._core import cdr_visitation_stays as _cdr_visitation_stays

from ..measures._common import _ROW_ORDER_COL, _build_user_ranges


_VISITATION_COLUMNS = [
    "user_id",
    "area",
    "lat",
    "lon",
    "start_timestamp",
    "end_timestamp",
    "duration_minutes",
    "date",
    "day_of_week",
    "purpose",
    "main_mode",
    "weight",
]

_TRIPS_COLUMNS = [
    "user_id",
    "trip_number",
    "origin_lat",
    "origin_lon",
    "origin_area",
    "destination_lat",
    "destination_lon",
    "destination_area",
    "start_timestamp",
    "end_timestamp",
    "duration_minutes",
    "Purpose_D",
    "main_mode",
    "date",
    "day_of_week",
]


def _require_columns(df: nw.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {df.columns}.")


def _with_datetime_column(df: nw.DataFrame, column: str) -> nw.DataFrame:
    try:
        return df.with_columns(nw.col(column).cast(nw.Datetime).alias(column))
    except Exception:
        return df.with_columns(nw.col(column).str.to_datetime().alias(column))


def _timestamp_seconds(df: nw.DataFrame, column: str) -> list[float]:
    return (
        df.with_columns((nw.col(column).dt.timestamp("ms") / 1000.0).alias("__skmob2_ts_s__"))
        .get_column("__skmob2_ts_s__")
        .to_list()
    )


def _ts_to_dt(ts: float | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def _date_midnight(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day)


def _stable_codes(values: list[Any]) -> list[int]:
    codes: dict[Any, int] = {}
    out: list[int] = []
    for value in values:
        if value not in codes:
            codes[value] = len(codes)
        out.append(codes[value])
    return out


def _empty_native(columns: list[str], backend: Any) -> Any:
    return nw.from_dict({col: [] for col in columns}, backend=backend).to_native()


def _validate_travel_parameters(avg_speed_kmh: float, circuity: float) -> None:
    if not np.isfinite(avg_speed_kmh) or avg_speed_kmh <= 0.0:
        raise ValueError("avg_speed_kmh must be a positive finite value.")
    if not np.isfinite(circuity) or circuity < 0.0:
        raise ValueError("circuity must be a non-negative finite value.")


def cdr_to_visitation_df(
    trajectory_df: Any,
    user_id_column: str = "user_id",
    timestamp_column: str = "timestamp",
    venue_column: str = "venueId",
    lat_column: str = "lat",
    lon_column: str = "long",
    avg_speed_kmh: float = 50.0,
    circuity: float = 1.2,
) -> Any:
    """Convert raw CDR records into a NetMob-like visitation dataframe.

    Consecutive records at the same venue for the same user are collapsed into
    one stay.  The stay end time is estimated from approximate travel time to
    the next stay, clamped to the observed gap between venue changes.

    Parameters
    ----------
    trajectory_df:
        Raw CDR-like records in any Narwhals-compatible eager dataframe.
    user_id_column:
        Column containing user IDs.
    timestamp_column:
        Column containing observation timestamps.
    venue_column:
        Column containing venue or cell-tower IDs.
    lat_column:
        Latitude column.
    lon_column:
        Longitude column.
    avg_speed_kmh:
        Assumed average travel speed used to estimate stay end times.
    circuity:
        Multiplier applied to straight-line travel distance.

    Returns
    -------
    DataFrame
        NetMob-like visitation dataframe in the caller's original backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.preprocessing import cdr_to_visitation_df
    >>> cdr = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u1"],
    ...         "timestamp": pd.to_datetime(
    ...             ["2020-01-01 08:00", "2020-01-01 08:15", "2020-01-01 09:00", "2020-01-01 10:00"]
    ...         ),
    ...         "venueId": ["home", "home", "work", "home"],
    ...         "lat": [0.0, 0.0, 0.01, 0.0],
    ...         "long": [0.0, 0.0, 0.01, 0.0],
    ...     }
    ... )
    >>> result = cdr_to_visitation_df(cdr)
    >>> preview = result[["user_id", "area", "start_timestamp", "end_timestamp", "duration_minutes"]]
    >>> print(preview.round({"duration_minutes": 2}).to_string(index=False))
    user_id area     start_timestamp              end_timestamp  duration_minutes
         u1 home 2020-01-01 08:00:00 2020-01-01 08:57:44.132898             57.74
         u1 work 2020-01-01 09:00:00 2020-01-01 09:57:44.132898             57.74
         u1 home 2020-01-01 10:00:00                        NaT               NaN
    """
    _validate_travel_parameters(avg_speed_kmh, circuity)

    df = nw.from_native(trajectory_df, eager_only=True).with_row_index(_ROW_ORDER_COL)
    _require_columns(df, [user_id_column, timestamp_column, venue_column, lat_column, lon_column])

    df = (
        _with_datetime_column(df, timestamp_column)
        .select([user_id_column, timestamp_column, venue_column, lat_column, lon_column, _ROW_ORDER_COL])
        .drop_nulls(subset=[user_id_column, timestamp_column, venue_column, lat_column, lon_column])
        .sort(user_id_column, timestamp_column, _ROW_ORDER_COL)
        .with_columns(
            nw.col(lat_column).cast(nw.Float64),
            nw.col(lon_column).cast(nw.Float64),
        )
        .drop(_ROW_ORDER_COL)
    )

    if len(df) == 0:
        return _empty_native(_VISITATION_COLUMNS, df.implementation)

    uid_values, ranges = _build_user_ranges(df, user_id_column)
    del uid_values

    user_values = df.get_column(user_id_column).to_list()
    venues = df.get_column(venue_column).to_list()
    lats = df.get_column(lat_column).to_list()
    lons = df.get_column(lon_column).to_list()
    timestamps_s = _timestamp_seconds(df, timestamp_column)

    starts, ends, _, has_end_timestamp = _cdr_visitation_stays(
        _stable_codes(venues),
        timestamps_s,
        ranges,
    )

    if not starts:
        return _empty_native(_VISITATION_COLUMNS, df.implementation)

    stay_lats = [float(np.median(lats[start:end])) for start, end in zip(starts, ends)]
    stay_lons = [float(np.median(lons[start:end])) for start, end in zip(starts, ends)]
    travel_pair_indices = [i for i, has_end in enumerate(has_end_timestamp) if has_end]
    estimated_travel_minutes = _cdr_approx_travel_minutes(
        [stay_lats[i] for i in travel_pair_indices],
        [stay_lons[i] for i in travel_pair_indices],
        [stay_lats[i + 1] for i in travel_pair_indices],
        [stay_lons[i + 1] for i in travel_pair_indices],
        avg_speed_kmh,
        circuity,
    )
    travel_minutes_by_stay = dict(zip(travel_pair_indices, estimated_travel_minutes))

    out: dict[str, list[Any]] = {col: [] for col in _VISITATION_COLUMNS}
    for stay_idx, (start, end, has_end) in enumerate(zip(starts, ends, has_end_timestamp)):
        start_dt = _ts_to_dt(timestamps_s[start])
        if has_end:
            next_start_ts_s = timestamps_s[end]
            last_origin_ts_s = timestamps_s[end - 1]
            observed_gap_minutes = max((next_start_ts_s - last_origin_ts_s) / 60.0, 0.0)
            travel_minutes = min(travel_minutes_by_stay[stay_idx], observed_gap_minutes)
            end_ts_s = next_start_ts_s - travel_minutes * 60.0
            end_dt = _ts_to_dt(end_ts_s)
            duration = max((end_ts_s - timestamps_s[start]) / 60.0, 0.0)
        else:
            end_dt = None
            duration = float("nan")

        out["user_id"].append(user_values[start])
        out["area"].append(venues[start])
        out["lat"].append(stay_lats[stay_idx])
        out["lon"].append(stay_lons[stay_idx])
        out["start_timestamp"].append(start_dt)
        out["end_timestamp"].append(end_dt)
        out["duration_minutes"].append(duration)
        out["date"].append(_date_midnight(start_dt))
        out["day_of_week"].append(start_dt.strftime("%A").lower())
        out["purpose"].append("UNKNOWN")
        out["main_mode"].append("UNKNOWN")
        out["weight"].append(1.0)

    return nw.from_dict(out, backend=df.implementation).to_native()


def cdr_to_trips_df(
    visitation_df: Any,
    user_id_column: str = "user_id",
    avg_speed_kmh: float = 50.0,
    circuity: float = 1.2,
) -> Any:
    """Derive a CDR trips dataframe from ``cdr_to_visitation_df`` output.

    Parameters
    ----------
    visitation_df:
        Visitation dataframe produced by :func:`cdr_to_visitation_df`.
    user_id_column:
        Column containing user IDs.
    avg_speed_kmh:
        Assumed average travel speed used to estimate trip durations.
    circuity:
        Multiplier applied to straight-line travel distance.

    Returns
    -------
    DataFrame
        NetMob-like trips dataframe in the caller's original backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.preprocessing import cdr_to_trips_df, cdr_to_visitation_df
    >>> cdr = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u1"],
    ...         "timestamp": pd.to_datetime(
    ...             ["2020-01-01 08:00", "2020-01-01 08:15", "2020-01-01 09:00", "2020-01-01 10:00"]
    ...         ),
    ...         "venueId": ["home", "home", "work", "home"],
    ...         "lat": [0.0, 0.0, 0.01, 0.0],
    ...         "long": [0.0, 0.0, 0.01, 0.0],
    ...     }
    ... )
    >>> visits = cdr_to_visitation_df(cdr)
    >>> result = cdr_to_trips_df(visits)
    >>> preview = result[["user_id", "trip_number", "origin_area", "destination_area", "duration_minutes"]]
    >>> print(preview.round({"duration_minutes": 2}).to_string(index=False))
    user_id  trip_number origin_area destination_area  duration_minutes
         u1            1        home             work              2.26
         u1            2        work             home              2.26
    """
    _validate_travel_parameters(avg_speed_kmh, circuity)

    df = nw.from_native(visitation_df, eager_only=True).with_row_index(_ROW_ORDER_COL)
    _require_columns(
        df,
        [
            user_id_column,
            "area",
            "lat",
            "lon",
            "start_timestamp",
            "end_timestamp",
            "purpose",
        ],
    )

    df = _with_datetime_column(df, "start_timestamp")
    df = _with_datetime_column(df, "end_timestamp")
    df = df.sort(user_id_column, "start_timestamp", _ROW_ORDER_COL).drop(_ROW_ORDER_COL)

    if len(df) == 0:
        return _empty_native(_TRIPS_COLUMNS, df.implementation)

    _, ranges = _build_user_ranges(df, user_id_column)

    users = df.get_column(user_id_column).to_list()
    areas = df.get_column("area").to_list()
    lats = df.get_column("lat").to_list()
    lons = df.get_column("lon").to_list()
    purposes = df.get_column("purpose").to_list()
    start_timestamps_s = _timestamp_seconds(df, "start_timestamp")

    has_departure = df.select((~nw.col("end_timestamp").is_null()).alias("__has_departure__")).get_column(
        "__has_departure__"
    ).to_list()
    end_timestamps_s = (
        df.with_columns(
            nw.when(nw.col("end_timestamp").is_null())
            .then(None)
            .otherwise(nw.col("end_timestamp").cast(nw.Datetime).dt.timestamp("ms") / 1000.0)
            .alias("__skmob2_end_ts_s__")
        )
        .get_column("__skmob2_end_ts_s__")
        .to_list()
    )

    origins, destinations = _cdr_trip_indices(has_departure, ranges)
    if not origins:
        return _empty_native(_TRIPS_COLUMNS, df.implementation)

    estimated_travel_minutes = _cdr_approx_travel_minutes(
        [lats[i] for i in origins],
        [lons[i] for i in origins],
        [lats[i] for i in destinations],
        [lons[i] for i in destinations],
        avg_speed_kmh,
        circuity,
    )

    out: dict[str, list[Any]] = {col: [] for col in _TRIPS_COLUMNS}
    trip_counts: dict[Any, int] = {}
    for origin_idx, dest_idx, travel_minutes in zip(origins, destinations, estimated_travel_minutes):
        departure_s = end_timestamps_s[origin_idx]
        arrival_s = start_timestamps_s[dest_idx]
        observed_gap_minutes = max((arrival_s - departure_s) / 60.0, 0.0)
        duration = min(travel_minutes, observed_gap_minutes)
        departure_s = arrival_s - duration * 60.0
        departure_dt = _ts_to_dt(departure_s)
        user_id = users[origin_idx]
        trip_counts[user_id] = trip_counts.get(user_id, 0) + 1

        out["user_id"].append(user_id)
        out["trip_number"].append(trip_counts[user_id])
        out["origin_lat"].append(lats[origin_idx])
        out["origin_lon"].append(lons[origin_idx])
        out["origin_area"].append(areas[origin_idx])
        out["destination_lat"].append(lats[dest_idx])
        out["destination_lon"].append(lons[dest_idx])
        out["destination_area"].append(areas[dest_idx])
        out["start_timestamp"].append(departure_dt)
        out["end_timestamp"].append(_ts_to_dt(arrival_s))
        out["duration_minutes"].append(duration)
        out["Purpose_D"].append(purposes[dest_idx])
        out["main_mode"].append("UNKNOWN")
        out["date"].append(_date_midnight(departure_dt))
        out["day_of_week"].append(departure_dt.strftime("%A").lower())

    return nw.from_dict(out, backend=df.implementation).to_native()
