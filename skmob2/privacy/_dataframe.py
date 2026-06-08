"""DataFrame helpers for privacy risk attacks."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from ._constants import FREQUENCY, LATITUDE, LONGITUDE, PROBABILITY, UID


def _as_frame(data: Any) -> nw.DataFrame:
    return nw.from_native(data, eager_only=True)


def _backend(data: Any) -> Any:
    return _as_frame(data).implementation


def _to_native(data: dict[str, list[Any]], backend: Any) -> Any:
    return nw.from_dict(data, backend=backend).to_native()


def _frequency_vector(traj: Any) -> Any:
    return (
        _as_frame(traj)
        .group_by([UID, LATITUDE, LONGITUDE])
        .agg(nw.len().alias(FREQUENCY))
        .sort([UID, FREQUENCY, LATITUDE, LONGITUDE])
        .to_native()
    )


def _probability_vector(traj: Any) -> Any:
    counts = (
        _as_frame(traj)
        .group_by([UID, LATITUDE, LONGITUDE])
        .agg(nw.len().alias(FREQUENCY))
    )
    return (
        counts.with_columns((nw.col(FREQUENCY) / nw.col(FREQUENCY).sum().over(UID)).alias(PROBABILITY))
        .drop(FREQUENCY)
        .sort([UID, PROBABILITY, LATITUDE, LONGITUDE])
        .to_native()
    )


def _datetime_precision_format(precision: str) -> str:
    normalized = precision.lower()
    formats = {
        "year": "%Y",
        "month": "%Y%m",
        "day": "%Y%m%d",
        "hour": "%Y%m%d%H",
        "minute": "%Y%m%d%H%M",
        "second": "%Y%m%d%H%M%S",
    }
    return formats[normalized]


def _with_date_time_precision(data: Any, datetime_col: str, output_col: str, precision: str) -> nw.DataFrame:
    return _as_frame(data).with_columns(nw.col(datetime_col).dt.to_string(_datetime_precision_format(precision)).alias(output_col))


def _date_time_precision(dt: Any, precision: str) -> str:
    result = ""
    if precision in ("Year", "year"):
        result += str(dt.year)
    elif precision in ("Month", "month"):
        result += str(dt.year) + str(dt.month)
    elif precision in ("Day", "day"):
        result += str(dt.year) + str(dt.month) + str(dt.day)
    elif precision in ("Hour", "hour"):
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour)
    elif precision in ("Minute", "minute"):
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour) + str(dt.minute)
    elif precision in ("Second", "second"):
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour) + str(dt.minute) + str(dt.second)
    return result


__all__ = [
    "_as_frame",
    "_backend",
    "_to_native",
    "_frequency_vector",
    "_probability_vector",
    "_with_date_time_precision",
    "_date_time_precision",
]
