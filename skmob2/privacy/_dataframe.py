"""DataFrame helpers for privacy risk attacks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import narwhals as nw

from ._constants import FREQUENCY, LATITUDE, LONGITUDE, PROBABILITY, UID


def _as_frame(data: Any) -> nw.DataFrame:
    return nw.from_native(data, eager_only=True)


def _records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, tuple):
        return list(data)
    return _as_frame(data).rows(named=True)


def _records_like(data: Any, columns: list[str]) -> list[dict[str, Any]]:
    if isinstance(data, list) and (not data or isinstance(data[0], dict)):
        return data
    if isinstance(data, tuple) and (not data or isinstance(data[0], dict)):
        return list(data)
    if hasattr(data, "tolist"):
        data = data.tolist()
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], dict):
            return data
        return [dict(zip(columns, row)) for row in data]
    return _records(data)


def _backend(data: Any) -> Any:
    return _as_frame(data).implementation


def _to_native(data: dict[str, list[Any]], backend: Any) -> Any:
    return nw.from_dict(data, backend=backend).to_native()


def _uid_values(data: Any) -> set[Any]:
    return {row[UID] for row in _records(data)}


def _rows_by_uid(rows: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[UID]].append(row)
    return dict(groups)


def _group_counts(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], int]:
    counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _frequency_vector(traj: Any) -> Any:
    rows = _records(traj)
    counts = _group_counts(rows, [UID, LATITUDE, LONGITUDE])
    out = {
        UID: [],
        LATITUDE: [],
        LONGITUDE: [],
        FREQUENCY: [],
    }
    for uid, lat, lng in sorted(counts, key=lambda key: (key[0], counts[key], key[1], key[2])):
        out[UID].append(uid)
        out[LATITUDE].append(lat)
        out[LONGITUDE].append(lng)
        out[FREQUENCY].append(counts[(uid, lat, lng)])
    return _to_native(out, _backend(traj))


def _probability_vector(traj: Any) -> Any:
    rows = _records(traj)
    counts = _group_counts(rows, [UID, LATITUDE, LONGITUDE])
    totals: dict[Any, int] = {}
    for uid, lat, lng in counts:
        totals[uid] = totals.get(uid, 0) + counts[(uid, lat, lng)]

    out = {
        UID: [],
        LATITUDE: [],
        LONGITUDE: [],
        PROBABILITY: [],
    }
    for uid, lat, lng in sorted(counts, key=lambda key: (key[0], counts[key] / totals[key[0]], key[1], key[2])):
        out[UID].append(uid)
        out[LATITUDE].append(lat)
        out[LONGITUDE].append(lng)
        out[PROBABILITY].append(counts[(uid, lat, lng)] / totals[uid])
    return _to_native(out, _backend(traj))


def _date_time_precision(dt: Any, precision: str) -> str:
    result = ""
    if precision in ("Year", "year"):
        result += str(dt.year)
    elif precision in ("Month", "month"):
        result += str(dt.year) + str(dt.month)
    elif precision in ("Day", "day"):
        result += str(dt.year) + str(dt.month) + str(dt.day)
    elif precision in ("Hour", "hour"):
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month)
    elif precision in ("Minute", "minute"):
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month) + str(dt.minute)
    elif precision in ("Second", "second"):
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month) + str(dt.minute) + str(dt.second)
    return result


__all__ = [
    "_as_frame",
    "_records",
    "_records_like",
    "_backend",
    "_to_native",
    "_uid_values",
    "_rows_by_uid",
    "_group_counts",
    "_frequency_vector",
    "_probability_vector",
    "_date_time_precision",
]
