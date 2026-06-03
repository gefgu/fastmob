"""Private temporal helpers for hue-based grouping."""

from __future__ import annotations

from typing import Any

from ._utils import _is_null

_DAY_PERIOD_COLUMN = "day_period"
_WEEKDAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_PERIOD_ORDER = [
    "Morning (06:00-11:59)",
    "Afternoon (12:00-17:59)",
    "Evening (18:00-23:59)",
    "Night (00:00-05:59)",
]


def _hour_from_value(value: Any) -> int | None:
    if _is_null(value):
        return None
    if hasattr(value, "hour"):
        return int(value.hour)
    text = str(value)
    try:
        if ":" in text:
            return int(text.split(":", 1)[0][-2:])
        import numpy as np

        return int(np.datetime64(text, "h").astype(object).hour)
    except Exception:
        return None


def _day_period(value: Any) -> str | None:
    hour = _hour_from_value(value)
    if hour is None:
        return None
    if 6 <= hour < 12:
        return _DAY_PERIOD_ORDER[0]
    if 12 <= hour < 18:
        return _DAY_PERIOD_ORDER[1]
    if 18 <= hour < 24:
        return _DAY_PERIOD_ORDER[2]
    return _DAY_PERIOD_ORDER[3]
