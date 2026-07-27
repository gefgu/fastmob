"""Private shared utilities for the comparison submodule."""

from __future__ import annotations

import math
from typing import Any

import narwhals as nw


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _series_values(df: nw.DataFrame, column: str) -> list[Any]:
    return df.get_column(column).to_list()
