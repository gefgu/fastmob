"""Private shared utilities for the comparison submodule."""

from __future__ import annotations

from typing import Any

import narwhals as nw


def _is_null(value: Any) -> bool:
    try:
        return bool(value is None or value != value)
    except TypeError:
        return value is None


def _series_values(df: nw.DataFrame, column: str) -> list[Any]:
    return df.get_column(column).to_list()
