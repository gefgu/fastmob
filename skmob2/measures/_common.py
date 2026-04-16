from __future__ import annotations

from typing import Iterable

_ROW_ORDER_COL = "__skmob2_row_order__"


def _pick_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None
