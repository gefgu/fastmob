"""Origin-Destination matrix measures."""
from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd

from ._common import _pick_existing_column

# Candidate column names for auto-detection
_ORIGIN_CANDIDATES = ["origin_area", "Origin_Area", "area_o", "ORIGIN_AREA"]
_DEST_CANDIDATES = ["destination_area", "Dest_Area", "area_d", "DESTINATION_AREA"]


def od_matrix(
    trips: Any,
    origin_col: str | None = None,
    destination_col: str | None = None,
) -> pd.DataFrame:
    """Compute an Origin-Destination matrix from a trips DataFrame.

    Groups trips by (origin, destination) pairs and counts them. The result
    is a wide-format pandas DataFrame with origins as index, destinations as
    columns, and counts as values (missing pairs filled with 0).

    Parameters
    ----------
    trips:
        A DataFrame (pandas, polars, or any Narwhals-compatible backend) with
        at least two columns representing origin and destination areas.
    origin_col:
        Column name for the origin area. Auto-detected if None.
    destination_col:
        Column name for the destination area. Auto-detected if None.

    Returns
    -------
    pd.DataFrame
        Wide-format OD matrix (plain pandas DataFrame).
    """
    nw_df = nw.from_native(trips, eager_only=True)

    if origin_col is None:
        origin_col = _pick_existing_column(nw_df.columns, _ORIGIN_CANDIDATES)
    if destination_col is None:
        destination_col = _pick_existing_column(nw_df.columns, _DEST_CANDIDATES)

    if origin_col is None or destination_col is None:
        missing = []
        if origin_col is None:
            missing.append("origin")
        if destination_col is None:
            missing.append("destination")
        raise ValueError(
            f"Could not auto-detect columns: {', '.join(missing)}. "
            f"Available columns: {nw_df.columns}"
        )

    # Drop rows where either column is null
    filtered = nw_df.drop_nulls(subset=[origin_col, destination_col])

    # Group by (origin, destination) and count
    counts = (
        filtered.select([origin_col, destination_col])
        .group_by([origin_col, destination_col])
        .agg(nw.len().alias("count"))
    )

    # Convert to pandas for the pivot
    counts_pd = counts.to_native() if isinstance(counts.to_native(), pd.DataFrame) else pd.DataFrame(counts.to_native())

    # Pivot to wide format
    od = counts_pd.pivot_table(
        index=origin_col,
        columns=destination_col,
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    od.index.name = None
    od.columns.name = None

    return od


def od_metrics_per_area(
    od_df: pd.DataFrame,
    origin_col: str = "origin_area",
    dest_col: str = "destination_area",
) -> pd.DataFrame:
    """Compute per-area mobility metrics from a wide OD matrix.

    Takes the wide OD matrix returned by :func:`od_matrix` and computes
    MoveInside, InComing, OutGoing, and Total flows for each area.

    Parameters
    ----------
    od_df:
        Wide OD matrix as returned by :func:`od_matrix` (origin as index,
        destination as columns, counts as values).
    origin_col:
        Unused — kept for API compatibility with mobility_analysis adapters.
    dest_col:
        Unused — kept for API compatibility with mobility_analysis adapters.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["area_code", "MoveInside", "InComing",
        "OutGoing", "Total"]``, one row per area.
    """
    areas = sorted(set(od_df.index.tolist()) | set(od_df.columns.tolist()))
    records = []

    for area in areas:
        # MoveInside: diagonal (self-loop count), 0 if area not in both axis
        move_inside = 0
        if area in od_df.index and area in od_df.columns:
            move_inside = int(od_df.loc[area, area])

        # OutGoing: sum of row for this area, excluding the diagonal
        if area in od_df.index:
            row = od_df.loc[area]
            out_going = int(row.sum()) - (int(row[area]) if area in row.index else 0)
        else:
            out_going = 0

        # InComing: sum of column for this area, excluding the diagonal
        if area in od_df.columns:
            col = od_df[area]
            in_coming = int(col.sum()) - (int(col[area]) if area in col.index else 0)
        else:
            in_coming = 0

        total = move_inside + in_coming + out_going
        records.append({
            "area_code": area,
            "MoveInside": move_inside,
            "InComing": in_coming,
            "OutGoing": out_going,
            "Total": total,
        })

    return pd.DataFrame(records, columns=["area_code", "MoveInside", "InComing", "OutGoing", "Total"])
