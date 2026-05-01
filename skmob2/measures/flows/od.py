"""Origin-Destination matrix measures."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _pick_existing_column, ORIGIN_CANDIDATES, DEST_CANDIDATES


def _detect_od_columns(
    nw_df: nw.DataFrame,
    origin_col: str | None,
    destination_col: str | None,
) -> tuple[str, str]:
    if origin_col is None:
        origin_col = _pick_existing_column(nw_df.columns, ORIGIN_CANDIDATES)
    if destination_col is None:
        destination_col = _pick_existing_column(nw_df.columns, DEST_CANDIDATES)

    if origin_col is None or destination_col is None:
        missing = []
        if origin_col is None:
            missing.append(f"origin (looked for: {ORIGIN_CANDIDATES})")
        if destination_col is None:
            missing.append(f"destination (looked for: {DEST_CANDIDATES})")
        raise ValueError(
            f"Could not auto-detect column(s): {', '.join(missing)}. "
            f"Available columns: {nw_df.columns}. "
            f"Pass the column name(s) explicitly."
        )

    return origin_col, destination_col


def od_matrix(
    trips: Any,
    origin_col: str | None = None,
    destination_col: str | None = None,
) -> Any:
    """Compute an Origin-Destination matrix from a trips DataFrame.

    Groups trips by (origin, destination) pairs and counts them. Returns a
    long-format DataFrame with one row per observed origin-destination pair,
    in the same backend as the input.

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
    DataFrame
        Long-format OD counts with columns
        ``[origin_col, destination_col, "count"]``,
        returned in the caller's original backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.flows import od_matrix
    >>> trips = pd.DataFrame(
    ...     {
    ...         "origin_area": ["A", "A", "A", "B", "B", "C"],
    ...         "destination_area": ["A", "B", "B", "A", "C", "A"],
    ...     }
    ... )
    >>> result = od_matrix(trips)
    >>> print(result.to_string(index=False))
    origin_area destination_area  count
              A                A      1
              A                B      2
              B                A      1
              B                C      1
              C                A      1
    """
    nw_df = nw.from_native(trips, eager_only=True)
    origin_col, destination_col = _detect_od_columns(nw_df, origin_col, destination_col)

    counts = (
        nw_df.drop_nulls(subset=[origin_col, destination_col])
        .select([origin_col, destination_col])
        .group_by([origin_col, destination_col])
        .agg(nw.len().alias("count"))
        .sort([origin_col, destination_col])
    )

    return counts.to_native()


def od_metrics_per_area(
    od_df: Any,
    origin_col: str | None = None,
    destination_col: str | None = None,
) -> Any:
    """Compute per-area mobility metrics from a long-format OD DataFrame.

    Takes the long-format OD DataFrame returned by :func:`od_matrix` and
    computes MoveInside, InComing, OutGoing, and Total flows for each area.

    Parameters
    ----------
    od_df:
        Long-format OD DataFrame as returned by :func:`od_matrix`, with
        columns ``[origin_col, destination_col, "count"]``.
    origin_col:
        Column name for the origin area. Auto-detected if None.
    destination_col:
        Column name for the destination area. Auto-detected if None.

    Returns
    -------
    DataFrame
        DataFrame with columns ``["area_code", "MoveInside", "InComing",
        "OutGoing", "Total"]``, one row per area, returned in the caller's
        original backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.flows import od_matrix, od_metrics_per_area
    >>> trips = pd.DataFrame(
    ...     {
    ...         "origin_area": ["A", "A", "A", "B", "B", "C"],
    ...         "destination_area": ["A", "B", "B", "A", "C", "A"],
    ...     }
    ... )
    >>> od = od_matrix(trips)
    >>> result = od_metrics_per_area(od)
    >>> print(result.to_string(index=False))
    area_code  MoveInside  InComing  OutGoing  Total
            A           1         2         2      5
            B           0         2         2      4
            C           0         1         1      2
    """
    nw_df = nw.from_native(od_df, eager_only=True)
    origin_col, destination_col = _detect_od_columns(nw_df, origin_col, destination_col)

    all_areas = (
        nw.concat(
            [
                nw_df.select(nw.col(origin_col).alias("area_code")),
                nw_df.select(nw.col(destination_col).alias("area_code")),
            ]
        )
        .unique()
        .sort("area_code")
    )

    self_loops = (
        nw_df.filter(nw.col(origin_col) == nw.col(destination_col))
        .group_by(origin_col)
        .agg(nw.col("count").sum().alias("MoveInside"))
        .rename({origin_col: "area_code"})
    )

    cross_trips = nw_df.filter(nw.col(origin_col) != nw.col(destination_col))

    outgoing = (
        cross_trips.group_by(origin_col).agg(nw.col("count").sum().alias("OutGoing")).rename({origin_col: "area_code"})
    )

    incoming = (
        cross_trips.group_by(destination_col)
        .agg(nw.col("count").sum().alias("InComing"))
        .rename({destination_col: "area_code"})
    )

    result = (
        all_areas.join(self_loops, on="area_code", how="left")
        .join(outgoing, on="area_code", how="left")
        .join(incoming, on="area_code", how="left")
        .with_columns(
            nw.col("MoveInside").fill_null(0).cast(nw.Int64),
            nw.col("OutGoing").fill_null(0).cast(nw.Int64),
            nw.col("InComing").fill_null(0).cast(nw.Int64),
        )
        .with_columns((nw.col("MoveInside") + nw.col("InComing") + nw.col("OutGoing")).alias("Total"))
        .select(["area_code", "MoveInside", "InComing", "OutGoing", "Total"])
        .sort("area_code")
    )

    return result.to_native()
