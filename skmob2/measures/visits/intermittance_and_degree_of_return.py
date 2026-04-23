from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import narwhals as nw
import math

from .._common import (
    _pick_existing_column,
    USER_ID_CANDIDATES,
    LOCATION_CANDIDATES,
)

def intermittance_and_degree_of_return(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    cold_start_strategy: Literal["frequency", "suffix", "none"] = "frequency", 
    known_suffixes: tuple[str, ...] = ("_HOME", "_WORK"),
) -> pd.DataFrame:
    """Compute intermittancy and degree of return per user using vectorized operations.

    For each user, partitions the visit sequence into alternating blocks of
    *explorations* (new places) and *returns* (revisits or home/work visits).
    Then computes summary statistics over those blocks.

    Parameters
    ----------
    visits:
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col:
        Column name for the user ID. Auto-detected if None.
    location_id_col:
        Column name for the location ID. Auto-detected if None.
    cold_start_strategy:
        How to initialize "known" places. 
        - "frequency": Uses the paper's 90% max-frequency threshold per user.
        - "suffix": Flags locations ending in `known_suffixes` as known.
        - "none": No cold-start; the first visit to ANY place is an exploration.
    known_suffixes:
        Used only if cold_start_strategy is "suffix".

    Returns
    -------
    pd.DataFrame
        One row per user with columns
        ``[user_id_col, "intermittency", "degree_of_return", "mean_return",
        "mean_exploration"]``.
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)

    # 1. Build the unique location key
    location_expr = nw.col(location_id_col).cast(nw.String).fill_null("unknown")
    nw_df = nw_df.with_columns(location_key=location_expr)

    # 2. Assign an absolute sequential row index to preserve the temporal order natively
    nw_df = (
        nw_df.with_columns(_dummy=nw.lit(1))
        .with_columns(_row_idx=nw.col("_dummy").cum_sum())
        .drop("_dummy")
    )

    # 3. Find the global first occurrence of each location per user
    first_visits = nw_df.group_by([user_id_col, "location_key"]).agg(
        nw.col("_row_idx").min().alias("_first_idx")
    )
    nw_df = nw_df.join(first_visits, on=[user_id_col, "location_key"])

    # 4. State tracking: Apply Cold-Start Strategy
    if cold_start_strategy == "frequency":
        # Count total visits per user per location
        loc_counts = nw_df.group_by([user_id_col, "location_key"]).agg(
            nw.col("location_key").count().alias("_visit_count")
        )
        
        # Find the maximum visitation frequency per user (l_max)
        max_counts = loc_counts.group_by(user_id_col).agg(
            nw.col("_visit_count").max().alias("_max_count")
        )
        
        # Join max counts and filter for locations >= 90% of l_max
        loc_counts = loc_counts.join(max_counts, on=user_id_col)
        cold_start_locs = loc_counts.filter(
            nw.col("_visit_count") >= (nw.col("_max_count") * 0.9)
        ).with_columns(_is_cold_start=nw.lit(True))

        # Join the cold-start flags back to the main DataFrame
        nw_df = nw_df.join(
            cold_start_locs.select(user_id_col, "location_key", "_is_cold_start"),
            on=[user_id_col, "location_key"],
            how="left"
        )
        is_known_expr = nw.col("_is_cold_start").fill_null(False)

    elif cold_start_strategy == "suffix":
        # Create a chained OR expression to check if location_key ends with any of the suffixes
        is_known_expr = nw.lit(False)
        for suffix in known_suffixes:
            is_known_expr = is_known_expr | nw.col("location_key").str.ends_with(suffix)
            
    else:
        is_known_expr = nw.lit(False)

    # 5. A place is known if its current index > its first seen index, OR if it triggers the cold start
    nw_df = nw_df.with_columns(
        is_known=(nw.col("_row_idx") > nw.col("_first_idx")) | is_known_expr
    )

    # 6. Sort chronologically per user to isolate sequential blocks
    nw_df = nw_df.sort([user_id_col, "_row_idx"])

    # 7. Detect state alternations (Exploration <-> Return) using rolling shifts
    nw_df = nw_df.with_columns(
        is_known_shifted=nw.col("is_known").shift(1),
        user_id_shifted=nw.col(user_id_col).shift(1)
    )

    nw_df = nw_df.with_columns(
        block_change=(
            (nw.col("is_known") != nw.col("is_known_shifted")) |
            (nw.col(user_id_col) != nw.col("user_id_shifted"))
        ).fill_null(True).cast(nw.Int32)
    )

    # Generate isolated block IDs
    nw_df = nw_df.with_columns(block_id=nw.col("block_change").cum_sum())

    # 8. Aggregate block volumes (counting pure visits as per the paper, discarding duration)
    blocks = nw_df.group_by([user_id_col, "block_id", "is_known"]).agg(
        nw.col("block_id").count().alias("block_length")
    )

    # 9. Compute the mean sequential lengths (#U and #R)
    explorations = blocks.filter(~nw.col("is_known")).group_by(user_id_col).agg(
        nw.col("block_length").mean().alias("mean_exploration")
    )
    returns = blocks.filter(nw.col("is_known")).group_by(user_id_col).agg(
        nw.col("block_length").mean().alias("mean_return")
    )

    # 10. Merge aggregations per unique user and zero-fill null edge cases
    users = nw_df.select(user_id_col).unique()
    res = users.join(explorations, on=user_id_col, how="left").join(returns, on=user_id_col, how="left")

    res = res.with_columns(
        nw.col("mean_exploration").fill_null(0.0),
        nw.col("mean_return").fill_null(0.0)
    )

    # 11. Intermittency calculation directly in Narwhals
    res = res.with_columns(
        intermittency=(nw.col("mean_exploration") + nw.col("mean_return")).alias("intermittency")
    )

    # 12. Convert to a pure Python dictionary of lists
    data_dict = res.to_dict(as_series=False)

    # 13. Compute degree_of_return using standard Python math
    data_dict["degree_of_return"] = [
        math.atan2(r, e) for r, e in zip(data_dict["mean_return"], data_dict["mean_exploration"])
    ]

    # 14. Reconstruct the Narwhals DataFrame, preserving the original backend
    final_nw_df = nw.from_dict(
        data_dict, 
        native_namespace=nw.get_native_namespace(res)
    )

    # Return final structured DataFrame safely to its native type
    return final_nw_df.select(
        user_id_col, 
        "intermittency", 
        "degree_of_return", 
        "mean_return", 
        "mean_exploration"
    ).to_native()