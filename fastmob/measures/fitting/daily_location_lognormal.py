"""Lognormal fit for the number of locations visited per day."""

from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from fastmob.utils._common import (
    LOCATION_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _detect_required_column,
)


def daily_location_lognormal_fit(
    visits: Any,
    *,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    timestamp_col: str | None = None,
) -> tuple[list[float], list[float], float, float]:
    """Fit a lognormal distribution to daily distinct-location counts.

    For each user and calendar day, count distinct visited locations and fit
    ``mu`` and ``sigma`` to the natural log of those counts.
    """
    df = nw.from_native(visits, eager_only=True)
    user_id_col = _detect_required_column(df, user_id_col, USER_ID_CANDIDATES)
    location_id_col = _detect_required_column(df, location_id_col, LOCATION_CANDIDATES)
    timestamp_col = _detect_required_column(df, timestamp_col, TIMESTAMP_CANDIDATES)

    daily = (
        df.with_columns(nw.col(timestamp_col).dt.truncate("1d").alias("__day__"))
        .group_by([user_id_col, "__day__"])
        .agg(nw.col(location_id_col).n_unique().alias("__count__"))
        .filter(nw.col("__count__") > 0)
    )

    counts_df = (
        daily.group_by("__count__")
        .agg(nw.len().alias("freq"))
        .sort("__count__")
    )

    # Convert to pure Python lists instead of NumPy arrays
    x_points_int = counts_df.get_column("__count__").to_list()
    freqs = counts_df.get_column("freq").to_list()
    
    total_counts = sum(freqs)
    if total_counts < 2:
        raise ValueError("At least two daily location counts are required to fit.")

    # Cast x_points to float to match the previous np.astype(float) behavior
    x_points = [float(x) for x in x_points_int]
    probs = [f / total_counts for f in freqs]

    # Calculate mu and sigma using standard Python math
    log_x = [math.log(x) for x in x_points]
    
    mu = sum(lx * f for lx, f in zip(log_x, freqs)) / total_counts
    var = sum(f * (lx - mu) ** 2 for lx, f in zip(log_x, freqs)) / total_counts
    sigma = math.sqrt(var)

    if not math.isfinite(sigma) or sigma <= 1e-12:
        raise ValueError("Daily location counts must have positive log variance.")

    return x_points, probs, mu, sigma
