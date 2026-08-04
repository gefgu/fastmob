"""Lognormal fit for the number of global or user-scoped locations visited per day."""

from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from fastmob._core import daily_unique_location_histogram_arrow
from fastmob.core.locations_dataframe import Locations
from fastmob.core.staypoints_dataframe import Staypoints
from fastmob.utils._common import (
    LOCATION_CANDIDATES,
    _detect_required_column,
    _strip_time_zone,
    _values_to_list,
    _with_datetime_column,
)


def fit_daily_location_lognormal(
    staypoints: Any | Staypoints,
    *,
    locations: Locations | None = None,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    timestamp_col: str | None = None,
) -> tuple[list[float], list[float], float, float]:
    """Fit a lognormal distribution to per-user daily distinct-location counts.

    ``staypoints`` may be a :class:`~fastmob.core.Staypoints` object or an
    eager dataframe. When a :class:`~fastmob.core.Locations` catalogue is
    supplied, its global or user-scoped location identities are validated
    before fitting. Timezone-aware starts are bucketed by their local
    wall-clock calendar day.
    """
    df, user_id_col, timestamp_col, _lat_col, _lng_col = Staypoints.resolve_dataframe(
        staypoints,
        user_id_col=user_id_col,
        timestamp_col=timestamp_col,
    )
    location_id_col = _detect_required_column(df, location_id_col, LOCATION_CANDIDATES)
    if locations is not None:
        locations.validate_staypoint_assignments(
            df,
            user_id_col=user_id_col,
            location_id_col=location_id_col,
        )

    df = _strip_time_zone(_with_datetime_column(df, timestamp_col), timestamp_col).with_columns(
        nw.col(timestamp_col).cast(nw.Datetime("us")).alias(timestamp_col)
    )
    counts_raw, frequencies_raw = daily_unique_location_histogram_arrow(
        df.get_column(user_id_col).to_arrow(),
        df.get_column(location_id_col).to_arrow(),
        df.get_column(timestamp_col).to_arrow(),
    )
    counts = [float(value) for value in _values_to_list(counts_raw)]
    frequencies = [int(value) for value in _values_to_list(frequencies_raw)]
    total_counts = sum(frequencies)
    if total_counts < 2:
        raise ValueError("At least two daily location counts are required to fit.")

    probs = [frequency / total_counts for frequency in frequencies]
    log_counts = [math.log(count) for count in counts]
    mu = sum(log_count * frequency for log_count, frequency in zip(log_counts, frequencies)) / total_counts
    var = sum(frequency * (log_count - mu) ** 2 for log_count, frequency in zip(log_counts, frequencies)) / total_counts
    sigma = math.sqrt(var)
    if not math.isfinite(sigma) or sigma <= 1e-12:
        raise ValueError("Daily location counts must have positive log variance.")

    return counts, probs, mu, sigma
