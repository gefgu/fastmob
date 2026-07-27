"""Column-level and visit-count distribution comparisons.

Uses Narwhals group_by for the hot path — group iteration is O(N_hue_groups)
instead of O(N_rows) Python loops.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import narwhals as nw
import numpy as np

from fastmob.utils._common import (
    ACTIVITY_CANDIDATES,
    DAY_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _pick_existing_column,
)

from ._utils import _is_null, _series_values
from .metrics import histogram_jensen_shannon_divergence, wasserstein_distance
from .temporal import (
    _DAY_PERIOD_COLUMN,
    _DAY_PERIOD_ORDER,
    _WEEKDAY_ORDER,
    _day_period,
)


def _resolve_column(df: nw.DataFrame, explicit: str | None, candidates: list[str], role: str) -> str:
    if explicit is not None and explicit in df.columns:
        return explicit
    column = _pick_existing_column(df.columns, candidates)
    if column is None:
        tried = [explicit] + candidates if explicit is not None else candidates
        raise ValueError(f"Could not detect {role} column. Tried: {tried}. Available columns: {df.columns}.")
    return column


def _group_labels(groups1: dict[Any, Any], groups2: dict[Any, Any], hue: str | None) -> list[Any]:
    labels = set(groups1) | set(groups2)
    if hue == "day_of_week":
        return [day for day in _WEEKDAY_ORDER if day in labels]
    if hue == "day_period":
        return [period for period in _DAY_PERIOD_ORDER if period in labels]
    if hue == "purpose":
        counts = Counter()
        for groups in (groups1, groups2):
            for label, values in groups.items():
                counts[label] += len(values)
        return [label for label, _ in counts.most_common()]
    return ["All Trips"] if "All Trips" in labels else sorted(labels, key=lambda value: str(value))


def _build_hue_column(
    df: nw.DataFrame,
    hue: str | None,
    *,
    trip_start_col: str | None,
    day_col: str | None,
    purpose_col: str | None,
    skip_day_period_creation: bool,
) -> nw.DataFrame:
    """Return df with a new ``__hue__`` column added."""
    if hue is None:
        return df.with_columns(nw.lit("All Trips").alias("__hue__"))

    if hue == "day_of_week":
        day_col = _resolve_column(df, day_col, DAY_CANDIDATES, "day")
        return df.with_columns(nw.col(day_col).cast(nw.String).str.to_lowercase().alias("__hue__"))

    if hue == "purpose":
        purpose_col = _resolve_column(df, purpose_col, ACTIVITY_CANDIDATES, "purpose")
        return df.with_columns(nw.col(purpose_col).cast(nw.String).alias("__hue__"))

    if hue == "day_period":
        if skip_day_period_creation and _DAY_PERIOD_COLUMN in df.columns:
            return df.with_columns(nw.col(_DAY_PERIOD_COLUMN).alias("__hue__"))

        ts_col = _resolve_column(df, trip_start_col, TIMESTAMP_CANDIDATES, "timestamp")
        ts_dtype = df.schema[ts_col]
        if isinstance(ts_dtype, nw.Datetime):
            hour_expr = nw.col(ts_col).dt.hour()
            day_period_expr = (
                nw.when(hour_expr.is_between(6, 11))
                .then(nw.lit(_DAY_PERIOD_ORDER[0]))
                .otherwise(
                    nw.when(hour_expr.is_between(12, 17))
                    .then(nw.lit(_DAY_PERIOD_ORDER[1]))
                    .otherwise(
                        nw.when(hour_expr.is_between(18, 23))
                        .then(nw.lit(_DAY_PERIOD_ORDER[2]))
                        .otherwise(nw.lit(_DAY_PERIOD_ORDER[3]))
                    )
                )
            ).alias("__hue__")
            return df.with_columns(day_period_expr)

        # Fallback: string timestamps — element-wise via _day_period helper.
        labels = [_day_period(v) for v in _series_values(df, ts_col)]
        return df.with_columns(nw.new_series("__hue__", labels, backend=df.implementation))

    raise ValueError(f"Unsupported hue value: {hue!r}")


def _group_numeric_values(
    data: Any,
    value_col: str,
    *,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_col: str | None = None,
    day_col: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> dict[Any, np.ndarray]:
    df = nw.from_native(data, eager_only=True)
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not found. Available columns: {df.columns}.")

    if hue is None:
        values_np = df.get_column(value_col).cast(nw.Float64).to_numpy()
        finite_vals = values_np[np.isfinite(values_np)]
        return {"All Trips": finite_vals} if len(finite_vals) > 0 else {}

    df = _build_hue_column(
        df,
        hue,
        trip_start_col=trip_start_col,
        day_col=day_col,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )

    df = df.with_columns(nw.col(value_col).cast(nw.Float64).alias("__val__"))

    unique_hues: list[Any] = df.get_column("__hue__").drop_nulls().unique().to_list()
    groups: dict[Any, np.ndarray] = {}
    for hue_val in unique_hues:
        if _is_null(hue_val):
            continue
        values_np = (
            df.filter(nw.col("__hue__") == hue_val).drop_nulls(subset=["__val__"]).get_column("__val__").to_numpy()
        )
        finite_vals = values_np[np.isfinite(values_np)]
        if len(finite_vals) > 0:
            groups[hue_val] = finite_vals
    return groups


def _visits_per_user_groups(
    data: Any,
    *,
    user_id_col: str | None = None,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_col: str | None = None,
    day_col: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> dict[Any, np.ndarray]:
    df = nw.from_native(data, eager_only=True)
    uid_col = _resolve_column(df, user_id_col, USER_ID_CANDIDATES, "user")

    if hue is None:
        counts = (
            df.drop_nulls(subset=[uid_col])
            .group_by(uid_col)
            .agg(nw.len().alias("__count__"))
            .get_column("__count__")
            .to_numpy()
        )
        return {"All Trips": counts}

    df = _build_hue_column(
        df,
        hue,
        trip_start_col=trip_start_col,
        day_col=day_col,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )

    # Count rows per (hue, user) pair — vectorized.
    counts_df = (
        df.drop_nulls(subset=["__hue__", uid_col]).group_by(["__hue__", uid_col]).agg(nw.len().alias("__count__"))
    )

    unique_hues: list[Any] = counts_df.get_column("__hue__").unique().to_list()
    groups: dict[Any, np.ndarray] = {}
    for hue_val in unique_hues:
        if _is_null(hue_val):
            continue
        groups[hue_val] = counts_df.filter(nw.col("__hue__") == hue_val).get_column("__count__").to_numpy()
    return groups


def column_distribution_jensen_shannon_divergence(
    df1: Any,
    df2: Any,
    column: str,
    *,
    bin_size: float = 1.0,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_col1: str | None = None,
    trip_start_col2: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped numeric column distributions with Jensen-Shannon divergence."""
    groups1 = _group_numeric_values(
        df1,
        column,
        hue=hue,
        trip_start_col=trip_start_col1,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _group_numeric_values(
        df2,
        column,
        hue=hue,
        trip_start_col=trip_start_col2,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, histogram_jensen_shannon_divergence(groups1[label], groups2[label], bin_size=bin_size))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2 and len(groups1[label]) > 0 and len(groups2[label]) > 0
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])


def column_distribution_wasserstein_distance(
    df1: Any,
    df2: Any,
    column: str,
    *,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_col1: str | None = None,
    trip_start_col2: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped numeric column distributions with Rust-backed Wasserstein distance."""
    groups1 = _group_numeric_values(
        df1,
        column,
        hue=hue,
        trip_start_col=trip_start_col1,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _group_numeric_values(
        df2,
        column,
        hue=hue,
        trip_start_col=trip_start_col2,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, wasserstein_distance(groups1[label], groups2[label]))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2 and len(groups1[label]) > 0 and len(groups2[label]) > 0
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])


def visits_per_user_jensen_shannon_divergence(
    df1: Any,
    df2: Any,
    *,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    user_id_col1: str | None = None,
    user_id_col2: str | None = None,
    bin_size: float = 1.0,
    trip_start_col: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped visits-per-user distributions with Jensen-Shannon divergence."""
    groups1 = _visits_per_user_groups(
        df1,
        user_id_col=user_id_col1,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _visits_per_user_groups(
        df2,
        user_id_col=user_id_col2,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, histogram_jensen_shannon_divergence(groups1[label], groups2[label], bin_size=bin_size))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])


def visits_per_user_wasserstein_distance(
    df1: Any,
    df2: Any,
    *,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    user_id_col1: str | None = None,
    user_id_col2: str | None = None,
    trip_start_col: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped visits-per-user distributions with Rust-backed Wasserstein distance."""
    groups1 = _visits_per_user_groups(
        df1,
        user_id_col=user_id_col1,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _visits_per_user_groups(
        df2,
        user_id_col=user_id_col2,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, wasserstein_distance(groups1[label], groups2[label]))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])
