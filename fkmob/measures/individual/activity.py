"""Activity transition matrix measure."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
from fkmob._core import (
    activity_counts_arrow,
    activity_counts_numpy,
    activity_transition_counts_arrow,
    activity_transition_counts_numpy,
    daily_activity_percentages_arrow,
    daily_activity_percentages_numpy,
)

from .._common import (
    ACTIVITY_CANDIDATES,
    DAY_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _pick_existing_column,
    _use_arrow_kernel_path,
)

_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday"}
_WEEKENDS = {"saturday", "sunday"}
_END_TIMESTAMP_CANDIDATES = ["end_timestamp", "leaving_datetime", "end_time", "stop_timestamp"]


def _allowed_days(day_filter: str | None) -> set[str] | None:
    if day_filter == "weekdays":
        return _WEEKDAYS
    if day_filter == "weekends":
        return _WEEKENDS
    return None


def _apply_day_filter(
    df: nw.DataFrame,
    day_col: str | None,
    day_filter: str | None,
) -> nw.DataFrame:
    allowed_days = _allowed_days(day_filter)
    if allowed_days is None:
        return df
    if day_col is None:
        raise ValueError(
            "day_col could not be auto-detected and is required when day_filter is set. "
            f"Tried: {DAY_CANDIDATES}. Available columns: {df.columns}"
        )
    days = df.get_column(day_col).to_list()
    keep = [str(day).lower() in allowed_days for day in days]
    return df.with_columns(nw.new_series("__fkmob_keep__", keep, backend=df.implementation)).filter(
        nw.col("__fkmob_keep__")
    ).drop("__fkmob_keep__")


def _with_activity_fallback(
    df: nw.DataFrame,
    activity_col: str | None,
    unknown_label: str,
) -> tuple[nw.DataFrame, str]:
    if activity_col is None or activity_col not in df.columns:
        warnings.warn(
            f"activity column could not be found; using {unknown_label!r} for all visits",
            UserWarning,
            stacklevel=2,
        )
        fallback_col = "__fkmob_activity__"
        values = [unknown_label] * len(df)
        return df.with_columns(nw.new_series(fallback_col, values, backend=df.implementation)), fallback_col

    values = df.get_column(activity_col).to_list()
    if any(pd.isna(value) for value in values):
        warnings.warn(
            f"null activity values found; replacing them with {unknown_label!r}",
            UserWarning,
            stacklevel=2,
        )
        values = [unknown_label if pd.isna(value) else value for value in values]
        df = df.with_columns(nw.new_series(activity_col, values, backend=df.implementation))
    return df, activity_col


def _resolve_start_column(columns: list[str], start_time_col: str | None) -> str | None:
    if start_time_col is not None:
        return start_time_col
    return _pick_existing_column(columns, TIMESTAMP_CANDIDATES)


def _resolve_end_column(columns: list[str], end_time_col: str | None) -> str | None:
    if end_time_col is not None:
        return end_time_col
    return _pick_existing_column(columns, _END_TIMESTAMP_CANDIDATES)


def _factorize_activities(df: nw.DataFrame, activity_col: str) -> tuple[list[Any], nw.Series]:
    values = df.get_column(activity_col).to_list()
    categories = sorted(set(values), key=lambda value: str(value))
    category_idx = {category: index for index, category in enumerate(categories)}
    codes = np.fromiter((category_idx[value] for value in values), dtype=np.uint64, count=len(values))
    return categories, nw.new_series("__fkmob_activity_codes__", codes, dtype=nw.UInt64, backend=df.implementation)


def _kernel_result(values: Any) -> np.ndarray:
    return np.asarray(_arrow_result_values(values) if hasattr(values, "to_pyarrow") else values)


def _activity_counts(df: nw.DataFrame, codes: nw.Series, n_activities: int) -> np.ndarray:
    if _use_arrow_kernel_path(df):
        return _kernel_result(activity_counts_arrow(codes.to_arrow(), n_activities)).astype(np.uint64, copy=False)
    data = np.ascontiguousarray(codes.to_numpy(), dtype=np.uint64)
    return np.asarray(activity_counts_numpy(data, n_activities), dtype=np.uint64)


def visit_purpose_distribution(
    visits: Any,
    activity_col: str | None = None,
    normalize: bool = True,
    day_col: str | None = None,
    day_filter: str | None = None,
    unknown_label: str = "UNKNOWN",
) -> Any:
    """Compute visit-purpose counts and percentages.

    Missing activity columns and null activity values are recoverable: the
    affected visits are labelled with ``unknown_label`` and a warning is emitted.
    """
    nw_df = nw.from_native(visits, eager_only=True)
    is_pandas_input = isinstance(nw_df.to_native(), pd.DataFrame)

    if activity_col is None:
        activity_col = _pick_existing_column(nw_df.columns, ACTIVITY_CANDIDATES)
    if day_col is None:
        day_col = _pick_existing_column(nw_df.columns, DAY_CANDIDATES)

    work_cols = []
    if activity_col:
        work_cols.append(activity_col)
    if day_col and _allowed_days(day_filter) is not None:
        work_cols.append(day_col)
    work_cols = list(dict.fromkeys(work_cols))

    if work_cols:
        df = nw_df.select(work_cols)
    elif nw_df.columns:
        df = nw_df.select([nw_df.columns[0]])
    else:
        df = nw_df.select([])
    df = _apply_day_filter(df, day_col, day_filter)
    df, resolved_activity_col = _with_activity_fallback(df, activity_col, unknown_label)

    categories, codes = _factorize_activities(df, resolved_activity_col)
    category_counts = _activity_counts(df, codes, len(categories))
    order = sorted(range(len(categories)), key=lambda index: (-int(category_counts[index]), str(categories[index])))
    labels = [categories[index] for index in order]
    counts = [int(category_counts[index]) for index in order]
    total = sum(counts)
    percentages = [(count / total) * 100.0 if normalize and total else float(count) for count in counts]

    output = {
        "activity": labels,
        "count": counts,
        "percentage": percentages,
    }
    if is_pandas_input:
        return pd.DataFrame(output)
    return nw.from_dict(output, backend=nw_df.implementation).to_native()


def daily_activity_distribution(
    visits: Any,
    activity_col: str | None = None,
    start_time_col: str | None = None,
    end_time_col: str | None = None,
    bin_size_minutes: int = 10,
    day_col: str | None = None,
    day_filter: str | None = None,
    unknown_label: str = "UNKNOWN",
) -> tuple[np.ndarray, list[Any], int]:
    """Compute a daily activity distribution matrix over fixed time bins."""
    if bin_size_minutes <= 0 or 1440 % bin_size_minutes != 0:
        raise ValueError("bin_size_minutes must be a positive divisor of 1440")

    nw_df = nw.from_native(visits, eager_only=True)
    if activity_col is None:
        activity_col = _pick_existing_column(nw_df.columns, ACTIVITY_CANDIDATES)
    if day_col is None:
        day_col = _pick_existing_column(nw_df.columns, DAY_CANDIDATES)
    start_time_col = _resolve_start_column(nw_df.columns, start_time_col)
    end_time_col = _resolve_end_column(nw_df.columns, end_time_col)
    if start_time_col is None or start_time_col not in nw_df.columns:
        raise ValueError(
            "start_time_col could not be auto-detected. "
            f"Tried: {TIMESTAMP_CANDIDATES}. Available columns: {nw_df.columns}"
        )

    work_cols = [start_time_col]
    if activity_col:
        work_cols.append(activity_col)
    if end_time_col:
        work_cols.append(end_time_col)
    if day_col and _allowed_days(day_filter) is not None:
        work_cols.append(day_col)
    work_cols = list(dict.fromkeys(work_cols))

    df = nw_df.select(work_cols)
    df = _apply_day_filter(df, day_col, day_filter)
    df, resolved_activity_col = _with_activity_fallback(df, activity_col, unknown_label)

    categories, codes = _factorize_activities(df, resolved_activity_col)
    n_bins = 1440 // bin_size_minutes
    starts = pd.to_datetime(df.get_column(start_time_col).to_list(), errors="coerce")
    valid_rows = np.asarray(~pd.isna(starts), dtype=bool)
    start_minutes = np.where(valid_rows, starts.hour * 60 + starts.minute, 0).astype(np.int64)
    if end_time_col and end_time_col in df.columns:
        ends = pd.to_datetime(df.get_column(end_time_col).to_list(), errors="coerce")
        end_valid = np.asarray(~pd.isna(ends), dtype=bool)
        end_minutes = np.where(end_valid, ends.hour * 60 + ends.minute, 1439).astype(np.int64)
    else:
        end_minutes = np.full(len(df), 1439, dtype=np.int64)

    if _use_arrow_kernel_path(df):
        start_series = nw.new_series("start_minutes", start_minutes, dtype=nw.Int64, backend=df.implementation)
        end_series = nw.new_series("end_minutes", end_minutes, dtype=nw.Int64, backend=df.implementation)
        valid_series = nw.new_series("valid_rows", valid_rows, dtype=nw.Boolean, backend=df.implementation)
        flat = _kernel_result(
            daily_activity_percentages_arrow(
                codes.to_arrow(), start_series.to_arrow(), end_series.to_arrow(), valid_series.to_arrow(),
                len(categories), bin_size_minutes,
            )
        )
    else:
        flat = daily_activity_percentages_numpy(
            np.ascontiguousarray(codes.to_numpy(), dtype=np.uint64),
            np.ascontiguousarray(start_minutes), np.ascontiguousarray(end_minutes),
            np.ascontiguousarray(valid_rows), len(categories), bin_size_minutes,
        )
    activity_matrix_pct = np.asarray(flat, dtype=float).reshape(len(categories), n_bins)

    return activity_matrix_pct, categories, n_bins


def activity_transition_matrix(
    visits: Any,
    activity_col: str | None = None,
    user_id_col: str | None = None,
    timestamp_col: str | None = None,
    day_col: str | None = None,
    day_filter: str | None = None,
    unknown_label: str = "UNKNOWN",
) -> Any:
    """Compute the activity transition matrix for a visits DataFrame.

    Counts how often each activity-type transition (from -> to) occurs across
    all users, then normalises to percentages. Returns a square DataFrame with
    activity labels as both index and columns.

    Parameters
    ----------
    visits : DataFrame-like
        A DataFrame (any Narwhals-compatible backend) with at least an activity
        column and, optionally, a user-ID and timestamp column.
    activity_col : str or None, optional
        Column name for the activity/purpose type. Auto-detected if None.
    user_id_col : str or None, optional
        Column name for the user ID. Auto-detected if None.
    timestamp_col : str or None, optional
        Column name for the visit timestamp used for sorting. Auto-detected if
        None (rows are used in their current order when no timestamp is found).
    day_col : str or None, optional
        Column name for the day-of-week string. Auto-detected if None.
    day_filter : str or None, optional
        One of ``None`` (all days), ``"weekdays"`` (Mon–Fri only), or
        ``"weekends"`` (Sat–Sun only). When not None, ``day_col`` must be
        resolvable.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        Transition matrix as percentages. Pandas inputs return a pandas matrix
        with activity labels as index/columns; other backends return a native
        dataframe with an ``activity`` label column plus one column per target
        activity.

    Raises
    ------
    ValueError
        When ``day_filter`` is requested but no day-of-week column can be found.

    Examples
    --------
    >>> import pandas as pd
    >>> from fkmob.measures.individual import activity_transition_matrix
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
    ...         "start_timestamp": pd.to_datetime(
    ...             [
    ...                 "2020-01-01 08:00",
    ...                 "2020-01-01 09:00",
    ...                 "2020-01-01 18:00",
    ...                 "2020-01-01 07:30",
    ...                 "2020-01-01 12:00",
    ...                 "2020-01-01 19:00",
    ...             ]
    ...         ),
    ...         "purpose": ["HOME", "WORK", "HOME", "HOME", "SHOP", "HOME"],
    ...     }
    ... )
    >>> result = activity_transition_matrix(visits)
    >>> print(result.round(1).to_string())
          HOME  SHOP  WORK
    HOME   0.0  25.0  25.0
    SHOP  25.0   0.0   0.0
    WORK  25.0   0.0   0.0
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if activity_col is None:
        activity_col = _pick_existing_column(nw_df.columns, ACTIVITY_CANDIDATES)
    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if timestamp_col is None:
        timestamp_col = _pick_existing_column(nw_df.columns, TIMESTAMP_CANDIDATES)
    if day_col is None:
        day_col = _pick_existing_column(nw_df.columns, DAY_CANDIDATES)

    is_pandas_input = isinstance(nw_df.to_native(), pd.DataFrame)

    work_cols = [activity_col] if activity_col else []
    if user_id_col:
        work_cols.append(user_id_col)
    if timestamp_col:
        work_cols.append(timestamp_col)
    if day_col and _allowed_days(day_filter) is not None:
        work_cols.append(day_col)
    work_cols = list(dict.fromkeys(work_cols))

    if work_cols:
        df = nw_df.select(work_cols)
    elif nw_df.columns:
        df = nw_df.select([nw_df.columns[0]])
    else:
        df = nw_df.select([])
    df = _apply_day_filter(df, day_col, day_filter)
    df, activity_col = _with_activity_fallback(df, activity_col, unknown_label)

    sort_cols = []
    if user_id_col:
        sort_cols.append(user_id_col)
    if timestamp_col:
        sort_cols.append(timestamp_col)
    if sort_cols:
        df = df.sort(sort_cols)

    if len(df) == 0 or activity_col is None:
        if is_pandas_input:
            return pd.DataFrame()
        return nw.from_dict({"activity": []}, backend=nw_df.implementation).to_native()

    activities, codes = _factorize_activities(df, activity_col)
    n_activities = len(activities)
    _, indices, ends = _build_indexed_user_ranges_fast(df, user_id_col)
    if _use_arrow_kernel_path(df):
        flat_counts = _kernel_result(
            activity_transition_counts_arrow(codes.to_arrow(), indices, ends, n_activities)
        )
    else:
        flat_counts = activity_transition_counts_numpy(
            np.ascontiguousarray(codes.to_numpy(), dtype=np.uint64), indices, ends, n_activities
        )
    transition_matrix = np.asarray(flat_counts, dtype=float).reshape(n_activities, n_activities)
    total = transition_matrix.sum()
    if total > 0:
        transition_matrix = (transition_matrix / total) * 100.0

    if is_pandas_input:
        return pd.DataFrame(transition_matrix, index=activities, columns=activities)

    output: dict[str, list[Any]] = {"activity": activities}
    for idx, activity in enumerate(activities):
        output[str(activity)] = transition_matrix[:, idx].tolist()
    return nw.from_dict(output, backend=nw_df.implementation).to_native()
