"""Activity transition matrix measure."""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np
import narwhals as nw
import pandas as pd

from .._common import (
    _pick_existing_column,
    ACTIVITY_CANDIDATES,
    USER_ID_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    DAY_CANDIDATES,
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
    return df.with_columns(nw.new_series("__skmob2_keep__", keep, backend=df.implementation)).filter(
        nw.col("__skmob2_keep__")
    ).drop("__skmob2_keep__")


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
        fallback_col = "__skmob2_activity__"
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

    values = df.get_column(resolved_activity_col).to_list()
    counts = Counter(values)
    labels = sorted(counts, key=lambda label: (-counts[label], str(label)))
    total = sum(counts.values())
    percentages = [(counts[label] / total) * 100.0 if normalize and total else float(counts[label]) for label in labels]

    output = {
        "activity": labels,
        "count": [counts[label] for label in labels],
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

    categories = sorted(df.get_column(resolved_activity_col).unique().to_list(), key=lambda value: str(value))
    n_bins = 1440 // bin_size_minutes
    activity_matrix = np.full((len(categories), n_bins), np.nan)
    category_idx = {category: idx for idx, category in enumerate(categories)}

    starts = df.get_column(start_time_col).to_list()
    ends = df.get_column(end_time_col).to_list() if end_time_col and end_time_col in df.columns else [None] * len(df)
    activities = df.get_column(resolved_activity_col).to_list()

    for activity, start_value, end_value in zip(activities, starts, ends):
        if pd.isna(start_value):
            continue

        start = pd.to_datetime(start_value)
        if end_value is None or pd.isna(end_value):
            end = start.replace(hour=23, minute=59, second=59)
        else:
            end = pd.to_datetime(end_value)

        row_idx = category_idx[activity]
        start_min = start.hour * 60 + start.minute
        end_min = end.hour * 60 + end.minute

        ranges = []
        if end_min < start_min:
            ranges.append((start_min // bin_size_minutes, n_bins - 1))
            ranges.append((0, min(end_min // bin_size_minutes, n_bins - 1)))
        else:
            ranges.append((start_min // bin_size_minutes, min(end_min // bin_size_minutes, n_bins - 1)))

        for start_bin, end_bin in ranges:
            for bin_idx in range(start_bin, end_bin + 1):
                activity_matrix[row_idx, bin_idx] = (
                    1 if np.isnan(activity_matrix[row_idx, bin_idx]) else activity_matrix[row_idx, bin_idx] + 1
                )

    col_sums = np.nansum(activity_matrix, axis=0)
    activity_matrix_pct = np.full_like(activity_matrix, np.nan, dtype=float)
    for col in range(n_bins):
        if col_sums[col] > 0:
            activity_matrix_pct[:, col] = (activity_matrix[:, col] / col_sums[col]) * 100.0

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
    visits:
        A DataFrame (any Narwhals-compatible backend) with at least an activity
        column and, optionally, a user-ID and timestamp column.
    activity_col:
        Column name for the activity/purpose type. Auto-detected if None.
    user_id_col:
        Column name for the user ID. Auto-detected if None.
    timestamp_col:
        Column name for the visit timestamp used for sorting. Auto-detected if
        None (rows are used in their current order when no timestamp is found).
    day_col:
        Column name for the day-of-week string. Auto-detected if None.
    day_filter:
        One of ``None`` (all days), ``"weekdays"`` (Mon–Fri only), or
        ``"weekends"`` (Sat–Sun only). When not None, ``day_col`` must be
        resolvable.

    Returns
    -------
    DataFrame
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
    >>> from skmob2.measures.individual import activity_transition_matrix
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

    activities = sorted(df.get_column(activity_col).unique().to_list())
    n_activities = len(activities)
    act_idx = {a: i for i, a in enumerate(activities)}

    transition_matrix = np.zeros((n_activities, n_activities))

    if user_id_col:
        uid_values = df.get_column(user_id_col).to_list()
        act_values = df.get_column(activity_col).to_list()
        start = 0
        while start < len(act_values):
            end = start + 1
            while end < len(act_values) and uid_values[end] == uid_values[start]:
                end += 1
            _count_transition_values(act_values[start:end], act_idx, transition_matrix)
            start = end
    else:
        _count_transition_values(df.get_column(activity_col).to_list(), act_idx, transition_matrix)

    total = transition_matrix.sum()
    if total > 0:
        transition_matrix = (transition_matrix / total) * 100.0

    if is_pandas_input:
        return pd.DataFrame(transition_matrix, index=activities, columns=activities)

    output: dict[str, list[Any]] = {"activity": activities}
    for idx, activity in enumerate(activities):
        output[str(activity)] = transition_matrix[:, idx].tolist()
    return nw.from_dict(output, backend=nw_df.implementation).to_native()


def _count_transition_values(acts: list[Any], act_idx: dict, matrix: np.ndarray) -> None:
    """Accumulate transition counts from one already-sorted activity sequence."""
    for i in range(len(acts) - 1):
        matrix[act_idx[acts[i]], act_idx[acts[i + 1]]] += 1
