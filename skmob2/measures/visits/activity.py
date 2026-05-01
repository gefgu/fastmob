"""Activity transition matrix measure."""

from __future__ import annotations

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


def activity_transition_matrix(
    visits: Any,
    activity_col: str | None = None,
    user_id_col: str | None = None,
    timestamp_col: str | None = None,
    day_col: str | None = None,
    day_filter: str | None = None,
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
    >>> from skmob2.measures.visits import activity_transition_matrix
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

    if day_filter is not None and day_col is None:
        raise ValueError(
            "day_col could not be auto-detected and is required when day_filter is set. "
            f"Tried: {DAY_CANDIDATES}. Available columns: {nw_df.columns}"
        )

    is_pandas_input = isinstance(nw_df.to_native(), pd.DataFrame)

    if day_filter == "weekdays":
        allowed_days = _WEEKDAYS
    elif day_filter == "weekends":
        allowed_days = _WEEKENDS
    else:
        allowed_days = None

    work_cols = [activity_col] if activity_col else []
    if user_id_col:
        work_cols.append(user_id_col)
    if timestamp_col:
        work_cols.append(timestamp_col)
    if day_col and allowed_days is not None:
        work_cols.append(day_col)
    work_cols = list(dict.fromkeys(work_cols))

    df = nw_df.select(work_cols) if work_cols else nw_df
    if activity_col:
        df = df.drop_nulls(subset=[activity_col])

    if allowed_days is not None and day_col:
        days = df.get_column(day_col).to_list()
        keep = [str(day).lower() in allowed_days for day in days]
        df = df.with_columns(nw.new_series("__skmob2_keep__", keep, backend=df.implementation))
        df = df.filter(nw.col("__skmob2_keep__")).drop("__skmob2_keep__")

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
