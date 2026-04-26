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
) -> pd.DataFrame:
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
    pd.DataFrame
        Transition matrix as percentages (rows = from, columns = to).

    Raises
    ------
    ValueError
        When ``day_filter`` is requested but no day-of-week column can be found.
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

    # Convert to pandas for the row-iterative transition-counting loop
    df = nw_df.to_native()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Apply day filter
    if day_filter == "weekdays":
        df = df[df[day_col].str.lower().isin(_WEEKDAYS)].copy()
    elif day_filter == "weekends":
        df = df[df[day_col].str.lower().isin(_WEEKENDS)].copy()

    # Sort by [user_id, timestamp] when both are available
    sort_cols = []
    if user_id_col:
        sort_cols.append(user_id_col)
    if timestamp_col:
        sort_cols.append(timestamp_col)
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Drop rows with null activity
    if activity_col:
        df = df[df[activity_col].notna()].copy()

    if len(df) == 0 or activity_col is None:
        return pd.DataFrame()

    # Collect sorted unique activity labels
    activities = sorted(df[activity_col].unique())
    n_activities = len(activities)
    act_idx = {a: i for i, a in enumerate(activities)}

    transition_matrix = np.zeros((n_activities, n_activities))

    # Count transitions per user
    if user_id_col:
        for _uid, user_visits in df.groupby(user_id_col, sort=False):
            _count_transitions(user_visits, activity_col, act_idx, transition_matrix)
    else:
        _count_transitions(df, activity_col, act_idx, transition_matrix)

    transition_df = pd.DataFrame(transition_matrix, index=activities, columns=activities)

    total = transition_df.values.sum()
    if total > 0:
        transition_df = (transition_df / total) * 100.0

    return transition_df


def _count_transitions(
    user_visits: pd.DataFrame,
    activity_col: str,
    act_idx: dict,
    matrix: np.ndarray,
) -> None:
    """Accumulate transition counts from a single user's visit sequence."""
    acts = user_visits[activity_col].tolist()
    for i in range(len(acts) - 1):
        from_idx = act_idx[acts[i]]
        to_idx = act_idx[acts[i + 1]]
        matrix[from_idx, to_idx] += 1
