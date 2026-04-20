"""Regularity measure for visit trajectories."""
from __future__ import annotations

from typing import Any

import pandas as pd
import narwhals as nw

from .._common import (
    _pick_existing_column,
    USER_ID_CANDIDATES,
    LOCATION_CANDIDATES,
    LOCATION_TYPE_CANDIDATES,
)


def regularity(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
) -> pd.DataFrame:
    """Compute regularity per user.

    Regularity measures how repetitively a user visits the same places.
    It is defined as:

        regularity = 1 - (unique_locations / total_visits)

    where ``unique_locations`` is the count of distinct
    ``(location_id, location_type)`` pairs (or just distinct ``location_id``
    values when ``location_type_col`` is None), and ``total_visits`` is the
    row count for the user.

    Parameters
    ----------
    visits:
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col:
        Column name for the user ID. Auto-detected if None.
    location_id_col:
        Column name for the location ID. Auto-detected if None.
    location_type_col:
        Column name for the location type / activity purpose.
        Auto-detected if None; set explicitly to ``None`` to disable.

    Returns
    -------
    pd.DataFrame
        One row per user with columns ``[user_id_col, "regularity"]``.
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)

    # Convert to pandas for groupby logic
    df = nw_df.to_native()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def _compute_single_regularity(user_df: pd.DataFrame) -> float:
        total = len(user_df)
        if total == 0:
            return 0.0
        if location_id_col and location_type_col:
            unique = user_df[[location_id_col, location_type_col]].drop_duplicates().shape[0]
        elif location_id_col:
            unique = user_df[location_id_col].nunique()
        else:
            unique = 0
        return 1.0 - unique / total

    results = []
    if user_id_col:
        for uid, group in df.groupby(user_id_col, sort=False):
            reg = _compute_single_regularity(group)
            results.append({user_id_col: uid, "regularity": reg})
        out_df = pd.DataFrame(results)
    else:
        reg = _compute_single_regularity(df)
        out_df = pd.DataFrame([{"regularity": reg}])

    return out_df
