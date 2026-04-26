"""Trajectory diversity measure using suffix-array entropy."""

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
from .fast_diversity import fast_diversity


def diversity(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
) -> pd.DataFrame:
    """Compute trajectory diversity per user using suffix-array entropy.

    Per-user: factorize the composite location string
    (``location_id + "_" + location_type`` when ``location_type_col`` is not
    None; otherwise just ``location_id``), then call :func:`fast_diversity` on
    the integer codes.

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
        One row per user with columns ``[user_id_col, "diversity"]``.
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

    def _get_sequence(user_df: pd.DataFrame) -> list:
        if location_id_col and location_type_col:
            return (user_df[location_id_col].astype(str) + "_" + user_df[location_type_col].astype(str)).tolist()
        elif location_id_col:
            return user_df[location_id_col].astype(str).tolist()
        else:
            return []

    results = []
    if user_id_col:
        for uid, group in df.groupby(user_id_col, sort=False):
            seq = _get_sequence(group)
            div = fast_diversity(seq) if seq else 0.0
            results.append({user_id_col: uid, "diversity": div})
        out_df = pd.DataFrame(results)
    else:
        seq = _get_sequence(df)
        div = fast_diversity(seq) if seq else 0.0
        out_df = pd.DataFrame([{"diversity": div}])

    return out_df
