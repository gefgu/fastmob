"""Regularity measure for visit trajectories."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import (
    LOCATION_CANDIDATES,
    LOCATION_TYPE_CANDIDATES,
    USER_ID_CANDIDATES,
    _pick_existing_column,
)


def regularity(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
) -> Any:
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
    visits : DataFrame-like
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col : str or None, optional
        Column name for the user ID. Auto-detected if None.
    location_id_col : str or None, optional
        Column name for the location ID. Auto-detected if None.
    location_type_col : str or None, optional
        Column name for the location type / activity purpose.
        Auto-detected if None; set explicitly to ``None`` to disable.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns ``[user_id_col, "regularity"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.individual import regularity
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2"],
    ...         "location_id": ["home", "work", "home", "gym", "home", "shop", "home"],
    ...     }
    ... )
    >>> result = regularity(visits)
    >>> print(result.round(3).to_string(index=False))
    user_id  regularity
         u1       0.250
         u2       0.333
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)

    if location_id_col and location_type_col:
        key_cols = [location_id_col, location_type_col]
    elif location_id_col:
        key_cols = [location_id_col]
    else:
        key_cols = []

    if len(nw_df) == 0:
        columns = [user_id_col, "regularity"] if user_id_col else ["regularity"]
        return nw.from_dict({col: [] for col in columns}, backend=nw_df.implementation).to_native()

    if user_id_col:
        totals = nw_df.group_by(user_id_col).agg(nw.len().alias("__total__"))
        if key_cols:
            uniques = (
                nw_df.select([user_id_col, *key_cols]).unique().group_by(user_id_col).agg(nw.len().alias("__unique__"))
            )
            result = totals.join(uniques, on=user_id_col, how="left")
        else:
            result = totals.with_columns(nw.lit(0).alias("__unique__"))

        result = (
            result.with_columns((nw.lit(1.0) - nw.col("__unique__") / nw.col("__total__")).alias("regularity"))
            .select([user_id_col, "regularity"])
            .sort(user_id_col)
        )
        return result.to_native()

    total = len(nw_df)
    unique = len(nw_df.select(key_cols).unique()) if key_cols else 0
    return nw.from_dict(
        {"regularity": [1.0 - unique / total if total else 0.0]},
        backend=nw_df.implementation,
    ).to_native()
