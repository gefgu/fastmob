"""Trajectory diversity measure using suffix-array entropy."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import diversity_users
from fastmob.utils._common import (
    LOCATION_CANDIDATES,
    USER_ID_CANDIDATES,
    _build_presorted_user_ends,
    _factorize_arrow_values,
    _pick_existing_column,
)


def diversity(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
) -> Any:
    """Compute trajectory diversity per user using suffix-array entropy.

    Per-user: factorize the location column, then call the Rust suffix-array
    kernel on the integer-coded token sequence to compute the ratio of
    distinct substrings to total substrings.

    Parameters
    ----------
    visits : DataFrame-like
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col : str or None, optional
        Column name for the user ID. Auto-detected if None.
    location_id_col : str or None, optional
        Column name for the location ID. Auto-detected if None.
    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns ``[user_id_col, "diversity"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.individual import diversity
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2"],
    ...         "location_id": ["home", "work", "home", "gym", "home", "shop", "home"],
    ...     }
    ... )
    >>> result = diversity(visits)
    >>> print(result.round(3).to_string(index=False))
    user_id  diversity
         u1      0.900
         u2      0.833
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)

    nw_df = nw_df.filter(~nw.col(location_id_col).is_null())
    if user_id_col:
        nw_df = nw_df.sort(user_id_col)
        location_codes, _ = _factorize_arrow_values(
            nw_df.get_column(location_id_col).to_arrow(), sort=False
        )
        uid_values, ends = _build_presorted_user_ends(nw_df, user_id_col)
        values = diversity_users(location_codes, ends)
        return nw.from_dict(
            {user_id_col: uid_values.to_pylist(), "diversity": values},
            backend=nw_df.implementation,
        ).to_native()

    location_codes, _ = _factorize_arrow_values(
        nw_df.get_column(location_id_col).to_arrow(), sort=False
    )
    _, ends = _build_presorted_user_ends(nw_df, None)
    div = diversity_users(location_codes, ends)[0]
    return nw.from_dict({"diversity": [div]}, backend=nw_df.implementation).to_native()
