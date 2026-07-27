"""Trajectory diversity measure using suffix-array entropy."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import diversity_batch as _diversity_batch_rust
from fastmob.utils._common import (
    LOCATION_CANDIDATES,
    LOCATION_TYPE_CANDIDATES,
    USER_ID_CANDIDATES,
    _build_user_ranges,
    _pick_existing_column,
)


def diversity(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
) -> Any:
    """Compute trajectory diversity per user using suffix-array entropy.

    Per-user: factorize the composite location string
    (``location_id + "_" + location_type`` when ``location_type_col`` is not
    None; otherwise just ``location_id``), then call the Rust suffix-array
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
    location_type_col : str or None, optional
        Column name for the location type / activity purpose.
        Auto-detected if None; set explicitly to ``None`` to disable.

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
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)

    location_key_col = "__fastmob_location_key__"
    if location_id_col and location_type_col:
        nw_df = nw_df.with_columns(
            (nw.col(location_id_col).cast(nw.String) + nw.lit("_") + nw.col(location_type_col).cast(nw.String)).alias(
                location_key_col
            )
        )
        nw_df = nw_df.filter(~nw.col(location_key_col).is_null())
    elif location_id_col:
        nw_df = nw_df.with_columns(nw.col(location_id_col).cast(nw.String).alias(location_key_col))
        nw_df = nw_df.filter(~nw.col(location_key_col).is_null())

    if user_id_col:
        nw_df = nw_df.sort(user_id_col)
        uid_values, ranges = _build_user_ranges(nw_df, user_id_col)
        if location_id_col:
            tokens = nw_df.get_column(location_key_col).to_list()
            values = _diversity_batch_rust(tokens, ranges)
        else:
            values = [0.0] * len(ranges)
        return nw.from_dict(
            {user_id_col: uid_values, "diversity": values},
            backend=nw_df.implementation,
        ).to_native()

    if location_id_col:
        tokens = nw_df.get_column(location_key_col).to_list()
        div = _diversity_batch_rust(tokens, [(0, len(tokens))])[0]
    else:
        div = 0.0
    return nw.from_dict({"diversity": [div]}, backend=nw_df.implementation).to_native()
