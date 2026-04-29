"""Trajectory diversity measure using suffix-array entropy."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import (
    LOCATION_CANDIDATES,
    LOCATION_TYPE_CANDIDATES,
    USER_ID_CANDIDATES,
    _build_user_ranges,
    _pick_existing_column,
)
from .fast_diversity import fast_diversity


def diversity(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
) -> Any:
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
    DataFrame
        One row per user with columns ``[user_id_col, "diversity"]``.
        The returned backend matches the input backend.
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)

    location_key_col = "__skmob2_location_key__"
    if location_id_col and location_type_col:
        nw_df = nw_df.with_columns(
            (
                nw.col(location_id_col).cast(nw.String)
                + nw.lit("_")
                + nw.col(location_type_col).cast(nw.String)
            ).alias(location_key_col)
        )
    elif location_id_col:
        nw_df = nw_df.with_columns(nw.col(location_id_col).cast(nw.String).alias(location_key_col))
    else:
        nw_df = nw_df.with_columns(nw.lit(None).alias(location_key_col))

    if user_id_col:
        nw_df = nw_df.sort(user_id_col)
        uid_values, ranges = _build_user_ranges(nw_df, user_id_col)
        tokens = nw_df.get_column(location_key_col).to_list()
        values = [fast_diversity(tokens[start:end]) if start < end and location_id_col else 0.0 for start, end in ranges]
        return nw.from_dict(
            {user_id_col: uid_values, "diversity": values},
            backend=nw_df.implementation,
        ).to_native()

    tokens = nw_df.get_column(location_key_col).to_list() if location_id_col else []
    div = fast_diversity(tokens) if tokens else 0.0
    return nw.from_dict({"diversity": [div]}, backend=nw_df.implementation).to_native()
