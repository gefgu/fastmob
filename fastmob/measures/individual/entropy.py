"""Trajectory entropy and predictability measures (Kontoyiannis + Fano)."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import pyarrow as pa

from fastmob._core import real_entropy_users as _real_entropy_users_rust
from fastmob._core import trajectory_predictability_batch as _trajectory_predictability_batch_rust
from fastmob.utils._common import (
    LOCATION_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _build_presorted_user_ends,
    _factorize_arrow_values,
    _pick_existing_column,
)


def _sort_visits(
    df: nw.DataFrame,
    user_id_col: str | None,
    timestamp_col: str | None,
) -> nw.DataFrame:
    """Sort visits chronologically while keeping user trajectories together."""
    if user_id_col and timestamp_col:
        return df.sort([user_id_col, timestamp_col])
    if timestamp_col:
        return df.sort(timestamp_col)
    return df



# ---------------------------------------------------------------------------
# Public measures
# ---------------------------------------------------------------------------


def trajectory_entropy(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    timestamp_col: str | None = None,
    normalized: bool = True,
) -> Any:
    """Compute Kontoyiannis entropy of mobility trajectories per user.

    Parameters
    ----------
    visits : DataFrame-like
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col : str or None, optional
        Column name for the user ID. Auto-detected if None.
    location_id_col : str or None, optional
        Column name for the location ID. Auto-detected if None.
    timestamp_col : str or None, optional
        Column name for ordering visits. Auto-detected if None; when no
        timestamp column is found the row order is preserved.
    normalized : bool, optional
        When True (default), divide raw entropy by ``log2(n)`` and clip to
        ``[0, 1]``. When False, return raw Kontoyiannis bits.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns ``[user_id_col, "entropy"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.individual import trajectory_entropy
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2"],
    ...         "start_timestamp": pd.to_datetime(
    ...             [
    ...                 "2020-01-01 08:00",
    ...                 "2020-01-01 09:00",
    ...                 "2020-01-01 18:00",
    ...                 "2020-01-02 08:00",
    ...                 "2020-01-01 07:30",
    ...                 "2020-01-01 12:00",
    ...                 "2020-01-01 19:00",
    ...             ]
    ...         ),
    ...         "location_id": ["home", "work", "home", "gym", "home", "shop", "home"],
    ...     }
    ... )
    >>> result = trajectory_entropy(visits)
    >>> print(result.round(3).to_string(index=False))
    user_id  entropy
         u1      0.8
         u2      1.0
    """
    nw_df = nw.from_native(visits, eager_only=True)

    user_id_col = user_id_col or _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    location_id_col = location_id_col or _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    timestamp_col = timestamp_col or _pick_existing_column(nw_df.columns, TIMESTAMP_CANDIDATES)

    df = _sort_visits(nw_df, user_id_col, timestamp_col)
    if location_id_col:
        df = df.filter(~nw.col(location_id_col).is_null())

    location_ids = (
        _factorize_arrow_values(df.get_column(location_id_col).to_arrow(), sort=False)[0]
        if location_id_col
        else pa.array([], type=pa.uint64())
    )

    if user_id_col:
        uid_values, ends = _build_presorted_user_ends(df, user_id_col)
        entropies = _real_entropy_users_rust(location_ids, ends, normalized)
        return nw.from_dict(
            {user_id_col: uid_values.to_pylist(), "entropy": entropies},
            backend=df.implementation,
        ).to_native()

    _, ends = _build_presorted_user_ends(df, None)
    entropies = _real_entropy_users_rust(location_ids, ends, normalized)
    return nw.from_dict(
        {"entropy": entropies},
        backend=df.implementation,
    ).to_native()


def trajectory_predictability(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    timestamp_col: str | None = None,
) -> Any:
    """Compute per-user maximum predictability via Fano's inequality.

    Follows the Song et al. (2010) approach: estimate real entropy using
    Kontoyiannis (1998), then solve Fano's inequality for the maximum
    predictability upper bound.

    Parameters
    ----------
    visits : DataFrame-like
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col : str or None, optional
        Column name for the user ID. Auto-detected if None.
    location_id_col : str or None, optional
        Column name for the location ID. Auto-detected if None.
    timestamp_col : str or None, optional
        Column name for ordering visits. Auto-detected if None.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns
        ``[user_id_col, "real_entropy", "predictability",
        "n_unique_locations", "n_steps"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.individual import trajectory_predictability
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2"],
    ...         "start_timestamp": pd.to_datetime(
    ...             [
    ...                 "2020-01-01 08:00",
    ...                 "2020-01-01 09:00",
    ...                 "2020-01-01 18:00",
    ...                 "2020-01-02 08:00",
    ...                 "2020-01-01 07:30",
    ...                 "2020-01-01 12:00",
    ...                 "2020-01-01 19:00",
    ...             ]
    ...         ),
    ...         "location_id": ["home", "work", "home", "gym", "home", "shop", "home"],
    ...     }
    ... )
    >>> result = trajectory_predictability(visits)
    >>> print(result.round(3).to_string(index=False))
    user_id  real_entropy  predictability  n_unique_locations  n_steps
         u1         1.600           0.333                   3        4
         u2         1.585           0.500                   2        3
    """
    nw_df = nw.from_native(visits, eager_only=True)

    user_id_col = user_id_col or _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    location_id_col = location_id_col or _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    timestamp_col = timestamp_col or _pick_existing_column(nw_df.columns, TIMESTAMP_CANDIDATES)

    df = _sort_visits(nw_df, user_id_col, timestamp_col)
    if location_id_col:
        df = df.filter(~nw.col(location_id_col).is_null())

    location_ids = (
        _factorize_arrow_values(df.get_column(location_id_col).to_arrow(), sort=False)[0]
        if location_id_col
        else pa.array([], type=pa.uint64())
    )

    if user_id_col:
        uid_values, ends = _build_presorted_user_ends(df, user_id_col)
        real_entropies, predictabilities, n_unique_locations, n_steps = _trajectory_predictability_batch_rust(
            location_ids,
            ends,
        )
        return nw.from_dict(
            {
                user_id_col: uid_values.to_pylist(),
                "real_entropy": real_entropies,
                "predictability": predictabilities,
                "n_unique_locations": n_unique_locations,
                "n_steps": n_steps,
            },
            backend=df.implementation,
        ).to_native()

    _, ends = _build_presorted_user_ends(df, None)
    real_entropies, predictabilities, n_unique_locations, n_steps = _trajectory_predictability_batch_rust(
        location_ids, ends
    )
    return nw.from_dict(
        {
            "real_entropy": real_entropies,
            "predictability": predictabilities,
            "n_unique_locations": n_unique_locations,
            "n_steps": n_steps,
        },
        backend=df.implementation,
    ).to_native()
