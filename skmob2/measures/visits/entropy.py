"""Trajectory entropy and predictability measures (Kontoyiannis + Fano)."""

from __future__ import annotations

from typing import Any

import numpy as np
import narwhals as nw
from skmob2._core import trajectory_entropy_batch as _trajectory_entropy_batch_rust
from skmob2._core import trajectory_predictability_batch as _trajectory_predictability_batch_rust

from .._common import (
    _pick_existing_column,
    _build_user_ranges,
    USER_ID_CANDIDATES,
    LOCATION_CANDIDATES,
    LOCATION_TYPE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
)


# ---------------------------------------------------------------------------
# Private helpers — entropy subsystem
# ---------------------------------------------------------------------------


def _kontoyiannis_entropy(sequence: list) -> float:
    """Kontoyiannis (1998) entropy estimator on a pre-tokenized sequence.

    Computes the Lempel-Ziv entropy rate estimate using the DP-based
    longest-match algorithm from Kontoyiannis et al. (1998).

    Parameters
    ----------
    sequence:
        A list of hashable tokens (strings or integers).

    Returns
    -------
    float
        Entropy in bits. Returns 0.0 for sequences of length <= 1 or
        constant sequences.

    Implementation notes
    --------------------
    The DP transition is ``dp[i][j] = dp[i-1][j-1] + 1`` when
    ``sequence[i-1] == sequence[j-1]``, else 1.  The full n×n table is not
    needed at once: only the previous row is required to compute the current
    row.  We therefore keep two 1D arrays (``prev_row`` and ``curr_row``) and
    accumulate the per-column maximum incrementally, reducing memory from
    O(n²) to O(n).
    """
    sequence = [str(elem) for elem in sequence]
    n = len(sequence)

    if n <= 1:
        return 0.0

    # col_max[j] tracks max(dp[0..i][j]) as we advance i.
    # Initialised to 1 because dp[0][j] == 1 for all j (base row).
    col_max = [1] * n
    prev_row = [1] * n

    for i in range(1, n):
        curr_row = [1] * n
        for j in range(i + 1, n):
            if sequence[i - 1] == sequence[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
            # else curr_row[j] remains 1 (initialised above)
            if curr_row[j] > col_max[j]:
                col_max[j] = curr_row[j]
        prev_row = curr_row

    lambdas = sum(col_max)

    if lambdas == 0:
        return 0.0

    return float((n / lambdas) * np.log2(n))


def _fano_equation_term(predictability: float, real_entropy: float, n_unique: int) -> float:
    """Fano inequality residual."""
    p = float(np.clip(predictability, 1e-12, 1 - 1e-12))
    binary_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return binary_entropy + (1 - p) * np.log2(n_unique - 1) - real_entropy


def _solve_max_predictability_with_fano(
    real_entropy: float,
    n_unique: int,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Solve Fano's inequality for max predictability via bisection.

    Parameters
    ----------
    real_entropy:
        Estimated entropy in bits.
    n_unique:
        Number of distinct location tokens in the user's trajectory.
    max_iter:
        Maximum bisection iterations.
    tol:
        Convergence tolerance.

    Returns
    -------
    float
        Maximum predictability in ``[1/n_unique, 1]``.
    """
    if n_unique <= 1:
        return 1.0

    entropy_upper_bound = float(np.log2(n_unique))
    bounded_entropy = float(np.clip(real_entropy, 0.0, entropy_upper_bound))

    if np.isclose(bounded_entropy, 0.0):
        return 1.0
    if np.isclose(bounded_entropy, entropy_upper_bound):
        return float(1.0 / n_unique)

    low = 1.0 / n_unique
    high = 1.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        value = _fano_equation_term(mid, bounded_entropy, n_unique)

        if abs(value) < tol:
            return float(mid)

        if value > 0:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


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


def _with_location_key(
    df: nw.DataFrame,
    location_id_col: str | None,
    location_type_col: str | None,
    location_key_col: str,
) -> nw.DataFrame:
    """Add the token column consumed by the trajectory entropy estimator."""
    if location_id_col and location_type_col:
        return df.with_columns(
            (
                nw.col(location_id_col).cast(nw.String)
                + nw.lit("_")
                + nw.col(location_type_col).cast(nw.String)
            ).alias(location_key_col)
        )
    if location_id_col:
        return df.with_columns(nw.col(location_id_col).cast(nw.String).alias(location_key_col))
    return df


# ---------------------------------------------------------------------------
# Public measures
# ---------------------------------------------------------------------------


def trajectory_entropy(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
    timestamp_col: str | None = None,
    normalized: bool = True,
) -> Any:
    """Compute Kontoyiannis entropy of mobility trajectories per user.

    Parameters
    ----------
    visits:
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col:
        Column name for the user ID. Auto-detected if None.
    location_id_col:
        Column name for the location ID. Auto-detected if None.
    location_type_col:
        Column name for the location type / activity purpose. Auto-detected
        if None; set explicitly to ``None`` to disable.
    timestamp_col:
        Column name for ordering visits. Auto-detected if None; when no
        timestamp column is found the row order is preserved.
    normalized:
        When True (default), divide raw entropy by ``log2(n)`` and clip to
        ``[0, 1]``. When False, return raw Kontoyiannis bits.

    Returns
    -------
    DataFrame
        One row per user with columns ``[user_id_col, "entropy"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.visits import trajectory_entropy
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

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)
    if timestamp_col is None:
        timestamp_col = _pick_existing_column(nw_df.columns, TIMESTAMP_CANDIDATES)

    location_key_col = "__skmob2_location_key__"
    has_location_key = bool(location_id_col)
    df = _sort_visits(nw_df, user_id_col, timestamp_col)
    df = _with_location_key(df, location_id_col, location_type_col, location_key_col)

    tokens = df.get_column(location_key_col).to_list() if has_location_key else []

    if user_id_col:
        uid_values, ranges = _build_user_ranges(df, user_id_col)
        entropies = _trajectory_entropy_batch_rust(tokens, ranges, normalized)
        return nw.from_dict(
            {user_id_col: uid_values, "entropy": entropies},
            backend=df.implementation,
        ).to_native()

    entropies = _trajectory_entropy_batch_rust(tokens, [(0, len(tokens))], normalized)
    return nw.from_dict(
        {"entropy": entropies},
        backend=df.implementation,
    ).to_native()


def trajectory_predictability(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
    timestamp_col: str | None = None,
) -> Any:
    """Compute per-user maximum predictability via Fano's inequality.

    Follows the Song et al. (2010) approach: estimate real entropy using
    Kontoyiannis (1998), then solve Fano's inequality for the maximum
    predictability upper bound.

    Parameters
    ----------
    visits:
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col:
        Column name for the user ID. Auto-detected if None.
    location_id_col:
        Column name for the location ID. Auto-detected if None.
    location_type_col:
        Column name for the location type / activity purpose. Auto-detected
        if None; set explicitly to ``None`` to disable.
    timestamp_col:
        Column name for ordering visits. Auto-detected if None.

    Returns
    -------
    DataFrame
        One row per user with columns
        ``[user_id_col, "real_entropy", "predictability",
        "n_unique_locations", "n_steps"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.visits import trajectory_predictability
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

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)
    if timestamp_col is None:
        timestamp_col = _pick_existing_column(nw_df.columns, TIMESTAMP_CANDIDATES)

    location_key_col = "__skmob2_location_key__"
    has_location_key = bool(location_id_col)
    df = _sort_visits(nw_df, user_id_col, timestamp_col)
    df = _with_location_key(df, location_id_col, location_type_col, location_key_col)

    tokens = df.get_column(location_key_col).to_list() if has_location_key else []

    if user_id_col:
        uid_values, ranges = _build_user_ranges(df, user_id_col)
        real_entropies, predictabilities, n_unique_locations, n_steps = _trajectory_predictability_batch_rust(
            tokens,
            ranges,
        )
        return nw.from_dict(
            {
                user_id_col: uid_values,
                "real_entropy": real_entropies,
                "predictability": predictabilities,
                "n_unique_locations": n_unique_locations,
                "n_steps": n_steps,
            },
            backend=df.implementation,
        ).to_native()

    real_entropies, predictabilities, n_unique_locations, n_steps = _trajectory_predictability_batch_rust(
        tokens,
        [(0, len(tokens))],
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
