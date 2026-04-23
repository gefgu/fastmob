"""Trajectory entropy and predictability measures (Kontoyiannis + Fano)."""
from __future__ import annotations

from typing import Any

import numpy as np
import narwhals as nw

from .._common import (
    _pick_existing_column,
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
    """Fano inequality residual.  Used by the bisection solver."""
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
    pandas.DataFrame
        One row per user with columns ``[user_id_col, "entropy"]``.
        Always returns a pandas DataFrame because the internal algorithm
        requires pandas groupby iteration.
    """
    import pandas as pd  # noqa: PLC0415 — deferred: pandas iteration is unavoidable here

    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)
    if timestamp_col is None:
        timestamp_col = _pick_existing_column(nw_df.columns, TIMESTAMP_CANDIDATES)

    df = nw_df.to_native()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Sort by [user, timestamp] if timestamp is available
    if user_id_col and timestamp_col:
        df = df.sort_values([user_id_col, timestamp_col])
    elif timestamp_col:
        df = df.sort_values(timestamp_col)

    def _build_sequence(user_df: pd.DataFrame) -> list:
        if location_id_col and location_type_col:
            return (
                user_df[location_id_col].astype(str)
                + "_"
                + user_df[location_type_col].astype(str)
            ).tolist()
        elif location_id_col:
            return user_df[location_id_col].astype(str).tolist()
        return []

    def _compute_entropy(user_df: pd.DataFrame) -> float:
        seq = _build_sequence(user_df)
        n = len(seq)
        raw = _kontoyiannis_entropy(seq)
        if not normalized:
            return raw
        if n <= 1:
            return 0.0
        return float(np.clip(raw / np.log2(n), 0.0, 1.0))

    results = []
    if user_id_col:
        for uid, group in df.groupby(user_id_col, sort=False):
            entropy = _compute_entropy(group)
            results.append({user_id_col: uid, "entropy": entropy})
        out_df = pd.DataFrame(results)
    else:
        entropy = _compute_entropy(df)
        out_df = pd.DataFrame([{"entropy": entropy}])

    return out_df


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
    pandas.DataFrame
        One row per user with columns
        ``[user_id_col, "real_entropy", "predictability",
        "n_unique_locations", "n_steps"]``.
        Always returns a pandas DataFrame because the internal algorithm
        requires pandas groupby iteration.
    """
    import pandas as pd  # noqa: PLC0415 — deferred: pandas iteration is unavoidable here

    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if location_type_col is None:
        location_type_col = _pick_existing_column(nw_df.columns, LOCATION_TYPE_CANDIDATES)
    if timestamp_col is None:
        timestamp_col = _pick_existing_column(nw_df.columns, TIMESTAMP_CANDIDATES)

    df = nw_df.to_native()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if user_id_col and timestamp_col:
        df = df.sort_values([user_id_col, timestamp_col])
    elif timestamp_col:
        df = df.sort_values(timestamp_col)

    def _build_sequence(user_df: pd.DataFrame) -> list:
        if location_id_col and location_type_col:
            return (
                user_df[location_id_col].astype(str)
                + "_"
                + user_df[location_type_col].astype(str)
            ).tolist()
        elif location_id_col:
            return user_df[location_id_col].astype(str).tolist()
        return []

    def _compute_single(user_df: pd.DataFrame) -> dict:
        seq = _build_sequence(user_df)
        n_steps = len(seq)
        n_unique = len(set(seq))
        real_entropy = _kontoyiannis_entropy(seq)
        predictability = _solve_max_predictability_with_fano(real_entropy, n_unique)
        return {
            "real_entropy": float(real_entropy),
            "predictability": float(predictability),
            "n_unique_locations": int(n_unique),
            "n_steps": int(n_steps),
        }

    results = []
    if user_id_col:
        for uid, group in df.groupby(user_id_col, sort=False):
            row = _compute_single(group)
            row[user_id_col] = uid
            results.append(row)
        out_df = pd.DataFrame(results)
        # Reorder columns to put user_id first
        cols = [user_id_col, "real_entropy", "predictability", "n_unique_locations", "n_steps"]
        out_df = out_df[cols]
    else:
        row = _compute_single(df)
        out_df = pd.DataFrame([row])

    return out_df
