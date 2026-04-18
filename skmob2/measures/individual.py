"""Individual-level mobility measures."""
from __future__ import annotations

from typing import Any

import numpy as np
import narwhals as nw
import pandas as pd

from ._common import (
    _pick_existing_column,
    USER_ID_CANDIDATES,
    LOCATION_CANDIDATES,
    DURATION_CANDIDATES,
    PURPOSE_CANDIDATES,
    LOCATION_TYPE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from pydivsufsort import divsufsort as _divsufsort, kasai as _kasai
except ImportError:
    _divsufsort = None
    _kasai = None


# ---------------------------------------------------------------------------
# Intermittance and Degree of Return
# ---------------------------------------------------------------------------

def intermittance_and_degree_of_return(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    duration_col: str | None = None,
    purpose_col: str | None = None,
    home_purposes: frozenset = frozenset({"HOME", "WORK"}),
    combine_purpose_with_location: bool = True,
) -> pd.DataFrame:
    """Compute intermittancy and degree of return per user.

    For each user, partitions the visit sequence into alternating blocks of
    *explorations* (new places) and *returns* (revisits or home/work visits).
    Then computes summary statistics over those blocks.

    Parameters
    ----------
    visits:
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col:
        Column name for the user ID. Auto-detected if None.
    location_id_col:
        Column name for the location ID. Auto-detected if None.
    duration_col:
        Column name for the visit duration (numeric). Auto-detected if None.
    purpose_col:
        Column name for the activity/purpose type. Auto-detected if None.
    home_purposes:
        Set of purpose strings treated as "known places" unconditionally
        (independent of visit history). Default ``{"HOME", "WORK"}``.
    combine_purpose_with_location:
        When True (default), the effective location key is
        ``str(location_id) + "_" + str(purpose)``. When False, just
        ``str(location_id)``.

    Returns
    -------
    pd.DataFrame
        One row per user with columns
        ``[user_id_col, "intermittency", "degree_of_return", "mean_return",
        "mean_exploration"]``.
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if duration_col is None:
        duration_col = _pick_existing_column(nw_df.columns, DURATION_CANDIDATES)
    if purpose_col is None:
        purpose_col = _pick_existing_column(nw_df.columns, PURPOSE_CANDIDATES)

    # Convert to pandas for the row-iterative inner loop
    df = nw_df.to_native()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    results = []
    if user_id_col:
        user_order = df[user_id_col].unique().tolist()
        for uid, group in df.groupby(user_id_col, sort=False):
            metrics = _compute_single_idr(
                group, location_id_col, duration_col, purpose_col,
                home_purposes, combine_purpose_with_location,
            )
            results.append((uid, *metrics))
        out_df = pd.DataFrame(
            results,
            columns=[user_id_col, "intermittency", "degree_of_return",
                     "mean_return", "mean_exploration"],
        )
        # Preserve original user order
        out_df[user_id_col] = pd.Categorical(out_df[user_id_col], categories=user_order, ordered=True)
        out_df = out_df.sort_values(user_id_col).reset_index(drop=True)
        out_df[user_id_col] = out_df[user_id_col].astype(df[user_id_col].dtype)
    else:
        metrics = _compute_single_idr(
            df, location_id_col, duration_col, purpose_col,
            home_purposes, combine_purpose_with_location,
        )
        out_df = pd.DataFrame(
            [metrics],
            columns=["intermittency", "degree_of_return", "mean_return", "mean_exploration"],
        )

    return out_df


# ---------------------------------------------------------------------------
# Regularity
# ---------------------------------------------------------------------------

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


def _compute_single_idr(
    user_visits: pd.DataFrame,
    location_id_col: str | None,
    duration_col: str | None,
    purpose_col: str | None,
    home_purposes: frozenset,
    combine_purpose_with_location: bool,
) -> tuple[float, float, float, float]:
    """Inner loop for a single user's intermittance computation."""
    successive_explorations: list[float] = []
    successive_returns: list[float] = []
    visited_locations: set = set()

    current_exploration: float = 0.0
    current_return: float = 0.0

    for _, row in user_visits.iterrows():
        # Build location key
        loc = str(row[location_id_col]) if location_id_col else "unknown"
        purpose = str(row[purpose_col]) if purpose_col else None

        if combine_purpose_with_location and purpose is not None:
            location_key = loc + "_" + purpose
        else:
            location_key = loc

        duration = float(row[duration_col]) if duration_col else 1.0

        # Determine if this is a known place (return) or new place (exploration)
        is_known_place = location_key in visited_locations or (
            purpose is not None and purpose in home_purposes
        )

        visited_locations.add(location_key)

        if is_known_place:
            current_return += duration
            if current_exploration > 0:
                successive_explorations.append(current_exploration)
                current_exploration = 0.0
        else:
            current_exploration += duration
            if current_return > 0:
                successive_returns.append(current_return)
                current_return = 0.0

    if current_exploration > 0:
        successive_explorations.append(current_exploration)
    if current_return > 0:
        successive_returns.append(current_return)

    if not successive_explorations:
        successive_explorations.append(0.0)
    if not successive_returns:
        successive_returns.append(0.0)

    mean_exploration = float(np.mean(successive_explorations))
    mean_return = float(np.mean(successive_returns))

    intermittency = mean_exploration + mean_return
    if mean_exploration == 0.0:
        degree_of_return = np.pi / 2
    else:
        degree_of_return = np.arctan(mean_return / mean_exploration)

    return intermittency, float(degree_of_return), mean_return, mean_exploration


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def fast_diversity(sequence) -> float:
    """Compute the diversity of a sequence using suffix arrays.

    Maps elements to integer codes, builds a suffix array and LCP array via
    ``pydivsufsort``, then returns ``distinct_substrings / total_substrings``.

    Parameters
    ----------
    sequence:
        An iterable of hashable elements.

    Returns
    -------
    float
        A value in ``[0, 1]`` where 0 means no diversity (constant sequence)
        and values approaching 1 mean high diversity.

    Raises
    ------
    ImportError
        When ``pydivsufsort`` is not installed.
    """
    if _divsufsort is None:
        raise ImportError(
            "pydivsufsort is required: pip install skmob2[diversity]"
        )

    items = list(sequence)
    n = len(items)
    if n <= 1:
        return 0.0

    # Map distinct elements to int32 codes
    unique_map = {x: i for i, x in enumerate(set(items))}
    int_seq = np.array([unique_map[x] for x in items], dtype=np.int32)

    sa = _divsufsort(int_seq)
    lcp = _kasai(int_seq, sa)

    # Standard formula: sum of (suffix_length - lcp) per suffix array entry
    # gives the total count of distinct substrings.
    distinct_substrings = int(np.sum((n - sa) - lcp))
    total_substrings = n * (n + 1) // 2

    if total_substrings == 0:
        return 0.0

    return distinct_substrings / total_substrings


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
            return (
                user_df[location_id_col].astype(str)
                + "_"
                + user_df[location_type_col].astype(str)
            ).tolist()
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


# ---------------------------------------------------------------------------
# Entropy and Predictability
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
    """
    sequence = [str(elem) for elem in sequence]
    n = len(sequence)

    if n <= 1:
        return 0.0

    # DP table: dp[i][j] = length of longest match ending at position j
    # with a match starting at position i
    dp = [[1] * n for _ in range(n)]

    for i in range(1, n):
        for j in range(i + 1, n):
            if sequence[i - 1] == sequence[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

    # lambdas = sum of max match length for each position
    lambdas = sum(max(column) for column in zip(*dp))

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


def trajectory_entropy(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    location_type_col: str | None = None,
    timestamp_col: str | None = None,
    normalized: bool = True,
) -> pd.DataFrame:
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
    pd.DataFrame
        One row per user with columns ``[user_id_col, "entropy"]``.
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
) -> pd.DataFrame:
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
    pd.DataFrame
        One row per user with columns
        ``[user_id_col, "real_entropy", "predictability",
        "n_unique_locations", "n_steps"]``.
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
