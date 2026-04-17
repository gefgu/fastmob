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
)


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
