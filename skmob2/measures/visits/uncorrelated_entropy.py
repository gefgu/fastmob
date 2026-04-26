from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory, _shannon_entropy


def uncorrelated_entropy(
    traj: Any,
    *,
    normalize: bool = False,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the uncorrelated entropy of mobility for each user.

    Uncorrelated entropy is the Shannon entropy over the distribution of
    visit probabilities across distinct locations::

        S_unc = -sum(p_i * log2(p_i))

    where ``p_i`` is the fraction of visits to location ``i`` out of the
    user's total visits.  A location is a unique exact ``(lat, lng)`` pair.

    When ``normalize=True`` the result is divided by ``log2(n)`` (the
    random entropy) so the output lies in ``[0, 1]``.  If the user visits
    only one distinct location the entropy is 0; dividing by 0 is avoided
    by returning 0 directly.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    normalize:
        When True, divide the Shannon entropy by ``log2(n_distinct_locations)``
        to normalise into ``[0, 1]``.  Default False.
    datetime_col:
        Explicit datetime column name.  Auto-detected when None.
    lat_col:
        Explicit latitude column name.  Auto-detected when None.
    lng_col:
        Explicit longitude column name.  Auto-detected when None.
    uid_col:
        Explicit user-ID column name.  Auto-detected when None.

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, "uncorrelated_entropy"]``.
        The returned backend matches the input backend.

    @usedBy
        skmob2.measures.visits.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    loc_key_col = "__skmob2_loc_key__"
    df = df.with_columns(
        (nw.col(lat_col).cast(nw.String) + nw.lit("_") + nw.col(lng_col).cast(nw.String)).alias(loc_key_col)
    )

    def _compute_entropy_for_user(user_df: nw.DataFrame) -> float:
        """Compute uncorrelated entropy for rows belonging to a single user."""
        loc_series = user_df.get_column(loc_key_col)
        loc_list = loc_series.to_list()
        counts: dict[str, int] = {}
        for loc in loc_list:
            counts[loc] = counts.get(loc, 0) + 1
        entropy = _shannon_entropy(list(counts.values()))
        if normalize:
            n = len(counts)
            if n > 1:
                entropy = entropy / math.log2(n)
            else:
                entropy = 0.0
        return entropy

    if uid_col is None:
        entropy = _compute_entropy_for_user(df)
        return nw.from_dict(
            {"uncorrelated_entropy": [entropy]},
            backend=df.implementation,
        ).to_native()

    uid_vals = df.get_column(uid_col).unique().sort().to_list()
    entropies = []
    for uid in uid_vals:
        user_df = df.filter(nw.col(uid_col) == uid)
        entropies.append(_compute_entropy_for_user(user_df))

    return nw.from_dict(
        {uid_col: uid_vals, "uncorrelated_entropy": entropies},
        backend=df.implementation,
    ).to_native()
