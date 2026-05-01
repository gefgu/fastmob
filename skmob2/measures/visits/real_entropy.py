"""Real entropy of individual mobility trajectories (Kontoyiannis estimator)."""

from __future__ import annotations

from typing import Any

import numpy as np
import narwhals as nw

from .._common import _prepare_trajectory


def _skmob_true_entropy(sequence: list) -> float:
    """Match scikit-mobility's private _true_entropy estimator."""
    n = len(sequence)
    if n <= 1:
        return 0.0

    sum_lambda = 3.0

    def in_seq(prefix: list, candidate: list) -> bool:
        for i in range(len(prefix) - len(candidate) + 1):
            if prefix[i : i + len(candidate)] == candidate:
                return True
        return False

    for i in range(1, n - 1):
        j = i + 1
        while j < n and in_seq(sequence[:i], sequence[i:j]):
            j += 1
        if j == n:
            j += 1
        sum_lambda += j - i

    return float(n * np.log2(n) / sum_lambda)


def real_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the real (true) entropy of mobility for each user.

    Real entropy is estimated using the Kontoyiannis (1998) Lempel-Ziv
    entropy rate estimator applied to the sequence of visited locations.
    Each location is encoded as the string ``"<lat>_<lng>"`` using exact
    float equality — matching the skmob convention (no spatial clustering).

    The estimator captures both the frequency and the order of visits,
    unlike random entropy (which ignores order) and uncorrelated entropy
    (which ignores temporal correlations).

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
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
        One row per user with columns ``[uid_col, "real_entropy"]``.
        The returned backend matches the input backend.


    References
    ----------
    - [SQBB2010] Song, C., Qu, Z., Blumm, N. & Barabasi, A. L. (2010) Limits of Predictability in Human Mobility. Science 327(5968), 1018-1021, https://science.sciencemag.org/content/327/5968/1018

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

    # Encode each (lat, lng) pair as a string token for the LZ estimator.
    loc_key_col = "__skmob2_loc_key__"
    df = df.with_columns(
        (nw.col(lat_col).cast(nw.String) + nw.lit("_") + nw.col(lng_col).cast(nw.String)).alias(loc_key_col)
    )

    if uid_col is None:
        sequence = df.get_column(loc_key_col).to_list()
        entropy = _skmob_true_entropy(sequence)
        return nw.from_dict(
            {"real_entropy": [entropy]},
            backend=df.implementation,
        ).to_native()

    # Iterate per user in chronological order (already sorted by _prepare_trajectory).
    uid_series = df.get_column(uid_col).to_list()
    loc_series = df.get_column(loc_key_col).to_list()

    # Collect per-user sequences preserving row order.
    user_sequences: dict[Any, list[str]] = {}
    user_order: list[Any] = []
    for uid_val, loc in zip(uid_series, loc_series):
        if uid_val not in user_sequences:
            user_sequences[uid_val] = []
            user_order.append(uid_val)
        user_sequences[uid_val].append(loc)

    uid_vals = user_order
    entropies = [_skmob_true_entropy(user_sequences[u]) for u in uid_vals]

    return nw.from_dict(
        {uid_col: uid_vals, "real_entropy": entropies},
        backend=df.implementation,
    ).to_native()
