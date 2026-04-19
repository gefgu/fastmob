from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


def uncorrelated_location_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the uncorrelated entropy for each distinct location across all users.

    For each location ``l``, computes the Shannon entropy over the distribution
    of visit probabilities::

        S_unc(l) = -sum_u p(u, l) * log2(p(u, l))

    where ``p(u, l)`` is the fraction of all visits to ``l`` that belong to
    user ``u``.  A location is a unique exact ``(lat, lng)`` pair.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent all rows are treated as
        coming from a single individual (entropy will be 0 everywhere).
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
        One row per distinct ``(lat, lng)`` location with columns
        ``[lat_col, lng_col, "uncorrelated_entropy"]``.
        The returned backend matches the input backend.

    @usedBy
        skmob2.measures.flows.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    backend = df.implementation

    def _shannon(counts: list[int]) -> float:
        """Compute Shannon entropy in bits for a list of visit counts."""
        total = sum(counts)
        if total == 0:
            return 0.0
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        return entropy

    if uid_col is None:
        # Single user: each location has probability 1 -> entropy = 0.
        locs = (
            df.select([lat_col, lng_col])
            .unique()
            .sort([lat_col, lng_col])
        )
        result = locs.with_columns(
            nw.lit(0.0).alias("uncorrelated_entropy")
        )
        return result.to_native()

    # Count visits per (uid, lat, lng) triplet.
    visit_counts = (
        df.select([uid_col, lat_col, lng_col])
        .group_by([uid_col, lat_col, lng_col])
        .agg(nw.len().alias("__visits__"))
    )

    # Group per location and compute Shannon entropy over the per-user counts.
    loc_lats = (
        visit_counts
        .select([lat_col, lng_col])
        .unique()
        .sort([lat_col, lng_col])
        .get_column(lat_col)
        .to_list()
    )
    loc_lngs = (
        visit_counts
        .select([lat_col, lng_col])
        .unique()
        .sort([lat_col, lng_col])
        .get_column(lng_col)
        .to_list()
    )

    entropies = []
    for lat_v, lng_v in zip(loc_lats, loc_lngs):
        loc_rows = visit_counts.filter(
            (nw.col(lat_col) == lat_v) & (nw.col(lng_col) == lng_v)
        )
        counts = loc_rows.get_column("__visits__").to_list()
        entropies.append(_shannon(counts))

    return nw.from_dict(
        {lat_col: loc_lats, lng_col: loc_lngs, "uncorrelated_entropy": entropies},
        backend=backend,
    ).to_native()
