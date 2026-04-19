from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


def location_frequency(
    traj: Any,
    *,
    normalize: bool = False,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return visit frequency for each distinct location per user.

    Counts how many times each ``(lat, lng)`` location was visited by each
    user, optionally normalizing counts to visit probabilities that sum to 1
    within each user.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    normalize:
        When ``True``, the ``"frequency"`` column contains the probability
        of visiting that location (count / total_visits), so each user's
        frequencies sum to 1.0.  When ``False`` (default) raw visit counts
        are returned.
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
        One row per ``(user, location)`` pair with columns
        ``[uid_col, lat_col, lng_col, "frequency"]``.
        When ``uid_col`` is None the uid column is omitted.
        Rows are sorted by user then by frequency descending.
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

    def _freq_for_user(user_df: nw.DataFrame) -> tuple[list, list, list]:
        """Compute per-location visit frequencies for a single user.

        Returns three parallel lists: lats, lngs, frequencies — one entry
        per distinct location, sorted by frequency descending.
        """
        lat_list = user_df.get_column(lat_col).to_list()
        lng_list = user_df.get_column(lng_col).to_list()

        counts: dict[tuple, int] = {}
        for lat, lng in zip(lat_list, lng_list):
            key = (lat, lng)
            counts[key] = counts.get(key, 0) + 1

        sorted_locs = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        total = sum(c for _, c in sorted_locs)
        lats = [loc[0] for loc, _ in sorted_locs]
        lngs = [loc[1] for loc, _ in sorted_locs]
        if normalize:
            freqs: list = [c / total for _, c in sorted_locs]
        else:
            freqs = [float(c) for _, c in sorted_locs]

        return lats, lngs, freqs

    if uid_col is None:
        lats, lngs, freqs = _freq_for_user(df)
        return nw.from_dict(
            {lat_col: lats, lng_col: lngs, "frequency": freqs},
            backend=df.implementation,
        ).to_native()

    uid_vals_all: list = []
    lats_all: list = []
    lngs_all: list = []
    freqs_all: list = []

    uid_list = df.get_column(uid_col).unique().sort().to_list()
    for uid in uid_list:
        user_df = df.filter(nw.col(uid_col) == uid)
        lats, lngs, freqs = _freq_for_user(user_df)
        uid_vals_all.extend([uid] * len(lats))
        lats_all.extend(lats)
        lngs_all.extend(lngs)
        freqs_all.extend(freqs)

    return nw.from_dict(
        {
            uid_col: uid_vals_all,
            lat_col: lats_all,
            lng_col: lngs_all,
            "frequency": freqs_all,
        },
        backend=df.implementation,
    ).to_native()
