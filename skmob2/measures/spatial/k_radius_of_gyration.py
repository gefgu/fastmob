from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import k_radius_of_gyration_km

from .._common import _prepare_trajectory


def k_radius_of_gyration(
    traj: Any,
    k: int = 2,
    show_progress: bool = True,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
):
    """Compute the k-radius of gyration (km) for each user in the trajectory.

    The k-radius of gyration is the radius of gyration computed using only the
    k most-visited locations for each user.  Formally::

        krg(u, k) = sqrt(
            sum_{i in top-k}( w_i * haversine(r_i, r_cm)^2 )
            / sum_{i in top-k}( w_i )
        )

    where ``r_cm`` is the weighted center of mass over the top-k locations and
    ``w_i`` is the visit count of location i.

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, ...).  Must have columns for datetime, latitude, and longitude.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    k:
        Number of most-visited locations to consider.  Defaults to 2.
    show_progress:
        Accepted for API compatibility with skmob; currently unused.
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
        One row per user with columns ``[uid_col, "k_radius_of_gyration"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.spatial.k_radius_of_gyration import k_radius_of_gyration
    >>> df = pd.DataFrame({
    ...     "uid": ["a", "a", "a", "a"],
    ...     "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
    ...     "lat": [0.0, 1.0, 0.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0, 0.0],
    ... })
    >>> k_radius_of_gyration(df, k=2)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    if uid_col is None:
        lats = df.get_column(lat_col).to_list()
        lngs = df.get_column(lng_col).to_list()
        krg = _k_rog_for_sequence(lats, lngs, k)
        return nw.from_dict({"k_radius_of_gyration": [krg]}, backend=df.implementation).to_native()

    # Build per-user index ranges from the already-sorted dataframe.
    uid_series = df.get_column(uid_col).to_list()
    lats_full = df.get_column(lat_col).to_list()
    lngs_full = df.get_column(lng_col).to_list()

    uid_values: list = []
    krg_values: list = []

    i = 0
    n = len(uid_series)
    while i < n:
        current_uid = uid_series[i]
        start = i
        while i < n and uid_series[i] == current_uid:
            i += 1
        lats = lats_full[start:i]
        lngs = lngs_full[start:i]
        krg = _k_rog_for_sequence(lats, lngs, k)
        uid_values.append(current_uid)
        krg_values.append(krg)

    result = nw.from_dict(
        {uid_col: uid_values, "k_radius_of_gyration": krg_values},
        backend=df.implementation,
    )
    return result.to_native()


def _k_rog_for_sequence(lats: list[float], lngs: list[float], k: int) -> float:
    """Compute k-radius of gyration for one user's trajectory sequence.

    Counts visits per distinct (lat, lng) location, then delegates to the
    Rust kernel ``k_radius_of_gyration_km`` with the k most-visited locations.

    Parameters
    ----------
    lats:
        List of latitude values for a single user, sorted by datetime.
    lngs:
        List of longitude values for a single user, sorted by datetime.
    k:
        Number of most-visited locations to include.

    Returns
    -------
    float
        The k-radius of gyration in kilometres.

    @usedBy skmob2/measures/spatial/k_radius_of_gyration.py
    """
    # Count visits per distinct location using an ordered dict for determinism.
    from collections import Counter

    counts: Counter[tuple[float, float]] = Counter(zip(lats, lngs))
    if not counts:
        return 0.0

    coords = list(counts.keys())
    visit_counts = [counts[c] for c in coords]

    return k_radius_of_gyration_km(coords, visit_counts, k)
