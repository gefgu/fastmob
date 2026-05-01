from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import k_radius_of_gyration_km

from .._common import _build_user_ranges, _prepare_trajectory


def k_radius_of_gyration(
    traj: Any,
    k: int = 2,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
):
    """Compute the k-radius of gyration (km) for each user in the trajectory.

    The k-radius of gyration is the radius of gyration computed using only the
    k most-visited locations for each user. Formally:

    \\[
    r_g^{(k)}(u) =
    \\sqrt{
        \\frac{\\sum_{i \\in \\mathrm{top}\\text{-}k} w_i d_\\mathrm{haversine}(r_i, r_\\mathrm{cm})^2}
             {\\sum_{i \\in \\mathrm{top}\\text{-}k} w_i}
    }
    \\]

    where \\(r_\\mathrm{cm}\\) is the weighted center of mass over the top-k
    locations and \\(w_i\\) is the visit count of location \\(i\\).

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, ...).  Must have columns for datetime, latitude, and longitude.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    k:
        Number of most-visited locations to consider.  Defaults to 2.
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


    References
    ----------
    - [PSRPGB2015] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F. & Barabasi, A. L. (2015) Returners and Explorers dichotomy in human mobility. Nature Communications 6, https://www.nature.com/articles/ncomms9166

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

    lats_full = df.get_column(lat_col).to_list()
    lngs_full = df.get_column(lng_col).to_list()
    datetimes_full = df.get_column(datetime_col).to_list()

    if uid_col is None:
        krg = _k_rog_for_sequence(lats_full, lngs_full, datetimes_full, k)
        return nw.from_dict({"k_radius_of_gyration": [krg]}, backend=df.implementation).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    krg_values = [
        _k_rog_for_sequence(lats_full[start:end], lngs_full[start:end], datetimes_full[start:end], k)
        for start, end in ranges
    ]

    result = nw.from_dict(
        {uid_col: uid_values, "k_radius_of_gyration": krg_values},
        backend=df.implementation,
    )
    return result.to_native()


def _k_rog_for_sequence(lats: list[float], lngs: list[float], datetimes: list, k: int) -> float:
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
    loc_stats: dict[tuple[float, float], list] = {}
    for order, (lat, lng, timestamp) in enumerate(zip(lats, lngs, datetimes)):
        key = (lat, lng)
        if key not in loc_stats:
            loc_stats[key] = [0, timestamp, order]
        loc_stats[key][0] += 1

    if not loc_stats:
        return 0.0

    sorted_locs = sorted(
        loc_stats.items(),
        key=lambda item: (-item[1][0], item[1][1], item[1][2]),
    )
    top_locs = sorted_locs[: min(k, len(sorted_locs))]
    coords = [loc for loc, _ in top_locs]
    visit_counts = [stats[0] for _, stats in top_locs]

    return k_radius_of_gyration_km(coords, visit_counts, len(coords))
