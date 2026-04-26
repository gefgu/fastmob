from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import radius_of_gyration_arrow, radius_of_gyration_numpy

from .._common import _build_user_ranges, _prepare_trajectory


def _is_polars_backed(nw_df: nw.DataFrame) -> bool:
    native = nw_df.to_native()
    return hasattr(native, "lazy")


def _route_and_call(
    lats: nw.Series,
    lngs: nw.Series,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
) -> list[float]:
    if use_arrow:
        return radius_of_gyration_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)

    return radius_of_gyration_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)


def radius_of_gyration(
    traj: Any,
    show_progress: bool = True,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
):
    """Compute the radius of gyration (km) for each user in the trajectory.

    The radius of gyration captures how far a user typically roams from their
    center of mass.  Formally::

        rg(u) = sqrt( mean_i( haversine(r_i, r_cm)^2 ) )

    where ``r_cm`` is the arithmetic mean of the user's lat/lng coordinates.

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, …).  Must have columns for datetime, latitude, and longitude.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
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
        One row per user with columns ``[uid_col, "radius_of_gyration"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures import radius_of_gyration
    >>> df = pd.DataFrame({
    ...     "uid": ["a", "a", "a"],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ... })
    >>> radius_of_gyration(df)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        (rg,) = _route_and_call(lats_full, lngs_full, [(0, len(df))], use_arrow=use_arrow)
        return nw.from_dict({"radius_of_gyration": [rg]}, backend=df.implementation).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    # Single Rust call covering all users — eliminates per-user boundary crossings
    rog_values = _route_and_call(lats_full, lngs_full, ranges, use_arrow=use_arrow)

    result = nw.from_dict(
        {uid_col: uid_values, "radius_of_gyration": rog_values},
        backend=df.implementation,
    )

    return result.to_native()
