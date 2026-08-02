from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import (
    individual_mobility_network_indexed,
    individual_mobility_network_presorted,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _as_arrow,
    _build_indexed_user_ranges,
    _build_presorted_user_ends,
    _detect_trajectory_columns,
    _extract_timestamps,
    _take_uid_values,
    _to_native,
    _with_datetime_column,
)

_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _unpack_network(raw: tuple[Any, ...]) -> tuple[Any, ...]:
    return (
        _as_arrow(raw[0]),
        _as_arrow(raw[1]),
        _as_arrow(raw[2]),
        _as_arrow(raw[3]),
        _as_arrow(raw[4]),
        raw[5],
    )


def individual_mobility_network(
    traj: Any,
    *,
    self_loops: bool = False,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
) -> Any:
    """Return the individual mobility network as a directed edge-list DataFrame.

    An Individual Mobility Network (IMN) of an individual :math:`u` is a
    directed weighted graph :math:`G_u = (V, E)` where :math:`V` is the set
    of distinct visited locations and :math:`E` is the set of directed trips
    between locations [RGNPPG2014]_ [BL2012]_ [SQBB2010]_.  The edge weight
    function

    .. math::

        \\omega: E \\to \\mathbb{N}

    returns the number of times :math:`u` travelled that edge.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    self_loops : bool, optional
        When ``False`` (default), consecutive visits to the same location are
        skipped so no self-loop edges are created.  When ``True``, a step
        that stays at the same location contributes to a ``(loc, loc)`` edge.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    lat_col : str or None, optional
        Explicit latitude column name.  Auto-detected when None.
    lng_col : str or None, optional
        Explicit longitude column name.  Auto-detected when None.
    uid_col : str or None, optional
        Explicit user-ID column name.  Auto-detected when None.
    presorted : bool, optional
        When True, trust that rows are already grouped by user and ordered by
        datetime within each user, then use the contiguous fast path.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per directed edge with columns
        ``[uid_col, "lat_origin", "lng_origin", "lat_dest", "lng_dest",
        "n_trips"]``.
        When ``uid_col`` is None the uid column is omitted.
        The returned backend matches the input backend.



    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.utils.constants.BRIGHTKITE_SAMPLE
    >>> df = pd.read_csv(
    ...     url,
    ...     sep="\\t",
    ...     header=0,
    ...     nrows=5000,
    ...     names=["uid", "datetime", "lat", "lng", "location id"],
    ... )
    >>> df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    >>> df["location_id"] = df["location id"].astype("string")
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng", "location_id"]
    ... ]
    >>> print(df.head().to_string(index=False))
     uid                  datetime       lat         lng                              location_id
       0 2010-10-16 06:02:04+00:00 39.891383 -105.070814         7a0f88982aa015062b95e3b4843f9ca2
       0 2010-10-16 03:48:54+00:00 39.891077 -105.068532         dd7cd3d264c2d063832db506fba8bf79
       0 2010-10-14 18:25:51+00:00 39.750469 -104.999073 9848afcc62e500a01cf6fbf24b797732f8963683
       0 2010-10-14 00:21:47+00:00 39.752713 -104.996337         2ef143e12038c870038df53e0478cefc
       0 2010-10-13 23:31:51+00:00 39.752508 -104.996637         424eb3dd143292f9e013efa00486c907
    >>> from fastmob import individual_mobility_network
    >>> result = individual_mobility_network(df)
    >>> print(result.round({"lat_origin": 3, "lng_origin": 3, "lat_dest": 3, "lng_dest": 3}).head().to_string(index=False))
     uid  lat_origin  lng_origin  lat_dest  lng_dest  n_trips
       0      37.775    -122.419    37.601  -122.382        1
       0      37.601    -122.382    37.615  -122.390        1
       0      37.615    -122.390    39.879  -104.682        1
       0      39.879    -104.682    39.739  -104.985        1
       0      39.739    -104.985    39.762  -104.982       19

    References
    ----------
    - [RGNPPG2014] Rinzivillo, S., Gabrielli, L., Nanni, M., Pappalardo, L., Pedreschi, D. & Giannotti, F. (2012) The purpose of motion: Learning activities from Individual Mobility Networks. Proceedings of the 2014 IEEE International Conference on Data Science and Advanced Analytics, 312-318, https://ieeexplore.ieee.org/document/7058090
    - [BL2012] Bagrow, J. P. & Lin, Y.-R. (2012) Mesoscopic Structure and Social Aspects of Human Mobility. PLOS ONE 7(5): e37676. https://doi.org/10.1371/journal.pone.0037676
    - [SQBB2010] Song, C., Qu, Z., Blumm, N. & Barabasi, A. L. (2010) Limits of Predictability in Human Mobility. Science 327(5968), 1018-1021, https://science.sciencemag.org/content/327/5968/1018
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = _with_datetime_column(df, datetime_col)
    df = df.drop_nulls(subset=[datetime_col]).with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )

    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()

    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        raw = individual_mobility_network_presorted(lats_data, lngs_data, ends, self_loops)
    else:
        timestamps = _extract_timestamps(df, datetime_col)
        uid_values, indices, ends = _build_indexed_user_ranges(df, uid_col, timestamps)
        raw = individual_mobility_network_indexed(lats_data, lngs_data, indices, ends, self_loops)
    lat_origins, lng_origins, lat_dests, lng_dests, n_trips, user_indices = _unpack_network(raw)

    result_dict: dict[str, Any] = {
        "lat_origin": lat_origins,
        "lng_origin": lng_origins,
        "lat_dest": lat_dests,
        "lng_dest": lng_dests,
        "n_trips": n_trips,
    }
    if uid_col is not None:
        result_dict = {uid_col: _take_uid_values(uid_values, user_indices), **result_dict}
    return _to_native(result_dict, df)
