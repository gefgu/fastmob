from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _build_user_ranges, _prepare_trajectory


def individual_mobility_network(
    traj: Any,
    *,
    self_loops: bool = False,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the individual mobility network as a directed edge-list DataFrame.

    For each user, builds a directed weighted graph where nodes are distinct
    locations (``lat``, ``lng`` pairs) and each edge ``(origin -> dest)``
    represents a transition between consecutive trajectory points.  The edge
    weight ``n_trips`` counts how many times that transition occurred.

    The output format matches the skmob ``individual.individual_mobility_network``
    return shape: one row per ``(uid, origin, destination)`` triple.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    self_loops:
        When ``False`` (default), consecutive visits to the same location are
        skipped so no self-loop edges are created.  When ``True``, a step
        that stays at the same location contributes to a ``(loc, loc)`` edge.
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
        One row per directed edge with columns
        ``[uid_col, "lat_origin", "lng_origin", "lat_dest", "lng_dest",
        "n_trips"]``.
        When ``uid_col`` is None the uid column is omitted.
        The returned backend matches the input backend.



    Examples
    --------
    >>> import pandas as pd
    >>> import skmob2
    >>> url = skmob2.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from skmob2 import individual_mobility_network
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

    def _network_for_values(lat_list: list, lng_list: list) -> tuple[list, list, list, list, list[int]]:
        """Build directed transition counts for a single user.

        Returns five parallel lists: lat_origins, lng_origins, lat_dests,
        lng_dests, n_trips — one entry per unique directed edge.
        """
        edge_counts: dict[tuple, int] = {}
        for i in range(len(lat_list) - 1):
            origin = (lat_list[i], lng_list[i])
            dest = (lat_list[i + 1], lng_list[i + 1])
            if not self_loops and origin == dest:
                continue
            edge = (origin, dest)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        lat_origins: list = []
        lng_origins: list = []
        lat_dests: list = []
        lng_dests: list = []
        n_trips: list[int] = []
        for (origin, dest), count in edge_counts.items():
            lat_origins.append(origin[0])
            lng_origins.append(origin[1])
            lat_dests.append(dest[0])
            lng_dests.append(dest[1])
            n_trips.append(count)

        return lat_origins, lng_origins, lat_dests, lng_dests, n_trips

    if uid_col is None:
        lat_origins, lng_origins, lat_dests, lng_dests, n_trips = _network_for_values(
            df.get_column(lat_col).to_list(),
            df.get_column(lng_col).to_list(),
        )
        return nw.from_dict(
            {
                "lat_origin": lat_origins,
                "lng_origin": lng_origins,
                "lat_dest": lat_dests,
                "lng_dest": lng_dests,
                "n_trips": n_trips,
            },
            backend=df.implementation,
        ).to_native()

    uid_vals_all: list = []
    lat_origins_all: list = []
    lng_origins_all: list = []
    lat_dests_all: list = []
    lng_dests_all: list = []
    n_trips_all: list[int] = []

    lat_full = df.get_column(lat_col).to_list()
    lng_full = df.get_column(lng_col).to_list()
    uid_values, ranges = _build_user_ranges(df, uid_col)
    for uid, (start, end) in zip(uid_values, ranges):
        lat_origins, lng_origins, lat_dests, lng_dests, n_trips = _network_for_values(
            lat_full[start:end],
            lng_full[start:end],
        )
        n = len(n_trips)
        uid_vals_all.extend([uid] * n)
        lat_origins_all.extend(lat_origins)
        lng_origins_all.extend(lng_origins)
        lat_dests_all.extend(lat_dests)
        lng_dests_all.extend(lng_dests)
        n_trips_all.extend(n_trips)

    return nw.from_dict(
        {
            uid_col: uid_vals_all,
            "lat_origin": lat_origins_all,
            "lng_origin": lng_origins_all,
            "lat_dest": lat_dests_all,
            "lng_dest": lng_dests_all,
            "n_trips": n_trips_all,
        },
        backend=df.implementation,
    ).to_native()
