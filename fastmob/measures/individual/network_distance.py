"""Road/rail-network distance mirrors of jump_lengths/radius_of_gyration.

fastmob's own :func:`~fastmob.measures.individual.jump_lengths.jump_lengths`/
:func:`~fastmob.measures.individual.radius_of_gyration.radius_of_gyration`
measure straight-line Haversine distance between stop points. This module
recomputes the two metrics as actual network distance over a prepared
:class:`fastmob.network.RoadNetwork`, falling back to Haversine per-pair
wherever a point is unsnapped or the graph is disconnected between the two
points.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from .._common import (
    LAT_CANDIDATES,
    LNG_CANDIDATES,
    UID_CANDIDATES,
    _detect_trajectory_columns,
    _pick_existing_column,
    _prepare_trajectory,
)
from ...network._util import haversine_m_batch
from ...network.road_graph import RoadNetwork
from ...network.snap import snap_locations_to_graph


def _road_or_haversine_km(
    network: RoadNetwork,
    from_node: np.ndarray,
    to_node: np.ndarray,
    from_lat: np.ndarray,
    from_lng: np.ndarray,
    to_lat: np.ndarray,
    to_lng: np.ndarray,
) -> np.ndarray:
    distances_m, connected = network.batch_distances(from_node, to_node)
    fallback_km = haversine_m_batch(from_lat, from_lng, to_lat, to_lng) / 1000.0
    road_km = distances_m / 1000.0
    return np.where(connected, road_km, fallback_km)


def jump_lengths_km(
    traj: Any,
    *,
    network: RoadNetwork,
    uid_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    datetime_col: str | None = None,
    snap_max_distance_m: float = 750.0,
) -> np.ndarray:
    """Road-network jump lengths (km): distance between consecutive stops
    for the same user, sorted by datetime -- mirrors
    :func:`~fastmob.measures.individual.jump_lengths.jump_lengths`'s sort
    key and its inclusion of zero-length jumps, but measures along the
    network instead of straight-line, falling back to Haversine per-pair
    when unsnapped or disconnected.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    network:
        A prepared :class:`fastmob.network.RoadNetwork`.
    uid_col, lat_col, lng_col, datetime_col:
        Explicit column name overrides; auto-detected when None.
    snap_max_distance_m:
        Maximum distance (metres) to snap a stop to the network; farther
        stops fall back to Haversine entirely for any jump touching them.

    Returns
    -------
    numpy.ndarray
        One value per consecutive same-user pair (length: `len(traj) - n_users`).
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df, datetime_col=datetime_col, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col
    )
    df = _prepare_trajectory(
        df, datetime_col=datetime_col, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, sort=True, drop_nulls=True
    )

    coords_pd = df.select([lat_col, lng_col]).to_pandas()
    node_idx = snap_locations_to_graph(coords_pd, network.nodes_df, snap_max_distance_m, lat_col=lat_col, lng_col=lng_col)

    uid_arr = df.get_column(uid_col).to_numpy() if uid_col is not None else None
    lat_arr = df.get_column(lat_col).to_numpy().astype(np.float64)
    lng_arr = df.get_column(lng_col).to_numpy().astype(np.float64)

    if lat_arr.size < 2:
        return np.empty(0, dtype=np.float64)

    if uid_arr is not None:
        same_uid = uid_arr[1:] == uid_arr[:-1]
    else:
        same_uid = np.ones(lat_arr.size - 1, dtype=bool)

    from_node = node_idx[:-1][same_uid]
    to_node = node_idx[1:][same_uid]
    if from_node.size == 0:
        return np.empty(0, dtype=np.float64)

    return _road_or_haversine_km(
        network,
        from_node,
        to_node,
        lat_arr[:-1][same_uid],
        lng_arr[:-1][same_uid],
        lat_arr[1:][same_uid],
        lng_arr[1:][same_uid],
    )


def radius_of_gyration_km(
    traj: Any,
    *,
    network: RoadNetwork,
    uid_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    snap_max_distance_m: float = 750.0,
) -> Any:
    """Road-network radius of gyration (km) per user: RMS network distance
    from each of a user's stops to the arithmetic-mean centroid of their
    stops -- mirrors the unweighted-centroid formula
    ``r_g(u) = sqrt(mean(d(r_i, r_cm)^2))`` used by
    :func:`~fastmob.measures.individual.radius_of_gyration.radius_of_gyration`,
    but measures ``d`` along the network instead of straight-line.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    network:
        A prepared :class:`fastmob.network.RoadNetwork`.
    uid_col, lat_col, lng_col:
        Explicit column name overrides; auto-detected when None.
    snap_max_distance_m:
        Maximum distance (metres) to snap a stop to the network.

    Returns
    -------
    DataFrame
        ``[uid_col, "radius_of_gyration"]``, one row per user, in the same
        backend as input.
    """
    df = nw.from_native(traj, eager_only=True)
    if lat_col is None:
        lat_col = _pick_existing_column(df.columns, LAT_CANDIDATES)
    if lng_col is None:
        lng_col = _pick_existing_column(df.columns, LNG_CANDIDATES)
    if uid_col is None:
        uid_col = _pick_existing_column(df.columns, UID_CANDIDATES)
    missing = [name for name, col in [("latitude", lat_col), ("longitude", lng_col)] if col is None]
    if missing:
        raise ValueError(
            f"Could not detect required column(s): {missing}. Available columns: {df.columns}. "
            "Pass the column name(s) explicitly."
        )
    if uid_col is None:
        raise ValueError(
            f"Could not detect a user-id column. Available columns: {df.columns}. Pass uid_col explicitly."
        )

    df = df.drop_nulls(subset=[uid_col, lat_col, lng_col]).with_columns(
        nw.col(lat_col).cast(nw.Float64), nw.col(lng_col).cast(nw.Float64)
    )
    if len(df) == 0:
        return nw.from_dict(
            {uid_col: [], "radius_of_gyration": []},
            backend=df.implementation,
        ).to_native()

    centroid = df.group_by(uid_col).agg(nw.col(lat_col).mean(), nw.col(lng_col).mean())
    centroid_pd = centroid.select([lat_col, lng_col]).to_pandas()
    centroid_node = snap_locations_to_graph(
        centroid_pd, network.nodes_df, snap_max_distance_m, lat_col=lat_col, lng_col=lng_col
    )
    centroid = centroid.with_columns(
        nw.new_series("__centroid_node__", centroid_node, backend=df.implementation).alias("__centroid_node__")
    ).rename({lat_col: "__centroid_lat__", lng_col: "__centroid_lng__"})

    merged = df.join(centroid, on=uid_col, how="left")
    stop_coords_pd = merged.select([lat_col, lng_col]).to_pandas()
    stop_node = snap_locations_to_graph(
        stop_coords_pd, network.nodes_df, snap_max_distance_m, lat_col=lat_col, lng_col=lng_col
    )

    dist_km = _road_or_haversine_km(
        network,
        stop_node,
        merged.get_column("__centroid_node__").to_numpy().astype(np.int64),
        merged.get_column(lat_col).to_numpy().astype(np.float64),
        merged.get_column(lng_col).to_numpy().astype(np.float64),
        merged.get_column("__centroid_lat__").to_numpy().astype(np.float64),
        merged.get_column("__centroid_lng__").to_numpy().astype(np.float64),
    )

    result = nw.from_dict(
        {uid_col: merged.get_column(uid_col).to_list(), "__dist_km__": dist_km},
        backend=df.implementation,
    ).with_columns((nw.col("__dist_km__") ** 2).alias("__dist_km_sq__"))
    # A single aggregation expression combining a power and a mean (as
    # `(col**2).mean()**0.5` would be) triggers a narwhals warning that
    # pandas can't push it down efficiently -- pre-squaring as its own
    # column keeps the aggregation itself to a simple `.mean()`.
    result = result.group_by(uid_col).agg(nw.col("__dist_km_sq__").mean().alias("__mean_sq__"))
    result = result.with_columns((nw.col("__mean_sq__") ** 0.5).alias("radius_of_gyration")).select(
        [uid_col, "radius_of_gyration"]
    )
    return result.to_native()


jump_lengths_km.__module__ = "fastmob.measures.individual"
radius_of_gyration_km.__module__ = "fastmob.measures.individual"
