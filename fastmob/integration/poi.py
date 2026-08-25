"""PyMove-style trajectory/POI spatial joins.

Mirrors ``pymove.utils.integration.join_with_pois`` /
``join_with_pois_by_category``: augments a trajectory dataframe with
columns describing its nearest point(s) of interest, rather than
aggregating the trajectory into a per-user summary like the measures under
``fastmob.measures``.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
import numpy as np

from fastmob.utils._common import LAT_CANDIDATES, LNG_CANDIDATES, _pick_existing_column

from ..network._nearest import nearest_candidate


def _detect_lat_lng(df: nw.DataFrame, lat_col: str | None, lng_col: str | None) -> tuple[str, str]:
    if lat_col is None:
        lat_col = _pick_existing_column(df.columns, LAT_CANDIDATES)
    if lng_col is None:
        lng_col = _pick_existing_column(df.columns, LNG_CANDIDATES)
    missing = [name for name, col in [("latitude", lat_col), ("longitude", lng_col)] if col is None]
    if missing:
        raise ValueError(
            f"Could not detect required column(s): {missing}. Available columns: {df.columns}. "
            "Pass the column name(s) explicitly."
        )
    return lat_col, lng_col


def join_with_pois(
    traj: Any,
    pois_df: Any,
    *,
    lat_col: str | None = None,
    lng_col: str | None = None,
    poi_lat_col: str = "lat",
    poi_lng_col: str = "lng",
    poi_id_col: str = "id",
    poi_name_col: str = "name_poi",
) -> Any:
    """Join each trajectory point with its single nearest point of interest.

    Mirrors PyMove's ``join_with_pois``: an unconditional single-nearest
    lookup with no distance cutoff (contrast with
    :func:`fastmob.network.snap_locations_to_graph`, which reports
    "unsnapped" past `max_distance_m`).

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    pois_df:
        Points of interest; any Narwhals-compatible eager backend, with
        `poi_lat_col`/`poi_lng_col` columns and, unless overridden,
        `poi_id_col`/`poi_name_col` columns.
    lat_col, lng_col:
        Explicit trajectory column overrides; auto-detected when None.
    poi_lat_col, poi_lng_col, poi_id_col, poi_name_col:
        Column names on `pois_df`.

    Returns
    -------
    DataFrame
        `traj`'s columns plus ``id_poi``, ``dist_poi``, ``name_poi``, in
        the same backend as `traj`. A `pois_df` with zero rows produces
        ``None``/``inf``/``None`` for every trajectory point.
    """
    df = nw.from_native(unwrap_native(traj), eager_only=True)
    lat_col, lng_col = _detect_lat_lng(df, lat_col, lng_col)

    pois = nw.from_native(pois_df, eager_only=True)
    poi_lat = pois.get_column(poi_lat_col).to_numpy().astype(np.float64)
    poi_lng = pois.get_column(poi_lng_col).to_numpy().astype(np.float64)
    poi_id = pois.get_column(poi_id_col).to_numpy()
    poi_name = pois.get_column(poi_name_col).to_numpy()

    query_lat = df.get_column(lat_col).to_numpy().astype(np.float64)
    query_lng = df.get_column(lng_col).to_numpy().astype(np.float64)

    nearest_idx, dist_m = nearest_candidate(query_lat, query_lng, poi_lat, poi_lng)

    if len(poi_id) == 0:
        id_poi = np.full(len(query_lat), None, dtype=object)
        name_poi = np.full(len(query_lat), None, dtype=object)
    else:
        id_poi = poi_id[nearest_idx]
        name_poi = poi_name[nearest_idx]

    result = df.with_columns(
        nw.new_series("id_poi", id_poi, backend=df.implementation),
        nw.new_series("dist_poi", dist_m, backend=df.implementation),
        nw.new_series("name_poi", name_poi, backend=df.implementation),
    )
    return result.to_native()


def join_with_pois_by_category(
    traj: Any,
    pois_df: Any,
    *,
    lat_col: str | None = None,
    lng_col: str | None = None,
    poi_lat_col: str = "lat",
    poi_lng_col: str = "lng",
    poi_id_col: str = "id",
    category_col: str = "type_poi",
) -> Any:
    """Join each trajectory point with its nearest POI in each category.

    Mirrors PyMove's ``join_with_pois_by_category``: for every distinct
    value in ``pois_df[category_col]``, adds an ``id_<category>``/
    ``dist_<category>`` column pair holding the nearest POI *of that
    category* to each trajectory point (unlike :func:`join_with_pois`,
    which only reports the single nearest POI overall).

    Returns
    -------
    DataFrame
        `traj`'s columns plus one ``id_<category>``/``dist_<category>``
        column pair per distinct category value present in `pois_df`, in
        the same backend as `traj`.
    """
    df = nw.from_native(unwrap_native(traj), eager_only=True)
    lat_col, lng_col = _detect_lat_lng(df, lat_col, lng_col)
    query_lat = df.get_column(lat_col).to_numpy().astype(np.float64)
    query_lng = df.get_column(lng_col).to_numpy().astype(np.float64)

    pois = nw.from_native(pois_df, eager_only=True)
    poi_lat_all = pois.get_column(poi_lat_col).to_numpy().astype(np.float64)
    poi_lng_all = pois.get_column(poi_lng_col).to_numpy().astype(np.float64)
    poi_id_all = pois.get_column(poi_id_col).to_numpy()
    poi_category_all = pois.get_column(category_col).to_numpy()

    new_series = []
    for category in np.unique(poi_category_all):
        mask = poi_category_all == category
        cat_lat = poi_lat_all[mask]
        cat_lng = poi_lng_all[mask]
        cat_id = poi_id_all[mask]

        nearest_idx, dist_m = nearest_candidate(query_lat, query_lng, cat_lat, cat_lng)
        id_col = cat_id[nearest_idx] if len(cat_id) else np.full(len(query_lat), None, dtype=object)

        new_series.append(nw.new_series(f"id_{category}", id_col, backend=df.implementation))
        new_series.append(nw.new_series(f"dist_{category}", dist_m, backend=df.implementation))

    result = df.with_columns(*new_series)
    return result.to_native()


join_with_pois.__module__ = "fastmob.integration"
join_with_pois_by_category.__module__ = "fastmob.integration"
