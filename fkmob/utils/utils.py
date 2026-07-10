"""Geo-spatial utility helpers for fkmob.

Only imported by code that already requires geopandas (tessellation, mapping, plot).
"""

from __future__ import annotations


def get_geom_centroid(geom, return_lat_lng: bool = False) -> list:
    """Return the centroid of a Polygon, MultiPolygon, or Point as [lng, lat].

    Parameters
    ----------
    geom : shapely geometry
        A Polygon, MultiPolygon, or Point whose centroid is computed.
    return_lat_lng : bool, optional
        If ``True`` the returned list is ``[lat, lng]``; otherwise ``[lng, lat]``.
        The default is ``False``.

    Returns
    -------
    list
        Two-element list with the centroid coordinates.
    """
    lng, lat = map(lambda x: x.pop(), geom.centroid.xy)
    if return_lat_lng:
        return [lat, lng]
    return [lng, lat]


def nearest(origin, tessellation, col: str):
    """Return the tessellation column value of the nearest Point for each origin row.

    Uses squared Euclidean distance on raw coordinates — suitable for finding the
    closest point within a localised region (not for long-distance comparisons).

    Parameters
    ----------
    origin : geopandas.GeoDataFrame
        GeoDataFrame whose geometry column contains the query points.
    tessellation : geopandas.GeoDataFrame
        GeoDataFrame with Point geometry to search.
    col : str
        Column in *tessellation* whose value to return for each nearest match.

    Returns
    -------
    pandas.Series
        Series aligned with *tessellation*, containing the *col* value for the
        nearest tessellation point for each row in *origin*.
    """

    def _nearest_idx(row, tess):
        oy, ox = row["geometry"].y, row["geometry"].x
        min_dist = float("+inf")
        nearest_idx = None
        for idx, trow in tess.iterrows():
            p = trow["geometry"]
            d = (oy - p.y) ** 2 + (ox - p.x) ** 2
            if d < min_dist:
                min_dist = d
                nearest_idx = idx
        return nearest_idx

    indices = origin.apply(_nearest_idx, args=(tessellation,), axis=1)
    return tessellation.iloc[indices][col]
