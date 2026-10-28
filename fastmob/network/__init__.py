"""Network-constrained distance: road/rail graph construction and routing."""

import numpy as np

from fastmob._core import haversine_m_batch as _haversine_m_batch

from .builder import build_rail_graph, build_road_graph, fetch_rail_network, fetch_road_network
from .match import match_trajectory
from .od_flow import od_desire_lines
from .road_graph import RoadNetwork
from .snap import snap_locations_to_graph
from .speeds import (
    CAR_SPEED_FACTOR,
    DEFAULT_RAIL_CLASSES,
    DEFAULT_RAIL_SPEED_KMH_BY_CLASS,
    DEFAULT_SPEED_KMH_BY_CLASS,
    DRIVABLE_CLASSES,
)


def haversine_m_batch(lat1, lng1, lat2, lng2):
    """Compute elementwise Haversine distances in metres.

    Each argument must be a one-dimensional, same-length array-like of
    coordinates in degrees, including lists, NumPy arrays and pandas Series.
    The result is a NumPy array of distances in metres.
    """
    coordinates = [np.asarray(value, dtype=np.float64) for value in (lat1, lng1, lat2, lng2)]
    if any(value.ndim != 1 for value in coordinates):
        raise ValueError("coordinates must be one-dimensional")
    return _haversine_m_batch(*(np.ascontiguousarray(value) for value in coordinates))

__all__ = [
    "CAR_SPEED_FACTOR",
    "DEFAULT_RAIL_CLASSES",
    "DEFAULT_RAIL_SPEED_KMH_BY_CLASS",
    "DEFAULT_SPEED_KMH_BY_CLASS",
    "DRIVABLE_CLASSES",
    "RoadNetwork",
    "build_rail_graph",
    "build_road_graph",
    "fetch_rail_network",
    "fetch_road_network",
    "haversine_m_batch",
    "match_trajectory",
    "od_desire_lines",
    "snap_locations_to_graph",
]
