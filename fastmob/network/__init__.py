"""Network-constrained distance: road/rail graph construction and routing."""

from ._util import haversine_m_batch
from .builder import build_rail_graph, build_road_graph, fetch_rail_network, fetch_road_network
from .road_graph import RoadNetwork
from .snap import snap_locations_to_graph
from .speeds import (
    CAR_SPEED_FACTOR,
    DEFAULT_RAIL_CLASSES,
    DEFAULT_RAIL_SPEED_KMH_BY_CLASS,
    DEFAULT_SPEED_KMH_BY_CLASS,
    DRIVABLE_CLASSES,
)

__all__ = [
    "RoadNetwork",
    "fetch_road_network",
    "build_road_graph",
    "fetch_rail_network",
    "build_rail_graph",
    "snap_locations_to_graph",
    "haversine_m_batch",
    "DEFAULT_SPEED_KMH_BY_CLASS",
    "DRIVABLE_CLASSES",
    "CAR_SPEED_FACTOR",
    "DEFAULT_RAIL_CLASSES",
    "DEFAULT_RAIL_SPEED_KMH_BY_CLASS",
]
