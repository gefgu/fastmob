"""Network-constrained distance: road/rail graph construction and routing."""

from fastmob._core import haversine_m_batch

from .builder import build_rail_graph, build_road_graph, fetch_rail_network, fetch_road_network
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
    "od_desire_lines",
    "snap_locations_to_graph",
]
