"""Speed defaults for Overture Maps transportation segments.

Applied whenever a segment's ``speed_limits`` field is empty. Values are
typical free-flow speeds by Overture Transportation ``class`` (km/h).
"""

from __future__ import annotations

DEFAULT_SPEED_KMH_BY_CLASS: dict[str, float] = {
    "motorway": 110.0,
    "trunk": 90.0,
    "primary": 70.0,
    "secondary": 55.0,
    "tertiary": 40.0,
    "unclassified": 35.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
}

DRIVABLE_CLASSES: tuple[str, ...] = tuple(DEFAULT_SPEED_KMH_BY_CLASS)

# "80% of the speed limit" per the car-only, no-traffic routing model.
CAR_SPEED_FACTOR = 0.8

DEFAULT_RAIL_CLASSES: list[str] = ["subway", "tram", "light_rail", "monorail", "standard_gauge"]
DEFAULT_RAIL_SPEED_KMH_BY_CLASS: dict[str, float] = {
    "subway": 35.0,
    "tram": 22.0,
    "light_rail": 30.0,
    "monorail": 28.0,
    "standard_gauge": 45.0,
}
