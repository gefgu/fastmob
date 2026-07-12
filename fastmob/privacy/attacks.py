"""Compatibility imports for privacy risk attacks.

The attack implementations live in focused modules under ``fastmob.privacy``.
This module preserves the original ``fastmob.privacy.attacks`` import path.
"""

from ._constants import (
    COUNT,
    DATETIME,
    FREQUENCY,
    INSTANCE,
    INSTANCE_ELEMENT,
    LATITUDE,
    LONGITUDE,
    PRECISION_LEVELS,
    PRIVACY_RISK,
    PROBABILITY,
    PROPORTION,
    TEMP,
    TOTAL_FREQ,
    UID,
)
from ._dataframe import (
    _as_frame,
    _backend,
    _date_time_precision,
    _frequency_vector,
    _probability_vector,
    _to_native,
    _with_date_time_precision,
)
from .base import Attack
from .frequency import (
    HomeWorkAttack,
    LocationFrequencyAttack,
    LocationProbabilityAttack,
    LocationProportionAttack,
    UniqueLocationAttack,
)
from .location import LocationAttack, LocationSequenceAttack, LocationTimeAttack

__all__ = [
    "Attack",
    "LocationAttack",
    "LocationSequenceAttack",
    "LocationTimeAttack",
    "UniqueLocationAttack",
    "LocationFrequencyAttack",
    "LocationProbabilityAttack",
    "LocationProportionAttack",
    "HomeWorkAttack",
    "UID",
    "LATITUDE",
    "LONGITUDE",
    "DATETIME",
    "FREQUENCY",
    "PROBABILITY",
    "TOTAL_FREQ",
    "COUNT",
    "TEMP",
    "PROPORTION",
    "PRIVACY_RISK",
    "INSTANCE",
    "INSTANCE_ELEMENT",
    "PRECISION_LEVELS",
    "_as_frame",
    "_backend",
    "_to_native",
    "_frequency_vector",
    "_probability_vector",
    "_with_date_time_precision",
    "_date_time_precision",
]
