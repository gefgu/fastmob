"""Generation models compatible with original scikit-mobility."""

from .epr import EPR, DensityEPR, Ditras, SpatialEPR
from .geosim import GeoSim
from .gravity import Gravity, ci
from .markov_diary_generator import MarkovDiaryGenerator
from .next_location import NextLocationPredictor
from .radiation import Radiation
from .sts_epr import STS_epr

__all__ = [
    "EPR",
    "DensityEPR",
    "Ditras",
    "GeoSim",
    "Gravity",
    "MarkovDiaryGenerator",
    "NextLocationPredictor",
    "Radiation",
    "STS_epr",
    "SpatialEPR",
    "ci",
]
