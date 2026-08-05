"""Generic, Narwhals-native visualization API for Fastmob."""

from .brand import PALETTES
from .charts import Chart, bar, boxplot, ecdf, heatmap, map, scatter
from .figure import EChartsFigure, get_resource_bundle

__all__ = [
    "PALETTES",
    "Chart",
    "EChartsFigure",
    "bar",
    "boxplot",
    "ecdf",
    "get_resource_bundle",
    "heatmap",
    "map",
    "scatter",
]
