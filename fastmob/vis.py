from __future__ import annotations

try:
    from fastmob_vis import (
        PALETTES,
        Chart,
        EChartsFigure,
        bar,
        boxplot,
        ecdf,
        get_resource_bundle,
        heatmap,
        map,
        scatter,
    )
except ImportError as exc:
    raise ImportError("fastmob-vis is required: pip install fastmob[vis]") from exc

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
