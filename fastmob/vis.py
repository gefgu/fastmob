"""Stable public facade for Fastmob's optional visualization support.

Install it with ``pip install \"fastmob[vis]\"``. The implementation lives in
the separately distributed :mod:`fastmob_vis` package; its direct imports
remain supported for compatibility.
"""

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
    from fastmob_vis.motifs import LITERATURE_MOTIF_PERCENTAGES, LITERATURE_TO_FASTMOB_MOTIF_ID
except ImportError as exc:
    raise ImportError("fastmob-vis is required: pip install fastmob[vis]") from exc

__all__ = [
    "LITERATURE_MOTIF_PERCENTAGES",
    "LITERATURE_TO_FASTMOB_MOTIF_ID",
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
