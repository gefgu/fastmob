"""Public API checks for the optional ``fastmob.vis`` facade."""

import pytest


def test_visualization_facade_reexports_fastmob_vis_public_api():
    vis = pytest.importorskip("fastmob.vis")
    import fastmob_vis
    from fastmob_vis import motifs

    assert vis.Chart is fastmob_vis.Chart
    assert vis.EChartsFigure is fastmob_vis.EChartsFigure
    assert vis.ecdf is fastmob_vis.ecdf
    assert vis.LITERATURE_MOTIF_PERCENTAGES is motifs.LITERATURE_MOTIF_PERCENTAGES
    assert vis.LITERATURE_TO_FASTMOB_MOTIF_ID is motifs.LITERATURE_TO_FASTMOB_MOTIF_ID


def test_visualization_facade_exports_documented_symbols():
    vis = pytest.importorskip("fastmob.vis")

    assert {
        "PALETTES",
        "Chart",
        "EChartsFigure",
        "bar",
        "boxplot",
        "ecdf",
        "get_resource_bundle",
        "heatmap",
        "LITERATURE_MOTIF_PERCENTAGES",
        "LITERATURE_TO_FASTMOB_MOTIF_ID",
        "map",
        "scatter",
    } <= set(vis.__all__)
