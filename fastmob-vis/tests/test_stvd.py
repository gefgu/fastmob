import copy

import pytest
from fastmob_vis import EChartsFigure, plot_stvd_comparison


def _feature(
    area="cell-a",
    volume_diff=-4.0,
    peak_shift=1.0,
    *,
    west=2.30,
    south=48.80,
    properties=None,
):
    return {
        "type": "Feature",
        "properties": properties
        or {
            "area": area,
            "volume_diff_pct": volume_diff,
            "peak_shift_hours": peak_shift,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [west, south],
                    [west + 0.02, south],
                    [west + 0.02, south + 0.02],
                    [west, south + 0.02],
                    [west, south],
                ]
            ],
        },
    }


def _layer(features=None):
    return {"type": "FeatureCollection", "features": features or [_feature()]}


def _layers():
    return {
        5: _layer([_feature("r5", -4, 1)]),
        7: _layer([_feature("r7", 0, 4, west=2.32)]),
        9: _layer([_feature("r9", 4, 8, west=2.34)]),
    }


def test_plot_stvd_comparison_returns_figure_and_embeds_layers():
    figure = plot_stvd_comparison(_layers())
    option = figure.to_dict()

    assert isinstance(figure, EChartsFigure)
    assert option["_meta"]["chartType"] == "stvd_comparison"
    assert set(option["_meta"]["layers"]) == {"5", "7", "9"}
    assert option["_meta"]["initialResolution"] == 7
    assert option["series"][0]["type"] == "custom"
    assert option["series"][0]["coordinateSystem"] == "leaflet"
    assert option["series"][0]["data"] == [[0]]


@pytest.mark.parametrize(
    ("volume_diff", "peak_shift", "expected_bins", "expected_color"),
    [
        (-3.01, 2.0, (0, 0, 0), "#91bfdb"),
        (-3.0, 2.01, (1, 1, 4), "#bdbdbd"),
        (3.0, 5.0, (1, 1, 4), "#bdbdbd"),
        (3.01, 5.01, (2, 2, 8), "#b2182b"),
    ],
)
def test_stvd_classification_boundaries(
    volume_diff, peak_shift, expected_bins, expected_color
):
    option = plot_stvd_comparison({7: _layer([_feature("cell", volume_diff, peak_shift)])}).to_dict()
    values = option["_meta"]["layers"]["7"]["features"][0]["properties"]["_fastmobVis"]

    assert (values["volumeBin"], values["peakBin"], values["class"]) == expected_bins
    assert values["color"] == expected_color


def test_stvd_uses_custom_properties_and_map_configuration():
    layers = {
        6: _layer(
            [
                _feature(
                    properties={"hex": "custom", "delta": 2.5, "shift": 6.0},
                    west=-74.1,
                    south=40.6,
                )
            ]
        )
    }
    figure = plot_stvd_comparison(
        layers,
        area_property="hex",
        volume_diff_property="delta",
        peak_shift_property="shift",
        volume_threshold=2.0,
        zoom_to_resolution={0: 6},
        center=(-74.0, 40.7),
        zoom=11,
        tile_url="https://tiles.example/{z}/{x}/{y}.png",
        attribution="Example tiles",
        title="Custom STVD",
        width=900,
        height=500,
    )
    option = figure.to_dict()
    values = option["_meta"]["layers"]["6"]["features"][0]["properties"]["_fastmobVis"]

    assert values["area"] == "custom"
    assert values["volumeBin"] == 2
    assert option["leaflet"]["center"] == [-74.0, 40.7]
    assert option["leaflet"]["zoom"] == 11
    assert option["leaflet"]["tiles"][0]["urlTemplate"].startswith("https://tiles.example")
    assert option["title"]["text"] == "Custom STVD"
    assert figure.width == "900px"
    assert figure.height == "500px"


def test_stvd_computes_center_from_all_layers():
    option = plot_stvd_comparison(
        {
            5: _layer([_feature(west=0.0, south=10.0)]),
            7: _layer([_feature(west=2.0, south=12.0)]),
        },
        zoom_to_resolution={0: 5, 8: 7},
    ).to_dict()

    assert option["leaflet"]["center"] == pytest.approx([1.01, 11.01])


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ({}, "non-empty"),
        ({7: {"type": "FeatureCollection", "features": []}}, "at least one"),
        ({7: {"type": "Feature", "features": []}}, "FeatureCollection"),
        ({7: _layer([{"type": "Feature", "properties": {}, "geometry": None}])}, "geometry"),
        (
            {
                7: _layer(
                    [
                        {
                            **_feature(),
                            "geometry": {"type": "Point", "coordinates": [2.3, 48.8]},
                        }
                    ]
                )
            },
            "Polygon or MultiPolygon",
        ),
    ],
)
def test_stvd_rejects_invalid_layers(layers, message):
    with pytest.raises(ValueError, match=message):
        plot_stvd_comparison(layers)


@pytest.mark.parametrize(
    ("property_name", "property_value", "message"),
    [
        ("volume_diff_pct", None, "finite"),
        ("volume_diff_pct", float("nan"), "finite"),
        ("peak_shift_hours", None, "finite"),
        ("peak_shift_hours", 13, r"\[0, 12\]"),
    ],
)
def test_stvd_rejects_missing_or_invalid_metrics(property_name, property_value, message):
    feature = _feature()
    feature["properties"][property_name] = property_value

    with pytest.raises(ValueError, match=message):
        plot_stvd_comparison({7: _layer([feature])})

    missing = copy.deepcopy(feature)
    del missing["properties"][property_name]
    with pytest.raises(ValueError, match="missing property"):
        plot_stvd_comparison({7: _layer([missing])})


def test_stvd_rejects_invalid_zoom_mapping():
    with pytest.raises(ValueError, match="zoom 0"):
        plot_stvd_comparison({7: _layer()}, zoom_to_resolution={8: 7})
    with pytest.raises(ValueError, match="missing layer resolution 9"):
        plot_stvd_comparison({7: _layer()}, zoom_to_resolution={0: 9})
    with pytest.raises(ValueError, match="non-negative integers"):
        plot_stvd_comparison({7: _layer()}, zoom_to_resolution={-1: 7})


def test_stvd_html_inlines_leaflet_extension_and_zoom_switching(tmp_path):
    content = plot_stvd_comparison(_layers()).to_html(tmp_path / "stvd.html").read_text()

    assert "Leaflet 1.9.4" in content
    assert "echarts.registerCoordinateSystem('leaflet'" in content
    assert "echarts.registerMap('fastmob-stvd-res-'" in content
    assert "leafletMap.on('zoomend'" in content
    assert "stvdResolutionForZoom(leafletMap.getZoom())" in content
    assert "fastmob-vis-stvd-legend" in content
    assert "fetch(" not in content
    assert "unpkg.com/leaflet" not in content


def test_svg_rendering_is_not_supported_for_stvd():
    figure = plot_stvd_comparison(_layers(), display="svg")

    with pytest.raises(ValueError, match="not supported"):
        figure.to_svg()


def test_plot_stvd_comparison_is_exported():
    import fastmob_vis
    import fastmob_vis.plots

    assert fastmob_vis.plot_stvd_comparison is plot_stvd_comparison
    assert fastmob_vis.plots.plot_stvd_comparison is plot_stvd_comparison
    assert "plot_stvd_comparison" in fastmob_vis.__all__
