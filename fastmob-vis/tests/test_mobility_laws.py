import math

import numpy as np
import pytest
from fastmob_vis import (
    PALETTES,
    EChartsFigure,
    plot_distance_frequency_law,
    plot_lognormal_fits,
    plot_truncated_powerlaw_fits,
)


def test_truncated_powerlaw_builds_empirical_fit_and_reference_series():
    option = plot_truncated_powerlaw_fits(
        ((2.0, 1.0, 2.0, 10.0), [1.0, 10.0], [0.4, 0.01], "observed"),
        n_points=2,
    ).to_dict()

    assert option["_meta"]["chartType"] == "mobility_law"
    assert option["xAxis"]["type"] == "log"
    assert option["yAxis"]["type"] == "log"
    assert option["legend"]["type"] == "scroll"
    assert option["legend"]["top"] == 54
    assert option["grid"]["top"] == 124
    assert [series["name"] for series in option["series"]] == [
        "observed",
        "observed fit",
        "González reference",
    ]
    assert option["series"][0]["data"] == [[1.0, 0.4], [10.0, 0.01]]
    expected = 2.0 * (1.0 + 1.0) ** -2.0 * math.exp(-1.0 / 10.0)
    assert option["series"][1]["data"][0][1] == pytest.approx(expected)
    assert option["series"][2]["reference"] is True


def test_truncated_powerlaw_supports_radius_of_gyration_labels_and_palette():
    option = plot_truncated_powerlaw_fits(
        ((1.0, 1.5, 1.75, 400.0), [1, 2], [0.2, 0.1], "reference"),
        ((1.1, 1.0, 1.6, 300.0), [1, 2], [0.25, 0.12], "model"),
        x_label="radius of gyration · km",
        y_label="P(r_g)",
        palette="forest",
        show_reference=False,
    ).to_dict()

    assert option["xAxis"]["name"] == "RADIUS OF GYRATION · KM"
    assert option["yAxis"]["name"] == "P(R_G)"
    assert option["series"][0]["itemStyle"]["borderColor"] == PALETTES["forest"]["observed"]
    assert option["series"][2]["itemStyle"]["borderColor"] == PALETTES["forest"]["synthetic"]
    assert option["series"][3]["lineStyle"]["type"] == [6, 4]


def test_lognormal_builds_exact_curve_values():
    option = plot_lognormal_fits(
        ([1.0, 2.0], [0.3, 0.2], 0.0, 1.0, "observed"),
        n_points=2,
        show_reference=False,
    ).to_dict()

    assert option["xAxis"]["type"] == "value"
    assert option["yAxis"]["type"] == "log"
    assert option["series"][1]["data"][0] == pytest.approx([1.0, 1.0 / math.sqrt(2.0 * math.pi)])
    assert option["series"][1]["fitParameters"] == {"mu": 0.0, "sigma": 1.0}


def test_lognormal_reference_parameters_are_customizable():
    option = plot_lognormal_fits(
        ([1, 2], [0.3, 0.2], 0.5, 0.7, "data"),
        reference_parameters=(2.0, 0.8),
    ).to_dict()

    assert option["series"][-1]["fitParameters"] == {"mu": 2.0, "sigma": 0.8}
    assert option["series"][-1]["reference"] is True


def test_distance_frequency_fit_and_scaled_reference():
    option = plot_distance_frequency_law(
        ([1.0, 10.0], [8.0, 0.08], 2.0, 8.0, "observed"),
        n_points=2,
    ).to_dict()

    np.testing.assert_allclose(option["series"][1]["data"], [[1.0, 8.0], [10.0, 0.08]])
    assert option["series"][1]["fitParameters"] == {"eta": 2.0, "mu": 8.0}
    assert option["series"][-1]["fitParameters"]["alpha"] == -2.0
    assert option["series"][-1]["fitParameters"]["scale"] == pytest.approx(8.0)
    np.testing.assert_allclose(option["series"][-1]["data"], [[1.0, 8.0], [10.0, 0.08]])


def test_mobility_law_figure_dimensions_and_formatter(tmp_path):
    figure = plot_distance_frequency_law(
        ([1, 2], [2, 1], 1.0, 2.0, "data"),
        width="100%",
        height="500px",
    )
    content = figure.to_html(tmp_path / "law.html").read_text(encoding="utf-8")

    assert isinstance(figure, EChartsFigure)
    assert figure.width == "100%"
    assert figure.height == "500px"
    assert "fitParameterText" in content
    assert "option.series[params.seriesIndex]" in content
    assert "function ecdfAt" not in content


def test_mobility_law_functions_are_exported():
    import fastmob_vis
    import fastmob_vis.plots

    assert fastmob_vis.plot_truncated_powerlaw_fits is plot_truncated_powerlaw_fits
    assert fastmob_vis.plot_lognormal_fits is plot_lognormal_fits
    assert fastmob_vis.plot_distance_frequency_law is plot_distance_frequency_law
    assert fastmob_vis.plots.plot_truncated_powerlaw_fits is plot_truncated_powerlaw_fits


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: plot_truncated_powerlaw_fits(), "at least one"),
        (
            lambda: plot_truncated_powerlaw_fits(((1, 1, 1, 1), [], [], "data")),
            "must not be empty",
        ),
        (
            lambda: plot_truncated_powerlaw_fits(((1, 1, 1, 1), [1, 2], [1], "data")),
            "same shape",
        ),
        (
            lambda: plot_truncated_powerlaw_fits(((1, 1, 1, 1), [1, float("nan")], [1, 2], "data")),
            "finite",
        ),
        (
            lambda: plot_truncated_powerlaw_fits(((1, 1, 1, 1), [0, 1], [1, 2], "data")),
            "positive",
        ),
        (
            lambda: plot_truncated_powerlaw_fits(((1, -1, 1, 1), [1, 2], [1, 2], "data")),
            "parameters require",
        ),
        (
            lambda: plot_lognormal_fits(([1, 2], [1, 0.5], 0, 0, "data")),
            "sigma must be positive",
        ),
        (
            lambda: plot_distance_frequency_law(([1, 2], [1, 0.5], -1, 1, "data")),
            "eta and mu must be positive",
        ),
    ],
)
def test_mobility_law_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call()
