import importlib.resources
import json

import pytest
from fastmob_vis import (
    PALETTES,
    EChartsFigure,
    plot_activity_transition_difference,
    plot_activity_transition_matrix,
    plot_daily_activity_difference,
    plot_daily_activity_distribution,
    plot_dwell_time_ecdf,
    plot_jump_lengths_ecdf,
    plot_motif_literature_comparison,
    plot_radius_of_gyration_ecdf,
    plot_trip_duration_ecdf,
    plot_visit_purpose_comparison,
    plot_visit_purpose_distribution,
    plot_visits_frequency_ecdf,
)
from fastmob_vis.motifs import (
    LITERATURE_TO_FASTMOB_MOTIF_ID,
    format_motif_hex_id,
    map_motif_distribution_to_literature_basis,
)

WARM = PALETTES["warm"]


# ---------------------------------------------------------------------------
# jump lengths
# ---------------------------------------------------------------------------

def test_plot_jump_lengths_ecdf_returns_figure():
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1], labels=("observed", "synthetic"))

    assert isinstance(figure, EChartsFigure)


def test_to_dict_contains_expected_series():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1], labels=("observed", "synthetic")).to_dict()

    assert option["backgroundColor"] == WARM["bg"]
    assert option["series"][0]["name"] == "observed"
    assert option["series"][0]["data"] == [[1.0, 0.5], [2.0, 0.98]]
    assert option["series"][1]["name"] == "synthetic"
    assert option["series"][1]["data"] == [[1.0, 0.5], [3.0, 0.98]]


def test_series_colors_match_palette():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    assert option["series"][0]["lineStyle"]["color"] == WARM["observed"]
    assert option["series"][1]["lineStyle"]["color"] == WARM["synthetic"]


def test_synthetic_series_is_dashed():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    assert "type" not in option["series"][0]["lineStyle"]
    assert option["series"][1]["lineStyle"]["type"] == [6, 4]


def test_series_uses_smooth_and_no_symbols():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    for s in option["series"]:
        assert s["smooth"] == 0.35
        assert s["showSymbol"] is False


def test_y_axis_is_always_zero_to_one():
    option = plot_jump_lengths_ecdf([4, 1, 3, 2], [10, 20, 30, 40], cdf_cutoff=0.75).to_dict()

    assert option["yAxis"]["min"] == 0
    assert option["yAxis"]["max"] == 1


def test_y_axis_name_and_layout():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    assert option["yAxis"]["name"] == "F(X)"
    assert option["yAxis"]["nameGap"] == 56
    assert option["yAxis"]["nameRotate"] == 90
    assert option["yAxis"]["interval"] == 0.2


def test_x_axis_name_uppercased_with_unit():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    assert option["xAxis"]["name"] == "JUMP LENGTH · KM"
    assert option["xAxis"]["nameGap"] == 38


def test_grid_padding_matches_design():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    assert option["grid"] == {"left": 76, "right": 28, "top": 64, "bottom": 64, "containLabel": False}


def test_animation_is_disabled():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    assert option["animation"] is False


def test_custom_cdf_cutoff_clips_series_data():
    option = plot_jump_lengths_ecdf([4, 1, 3, 2], [10, 20, 30, 40], cdf_cutoff=0.75).to_dict()

    assert option["series"][0]["data"] == [[1.0, 0.25], [2.0, 0.5], [3.0, 0.75]]
    assert option["series"][1]["data"] == [[10.0, 0.25], [20.0, 0.5], [30.0, 0.75]]


def test_palette_warm_is_default():
    option = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_dict()

    assert option["backgroundColor"] == WARM["bg"]


def test_palette_forest_applies_colors():
    forest = PALETTES["forest"]
    option = plot_jump_lengths_ecdf([2, 1], [3, 1], palette="forest").to_dict()

    assert option["backgroundColor"] == forest["bg"]
    assert option["series"][0]["lineStyle"]["color"] == forest["observed"]
    assert option["series"][1]["lineStyle"]["color"] == forest["synthetic"]


def test_unknown_palette_raises():
    with pytest.raises(ValueError, match="unknown palette"):
        plot_jump_lengths_ecdf([2, 1], [3, 1], palette="neon")


def test_background_field_matches_palette_bg():
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1], palette="forest")

    assert figure.background == PALETTES["forest"]["bg"]


def test_default_width_is_600px():
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1])

    assert figure.width == "600px"


def test_int_width_is_converted_to_px():
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1], width=900)

    assert figure.width == "900px"


def test_string_width_passes_through():
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1], width="100%")

    assert figure.width == "100%"


def test_width_appears_in_repr_html():
    html = plot_jump_lengths_ecdf([2, 1], [3, 1], width=800)._repr_html_()

    assert "800px" in html


def test_to_json_is_valid_json():
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1])

    assert json.loads(figure.option_json) == figure.to_dict()


def test_figure_requires_dictionary_options():
    with pytest.raises(TypeError, match="option must be a dictionary"):
        EChartsFigure("{}")


def test_figure_copies_constructor_option():
    option = {"series": [{"data": [1, 2]}]}
    figure = EChartsFigure(option)

    option["series"][0]["data"].append(3)

    assert figure.to_dict()["series"][0]["data"] == [1, 2]


def test_figure_option_accessors_return_copies():
    figure = EChartsFigure({"series": [{"data": [1, 2]}]})
    option = figure.option
    option_dict = figure.to_dict()

    option["series"][0]["data"].append(3)
    option_dict["series"][0]["data"].append(4)

    assert figure.to_dict()["series"][0]["data"] == [1, 2]


def test_repr_html_contains_echarts_container_and_option():
    html = plot_jump_lengths_ecdf([2, 1], [3, 1])._repr_html_()

    assert "iframe" in html
    assert "srcdoc=" in html
    assert "fastmob-vis-" in html
    assert "echarts.init" in html
    assert "Jump length ECDF" in html


def test_to_html_writes_standalone_document(tmp_path):
    path = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_html(tmp_path / "chart.html")
    content = path.read_text(encoding="utf-8")

    assert path == tmp_path / "chart.html"
    assert "<!doctype html>" in content
    assert "echarts.init" in content
    assert "fonts.googleapis.com" in content
    # ECharts is bundled inline — no CDN dependency
    assert "cdn.jsdelivr.net" not in content


def test_to_svg_returns_static_svg():
    svg = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_svg()

    assert isinstance(svg, str)
    assert "<svg" in svg
    assert "</svg>" in svg


def test_to_svg_writes_file(tmp_path):
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1])
    path = figure.to_svg(tmp_path / "chart.svg")

    assert path == tmp_path / "chart.svg"
    assert "<svg" in path.read_text(encoding="utf-8")


def test_svg_display_returns_svg_mimebundle():
    bundle = plot_jump_lengths_ecdf([2, 1], [3, 1], display="svg")._repr_mimebundle_()

    assert set(bundle) == {"image/svg+xml"}
    assert "<svg" in bundle["image/svg+xml"]


def test_html_display_remains_default_mimebundle():
    bundle = plot_jump_lengths_ecdf([2, 1], [3, 1])._repr_mimebundle_()

    assert set(bundle) == {"text/html"}
    assert "iframe" in bundle["text/html"]


def test_svg_requires_fixed_pixel_dimensions():
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1], width="100%")

    with pytest.raises(ValueError, match="fixed pixel width"):
        figure.to_svg()


def test_to_html_injects_formatter_js(tmp_path):
    path = plot_jump_lengths_ecdf([2, 1], [3, 1]).to_html(tmp_path / "chart.html")
    content = path.read_text(encoding="utf-8")

    assert "axisLabel.formatter" in content
    assert "tooltip.formatter" in content


@pytest.mark.parametrize(
    ("chart_type", "expected_marker", "excluded_markers"),
    [
        ("ecdf", "function ecdfAt", ("function formatPercent", "function formatDifference")),
        ("bar", "formatPercent(params.value)", ("function ecdfAt", "function heatmapAxisData")),
        ("visit_purpose_comparison", "data.count", ("function ecdfAt", "function heatmapAxisData")),
        ("transition", "function heatmapAxisData", ("function ecdfAt", "function formatDifference")),
        ("daily_activity", "function heatmapAxisData", ("function ecdfAt", "function formatDifference")),
        (
            "transition_difference",
            "function formatDifference",
            ("function ecdfAt", "function formatPercent"),
        ),
        (
            "daily_activity_difference",
            "function formatDifference",
            ("function ecdfAt", "function formatPercent"),
        ),
        (
            "motif_literature_comparison",
            "Packed motif ID",
            ("function ecdfAt", "function heatmapAxisData", "function formatDifference"),
        ),
    ],
)
def test_formatter_registry_injects_only_expected_resources(chart_type, expected_marker, excluded_markers):
    content = EChartsFigure({"_meta": {"chartType": chart_type}})._to_html_string()

    assert expected_marker in content
    for excluded_marker in excluded_markers:
        assert excluded_marker not in content


def test_unknown_chart_type_does_not_inject_formatter():
    content = EChartsFigure({"_meta": {"chartType": "unknown"}})._to_html_string()

    assert "function ecdfAt" not in content
    assert "function formatPercent" not in content
    assert "function formatDifference" not in content
    assert "function heatmapAxisData" not in content


def test_missing_chart_type_defaults_to_ecdf_formatter():
    content = EChartsFigure({})._to_html_string()

    assert "function ecdfAt" in content


def test_to_json_writes_full_payload(tmp_path):
    figure = plot_jump_lengths_ecdf([2, 1], [3, 1])
    path = figure.to_json(tmp_path / "chart.json")

    assert path == tmp_path / "chart.json"
    assert json.loads(path.read_text(encoding="utf-8")) == figure.to_dict()


def test_invalid_input_fails_loudly():
    with pytest.raises(ValueError, match="finite"):
        plot_jump_lengths_ecdf([1.0, float("nan")], [1.0])


# ---------------------------------------------------------------------------
# visits frequency
# ---------------------------------------------------------------------------

def test_plot_visits_frequency_ecdf_returns_figure():
    figure = plot_visits_frequency_ecdf([5, 1], [8, 2], labels=("observed", "synthetic"))

    assert isinstance(figure, EChartsFigure)


def test_visits_frequency_ecdf_series_and_axis_labels():
    option = plot_visits_frequency_ecdf([5, 1], [8, 2], labels=("observed", "synthetic")).to_dict()

    assert option["series"][0]["name"] == "observed"
    assert option["series"][0]["data"] == [[1.0, 0.5], [5.0, 0.98]]
    assert option["series"][1]["name"] == "synthetic"
    assert option["series"][1]["data"] == [[2.0, 0.5], [8.0, 0.98]]
    assert option["xAxis"]["name"] == "NUMBER OF VISITS"
    assert option["yAxis"]["name"] == "F(X)"


def test_visits_frequency_ecdf_custom_cdf_cutoff():
    option = plot_visits_frequency_ecdf([4, 1, 3, 2], [10, 20, 30, 40], cdf_cutoff=0.75).to_dict()

    assert option["series"][0]["data"] == [[1.0, 0.25], [2.0, 0.5], [3.0, 0.75]]
    assert option["series"][1]["data"] == [[10.0, 0.25], [20.0, 0.5], [30.0, 0.75]]
    assert option["yAxis"]["max"] == 1


def test_visits_frequency_ecdf_default_title():
    option = plot_visits_frequency_ecdf([1, 2], [3, 4]).to_dict()

    title = option["title"]
    text = title[0]["text"] if isinstance(title, list) else title["text"]
    assert text == "Visits frequency ECDF"


def test_visits_frequency_ecdf_invalid_input_fails_loudly():
    with pytest.raises(ValueError, match="finite"):
        plot_visits_frequency_ecdf([1.0, float("nan")], [1.0])


# ---------------------------------------------------------------------------
# radius of gyration
# ---------------------------------------------------------------------------

def test_plot_radius_of_gyration_ecdf_returns_figure():
    figure = plot_radius_of_gyration_ecdf([2.5, 1.0], [3.0, 1.5])

    assert isinstance(figure, EChartsFigure)


def test_radius_of_gyration_ecdf_axis_labels():
    option = plot_radius_of_gyration_ecdf([2.5, 1.0], [3.0, 1.5]).to_dict()

    assert option["xAxis"]["name"] == "RADIUS OF GYRATION · KM"
    assert option["yAxis"]["name"] == "F(X)"


def test_radius_of_gyration_ecdf_default_title():
    option = plot_radius_of_gyration_ecdf([1.0, 2.0], [1.5, 2.5]).to_dict()

    title = option["title"]
    text = title[0]["text"] if isinstance(title, list) else title["text"]
    assert text == "Radius of gyration ECDF"


def test_radius_of_gyration_ecdf_invalid_input_fails_loudly():
    with pytest.raises(ValueError, match="finite"):
        plot_radius_of_gyration_ecdf([1.0, float("inf")], [1.0])


# ---------------------------------------------------------------------------
# trip duration
# ---------------------------------------------------------------------------

def test_plot_trip_duration_ecdf_returns_figure():
    figure = plot_trip_duration_ecdf([10, 30], [15, 45])

    assert isinstance(figure, EChartsFigure)


def test_trip_duration_ecdf_axis_labels():
    option = plot_trip_duration_ecdf([10, 30], [15, 45]).to_dict()

    assert option["xAxis"]["name"] == "TRIP DURATION · MIN"
    assert option["yAxis"]["name"] == "F(X)"


def test_trip_duration_ecdf_default_title():
    option = plot_trip_duration_ecdf([10, 20], [15, 25]).to_dict()

    title = option["title"]
    text = title[0]["text"] if isinstance(title, list) else title["text"]
    assert text == "Trip duration ECDF"


def test_trip_duration_ecdf_invalid_input_fails_loudly():
    with pytest.raises(ValueError, match="finite"):
        plot_trip_duration_ecdf([float("nan")], [1.0])


# ---------------------------------------------------------------------------
# dwell time
# ---------------------------------------------------------------------------

def test_plot_dwell_time_ecdf_returns_figure():
    figure = plot_dwell_time_ecdf([5, 20], [8, 30])

    assert isinstance(figure, EChartsFigure)


def test_dwell_time_ecdf_axis_labels():
    option = plot_dwell_time_ecdf([5, 20], [8, 30]).to_dict()

    assert option["xAxis"]["name"] == "DWELL TIME · MIN"
    assert option["yAxis"]["name"] == "F(X)"


def test_dwell_time_ecdf_default_title():
    option = plot_dwell_time_ecdf([5, 10], [8, 12]).to_dict()

    title = option["title"]
    text = title[0]["text"] if isinstance(title, list) else title["text"]
    assert text == "Dwell time ECDF"


def test_dwell_time_ecdf_invalid_input_fails_loudly():
    with pytest.raises(ValueError, match="finite"):
        plot_dwell_time_ecdf([1.0, float("nan")], [1.0])


# ---------------------------------------------------------------------------
# activity plots
# ---------------------------------------------------------------------------

def test_plot_visit_purpose_distribution_returns_figure():
    figure = plot_visit_purpose_distribution((["HOME", "WORK"], [60.0, 40.0]))

    assert isinstance(figure, EChartsFigure)


def test_visit_purpose_distribution_uses_bar_series():
    option = plot_visit_purpose_distribution((["HOME", "WORK"], [60.0, 40.0])).to_dict()

    assert option["_meta"]["chartType"] == "bar"
    assert option["series"][0]["type"] == "bar"
    assert option["xAxis"]["data"] == ["HOME", "WORK"]


def test_visit_purpose_distribution_missing_purpose_warns_unknown():
    visits = {"user_id": [1, 2], "start_timestamp": ["2020-01-01 08:00", "2020-01-01 09:00"]}

    with pytest.warns(UserWarning, match="activity column"):
        option = plot_visit_purpose_distribution(visits).to_dict()

    assert option["xAxis"]["data"] == ["UNKNOWN"]
    assert option["series"][0]["data"][0]["value"] == 100.0


def test_raw_activity_mapping_delegates_to_top_level_fastmob(monkeypatch):
    import fastmob
    import pandas as pd

    received = {}

    def fake_distribution(visits, unknown_label):
        received["visits"] = visits
        received["unknown_label"] = unknown_label
        return pd.DataFrame({"activity": ["HOME"], "count": [2], "percentage": [100.0]})

    monkeypatch.setattr(fastmob, "visit_purpose_distribution", fake_distribution)

    option = plot_visit_purpose_distribution({"purpose": ["HOME", "HOME"]}).to_dict()

    assert isinstance(received["visits"], pd.DataFrame)
    assert received["unknown_label"] == "UNKNOWN"
    assert option["series"][0]["data"][0]["count"] == 2


def test_raw_activity_metric_errors_are_not_swallowed(monkeypatch):
    import fastmob

    def fail(*args, **kwargs):
        raise RuntimeError("metric failed")

    monkeypatch.setattr(fastmob, "activity_transition_matrix", fail)

    with pytest.raises(RuntimeError, match="metric failed"):
        plot_activity_transition_matrix({"purpose": ["HOME"]})


def test_visit_purpose_distribution_rounds_percentages_to_two_decimals(tmp_path):
    figure = plot_visit_purpose_distribution((["HOME"], [33.333333]))
    option = figure.to_dict()
    path = figure.to_html(tmp_path / "activity.html")
    content = path.read_text(encoding="utf-8")

    assert option["series"][0]["data"][0]["value"] == 33.33
    assert "toFixed(2) + '%'" in content


def test_visit_purpose_comparison_aligns_categories_and_series():
    option = plot_visit_purpose_comparison(
        {
            "Observed": (["HOME", "WORK"], [60.0, 40.0], [6, 4]),
            "Model": (["WORK", "LEISURE"], [25.0, 75.0], [1, 3]),
        }
    ).to_dict()

    assert option["_meta"]["chartType"] == "visit_purpose_comparison"
    assert option["xAxis"]["data"] == ["HOME", "WORK", "LEISURE"]
    assert [series["name"] for series in option["series"]] == ["Observed", "Model"]
    assert [item["value"] for item in option["series"][0]["data"]] == [60.0, 40.0, 0.0]
    assert [item["value"] for item in option["series"][1]["data"]] == [0.0, 25.0, 75.0]
    assert "count" not in option["series"][0]["data"][2]


def test_visit_purpose_comparison_colors_by_purpose_and_shades_by_dataset():
    option = plot_visit_purpose_comparison(
        {
            "A": (["HOME", "WORK"], [50.0, 50.0]),
            "B": (["HOME", "WORK"], [40.0, 60.0]),
            "C": (["HOME", "WORK"], [30.0, 70.0]),
        }
    ).to_dict()

    assert option["series"][0]["data"][0]["itemStyle"]["color"] == WARM["observed"]
    assert option["series"][1]["data"][0]["itemStyle"]["color"] == WARM["observed"]
    assert option["series"][0]["data"][1]["itemStyle"]["color"] == WARM["synthetic"]
    assert [series["data"][0]["itemStyle"]["opacity"] for series in option["series"]] == [1.0, 0.7, 0.4]


def test_visit_purpose_comparison_supports_count_and_percentage_inputs():
    option = plot_visit_purpose_comparison(
        {
            "Counts": {"activity": ["HOME", "WORK"], "count": [3, 1]},
            "Percentages": {"activity": ["HOME", "WORK"], "percentage": [25.0, 75.0]},
        }
    ).to_dict()

    assert [item["value"] for item in option["series"][0]["data"]] == [75.0, 25.0]
    assert option["series"][0]["data"][0]["count"] == 3
    assert "count" not in option["series"][1]["data"][0]


def test_visit_purpose_comparison_supports_raw_inputs(monkeypatch):
    import fastmob
    import pandas as pd

    def fake_distribution(visits, unknown_label):
        assert isinstance(visits, pd.DataFrame)
        return pd.DataFrame({"activity": ["HOME"], "count": [len(visits)], "percentage": [100.0]})

    monkeypatch.setattr(fastmob, "visit_purpose_distribution", fake_distribution)
    option = plot_visit_purpose_comparison(
        {"A": {"purpose": ["HOME"]}, "B": {"purpose": ["WORK", "WORK"]}}
    ).to_dict()

    assert option["series"][0]["data"][0]["count"] == 1
    assert option["series"][1]["data"][0]["count"] == 2


def test_visit_purpose_comparison_validates_mapping_and_duplicates():
    with pytest.raises(ValueError, match="non-empty mapping"):
        plot_visit_purpose_comparison({})

    with pytest.raises(ValueError, match="duplicate purpose labels"):
        plot_visit_purpose_comparison({"A": (["HOME", "HOME"], [50.0, 50.0])})


def test_visit_purpose_comparison_rounds_and_formats_tooltip(tmp_path):
    figure = plot_visit_purpose_comparison(
        {"A": (["HOME"], [33.333333], [2]), "B": (["HOME"], [66.666666])},
        palette="forest",
        width="100%",
        height="500px",
    )
    option = figure.to_dict()
    content = figure.to_html(tmp_path / "comparison.html").read_text(encoding="utf-8")

    assert option["series"][0]["data"][0]["value"] == 33.33
    assert option["series"][1]["data"][0]["value"] == 66.67
    assert option["series"][0]["data"][0]["itemStyle"]["color"] == PALETTES["forest"]["observed"]
    assert option["legend"]["data"] == ["A", "B"]
    assert option["series"][0]["label"]["show"] is False
    assert figure.width == "100%"
    assert figure.height == "500px"
    assert "data.count !== undefined" in content
    assert "formatPercent(data.value)" in content


def test_plot_visit_purpose_comparison_is_exported():
    import fastmob_vis
    import fastmob_vis.plots

    assert fastmob_vis.plot_visit_purpose_comparison is plot_visit_purpose_comparison
    assert fastmob_vis.plots.plot_visit_purpose_comparison is plot_visit_purpose_comparison


def test_activity_transition_matrix_uses_heatmap_series():
    option = plot_activity_transition_matrix((["HOME", "WORK"], [[0.0, 50.0], [50.0, 0.0]])).to_dict()

    assert option["_meta"]["chartType"] == "transition"
    assert option["series"][0]["type"] == "heatmap"
    assert option["xAxis"]["data"] == ["HOME", "WORK"]
    assert option["yAxis"]["data"] == ["HOME", "WORK"]


def test_activity_transition_matrix_missing_purpose_warns_unknown():
    visits = {
        "user_id": [1, 1, 1],
        "start_timestamp": ["2020-01-01 08:00", "2020-01-01 09:00", "2020-01-01 10:00"],
    }

    with pytest.warns(UserWarning, match="activity column"):
        option = plot_activity_transition_matrix(visits).to_dict()

    assert option["xAxis"]["data"] == ["UNKNOWN"]
    assert option["yAxis"]["data"] == ["UNKNOWN"]
    assert option["series"][0]["data"] == [[0, 0, 100.0]]


def test_activity_transition_matrix_rounds_percentages_to_two_decimals():
    option = plot_activity_transition_matrix((["HOME"], [[33.333333]])).to_dict()

    assert option["series"][0]["data"] == [[0, 0, 33.33]]


def test_daily_activity_distribution_uses_heatmap_series():
    option = plot_daily_activity_distribution(([[100.0, float("nan")]], ["HOME"], 2)).to_dict()

    assert option["_meta"]["chartType"] == "daily_activity"
    assert option["series"][0]["type"] == "heatmap"
    assert option["yAxis"]["data"] == ["HOME"]
    assert option["series"][0]["data"] == [[0, 0, 100.0]]


def test_daily_activity_distribution_missing_purpose_warns_unknown():
    visits = {
        "start_timestamp": ["2020-01-01 08:00"],
        "end_timestamp": ["2020-01-01 08:10"],
    }

    with pytest.warns(UserWarning, match="activity column"):
        option = plot_daily_activity_distribution(visits).to_dict()

    assert option["yAxis"]["data"] == ["UNKNOWN"]
    assert [48, 0, 100.0] in option["series"][0]["data"]


def test_daily_activity_distribution_rounds_percentages_to_two_decimals():
    option = plot_daily_activity_distribution(([[33.333333]], ["HOME"], 1)).to_dict()

    assert option["series"][0]["data"] == [[0, 0, 33.33]]


def test_activity_transition_matrix_html_uses_dynamic_tooltip(tmp_path):
    path = plot_activity_transition_matrix((["HOME", "WORK"], [[0.0, 75.0], [25.0, 0.0]])).to_html(
        tmp_path / "transition.html"
    )
    content = path.read_text(encoding="utf-8")

    assert "option.tooltip.formatter = function(params)" in content
    assert "value[2]" in content
    assert "yLabel + ' -> ' + xLabel" in content
    assert "formatPercent(params.value[2])" in content


def test_daily_activity_distribution_html_uses_dynamic_tooltip(tmp_path):
    path = plot_daily_activity_distribution(([[25.0, 75.0]], ["HOME"], 2)).to_html(tmp_path / "daily.html")
    content = path.read_text(encoding="utf-8")

    assert "option.tooltip.formatter = function(params)" in content
    assert "value[2]" in content
    assert "Percentage: ' + percent" in content


def test_activity_transition_difference_aligns_union_and_preserves_sign():
    option = plot_activity_transition_difference(
        (["HOME", "WORK"], [[20.0, 80.0], [40.0, 60.0]]),
        (["WORK", "SHOP"], [[10.0, 90.0], [30.0, 70.0]]),
        labels=("observed", "synthetic"),
    ).to_dict()

    assert option["_meta"] == {
        "chartType": "transition_difference",
        "differenceLabels": ["observed", "synthetic"],
    }
    assert option["xAxis"]["data"] == ["HOME", "WORK", "SHOP"]
    assert option["yAxis"]["data"] == ["HOME", "WORK", "SHOP"]
    assert [0, 0, -20.0] in option["series"][0]["data"]
    assert [1, 0, -80.0] in option["series"][0]["data"]
    assert [2, 1, 90.0] in option["series"][0]["data"]
    assert [2, 2, 70.0] in option["series"][0]["data"]
    assert option["visualMap"]["min"] == -90.0
    assert option["visualMap"]["max"] == 90.0
    assert option["visualMap"]["inRange"]["color"] == [WARM["observed"], WARM["bg"], WARM["synthetic"]]
    assert option["series"][0]["name"] == "synthetic - observed"


def test_activity_transition_difference_rounds_and_omits_nan():
    option = plot_activity_transition_difference(
        (["HOME"], [[float("nan")]]),
        (["HOME"], [[33.333333]]),
    ).to_dict()

    assert option["series"][0]["data"] == []
    assert option["visualMap"]["min"] == -1.0
    assert option["visualMap"]["max"] == 1.0


def test_activity_transition_difference_accepts_raw_inputs(monkeypatch):
    calls = []

    def fake_transition(visits, unknown_label):
        calls.append((visits["source"].iloc[0], unknown_label))
        if visits["source"].iloc[0] == "first":
            return (["HOME"], [[25.0]])
        return (["HOME"], [[75.0]])

    import fastmob

    monkeypatch.setattr(fastmob, "activity_transition_matrix", fake_transition)
    option = plot_activity_transition_difference(
        {"source": ["first"]}, {"source": ["second"]}
    ).to_dict()

    assert calls == [("first", "UNKNOWN"), ("second", "UNKNOWN")]
    assert option["series"][0]["data"] == [[0, 0, 50.0]]


def test_activity_transition_difference_accepts_dataframe_matrices():
    pd = pytest.importorskip("pandas")
    first = pd.DataFrame([[10.0, 90.0], [30.0, 70.0]], index=["HOME", "WORK"], columns=["HOME", "WORK"])
    second = pd.DataFrame([[80.0, 20.0], [60.0, 40.0]], index=["HOME", "WORK"], columns=["WORK", "HOME"])

    option = plot_activity_transition_difference(first, second).to_dict()

    assert option["series"][0]["data"] == [
        [0, 0, 10.0],
        [1, 0, -10.0],
        [0, 1, 10.0],
        [1, 1, -10.0],
    ]


def test_activity_transition_difference_validates_inputs():
    with pytest.raises(ValueError, match="exactly two"):
        plot_activity_transition_difference((["HOME"], [[100.0]]), (["HOME"], [[100.0]]), labels=("one",))
    with pytest.raises(ValueError, match="duplicate"):
        plot_activity_transition_difference(
            (["HOME", "HOME"], [[1.0, 2.0], [3.0, 4.0]]),
            (["HOME"], [[100.0]]),
        )
    with pytest.raises(ValueError, match="must have shape"):
        plot_activity_transition_difference((["HOME"], [[1.0, 2.0]]), (["HOME"], [[100.0]]))
    with pytest.raises(ValueError, match="finite values"):
        plot_activity_transition_difference((["HOME"], [[float("inf")]]), (["HOME"], [[100.0]]))


def test_daily_activity_difference_aligns_union_and_requires_matching_bins():
    option = plot_daily_activity_difference(
        ([[10.0, 20.0], [30.0, float("nan")]], ["HOME", "WORK"], 2),
        ([[50.0, 10.0], [40.0, 60.0]], ["WORK", "SHOP"], 2),
    ).to_dict()

    assert option["yAxis"]["data"] == ["HOME", "WORK", "SHOP"]
    assert [0, 0, -10.0] in option["series"][0]["data"]
    assert [0, 1, 20.0] in option["series"][0]["data"]
    assert [1, 1, 10.0] not in option["series"][0]["data"]
    assert [0, 2, 40.0] in option["series"][0]["data"]
    assert [1, 2, 60.0] in option["series"][0]["data"]
    assert option["visualMap"]["min"] == -60.0
    assert option["visualMap"]["max"] == 60.0

    with pytest.raises(ValueError, match="same n_bins"):
        plot_daily_activity_difference(
            ([[100.0]], ["HOME"], 1),
            ([[50.0, 50.0]], ["HOME"], 2),
        )


def test_daily_activity_difference_accepts_raw_and_explicit_matrix_inputs(monkeypatch):
    def fake_daily(visits, unknown_label):
        value = 25.0 if visits["source"].iloc[0] == "first" else 75.0
        return ([[value]], ["HOME"], 1)

    import fastmob

    monkeypatch.setattr(fastmob, "daily_activity_distribution", fake_daily)
    raw_option = plot_daily_activity_difference(
        {"source": ["first"]}, {"source": ["second"]}
    ).to_dict()
    matrix_option = plot_daily_activity_difference(
        [[25.0]], [[75.0]], categories=["HOME"], n_bins=1
    ).to_dict()

    assert raw_option["series"][0]["data"] == [[0, 0, 50.0]]
    assert matrix_option["series"][0]["data"] == [[0, 0, 50.0]]


def test_daily_activity_difference_validates_categories_and_shapes():
    with pytest.raises(ValueError, match="duplicate"):
        plot_daily_activity_difference(
            ([[10.0], [20.0]], ["HOME", "HOME"], 1),
            ([[30.0]], ["HOME"], 1),
        )
    with pytest.raises(ValueError, match="must have shape"):
        plot_daily_activity_difference(
            ([[10.0, 20.0]], ["HOME"], 1),
            ([[30.0]], ["HOME"], 1),
        )


def test_difference_heatmap_html_uses_signed_percentage_points(tmp_path):
    path = plot_activity_transition_difference(
        (["HOME"], [[75.0]]),
        (["HOME"], [[25.0]]),
        labels=("reference", "candidate"),
    ).to_html(tmp_path / "difference.html")
    content = path.read_text(encoding="utf-8")

    assert "number > 0 ? '+' : ''" in content
    # Cells show the bare signed number; the tooltip keeps the " pp" suffix.
    assert "number.toFixed(2) + (suffix || '')" in content
    assert "formatDifference(value[2], ' pp')" in content
    assert "labels[1] + ' - ' + labels[0]" in content
    assert "meta.chartType === 'transition_difference'" in content


def test_activity_difference_plots_are_exported():
    import fastmob_vis
    import fastmob_vis.plots

    assert fastmob_vis.plot_activity_transition_difference is plot_activity_transition_difference
    assert fastmob_vis.plot_daily_activity_difference is plot_daily_activity_difference
    assert fastmob_vis.plots.plot_activity_transition_difference is plot_activity_transition_difference
    assert fastmob_vis.plots.plot_daily_activity_difference is plot_daily_activity_difference


def test_non_ecdf_html_does_not_include_ecdf_formatter(tmp_path):
    path = plot_visit_purpose_distribution((["HOME"], [100.0])).to_html(tmp_path / "activity.html")
    content = path.read_text(encoding="utf-8")

    assert "ecdfAt" not in content
    assert "axisLabel.formatter = function" not in content


# ---------------------------------------------------------------------------
# motif literature comparison
# ---------------------------------------------------------------------------

def test_format_motif_hex_id_uses_lowercase_hex():
    assert format_motif_hex_id(LITERATURE_TO_FASTMOB_MOTIF_ID[2]) == "0x2000000006"
    assert format_motif_hex_id("other") == "other"


def test_all_literature_motif_svgs_are_packaged():
    resource_names = {
        name
        for name in importlib.resources.contents("fastmob_vis.assets.motifs")
        if name.endswith(".svg")
    }
    expected_names = {
        f"{format_motif_hex_id(motif_id)}.svg"
        for motif_id in LITERATURE_TO_FASTMOB_MOTIF_ID.values()
    }

    assert resource_names == expected_names


def test_motif_distribution_maps_known_ids_and_other():
    known_id = LITERATURE_TO_FASTMOB_MOTIF_ID[2]
    mapped = map_motif_distribution_to_literature_basis(
        {"motif_id": [known_id, 12345], "percentage": [25.0, 75.0], "count": [2, 6]}
    )

    assert len(mapped) == 18
    assert mapped[1]["motif_id"] == known_id
    assert mapped[1]["percentage"] == 25.0
    assert mapped[1]["count"] == 2
    assert mapped[-1]["motif_id"] == "other"
    assert mapped[-1]["percentage"] == 75.0
    assert mapped[-1]["count"] == 6


def test_motif_distribution_count_only_computes_percentages():
    known_id = LITERATURE_TO_FASTMOB_MOTIF_ID[1]
    mapped = map_motif_distribution_to_literature_basis(
        {"motif_id": [known_id, 999], "count": [1, 3]}
    )

    assert mapped[0]["percentage"] == 25.0
    assert mapped[-1]["percentage"] == 75.0


def test_motif_distribution_rounds_percentages_to_two_decimals():
    known_id = LITERATURE_TO_FASTMOB_MOTIF_ID[1]
    mapped = map_motif_distribution_to_literature_basis(
        {"motif_id": [known_id], "percentage": [33.333333]}
    )

    assert mapped[0]["percentage"] == 33.33


def test_plot_motif_literature_comparison_accepts_sibling_literature_schema():
    option = plot_motif_literature_comparison(
        literature_df={
            "motif_id": [1],
            "fastmob_motif_id": [LITERATURE_TO_FASTMOB_MOTIF_ID[1]],
            "percentage": [100.0],
        }
    ).to_dict()

    assert option["series"][0]["data"][0][1] == 100.0
    assert option["series"][0]["data"][-1][1] == 0.0


def test_plot_motif_literature_comparison_returns_figure():
    figure = plot_motif_literature_comparison(
        reference_distribution={"motif_id": [LITERATURE_TO_FASTMOB_MOTIF_ID[1]], "percentage": [100.0]},
        comparison_distribution={"motif_id": [LITERATURE_TO_FASTMOB_MOTIF_ID[2]], "percentage": [100.0]},
    )

    assert isinstance(figure, EChartsFigure)


def test_motif_literature_comparison_option_shape():
    option = plot_motif_literature_comparison(
        reference_distribution={"motif_id": [LITERATURE_TO_FASTMOB_MOTIF_ID[1]], "count": [4]},
        comparison_distribution={"motif_id": [999], "count": [2]},
        labels=("NetMob", "Simulation"),
    ).to_dict()

    assert option["_meta"]["chartType"] == "motif_literature_comparison"
    assert len(option["xAxis"]["data"]) == 18
    assert option["xAxis"]["data"][0] == "0x1000000000"
    assert option["xAxis"]["data"][-1] == "Other"
    assert [series["name"] for series in option["series"]] == ["Literature", "NetMob", "Simulation"]
    assert option["series"][0]["itemStyle"]["color"] == WARM["baseline"]
    assert option["series"][1]["itemStyle"]["color"] == WARM["observed"]
    assert option["series"][2]["itemStyle"]["color"] == WARM["synthetic"]
    assert option["series"][1]["data"][0][1] == 100.0
    assert option["series"][2]["data"][-1][1] == 100.0


def test_motif_literature_comparison_uses_svg_axis_labels():
    option = plot_motif_literature_comparison().to_dict()
    label_keys = option["_meta"]["motifLabelKeys"]
    rich_styles = option["xAxis"]["axisLabel"]["rich"]

    assert len(label_keys) == 17
    assert len(rich_styles) == 17
    assert "Other" not in label_keys
    for hex_id, style_key in label_keys.items():
        assert hex_id.startswith("0x")
        assert rich_styles[style_key]["width"] == 88
        assert rich_styles[style_key]["height"] == 88
        assert rich_styles[style_key]["backgroundColor"]["image"].startswith(
            "data:image/svg+xml;base64,"
        )


def test_motif_literature_comparison_html_uses_two_decimal_percentages(tmp_path):
    path = plot_motif_literature_comparison(
        reference_distribution={"motif_id": [LITERATURE_TO_FASTMOB_MOTIF_ID[1]], "percentage": [33.333333]},
    ).to_html(tmp_path / "motifs.html")
    content = path.read_text(encoding="utf-8")

    assert "toFixed(2) + '%'" in content
    assert "formatPercent(value[1])" in content
    assert "data:image/svg+xml;base64," in content
    assert "const motifLabelKeys = meta.motifLabelKeys || {}" in content
    assert "styleKey ? '{' + styleKey + '| }' : value" in content
    assert "fastmob_vis/assets/motifs" not in content


def test_plot_motif_literature_comparison_is_exported():
    import fastmob_vis

    assert "plot_motif_literature_comparison" in fastmob_vis.__all__
