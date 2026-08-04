import numpy as np
import pytest
from fastmob_vis import (
    PALETTES,
    plot_mobility_profiles,
    plot_profile_metrics,
)


def _profiles(seed: int = 0):
    rng = np.random.default_rng(seed)
    types = ["Scouter", "Regular", "Routiner"]
    rows = {"degree_of_return": [], "intermittency": [], "agent_type": [],
            "regularity": [], "diversity": [], "stationarity": [], "entropy": []}
    for index, agent_type in enumerate(types):
        for _ in range(6):
            rows["degree_of_return"].append(0.2 + 0.3 * index + rng.random() * 0.1)
            rows["intermittency"].append(0.1 + 0.3 * index + rng.random() * 0.1)
            rows["agent_type"].append(agent_type)
            for metric in ("regularity", "diversity", "stationarity", "entropy"):
                rows[metric].append(rng.random())
    return rows


def test_mobility_profiles_one_series_per_profile():
    option = plot_mobility_profiles(_profiles()).to_dict()

    assert option["_meta"]["chartType"] == "mobility_profiles"
    assert option["xAxis"]["name"] == "DEGREE OF RETURN"
    assert option["yAxis"]["name"] == "INTERMITTENCY"
    # Canonical ordering: Routiner, Regular, Scouter.
    assert [series["name"] for series in option["series"]] == ["Routiner", "Regular", "Scouter"]
    assert all(series["type"] == "scatter" for series in option["series"])
    total_points = sum(len(series["data"]) for series in option["series"])
    assert total_points == 18
    assert option["series"][0]["data"][0] == pytest.approx(
        [option["series"][0]["data"][0][0], option["series"][0]["data"][0][1]]
    )


def test_mobility_profiles_default_colors_follow_palette_roles():
    option = plot_mobility_profiles(_profiles(), palette="warm").to_dict()
    by_name = {series["name"]: series for series in option["series"]}
    assert by_name["Scouter"]["itemStyle"]["color"] == PALETTES["warm"]["synthetic"]
    assert by_name["Regular"]["itemStyle"]["color"] == PALETTES["warm"]["syn3"]
    assert by_name["Routiner"]["itemStyle"]["color"] == PALETTES["warm"]["syn2"]


def test_mobility_profiles_custom_colors_and_length_validation():
    profiles = _profiles()
    custom = {"Scouter": "#111111", "Regular": "#222222", "Routiner": "#333333"}
    option = plot_mobility_profiles(profiles, profile_colors=custom).to_dict()
    by_name = {series["name"]: series for series in option["series"]}
    assert by_name["Scouter"]["itemStyle"]["color"] == "#111111"

    broken = dict(profiles)
    broken["intermittency"] = profiles["intermittency"][:-1]
    with pytest.raises(ValueError):
        plot_mobility_profiles(broken)


def test_profile_metrics_grid_and_boxplot_series():
    option = plot_profile_metrics({"synthetic": _profiles(1), "observed": _profiles(2)}).to_dict()

    assert option["_meta"]["chartType"] == "profile_metrics"
    # 4 grids (one per metric), 4 x/y axes, 2 datasets * 4 metrics = 8 boxplot series.
    assert len(option["grid"]) == 4
    assert len(option["xAxis"]) == 4
    assert len(option["yAxis"]) == 4
    assert len(option["series"]) == 8
    assert all(series["type"] == "boxplot" for series in option["series"])
    # main title + one title per metric.
    assert len(option["title"]) == 5
    assert option["title"][1]["text"] == "Regularity"

    first = option["series"][0]
    assert first["name"] == "synthetic"
    assert first["xAxisIndex"] == 0 and first["yAxisIndex"] == 0
    # one box (5-number summary) per profile in profile_order.
    assert len(first["data"]) == 3
    assert all(len(box) == 5 for box in first["data"])


def test_profile_metrics_handles_empty_profile_and_validates_input():
    profiles = _profiles(3)
    # No 'All' rows -> that profile yields a None box rather than erroring.
    option = plot_profile_metrics(
        {"only": profiles}, profile_order=("Scouter", "Regular", "Routiner", "All")
    ).to_dict()
    last_box = option["series"][0]["data"][-1]
    assert last_box is None

    with pytest.raises(ValueError):
        plot_profile_metrics({})
