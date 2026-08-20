"""Correctness tests for function-first privacy risk APIs."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
from fastmob import privacy


@pytest.fixture
def trajectory():
    return pd.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2, 2],
            "lat": [1.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "lng": [4.0, 5.0, 4.0, 4.0, 5.0, 6.0],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 2),
        }
    )


FUNCTIONS = (
    privacy.location_risk,
    privacy.location_sequence_risk,
    privacy.location_time_risk,
    privacy.unique_location_risk,
    privacy.location_frequency_risk,
    privacy.location_probability_risk,
    privacy.location_proportion_risk,
    privacy.home_work_risk,
)


def test_function_api_is_public_and_classes_are_removed():
    assert set(privacy.__all__) == {function.__name__ for function in FUNCTIONS}
    assert not hasattr(privacy, "LocationAttack")
    assert "knowledge_length" in inspect.signature(privacy.location_risk).parameters
    assert inspect.signature(privacy.location_risk).parameters["h3_resolution"].default == 12


@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda function: function.__name__)
def test_risk_functions_return_expected_schema(trajectory, function):
    kwargs = {} if function is privacy.home_work_risk else {"knowledge_length": 1}
    result = function(trajectory, **kwargs)
    assert list(result.columns) == ["uid", "risk"]
    assert set(result["uid"]) == {1, 2}


def test_sequence_and_time_risks_require_datetime(trajectory):
    no_datetime = trajectory.drop(columns="datetime")
    with pytest.raises(ValueError, match="datetime"):
        privacy.location_sequence_risk(no_datetime, 1)
    with pytest.raises(ValueError, match="datetime"):
        privacy.location_time_risk(no_datetime, 1)


def test_custom_columns_targets_and_force_instances(trajectory):
    renamed = trajectory.rename(
        columns={"uid": "user_id", "lat": "latitude", "lng": "longitude", "datetime": "timestamp"}
    )
    result = privacy.location_risk(renamed.drop(columns="timestamp"), 1, targets=[1], force_instances=True)
    assert list(result.columns) == ["latitude", "longitude", "user_id", "instance", "instance_elem", "prob"]
    assert set(result["user_id"]) == {1}
    timed = privacy.location_time_risk(renamed, 1, datetime_col="timestamp", force_instances=True)
    assert "timestamp" in timed.columns


@pytest.mark.parametrize("function", FUNCTIONS[:-1], ids=lambda function: function.__name__)
def test_invalid_knowledge_length(function, trajectory):
    with pytest.raises(ValueError, match="knowledge_length"):
        function(trajectory, 0)


@pytest.mark.parametrize(
    "function", [privacy.location_frequency_risk, privacy.location_probability_risk, privacy.location_proportion_risk]
)
def test_invalid_tolerance(function, trajectory):
    with pytest.raises(ValueError, match="tolerance"):
        function(trajectory, 1, tolerance=1.1)


def test_presorted_path_matches_default(trajectory):
    expected = privacy.location_sequence_risk(trajectory, 2)
    actual = privacy.location_sequence_risk(trajectory, 2)
    pd.testing.assert_frame_equal(actual, expected)


def test_default_h3_groups_gps_jitter():
    trajectory = pd.DataFrame(
        {
            "uid": [1, 2],
            "lat": [37.769377, 37.769378],
            "lng": [-122.388519, -122.388518],
        }
    )

    result = privacy.location_risk(trajectory, 1)

    assert set(result["risk"]) == {0.5}


def test_h3_force_instances_return_cell_centers():
    h3 = pytest.importorskip("h3")
    trajectory = pd.DataFrame({"uid": [1], "lat": [37.769377], "lng": [-122.388519]})

    result = privacy.location_risk(trajectory, 1, force_instances=True)
    cell = h3.latlng_to_cell(37.769377, -122.388519, 12)
    expected_lat, expected_lng = h3.cell_to_latlng(cell)

    assert result.loc[0, "lat"] == pytest.approx(expected_lat)
    assert result.loc[0, "lng"] == pytest.approx(expected_lng)


@pytest.mark.parametrize("resolution", [None, -1, 16, 1.5, True])
def test_invalid_h3_resolution(trajectory, resolution):
    with pytest.raises(ValueError, match="h3_resolution"):
        privacy.location_risk(trajectory, 1, h3_resolution=resolution)
