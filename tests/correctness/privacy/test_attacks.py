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
    renamed = trajectory.rename(columns={"uid": "user_id", "lat": "latitude", "lng": "longitude", "datetime": "timestamp"})
    result = privacy.location_risk(renamed.drop(columns="timestamp"), 1, targets=[1], force_instances=True)
    assert list(result.columns) == ["latitude", "longitude", "user_id", "instance", "instance_elem", "prob"]
    assert set(result["user_id"]) == {1}
    timed = privacy.location_time_risk(renamed, 1, datetime_col="timestamp", force_instances=True)
    assert "timestamp" in timed.columns


@pytest.mark.parametrize("function", FUNCTIONS[:-1], ids=lambda function: function.__name__)
def test_invalid_knowledge_length(function, trajectory):
    with pytest.raises(ValueError, match="knowledge_length"):
        function(trajectory, 0)


@pytest.mark.parametrize("function", [privacy.location_frequency_risk, privacy.location_probability_risk, privacy.location_proportion_risk])
def test_invalid_tolerance(function, trajectory):
    with pytest.raises(ValueError, match="tolerance"):
        function(trajectory, 1, tolerance=1.1)


def test_presorted_path_matches_default(trajectory):
    expected = privacy.location_sequence_risk(trajectory, 2)
    actual = privacy.location_sequence_risk(trajectory, 2, presorted=True)
    pd.testing.assert_frame_equal(actual, expected)
