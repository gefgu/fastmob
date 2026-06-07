"""Correctness checks for trusted sorted spatial fast paths."""

from __future__ import annotations

import inspect

import numpy as np
import pytest


SPATIAL_FUNCTIONS = (
    "distance_straight_line",
    "home_location",
    "jump_lengths",
    "k_radius_of_gyration",
    "max_distance_from_home",
    "maximum_distance",
    "number_of_locations",
    "number_of_visits",
    "radius_of_gyration",
    "waiting_times",
)


def _tiny_unsorted_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "uid": ["b", "a", "a", "c", "b", "a"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 01:00:00",
                    "2020-01-01 03:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 02:00:00",
                ],
                utc=True,
            ),
            "lat": [10.0, 0.0, 0.0, 5.0, 10.0, 0.0],
            "lng": [0.0, 3.0, 0.0, 5.0, 2.0, 1.0],
        }
    )


def _sorted_df(df):
    return df.assign(__row_order=np.arange(len(df))).sort_values(
        ["uid", "datetime", "__row_order"], kind="mergesort"
    ).drop(columns=["__row_order"])


def _metric_kwargs(name: str) -> dict:
    if name in {"jump_lengths", "waiting_times"}:
        return {"merge": False}
    if name == "k_radius_of_gyration":
        return {"k": 2}
    return {}


def _to_pandas(result):
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    return result


def _assert_frames_equivalent(left, right):
    left = _to_pandas(left).sort_values(left.columns[0]).reset_index(drop=True)
    right = _to_pandas(right).sort_values(right.columns[0]).reset_index(drop=True)
    assert list(left.columns) == list(right.columns)
    assert left[left.columns[0]].tolist() == right[right.columns[0]].tolist()

    for column in left.columns[1:]:
        for left_value, right_value in zip(left[column], right[column]):
            if isinstance(left_value, (list, tuple, np.ndarray)):
                np.testing.assert_allclose(
                    np.asarray(left_value, dtype=float),
                    np.asarray(right_value, dtype=float),
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )
            else:
                np.testing.assert_allclose(
                    float(left_value),
                    float(right_value),
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )


@pytest.mark.parametrize("name", SPATIAL_FUNCTIONS)
def test_sorted_spatial_fast_path_matches_default_pandas(name):
    module = __import__(f"skmob2.measures.individual.{name}", fromlist=[name])
    metric = getattr(module, name)
    raw = _tiny_unsorted_df()
    sorted_input = _sorted_df(raw)
    kwargs = _metric_kwargs(name)

    default_result = metric(raw, **kwargs)
    sorted_result = metric(sorted_input, **kwargs, presorted=True)

    _assert_frames_equivalent(default_result, sorted_result)


@pytest.mark.parametrize("name", SPATIAL_FUNCTIONS)
def test_spatial_metric_accepts_presorted_keyword(name):
    module = __import__(f"skmob2.measures.individual.{name}", fromlist=[name])
    metric = getattr(module, name)
    assert "presorted" in inspect.signature(metric).parameters


def test_sorted_waiting_times_and_jump_lengths_merge_match_default_single_user():
    pd = pytest.importorskip("pandas")
    from skmob2.measures.individual.jump_lengths import jump_lengths
    from skmob2.measures.individual.waiting_times import waiting_times

    raw = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2020-01-01 03:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"],
                utc=True,
            ),
            "lat": [0.0, 0.0, 0.0],
            "lng": [3.0, 0.0, 1.0],
        }
    )
    sorted_input = raw.sort_values("datetime", kind="mergesort")

    np.testing.assert_allclose(
        jump_lengths(sorted_input, merge=True, presorted=True),
        jump_lengths(raw, merge=True),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        waiting_times(sorted_input, merge=True, presorted=True),
        waiting_times(raw, merge=True),
        rtol=1e-10,
        atol=1e-10,
    )


def test_sorted_radius_of_gyration_requires_clean_input_for_invalid_user_filtering():
    pd = pytest.importorskip("pandas")
    from skmob2.measures.individual.radius_of_gyration import radius_of_gyration

    raw = pd.DataFrame(
        {
            "uid": ["a", "a", "b"],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"], utc=True),
            "lat": [np.nan, np.nan, 0.0],
            "lng": [np.nan, np.nan, 0.0],
        }
    )
    sorted_input = _sorted_df(raw.dropna(subset=["lat", "lng"]))

    default_result = radius_of_gyration(raw)
    sorted_result = radius_of_gyration(sorted_input, presorted=True)

    _assert_frames_equivalent(default_result, sorted_result)


def test_sorted_spatial_fast_path_polars_smoke():
    pl = pytest.importorskip("polars")
    from skmob2.measures.individual.distance_straight_line import distance_straight_line

    raw = _tiny_unsorted_df()
    sorted_input = pl.from_pandas(_sorted_df(raw))
    result = distance_straight_line(sorted_input, presorted=True)

    assert result.height == 3
    assert "distance_straight_line" in result.columns
