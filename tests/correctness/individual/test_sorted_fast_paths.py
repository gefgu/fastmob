"""Correctness checks for trusted sorted spatial fast paths."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

SPATIAL_FUNCTIONS = (
    "distance_straight_line",
    "frequency_rank",
    "home_location",
    "individual_mobility_network",
    "k_radius_of_gyration",
    "location_frequency",
    "max_distance_from_home",
    "maximum_distance",
    "number_of_locations",
    "number_of_visits",
    "recency_rank",
    "waiting_times",
)

AUTO_DISPATCH_FUNCTIONS = ("jump_lengths", "radius_of_gyration")


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
    return (
        df.assign(__row_order=np.arange(len(df)))
        .sort_values(["uid", "datetime", "__row_order"], kind="mergesort")
        .drop(columns=["__row_order"])
    )


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
    module = __import__(f"fastmob.measures.individual.{name}", fromlist=[name])
    metric = getattr(module, name)
    raw = _tiny_unsorted_df()
    sorted_input = _sorted_df(raw)
    kwargs = _metric_kwargs(name)

    default_result = metric(raw, **kwargs)
    sorted_result = metric(sorted_input, **kwargs)

    _assert_frames_equivalent(default_result, sorted_result)


@pytest.mark.parametrize("name", SPATIAL_FUNCTIONS)
def test_spatial_metric_has_no_presorted_keyword(name):
    module = __import__(f"fastmob.measures.individual.{name}", fromlist=[name])
    metric = getattr(module, name)
    assert "presorted" not in inspect.signature(metric).parameters


@pytest.mark.parametrize("name", AUTO_DISPATCH_FUNCTIONS)
def test_auto_dispatch_matches_between_sorted_and_unsorted_input(name):
    module = __import__(f"fastmob.measures.individual.{name}", fromlist=[name])
    metric = getattr(module, name)
    raw = _tiny_unsorted_df()
    kwargs = _metric_kwargs(name)

    _assert_frames_equivalent(metric(raw, **kwargs), metric(_sorted_df(raw), **kwargs))


@pytest.mark.parametrize("name", AUTO_DISPATCH_FUNCTIONS)
def test_auto_dispatch_takes_the_contiguous_path_only_when_it_may(name, monkeypatch):
    """Matching numbers alone cannot tell the two kernels apart -- watch which runs."""
    pd = pytest.importorskip("pandas")
    module = __import__(f"fastmob.measures.individual.{name}", fromlist=[name])
    metric = getattr(module, name)
    kwargs = _metric_kwargs(name)

    taken = []
    for kernel in (f"{name}_presorted", f"{name}_indexed"):
        original = getattr(module, kernel)

        def spy(*args, _kernel=kernel, _original=original, **kw):
            taken.append(_kernel)
            return _original(*args, **kw)

        monkeypatch.setattr(module, kernel, spy)

    raw = _tiny_unsorted_df()
    sorted_input = _sorted_df(raw)

    metric(raw, **kwargs)
    assert taken == [f"{name}_indexed"]

    taken.clear()
    metric(sorted_input, **kwargs)
    assert taken == [f"{name}_presorted"]

    # A NaN coordinate must fall back: the contiguous kernels have no per-row
    # validity test, so they would fold the NaN into a whole user's result.
    taken.clear()
    with_nan = sorted_input.copy()
    with_nan.loc[with_nan.index[0], "lat"] = np.nan
    metric(with_nan, **kwargs)
    assert taken == [f"{name}_indexed"]

    # Grouped by user but with one user's rows split in two runs.
    taken.clear()
    split_runs = pd.concat([sorted_input, sorted_input.iloc[:1]], ignore_index=True)
    metric(split_runs, **kwargs)
    assert taken == [f"{name}_indexed"]


def test_sorted_waiting_times_and_jump_lengths_merge_match_default_single_user():
    pd = pytest.importorskip("pandas")
    from fastmob.measures.individual.jump_lengths import jump_lengths
    from fastmob.measures.individual.waiting_times import waiting_times

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
        jump_lengths(sorted_input, merge=True),
        jump_lengths(raw, merge=True),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        waiting_times(sorted_input, merge=True),
        waiting_times(raw, merge=True),
        rtol=1e-10,
        atol=1e-10,
    )


def test_auto_dispatch_radius_of_gyration_filters_invalid_users_either_way():
    pd = pytest.importorskip("pandas")
    from fastmob.measures.individual.radius_of_gyration import radius_of_gyration

    raw = pd.DataFrame(
        {
            "uid": ["a", "a", "b"],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"], utc=True),
            "lat": [np.nan, np.nan, 0.0],
            "lng": [np.nan, np.nan, 0.0],
        }
    )
    # The contiguous kernel cannot filter users whose every coordinate is
    # invalid, so auto-dispatch has to keep this frame on the indexed path; a
    # caller who cleans the frame first gets the fast path and the same answer.
    sorted_input = _sorted_df(raw.dropna(subset=["lat", "lng"]))

    default_result = radius_of_gyration(raw)
    sorted_result = radius_of_gyration(sorted_input)

    _assert_frames_equivalent(default_result, sorted_result)


def test_sorted_spatial_fast_path_polars_smoke():
    pl = pytest.importorskip("polars")
    from fastmob.measures.individual.distance_straight_line import distance_straight_line

    raw = _tiny_unsorted_df()
    sorted_input = pl.from_pandas(_sorted_df(raw))
    result = distance_straight_line(sorted_input)

    assert result.height == 3
    assert "distance_straight_line" in result.columns
