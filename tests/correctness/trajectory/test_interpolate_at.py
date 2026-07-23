"""Correctness tests for fastmob.trajectory.interpolate_at.

Hand-crafted fixture in ``conftest.py`` (``interpolate_at_tdf``): a single
user with 3 points, 1 hour apart, moving at a constant 1 degree lat/hour.
Cached ``movingpandas_reference`` comparisons validate against MovingPandas'
real ``Trajectory.get_position_at`` (see
``tests/populate_movingpandas_cache.py``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from fastmob.trajectory import interpolate_at


def test_linear_interpolates_between_points(interpolate_at_tdf):
    result = interpolate_at(interpolate_at_tdf, at="2020-01-01 00:30", method="linear")
    assert len(result) == 1
    assert bool(result["valid"].iloc[0]) is True
    assert result["lat"].iloc[0] == pytest.approx(0.5)


def test_nearest_snaps_to_closer_point(interpolate_at_tdf):
    result = interpolate_at(interpolate_at_tdf, at="2020-01-01 00:40", method="nearest")
    assert result["lat"].iloc[0] == pytest.approx(1.0)


def test_exact_timestamp_match_returns_exact_position(interpolate_at_tdf):
    result = interpolate_at(interpolate_at_tdf, at="2020-01-01 01:00", method="linear")
    assert result["lat"].iloc[0] == pytest.approx(1.0)


def test_out_of_range_query_is_invalid(interpolate_at_tdf):
    result = interpolate_at(interpolate_at_tdf, at="2019-12-31 00:00", method="linear")
    assert bool(result["valid"].iloc[0]) is False
    assert pd.isna(result["lat"].iloc[0])


def test_multiple_query_times_return_one_row_each(interpolate_at_tdf):
    result = interpolate_at(interpolate_at_tdf, at=["2020-01-01 00:30", "2020-01-01 01:30"], method="linear")
    assert len(result) == 2
    assert sorted(result["lat"].tolist()) == pytest.approx([0.5, 1.5])


def test_multi_user_returns_one_row_per_user_per_query():
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "b", "b"],
            "datetime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 00:00", "2020-01-01 01:00"]
            ),
            "lat": [0.0, 1.0, 10.0, 11.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = interpolate_at(df, at="2020-01-01 00:30", method="linear")
    assert len(result) == 2
    assert sorted(result["lat"].tolist()) == pytest.approx([0.5, 10.5])


def test_unknown_method_raises(interpolate_at_tdf):
    with pytest.raises(ValueError, match="unknown interpolate_at method"):
        interpolate_at(interpolate_at_tdf, at="2020-01-01 00:30", method="bogus")


def test_matches_movingpandas_reference(movingpandas_reference):
    """Numeric-value comparison against MovingPandas' real
    `Trajectory.get_position_at` for both `"linear"` and `"nearest"`."""
    input_df = movingpandas_reference.input_df
    checked_any = False
    for method in ("linear", "nearest"):
        reference = movingpandas_reference.position_at(method)
        if reference is None:
            pytest.skip(f"No MovingPandas interpolate_at({method}) reference cached.")

        for uid, ref_group in reference.groupby("uid", sort=False):
            user_df = input_df[input_df["uid"] == uid][["uid", "datetime", "lat", "lng"]].reset_index(drop=True)
            ref_group = ref_group.sort_values("query_time").reset_index(drop=True)
            result = interpolate_at(user_df, at=ref_group["query_time"].tolist(), method=method)
            result = result.sort_values("query_time").reset_index(drop=True)

            assert result["valid"].tolist() == ref_group["valid"].tolist()
            valid_mask = ref_group["valid"].to_numpy()
            assert result.loc[valid_mask, "lat"].to_numpy() == pytest.approx(
                ref_group.loc[valid_mask, "lat"].to_numpy(), abs=1e-6
            )
            assert result.loc[valid_mask, "lng"].to_numpy() == pytest.approx(
                ref_group.loc[valid_mask, "lon"].to_numpy(), abs=1e-6
            )
            checked_any = True

    assert checked_any, "no users/methods had cached reference data to compare"
