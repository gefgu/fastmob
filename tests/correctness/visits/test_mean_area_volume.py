"""Correctness tests for mean_area_volume."""
from __future__ import annotations

import pandas as pd
import pytest

from skmob2.measures.visits.mean_area_volume import mean_area_volume


def _df(**kwargs):
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# Basic single-slot tests
# ---------------------------------------------------------------------------


def test_single_visit_single_bin():
    df = _df(
        area=["A"],
        user_id=["u1"],
        start_timestamp=[pd.Timestamp("2020-01-06 08:00")],  # Monday
        end_timestamp=[pd.Timestamp("2020-01-06 08:10")],
    )
    result = mean_area_volume(df)
    assert set(result.columns) == {"area", "time_bin", "mean_volume"}
    row = result[(result["area"] == "A") & (result["time_bin"] == "08:00")]
    assert len(row) == 1
    assert row["mean_volume"].iloc[0] == pytest.approx(1.0 / 7.0)


def test_two_users_same_slot():
    df = _df(
        area=["A", "A"],
        user_id=["u1", "u2"],
        start_timestamp=[pd.Timestamp("2020-01-06 08:00")] * 2,
        end_timestamp=[pd.Timestamp("2020-01-06 08:10")] * 2,
    )
    result = mean_area_volume(df)
    row = result[(result["area"] == "A") & (result["time_bin"] == "08:00")]
    assert row["mean_volume"].iloc[0] == pytest.approx(2.0 / 7.0)


def test_same_user_two_mondays():
    # Two Mondays → n_dates=2, sum=2, DOW mean=1.0, overall=1/7
    df = _df(
        area=["A", "A"],
        user_id=["u1", "u1"],
        start_timestamp=[
            pd.Timestamp("2020-01-06 08:00"),
            pd.Timestamp("2020-01-13 08:00"),
        ],
        end_timestamp=[
            pd.Timestamp("2020-01-06 08:10"),
            pd.Timestamp("2020-01-13 08:10"),
        ],
    )
    result = mean_area_volume(df)
    row = result[(result["area"] == "A") & (result["time_bin"] == "08:00")]
    assert row["mean_volume"].iloc[0] == pytest.approx(1.0 / 7.0)


def test_two_users_two_mondays_unequal():
    # Day1: u1+u2 → count=2; Day2: u1 → count=1; DOW mean=(2+1)/2=1.5 → 1.5/7
    df = _df(
        area=["A", "A", "A"],
        user_id=["u1", "u2", "u1"],
        start_timestamp=[
            pd.Timestamp("2020-01-06 08:00"),
            pd.Timestamp("2020-01-06 08:00"),
            pd.Timestamp("2020-01-13 08:00"),
        ],
        end_timestamp=[
            pd.Timestamp("2020-01-06 08:10"),
            pd.Timestamp("2020-01-06 08:10"),
            pd.Timestamp("2020-01-13 08:10"),
        ],
    )
    result = mean_area_volume(df)
    row = result[(result["area"] == "A") & (result["time_bin"] == "08:00")]
    assert row["mean_volume"].iloc[0] == pytest.approx(1.5 / 7.0)


def test_full_week_single_user():
    # u1 visits A 08:00-08:10 on each of Mon-Sun; sum=7, mean=7/7=1.0
    dates = pd.date_range("2020-01-06", periods=7, freq="D")
    df = _df(
        area=["A"] * 7,
        user_id=["u1"] * 7,
        start_timestamp=list(dates.map(lambda d: d.replace(hour=8, minute=0))),
        end_timestamp=list(dates.map(lambda d: d.replace(hour=8, minute=10))),
    )
    result = mean_area_volume(df)
    row = result[(result["area"] == "A") & (result["time_bin"] == "08:00")]
    assert row["mean_volume"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bin expansion
# ---------------------------------------------------------------------------


def test_multi_bin_expansion():
    # start=08:05 → floor 08:00; end=08:35 → floor 08:30 → 4 bins
    df = _df(
        area=["A"],
        user_id=["u1"],
        start_timestamp=[pd.Timestamp("2020-01-06 08:05")],
        end_timestamp=[pd.Timestamp("2020-01-06 08:35")],
    )
    result = mean_area_volume(df)
    area_rows = result[result["area"] == "A"]
    bins_found = set(area_rows["time_bin"])
    assert bins_found == {"08:00", "08:10", "08:20", "08:30"}
    for mv in area_rows["mean_volume"]:
        assert mv == pytest.approx(1.0 / 7.0)


def test_midnight_spanning_visit():
    # start=2020-01-06 23:50 (Mon), end=2020-01-07 00:20 (Tue)
    # floor(23:50)=23:50, floor(00:20)=00:20
    # bins: 23:50 on Jan6 (Mon), 00:00/00:10/00:20 on Jan7 (Tue)
    df = _df(
        area=["A"],
        user_id=["u1"],
        start_timestamp=[pd.Timestamp("2020-01-06 23:50")],
        end_timestamp=[pd.Timestamp("2020-01-07 00:20")],
    )
    result = mean_area_volume(df)
    area_rows = result[result["area"] == "A"]
    bins_found = set(area_rows["time_bin"])
    assert "23:50" in bins_found
    assert "00:00" in bins_found
    assert "00:10" in bins_found
    assert "00:20" in bins_found
    assert len(area_rows) == 4
    for mv in area_rows["mean_volume"]:
        assert mv == pytest.approx(1.0 / 7.0)


def test_zero_duration_single_bin():
    # start == end → floor both to same bin → exactly 1 bin
    df = _df(
        area=["A"],
        user_id=["u1"],
        start_timestamp=[pd.Timestamp("2020-01-06 09:15")],
        end_timestamp=[pd.Timestamp("2020-01-06 09:15")],
    )
    result = mean_area_volume(df)
    area_rows = result[result["area"] == "A"]
    assert len(area_rows) == 1
    assert area_rows["time_bin"].iloc[0] == "09:10"
    assert area_rows["mean_volume"].iloc[0] == pytest.approx(1.0 / 7.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dataframe():
    df = pd.DataFrame(
        columns=["area", "user_id", "start_timestamp", "end_timestamp"]
    )
    result = mean_area_volume(df)
    assert list(result.columns) == ["area", "time_bin", "mean_volume"]
    assert len(result) == 0


def test_two_areas_independent():
    df = _df(
        area=["A", "B"],
        user_id=["u1", "u2"],
        start_timestamp=[
            pd.Timestamp("2020-01-06 08:00"),  # Monday
            pd.Timestamp("2020-01-08 09:00"),  # Wednesday
        ],
        end_timestamp=[
            pd.Timestamp("2020-01-06 08:10"),
            pd.Timestamp("2020-01-08 09:10"),
        ],
    )
    result = mean_area_volume(df)
    row_a = result[(result["area"] == "A") & (result["time_bin"] == "08:00")]
    row_b = result[(result["area"] == "B") & (result["time_bin"] == "09:00")]
    assert row_a["mean_volume"].iloc[0] == pytest.approx(1.0 / 7.0)
    assert row_b["mean_volume"].iloc[0] == pytest.approx(1.0 / 7.0)
    # No cross-contamination: A only has "08:00"/"08:10", B only has "09:00"/"09:10"
    assert set(result[result["area"] == "A"]["time_bin"]) == {"08:00", "08:10"}
    assert set(result[result["area"] == "B"]["time_bin"]) == {"09:00", "09:10"}


def test_mean_volume_always_positive():
    df = _df(
        area=["A", "A"],
        user_id=["u1", "u2"],
        start_timestamp=[pd.Timestamp("2020-01-06 08:00")] * 2,
        end_timestamp=[pd.Timestamp("2020-01-06 08:20")] * 2,
    )
    result = mean_area_volume(df)
    assert (result["mean_volume"] > 0).all()


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------


def test_column_autodetection_variants():
    df = _df(
        venueId=["A"],
        uid=["u1"],
        timestamp=[pd.Timestamp("2020-01-06 08:00")],
        end_timestamp=[pd.Timestamp("2020-01-06 08:10")],
    )
    result = mean_area_volume(df)
    assert "mean_volume" in result.columns
    # start=08:00, end=08:10 → floor both → 2 bins (08:00, 08:10)
    assert len(result) == 2


def test_output_columns_fixed_regardless_of_input():
    df = _df(
        location_id=["X"],
        agent_id=["u99"],
        start_timestamp=[pd.Timestamp("2020-01-06 10:00")],
        end_timestamp=[pd.Timestamp("2020-01-06 10:10")],
    )
    result = mean_area_volume(df)
    assert set(result.columns) == {"area", "time_bin", "mean_volume"}


def test_missing_area_column_raises():
    df = _df(
        no_area_here=["X"],
        user_id=["u1"],
        start_timestamp=[pd.Timestamp("2020-01-06 08:00")],
        end_timestamp=[pd.Timestamp("2020-01-06 08:10")],
    )
    with pytest.raises(ValueError, match="area"):
        mean_area_volume(df)


# ---------------------------------------------------------------------------
# Backend invariance
# ---------------------------------------------------------------------------


def test_polars_backend():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "area": ["A", "A"],
            "user_id": ["u1", "u2"],
            "start_timestamp": [
                pd.Timestamp("2020-01-06 08:00"),
                pd.Timestamp("2020-01-06 08:00"),
            ],
            "end_timestamp": [
                pd.Timestamp("2020-01-06 08:10"),
                pd.Timestamp("2020-01-06 08:10"),
            ],
        }
    )
    result = mean_area_volume(df)
    assert isinstance(result, pl.DataFrame)
    row = result.filter(
        (pl.col("area") == "A") & (pl.col("time_bin") == "08:00")
    )
    assert row["mean_volume"][0] == pytest.approx(2.0 / 7.0)
