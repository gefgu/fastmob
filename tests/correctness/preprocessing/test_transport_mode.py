"""Correctness tests for fastmob.preprocessing.predict_transport_mode /
calculate_modal_split.

Uses a small, hand-built `Triplegs`-shaped table (rather than the full
Positionfixes->Staypoints->Triplegs pipeline) so speed values sit exactly on
and around each bin boundary -- deterministic, unambiguous ground truth.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fastmob.core.triplegs_dataframe import Triplegs
from fastmob.preprocessing import calculate_modal_split, predict_transport_mode


def _triplegs_df():
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1", "u2"],
            "tripleg_id": [1, 2, 3, 4],
            "started_at": pd.to_datetime(
                ["2020-01-06 08:00", "2020-01-06 09:00", "2020-01-06 10:00", "2020-01-06 08:00"]
            ),
            "finished_at": pd.to_datetime(
                ["2020-01-06 08:10", "2020-01-06 09:05", "2020-01-06 10:02", "2020-01-06 08:20"]
            ),
            # Exactly at/near each bin boundary: 10 (slow), 100 (motorized, boundary), 250 (fast)
            "length_km": [10.0 / 6.0, 100.0 / 12.0, 250.0 / 30.0, 5.0 / 6.0],
            "duration_s": [600.0, 300.0, 120.0, 1200.0],
            "mean_speed_kmh": [10.0, 100.0, 250.0, 5.0],
        }
    )


def test_predict_transport_mode_bins_by_default_thresholds():
    result = predict_transport_mode(_triplegs_df())
    assert result["mode"].tolist() == [
        "slow_mobility",  # 10 km/h <= 15
        "motorized_mobility",  # 100 km/h <= 100 (inclusive upper bound)
        "fast_mobility",  # 250 km/h > 100
        "slow_mobility",  # 5 km/h <= 15
    ]


def test_predict_transport_mode_accepts_triplegs_wrapper_and_returns_one():
    tl = Triplegs(_triplegs_df(), uid_col="uid")
    result = tl.predict_transport_mode()
    assert isinstance(result, Triplegs)
    assert result.df["mode"].tolist()[0] == "slow_mobility"


def test_predict_transport_mode_categories_fully_overridable():
    custom = {5.0: "walking", math.inf: "everything_else"}
    result = predict_transport_mode(_triplegs_df(), categories=custom)
    assert result["mode"].tolist() == ["everything_else", "everything_else", "everything_else", "walking"]


def test_predict_transport_mode_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown transport-mode method"):
        predict_transport_mode(_triplegs_df(), method="bogus")


def test_calculate_modal_split_requires_mode_column():
    with pytest.raises(ValueError, match="requires a 'mode' column"):
        calculate_modal_split(_triplegs_df())


def test_calculate_modal_split_count():
    labeled = predict_transport_mode(_triplegs_df())
    split = calculate_modal_split(labeled, metric="count")
    counts = dict(zip(split["mode"], split["value"]))
    assert counts == {"slow_mobility": 2, "motorized_mobility": 1, "fast_mobility": 1}


def test_calculate_modal_split_distance_and_duration():
    df = _triplegs_df()
    labeled = predict_transport_mode(df)

    by_distance = calculate_modal_split(labeled, metric="distance")
    distances = dict(zip(by_distance["mode"], by_distance["value"]))
    assert distances["slow_mobility"] == pytest.approx(10.0 / 6.0 + 5.0 / 6.0)

    by_duration = calculate_modal_split(labeled, metric="duration")
    durations = dict(zip(by_duration["mode"], by_duration["value"]))
    assert durations["slow_mobility"] == pytest.approx(600.0 + 1200.0)


def test_calculate_modal_split_per_user_requires_uid_column():
    labeled = predict_transport_mode(_triplegs_df().drop(columns=["uid"]))
    with pytest.raises(ValueError, match="per_user=True requires"):
        calculate_modal_split(labeled, per_user=True)


def test_calculate_modal_split_per_user():
    tl = Triplegs(predict_transport_mode(_triplegs_df()), uid_col="uid")
    split = tl.calculate_modal_split(per_user=True, metric="count")
    rows = {(r["uid"], r["mode"]): r["value"] for _, r in split.iterrows()}
    assert rows[("u1", "slow_mobility")] == 1
    assert rows[("u1", "motorized_mobility")] == 1
    assert rows[("u1", "fast_mobility")] == 1
    assert rows[("u2", "slow_mobility")] == 1


def test_calculate_modal_split_normalize_sums_to_one():
    labeled = predict_transport_mode(_triplegs_df())
    split = calculate_modal_split(labeled, metric="count", normalize=True)
    assert split["value"].sum() == pytest.approx(1.0)


def test_calculate_modal_split_unknown_metric_raises():
    labeled = predict_transport_mode(_triplegs_df())
    with pytest.raises(ValueError, match="unknown modal-split metric"):
        calculate_modal_split(labeled, metric="bogus")


def test_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = _triplegs_df()

    result_pd = predict_transport_mode(df)
    result_pl = predict_transport_mode(pl.from_pandas(df)).to_pandas()

    assert result_pl["mode"].tolist() == result_pd["mode"].tolist()
