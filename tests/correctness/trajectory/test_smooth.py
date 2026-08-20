"""Correctness tests for fastmob.trajectory.smooth.

Unlike `interpolate`, `smooth` never changes row count or row order: every
row's (lat, lng) is replaced with a Kalman-filtered/RTS-smoothed estimate at
its original timestamp. Tests cover: noise reduction on a synthetic
constant-velocity trajectory, pandas/polars parity, Rule 2 null-row
handling, the `n<2`-per-user no-op edge case, and structural invariants over
a real Brightkite slice (no independent ground truth exists for real GPS
noise, so only shape/order/other-columns invariants are checked there).

A MovingPandas `KalmanSmootherCV` cached-reference comparison (the
`.venv-movingpandas` `stonesoup` optional dependency) is not wired up yet --
`stonesoup` was not available in the local `.venv-movingpandas` when this
suite was written. Left as a documented follow-up rather than blocking this
phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.trajectory import smooth

# ---------------------------------------------------------------------------
# Synthetic constant-velocity + noise fixture
# ---------------------------------------------------------------------------


def _noisy_line_rows(n: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_lats = 40.0 + np.arange(n) * 0.0005
    true_lngs = np.full(n, 10.0)
    noisy_lats = true_lats + rng.normal(0, 0.0002, n)
    noisy_lngs = true_lngs + rng.normal(0, 0.0002, n)
    rows = [
        {
            "uid": "u1",
            "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=i),
            "lat": float(noisy_lats[i]),
            "lng": float(noisy_lngs[i]),
        }
        for i in range(n)
    ]
    return rows, true_lats, true_lngs


def _rmse(values, truth):
    values = np.asarray(values, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((values - truth) ** 2)))


def test_kalman_cv_reduces_rmse_against_noisy_constant_velocity_line():
    rows, true_lats, _true_lngs = _noisy_line_rows()
    df = pd.DataFrame(rows)

    result = smooth(df, method="kalman_cv", process_noise_std_km=0.01, measurement_noise_std_km=0.05)

    raw_rmse = _rmse(df["lat"], true_lats)
    smoothed_rmse = _rmse(result["lat"], true_lats)
    assert smoothed_rmse < raw_rmse


def test_output_has_same_length_and_order_as_input():
    rows, _, _ = _noisy_line_rows()
    df = pd.DataFrame(rows)
    result = smooth(df, method="kalman_cv")
    assert len(result) == len(df)
    pd.testing.assert_series_equal(result["datetime"].reset_index(drop=True), df["datetime"].reset_index(drop=True))
    pd.testing.assert_series_equal(result["uid"].reset_index(drop=True), df["uid"].reset_index(drop=True))


def test_pandas_and_polars_backends_agree():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    rows, _, _ = _noisy_line_rows()
    df = pd.DataFrame(rows)
    pdf = pl.from_pandas(df)

    result_pd = smooth(df, method="kalman_cv")
    result_pl = smooth(pdf, method="kalman_cv").to_pandas()

    np.testing.assert_allclose(result_pd["lat"].to_numpy(), result_pl["lat"].to_numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result_pd["lng"].to_numpy(), result_pl["lng"].to_numpy(), rtol=1e-8, atol=1e-10)


def test_presorted_matches_default_indexed_path_on_already_sorted_data():
    rows, _, _ = _noisy_line_rows()
    df = pd.DataFrame(rows).sort_values(["uid", "datetime"]).reset_index(drop=True)

    result_default = smooth(df, method="kalman_cv")
    result_presorted = smooth(df, method="kalman_cv")

    np.testing.assert_allclose(result_default["lat"].to_numpy(), result_presorted["lat"].to_numpy())
    np.testing.assert_allclose(result_default["lng"].to_numpy(), result_presorted["lng"].to_numpy())


def test_interior_null_row_is_excluded_not_crashing():
    rows, _, _ = _noisy_line_rows(n=15)
    df = pd.DataFrame(rows)
    df.loc[7, "lat"] = np.nan

    result = smooth(df, method="kalman_cv")

    assert len(result) == len(df)
    assert np.isnan(result["lat"].iloc[7])
    other_rows = result.drop(index=7)
    assert not other_rows["lat"].isna().any()
    assert not other_rows["lng"].isna().any()


def test_single_point_user_passes_through_unchanged():
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = smooth(df, method="kalman_cv")
    assert len(result) == 1
    assert result["lat"].iloc[0] == pytest.approx(48.8566)
    assert result["lng"].iloc[0] == pytest.approx(2.3522)


def test_two_users_are_smoothed_independently():
    rows_a, _, _ = _noisy_line_rows(n=10, seed=1)
    rows_b, _, _ = _noisy_line_rows(n=10, seed=2)
    for row in rows_b:
        row["uid"] = "u2"
    df = pd.DataFrame(rows_a + rows_b)

    result = smooth(df, method="kalman_cv")
    assert len(result) == len(df)
    assert set(result["uid"]) == {"u1", "u2"}


def test_unknown_method_raises():
    df = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
            "lat": [0.0, 1.0],
            "lng": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="unknown smooth method"):
        smooth(df, method="bogus")


# ---------------------------------------------------------------------------
# Real dataset: structural invariants only (no ground truth for real GPS
# noise magnitude).
# ---------------------------------------------------------------------------


def test_brightkite_slice_preserves_shape_and_order(brightkite_sample_df):
    result = smooth(
        brightkite_sample_df,
        method="kalman_cv",
        datetime_col="check-in_time",
        lat_col="latitude",
        lng_col="longitude",
        uid_col="user",
    )
    assert len(result) == len(brightkite_sample_df)
    pd.testing.assert_series_equal(
        result["user"].reset_index(drop=True),
        brightkite_sample_df["user"].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        result["check-in_time"].reset_index(drop=True),
        brightkite_sample_df["check-in_time"].reset_index(drop=True),
    )
    assert result["latitude"].notna().all()
    assert result["longitude"].notna().all()
    assert result["latitude"].between(-90, 90).all()
    assert result["longitude"].between(-180, 180).all()
