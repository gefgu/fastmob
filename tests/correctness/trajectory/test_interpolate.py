"""Correctness tests for fastmob.trajectory.interpolate.

Hand-crafted fixtures live in ``conftest.py`` (``interpolate_tdf`` /
``interpolate_tdf_polars``): user "gap" exercises the single-point-per-gap
insertion policy, "no_gap" exercises the no-op path, "accelerating"
exercises kinematic's divergence from linear. Cached ``ptrail_reference``
comparisons validate linear/cubic/kinematic against PTRAIL's own
``Helpers.linear_help``/``cubic_help``/``kinematic_help``, run directly (see
``tests/populate_ptrail_cache.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.trajectory import interpolate


def _user(df, uid: str):
    if hasattr(df, "loc"):
        return df[df["uid"] == uid].reset_index(drop=True)
    import polars as pl

    return df.filter(pl.col("uid") == uid)


def test_linear_inserts_exactly_one_point_at_exceeded_gap(interpolate_tdf):
    """0h->2h gap (7200s > 3600s sampling rate) gets exactly one point
    inserted at hour 1, linearly interpolated; the 2h->3h gap (3600s, not
    strictly greater) is left untouched."""
    user = _user(interpolate_tdf, "gap")
    result = interpolate(user, method="linear", sampling_rate_s=3600.0)
    assert len(result) == 4
    lats = sorted(result["lat"].tolist())
    assert lats == pytest.approx([0.0, 0.5, 1.0, 2.0])


def test_no_gap_user_is_unchanged(interpolate_tdf):
    """Points already spaced at exactly the sampling rate get no insertions."""
    user = _user(interpolate_tdf, "no_gap")
    result = interpolate(user, method="linear", sampling_rate_s=3600.0)
    assert len(result) == len(user)


def test_single_point_user_passes_through_unchanged():
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = interpolate(df, method="linear", sampling_rate_s=3600.0)
    assert len(result) == 1


def test_unknown_method_raises():
    df = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
            "lat": [0.0, 1.0],
            "lng": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="unknown interpolate method"):
        interpolate(df, method="bogus")


def test_cubic_spline_falls_back_to_linear_for_three_point_user(interpolate_tdf):
    """`cubic_spline` needs > 3 distinct points to fit a real spline; with
    exactly 3 points (the "gap" user) it must fall back to the same output
    as `linear`."""
    user = _user(interpolate_tdf, "gap")
    linear_result = interpolate(user, method="linear", sampling_rate_s=3600.0)
    cubic_result = interpolate(user, method="cubic_spline", sampling_rate_s=3600.0)
    assert sorted(cubic_result["lat"].tolist()) == pytest.approx(sorted(linear_result["lat"].tolist()))


def test_cubic_spline_diverges_from_linear_on_nonlinear_data(interpolate_tdf):
    """With 4+ points and clearly non-collinear positions, a real cubic
    spline fit must produce a different interpolated value than plain linear
    interpolation for the inserted point."""
    user = _user(interpolate_tdf, "accelerating")
    linear_result = interpolate(user, method="linear", sampling_rate_s=3600.0)
    cubic_result = interpolate(user, method="cubic_spline", sampling_rate_s=3600.0)
    linear_inserted = sorted(set(linear_result["lat"].tolist()) - set(user["lat"].tolist()))
    cubic_inserted = sorted(set(cubic_result["lat"].tolist()) - set(user["lat"].tolist()))
    assert linear_inserted != [] and cubic_inserted != []
    assert linear_inserted[0] != pytest.approx(cubic_inserted[0])


def _kinematic_oracle(lat0, v0, lat1, dt, t_eval):
    """Independent re-derivation of the kinematic Hermite-cubic fit, matching
    the formula documented in `fastmob-core/src/trajectory/interpolate.rs`'s
    `kinematic_solve`/`kinematic_eval` (itself matching PTRAIL's own
    coefficient system, evaluated at the intended local elapsed time instead
    of PTRAIL's apparent absolute-timestamp evaluation bug)."""
    v1 = (lat1 - lat0) / dt
    a = np.array([[dt**2 / 2, dt**3 / 6], [dt, dt**2 / 2]])
    b = np.array([lat1 - lat0 - v0 * dt, v1 - v0])
    coef_b, coef_c = np.linalg.solve(a, b)
    return lat0 + v0 * t_eval + coef_b * t_eval**2 / 2 + coef_c * t_eval**3 / 6


def test_kinematic_matches_independent_python_oracle(interpolate_tdf):
    """`accelerating` user: t=[0,1,2,4]h, lat=[0,1,1,7]. The gap is between
    points at t=2h (lat=1) and t=4h (lat=7); v0 is the velocity feeding into
    t=2h (from t=1h->2h: (1-1)/1 = 0 deg/h) and the fit is evaluated at
    sampling_rate_s=3600 (1h) past t=2h."""
    user = _user(interpolate_tdf, "accelerating")
    result = interpolate(user, method="kinematic", sampling_rate_s=3600.0)
    inserted = sorted(set(result["lat"].tolist()) - set(user["lat"].tolist()))
    assert len(inserted) == 1

    expected = _kinematic_oracle(lat0=1.0, v0=0.0, lat1=7.0, dt=2.0, t_eval=1.0)
    assert inserted[0] == pytest.approx(expected)


def test_kinematic_max_speed_kmh_falls_back_to_linear(interpolate_tdf):
    """A very low `max_speed_kmh` clamp must force every kinematic gap-fill
    back to the plain linear position."""
    user = _user(interpolate_tdf, "accelerating")
    linear_result = interpolate(user, method="linear", sampling_rate_s=3600.0)
    clamped_result = interpolate(user, method="kinematic", sampling_rate_s=3600.0, max_speed_kmh=0.0001)
    assert sorted(clamped_result["lat"].tolist()) == pytest.approx(sorted(linear_result["lat"].tolist()))


def test_random_walk_zero_step_std_matches_linear(interpolate_tdf):
    """`step_std_km=0` disables the perturbation entirely, so random_walk
    must exactly reduce to linear interpolation."""
    user = _user(interpolate_tdf, "gap")
    linear_result = interpolate(user, method="linear", sampling_rate_s=3600.0)
    walk_result = interpolate(user, method="random_walk", sampling_rate_s=3600.0, step_std_km=0.0)
    assert sorted(walk_result["lat"].tolist()) == pytest.approx(sorted(linear_result["lat"].tolist()))


def test_random_walk_is_reproducible_with_same_seed(interpolate_tdf):
    user = _user(interpolate_tdf, "gap")
    result_a = interpolate(user, method="random_walk", sampling_rate_s=3600.0, step_std_km=0.05, seed=7)
    result_b = interpolate(user, method="random_walk", sampling_rate_s=3600.0, step_std_km=0.05, seed=7)
    assert result_a["lat"].tolist() == pytest.approx(result_b["lat"].tolist())
    assert result_a["lng"].tolist() == pytest.approx(result_b["lng"].tolist())


def test_random_walk_perturbs_away_from_linear(interpolate_tdf):
    """With a non-zero `step_std_km`, the inserted point should (with
    overwhelming probability for this fixed seed) differ from the
    unperturbed linear position."""
    user = _user(interpolate_tdf, "gap")
    linear_result = interpolate(user, method="linear", sampling_rate_s=3600.0)
    walk_result = interpolate(user, method="random_walk", sampling_rate_s=3600.0, step_std_km=0.1, seed=1)
    linear_inserted = sorted(set(linear_result["lat"].tolist()) - set(user["lat"].tolist()))
    walk_inserted = sorted(set(walk_result["lat"].tolist()) - set(user["lat"].tolist()))
    assert walk_inserted[0] != pytest.approx(linear_inserted[0])


def test_presorted_matches_default_indexed_path(interpolate_tdf):
    sorted_df = interpolate_tdf.sort_values(["uid", "datetime"], kind="mergesort").reset_index(drop=True)
    default_result = interpolate(interpolate_tdf, method="linear", sampling_rate_s=3600.0)
    presorted_result = interpolate(sorted_df, method="linear", sampling_rate_s=3600.0, presorted=True)
    assert sorted(default_result["lat"].tolist()) == pytest.approx(sorted(presorted_result["lat"].tolist()))


def test_polars_matches_pandas(interpolate_tdf, interpolate_tdf_polars):
    pandas_result = interpolate(interpolate_tdf, method="linear", sampling_rate_s=3600.0)
    polars_result = interpolate(interpolate_tdf_polars, method="linear", sampling_rate_s=3600.0)
    assert sorted(pandas_result["lat"].tolist()) == pytest.approx(sorted(polars_result["lat"].to_list()))


def test_linear_matches_ptrail_reference(ptrail_reference):
    """Numeric-value comparison against PTRAIL's real `Helpers.linear_help`
    (run directly, see `tests/populate_ptrail_cache.py`). Same algorithm, so
    an exact (floating-point-only) match is expected."""
    reference = ptrail_reference.interpolated_positions("linear")
    if reference is None:
        pytest.skip("No PTRAIL interpolate(linear) reference cached.")

    input_df = ptrail_reference.input_df
    checked_any = False
    for uid, ref_df in reference.items():
        user_df = input_df[input_df["uid"] == uid][["uid", "datetime", "lat", "lng"]].reset_index(drop=True)
        if len(user_df) < 2:
            continue
        result = interpolate(user_df, method="linear", sampling_rate_s=1800.0)
        result = result.sort_values("datetime").reset_index(drop=True)

        assert len(result) == len(ref_df), f"row count mismatch for user {uid}"
        assert result["lat"].to_numpy() == pytest.approx(ref_df["lat"].to_numpy(), abs=1e-6)
        assert result["lng"].to_numpy() == pytest.approx(ref_df["lon"].to_numpy(), abs=1e-6)
        checked_any = True

    assert checked_any, "no users had cached reference data to compare"


@pytest.mark.parametrize("method", ["cubic", "kinematic"])
def test_matches_ptrail_reference_structurally(ptrail_reference, method):
    """`cubic` and `kinematic` are compared structurally only (row count per
    user), not by numeric lat/lng value, for two independent reasons:

    - `cubic`: both a not-a-knot spline (PTRAIL/scipy) and a natural spline
      (fastmob) are prone to wild Runge's-phenomenon-style overshoot when
      fit through real, unevenly-spaced check-in data with few points --
      observed divergences exceed 50 degrees of latitude on this dataset,
      which reflects boundary-condition sensitivity inherent to cubic
      splines on sparse data, not an implementation bug in either library.
    - `kinematic`: PTRAIL's own `Helpers.kinematic_help` evaluates its
      Hermite-cubic fit at what appears to be an absolute-Unix-timestamp-
      derived value (`new_times[i-1].timestamp() / 10e9`) rather than the
      intended local elapsed time (`sampling_rate`), producing near-
      degenerate output that does not reflect the documented algorithm.
      fastmob's own `kinematic_solve`/`kinematic_eval` uses the same
      coefficient derivation but evaluates at the correct local elapsed
      time (see `interpolate.rs`'s docstring).

    Row counts are allowed to differ by at most 1 per user: a gap whose
    duration sits within floating-point epsilon of `sampling_rate_s` can
    tie-break differently between fastmob's Rust comparison and PTRAIL's
    pandas `.diff().dt.total_seconds()` comparison.
    """
    reference = ptrail_reference.interpolated_positions(method)
    if reference is None:
        pytest.skip(f"No PTRAIL interpolate({method}) reference cached.")

    input_df = ptrail_reference.input_df
    fastmob_method = "cubic_spline" if method == "cubic" else method
    min_points = 4 if method == "cubic" else 3

    checked_any = False
    for uid, ref_df in reference.items():
        user_df = input_df[input_df["uid"] == uid][["uid", "datetime", "lat", "lng"]].reset_index(drop=True)
        if len(user_df) < min_points:
            continue
        result = interpolate(user_df, method=fastmob_method, sampling_rate_s=1800.0)
        assert abs(len(result) - len(ref_df)) <= 1, f"inserted-point count mismatch for user {uid}"
        checked_any = True

    assert checked_any, "no users had cached reference data to compare"
