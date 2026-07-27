"""Correctness tests for fastmob/measures/spatial/k_radius_of_gyration.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Pre-computed expected values for the synthetic 3-user fixture (5 pts each),
# computed via the Rust kernel directly.
EXPECTED_K2: dict[str, float] = {
    "user_a": 55.59754011676645,
    "user_b": 55.59754011676645,
    "user_c": 0.3422542593102871,
}

EXPECTED_K3: dict[str, float] = {
    "user_a": 90.79040282666378,
    "user_b": 90.79040282666378,
    "user_c": 0.6566571388235487,
}


def test_k_radius_of_gyration_known_values_k2_pandas(synthetic_tdf):
    """k-RoG (k=2) on synthetic fixture (pandas backend), hardcoded expected values."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration

    result = k_radius_of_gyration(synthetic_tdf, k=2)
    assert "k_radius_of_gyration" in result.columns

    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)

    for uid, expected in EXPECTED_K2.items():
        row = result[result[uid_col] == uid]
        assert len(row) == 1, f"Expected exactly one row for uid={uid!r}"
        actual = float(row["k_radius_of_gyration"].iloc[0])
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"k-RoG (k=2) mismatch for uid={uid!r}",
        )


def test_k_radius_of_gyration_known_values_k3_pandas(synthetic_tdf):
    """k-RoG (k=3) on synthetic fixture (pandas backend), hardcoded expected values."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration

    result = k_radius_of_gyration(synthetic_tdf, k=3)
    assert "k_radius_of_gyration" in result.columns

    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)

    for uid, expected in EXPECTED_K3.items():
        row = result[result[uid_col] == uid]
        assert len(row) == 1, f"Expected exactly one row for uid={uid!r}"
        actual = float(row["k_radius_of_gyration"].iloc[0])
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"k-RoG (k=3) mismatch for uid={uid!r}",
        )


def test_k_radius_of_gyration_no_uid_column():
    """When uid column is absent the result has a single k-RoG value."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration

    # loc (0,0) visited twice, (1,0) once, (2,0) once => top-2 are (0,0) and (1,0)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [0.0, 1.0, 0.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = k_radius_of_gyration(df, k=2)
    assert "k_radius_of_gyration" in result.columns
    assert len(result) == 1
    # Expected value computed directly via the Rust kernel
    np.testing.assert_allclose(
        float(result["k_radius_of_gyration"].iloc[0]),
        52.4178635118089,
        rtol=1e-5,
        atol=1e-5,
    )


def test_k_radius_of_gyration_k_exceeds_locations():
    """When k >= number of distinct locations, result equals regular RoG for those locations."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration

    # Only 2 distinct locations; k=10 should yield the same as using all locations
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 5,
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [0.0, 1.0, 0.0, 1.0, 0.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result_k10 = k_radius_of_gyration(df, k=10)
    uid_col = next(c for c in ("uid", "user", "user_id") if c in result_k10.columns)
    val_k10 = float(result_k10[result_k10[uid_col] == "u1"]["k_radius_of_gyration"].iloc[0])

    # k=2 equals k=10 when there are only 2 distinct locations
    result_k2 = k_radius_of_gyration(df, k=2)
    val_k2 = float(result_k2[result_k2[uid_col] == "u1"]["k_radius_of_gyration"].iloc[0])

    np.testing.assert_allclose(val_k10, val_k2, rtol=1e-10, atol=1e-10)


def test_k_radius_of_gyration_top_k_ties_use_first_datetime():
    """Equal-count top-k ties match skmob's first-visit datetime ordering."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob._core import k_radius_of_gyration_km
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration

    df = pd.DataFrame(
        {
            "uid": ["u1"] * 6,
            "datetime": [
                pd.Timestamp("2020-01-01 03:00"),
                pd.Timestamp("2020-01-01 01:00"),
                pd.Timestamp("2020-01-01 02:00"),
                pd.Timestamp("2020-01-01 04:00"),
                pd.Timestamp("2020-01-01 05:00"),
                pd.Timestamp("2020-01-01 06:00"),
            ],
            "lat": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            "lng": [0.0] * 6,
        }
    )
    result = k_radius_of_gyration(df, k=2)
    value = float(result["k_radius_of_gyration"].iloc[0])
    expected = k_radius_of_gyration_km([(10.0, 0.0), (20.0, 0.0)], [2, 2], 2)
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)


def test_k_radius_of_gyration_single_location():
    """A user who visits only one location should have k-RoG = 0."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration

    df = pd.DataFrame(
        {
            "uid": ["u1"] * 3,
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [10.0, 10.0, 10.0],
            "lng": [50.0, 50.0, 50.0],
        }
    )
    result = k_radius_of_gyration(df, k=2)
    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)
    val = float(result[result[uid_col] == "u1"]["k_radius_of_gyration"].iloc[0])
    assert val == pytest.approx(0.0, abs=1e-10), "Single-location user must have k-RoG = 0"


def test_k_radius_of_gyration_pandas_polars_agree(synthetic_tdf, synthetic_tdf_polars):
    """Polars and pandas backends must produce identical k-RoG values."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    pytest.importorskip("polars", reason="Polars not installed")
    pytest.importorskip("pyarrow", reason="pyarrow required for polars.to_pandas()")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration

    res_pd = k_radius_of_gyration(synthetic_tdf, k=2)
    res_pl = k_radius_of_gyration(synthetic_tdf_polars, k=2).to_pandas()

    uid_col = next(c for c in ("uid", "user", "user_id") if c in res_pd.columns)
    for uid in ("user_a", "user_b", "user_c"):
        val_pd = float(res_pd[res_pd[uid_col] == uid]["k_radius_of_gyration"].iloc[0])
        val_pl = float(res_pl[res_pl[uid_col] == uid]["k_radius_of_gyration"].iloc[0])
        np.testing.assert_allclose(
            val_pd,
            val_pl,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"pandas/polars disagree for uid={uid}",
        )


@pytest.mark.skmob
def test_k_radius_of_gyration_matches_skmob(comparison_skmob):
    """fastmob k-RoG must agree with skmob's reference implementation within rtol=1e-5."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration as fastmob_krg
    from skmob.measures.individual import k_radius_of_gyration as skmob_krg

    k = 2
    skmob_result = skmob_krg(comparison_skmob, k=k, show_progress=False)
    fastmob_input = pd.DataFrame(comparison_skmob).copy()
    fastmob_result = fastmob_krg(fastmob_input, k=k)

    skmob_uid = next(c for c in ("uid", "user", "user_id") if c in skmob_result.columns)
    fastmob_uid = next(c for c in ("uid", "user", "user_id") if c in fastmob_result.columns)
    skmob_value_col = next(c for c in (f"{k}k_radius_of_gyration", "k_radius_of_gyration") if c in skmob_result.columns)
    fastmob_value_col = next(
        c for c in ("k_radius_of_gyration", f"{k}k_radius_of_gyration") if c in fastmob_result.columns
    )

    skmob_map = dict(zip(skmob_result[skmob_uid], skmob_result[skmob_value_col]))
    fastmob_map = dict(zip(fastmob_result[fastmob_uid], fastmob_result[fastmob_value_col]))

    assert set(skmob_map.keys()) == set(fastmob_map.keys()), "User sets differ"
    for uid, skmob_value in skmob_map.items():
        np.testing.assert_allclose(
            fastmob_map[uid],
            skmob_value,
            rtol=1e-5,
            atol=1e-12,
            err_msg=f"k-RoG mismatch for uid={uid}",
        )


def test_k_radius_of_gyration_matches_cached_reference(comparison_skmob_reference):
    """k-RoG (k=2) matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.k_radius_of_gyration import k_radius_of_gyration as fastmob_krg

    k = 2
    ref = comparison_skmob_reference
    skmob_result = ref.result("k_radius_of_gyration_k2")
    fastmob_result = fastmob_krg(ref.input_df, k=k)

    skmob_uid = next(c for c in ("uid", "user", "user_id") if c in skmob_result.columns)
    fastmob_uid = next(c for c in ("uid", "user", "user_id") if c in fastmob_result.columns)
    skmob_value_col = next(c for c in (f"{k}k_radius_of_gyration", "k_radius_of_gyration") if c in skmob_result.columns)
    fastmob_value_col = next(
        c for c in ("k_radius_of_gyration", f"{k}k_radius_of_gyration") if c in fastmob_result.columns
    )

    skmob_map = dict(zip(skmob_result[skmob_uid], skmob_result[skmob_value_col]))
    fastmob_map = dict(zip(fastmob_result[fastmob_uid], fastmob_result[fastmob_value_col]))

    assert set(skmob_map.keys()) == set(fastmob_map.keys()), "User sets differ"
    for uid, skmob_value in skmob_map.items():
        np.testing.assert_allclose(
            fastmob_map[uid],
            skmob_value,
            rtol=1e-5,
            atol=1e-12,
            err_msg=f"k-RoG mismatch for uid={uid}",
        )
