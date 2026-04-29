"""Correctness tests for skmob2/measures/jump_lengths.py."""

from __future__ import annotations

import numpy as np
import pytest

# Pre-computed expected jump lengths — keep in sync with conftest.EXPECTED_JUMP_LENGTHS.
EXPECTED_JUMP_LENGTHS: dict[str, list[float]] = {
    "user_a": [
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
    ],
    "user_b": [
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
    ],
    "user_c": [
        0.6845085181899847,
        0.9188177472926384,
        0.9187595626478804,
        0.9187013756985495,
    ],
}


def _normalize_result(df) -> dict:
    """Return {uid: np.array(jump_lengths)} from a jump_lengths result."""
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    for col in ("uid", "user", "user_id"):
        if col in df.columns:
            uid_col = col
            break
    else:
        raise AssertionError("Result lacks uid/user/user_id column")

    normalized = {}
    for _, row in df.iterrows():
        normalized[row[uid_col]] = np.asarray(row["jump_lengths"], dtype=float)
    return normalized


def test_jump_lengths_known_values(synthetic_tdf):
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.spatial.jump_lengths import jump_lengths

    result = jump_lengths(synthetic_tdf, merge=False)
    normalized = _normalize_result(result)

    assert set(normalized.keys()) == set(EXPECTED_JUMP_LENGTHS.keys()), (
        f"UID mismatch: got {set(normalized.keys())}, expected {set(EXPECTED_JUMP_LENGTHS.keys())}"
    )

    for uid, expected in EXPECTED_JUMP_LENGTHS.items():
        actual = normalized[uid]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"jump_lengths mismatch for uid={uid!r}",
        )


def test_jump_lengths_merge_returns_flat_list(synthetic_tdf):
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.spatial.jump_lengths import jump_lengths

    result = jump_lengths(synthetic_tdf, merge=True)
    assert isinstance(result, list), "merge=True should return a plain list"
    # 3 users × 4 jumps each = 12
    assert len(result) == 12


def test_jump_lengths_no_uid_column():
    """When no uid column is present the whole frame is treated as one user."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    import pandas as pd
    from skmob2.measures.spatial.jump_lengths import jump_lengths

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lng": [0.0, 1.0, 2.0, 3.0],
        }
    )
    result = jump_lengths(df, merge=False)
    assert "jump_lengths" in result.columns
    assert len(result) == 1  # single row, no uid column


def test_jump_lengths_polars_known_values(synthetic_tdf_polars):
    """Test skmob2.jump_lengths on Polars DataFrame with known values."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.spatial.jump_lengths import jump_lengths

    result = jump_lengths(synthetic_tdf_polars, merge=False)
    normalized = _normalize_result(result)

    assert set(normalized.keys()) == set(EXPECTED_JUMP_LENGTHS.keys()), (
        f"UID mismatch: got {set(normalized.keys())}, expected {set(EXPECTED_JUMP_LENGTHS.keys())}"
    )

    for uid, expected in EXPECTED_JUMP_LENGTHS.items():
        actual = normalized[uid]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"jump_lengths mismatch for uid={uid!r} (Polars)",
        )


def test_jump_lengths_polars_no_uid_column():
    """Test that Polars DataFrames work without uid column."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    import pandas as pd
    from skmob2.measures.spatial.jump_lengths import jump_lengths

    df_pandas = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lng": [0.0, 1.0, 2.0, 3.0],
        }
    )
    df_polars = polars.from_pandas(df_pandas)
    result = jump_lengths(df_polars, merge=False)
    assert "jump_lengths" in result.columns
    assert len(result) == 1  # single row, no uid column


def test_jump_lengths_polars_vs_pandas_skmob2():
    """Verify that Polars and pandas produce identical skmob2 results."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    import pandas as pd
    from skmob2.measures.spatial.jump_lengths import jump_lengths as skmob2_jl

    df_pandas = pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 2],
            "datetime": pd.date_range("2020-01-01", periods=6, freq="h"),
            "lat": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "lng": [0.0, 0.0, 0.0, 20.0, 20.0, 20.0],
        }
    )
    df_polars = polars.from_pandas(df_pandas)

    result_pandas = skmob2_jl(df_pandas, merge=False)
    result_polars = skmob2_jl(df_polars, merge=False)
    result_polars_pd = result_polars.to_pandas()

    pandas_norm = _normalize_result(result_pandas)
    polars_norm = _normalize_result(result_polars_pd)

    assert set(pandas_norm.keys()) == set(polars_norm.keys())
    for uid in pandas_norm:
        left = pandas_norm[uid]
        right = polars_norm[uid]
        assert left.shape == right.shape, f"Shape mismatch for uid={uid}"
        assert np.allclose(left, right, rtol=1e-10, atol=1e-10), f"Polars and pandas should be identical for uid={uid}"


@pytest.mark.skmob
def test_jump_lengths_matches_skmob(brightkite_skmob):
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    import pandas as pd
    from skmob.measures.individual import jump_lengths as skmob_jl
    from skmob2.measures.spatial.jump_lengths import jump_lengths as skmob2_jl

    skmob_result = skmob_jl(brightkite_skmob, show_progress=False, merge=False)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_jl(skmob2_input, merge=False)

    baseline = _normalize_result(skmob_result)
    candidate = _normalize_result(skmob2_result)

    assert set(baseline.keys()) == set(candidate.keys())
    for uid in baseline:
        left = baseline[uid]
        right = candidate[uid]
        assert left.shape == right.shape, f"Shape mismatch for uid={uid}"
        assert np.allclose(left, right, rtol=1e-5, atol=1e-5), f"Value mismatch for uid={uid}"
