"""Correctness tests for fastmob/measures/jump_lengths.py."""

from __future__ import annotations

import numpy as np
import pytest

from tests.correctness.conftest import EXPECTED_JUMP_LENGTHS


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


def _arrow_to_numpy(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    try:
        return values.to_numpy(zero_copy_only=False)
    except TypeError:
        return values.to_numpy()


def _grouped_from_offsets(starts, ends, values) -> list[np.ndarray]:
    values = np.asarray(values, dtype=float)
    return [values[int(start) : int(end)] for start, end in zip(starts, ends)]


def test_jump_lengths_known_values(synthetic_tdf):
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    from fastmob.measures.individual.jump_lengths import jump_lengths

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


def test_jump_lengths_merge_returns_numpy_array(synthetic_tdf):
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    from fastmob.measures.individual.jump_lengths import jump_lengths

    result = jump_lengths(synthetic_tdf, merge=True)
    assert isinstance(result, np.ndarray), "pandas merge=True should return a NumPy array"
    assert result.dtype == np.float64
    # 3 users × 4 jumps each = 12
    assert len(result) == 12


def test_jump_lengths_sorts_by_user_time_without_full_dataframe_ordering():
    """Interleaved input rows are ordered chronologically within each user."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    import pandas as pd
    from fastmob._core import jump_lengths_km
    from fastmob.measures.individual.jump_lengths import jump_lengths

    df = pd.DataFrame(
        {
            "uid": ["b", "a", "b", "a", "a", "b"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 02:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 01:00:00",
                ]
            ),
            "lat": [10.0, 0.0, 10.0, 0.0, 0.0, 10.0],
            "lng": [4.0, 3.0, 0.0, 0.0, 1.0, 2.0],
            "payload": ["large-string-column"] * 6,
        }
    )

    result = _normalize_result(jump_lengths(df, merge=False))

    np.testing.assert_allclose(
        result["a"],
        jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["b"],
        jump_lengths_km([10.0, 10.0, 10.0], [0.0, 2.0, 4.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_jump_lengths_polars_sorts_by_user_time_without_full_dataframe_ordering():
    """Polars input uses Rust-side user/time ordering."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    from fastmob._core import jump_lengths_km
    from fastmob.measures.individual.jump_lengths import jump_lengths

    df = polars.DataFrame(
        {
            "uid": ["b", "a", "b", "a", "a", "b"],
            "datetime": [
                "2020-01-01 02:00:00",
                "2020-01-01 02:00:00",
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 01:00:00",
            ],
            "lat": [10.0, 0.0, 10.0, 0.0, 0.0, 10.0],
            "lng": [4.0, 3.0, 0.0, 0.0, 1.0, 2.0],
            "payload": ["large-string-column"] * 6,
        }
    ).with_columns(polars.col("datetime").str.to_datetime())

    result = _normalize_result(jump_lengths(df, merge=False))

    np.testing.assert_allclose(
        result["a"],
        jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["b"],
        jump_lengths_km([10.0, 10.0, 10.0], [0.0, 2.0, 4.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_jump_lengths_equal_timestamps_keep_input_order():
    """Rows tied on timestamp keep their original post-cleaning order."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    import pandas as pd
    from fastmob._core import jump_lengths_km
    from fastmob.measures.individual.jump_lengths import jump_lengths

    df = pd.DataFrame(
        {
            "uid": [1, 1, 1],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00"]),
            "lat": [0.0, 0.0, 0.0],
            "lng": [0.0, 2.0, 3.0],
        }
    )

    result = _normalize_result(jump_lengths(df, merge=False))
    np.testing.assert_allclose(
        result[1],
        jump_lengths_km([0.0, 0.0, 0.0], [0.0, 2.0, 3.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_jump_lengths_no_uid_column():
    """When no uid column is present the whole frame is treated as one user."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    import pandas as pd
    from fastmob.measures.individual.jump_lengths import jump_lengths

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


def test_jump_lengths_no_uid_column_sorts_by_time():
    """Without uid, the whole trajectory is still ordered by timestamp."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    import pandas as pd
    from fastmob._core import jump_lengths_km
    from fastmob.measures.individual.jump_lengths import jump_lengths

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01 02:00", "2020-01-01 00:00", "2020-01-01 01:00"]),
            "lat": [0.0, 0.0, 0.0],
            "lng": [3.0, 0.0, 1.0],
        }
    )

    result = jump_lengths(df, merge=False)
    np.testing.assert_allclose(
        result["jump_lengths"].iloc[0],
        jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_jump_lengths_auto_dispatch_uses_existing_grouped_order():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    import pandas as pd
    from fastmob._core import jump_lengths_km
    from fastmob.measures.individual.jump_lengths import jump_lengths

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "b", "b", "b"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 00:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                ]
            ),
            "lat": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "lng": [0.0, 1.0, 3.0, 0.0, 2.0, 4.0],
        }
    )

    result = _normalize_result(jump_lengths(df, merge=False))
    np.testing.assert_allclose(result["a"], jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]))
    np.testing.assert_allclose(result["b"], jump_lengths_km([10.0, 10.0, 10.0], [0.0, 2.0, 4.0]))


def test_jump_lengths_single_user_auto_dispatch_matches_unsorted_input():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    import pandas as pd
    from fastmob.measures.individual.jump_lengths import jump_lengths

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lng": [0.0, 1.0, 2.0, 4.0],
        }
    )

    # Same rows, reversed: the sorted frame takes the contiguous path and the
    # shuffled one the indexed path, and both must agree.
    shuffled = df.iloc[::-1].reset_index(drop=True)
    sorted_result = jump_lengths(df, merge=False)
    normal_result = jump_lengths(shuffled, merge=False)
    np.testing.assert_allclose(
        sorted_result["jump_lengths"].iloc[0],
        normal_result["jump_lengths"].iloc[0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jump_lengths(df, merge=True),
        jump_lengths(shuffled, merge=True),
        rtol=0.0,
        atol=1e-12,
    )


def test_jump_lengths_polars_known_values(synthetic_tdf_polars):
    """Test fastmob.jump_lengths on Polars DataFrame with known values."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    from fastmob.measures.individual.jump_lengths import jump_lengths

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


def test_jump_lengths_polars_merge_returns_arrow_array(synthetic_tdf_polars):
    """Polars-backed merge=True keeps the flat result on the Arrow path."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob.measures.individual.jump_lengths import jump_lengths

    result = jump_lengths(synthetic_tdf_polars, merge=True)
    assert isinstance(result, pa.Array)
    assert result.type == pa.float64()
    assert len(result) == 12


def test_jump_lengths_polars_no_uid_column():
    """Test that Polars DataFrames work without uid column."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    import pandas as pd
    from fastmob.measures.individual.jump_lengths import jump_lengths

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


def test_jump_lengths_polars_vs_pandas_fastmob():
    """Verify that Polars and pandas produce identical fastmob results."""
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    import pandas as pd
    from fastmob.measures.individual.jump_lengths import jump_lengths as fastmob_jl

    df_pandas = pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 2],
            "datetime": pd.date_range("2020-01-01", periods=6, freq="h"),
            "lat": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "lng": [0.0, 0.0, 0.0, 20.0, 20.0, 20.0],
        }
    )
    df_polars = polars.from_pandas(df_pandas)

    result_pandas = fastmob_jl(df_pandas, merge=False)
    result_polars = fastmob_jl(df_polars, merge=False)
    result_polars_pd = result_polars.to_pandas()

    pandas_norm = _normalize_result(result_pandas)
    polars_norm = _normalize_result(result_polars_pd)

    assert set(pandas_norm.keys()) == set(polars_norm.keys())
    for uid in pandas_norm:
        left = pandas_norm[uid]
        right = polars_norm[uid]
        assert left.shape == right.shape, f"Shape mismatch for uid={uid}"
        assert np.allclose(left, right, rtol=1e-10, atol=1e-10), f"Polars and pandas should be identical for uid={uid}"


def test_jump_lengths_presorted_helper_groups_by_ranges():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    from fastmob._core import jump_lengths_km, jump_lengths_presorted

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ranges = [(0, 3), (3, 5)]
    ends = np.array([3, 5], dtype=np.uintp)

    value_starts, value_ends, values = jump_lengths_presorted(lats, lngs, ends)
    values = _arrow_to_numpy(values)
    grouped = _grouped_from_offsets(value_starts, value_ends, values)
    expected = [jump_lengths_km(lats[start:end].tolist(), lngs[start:end].tolist()) for start, end in ranges]

    assert len(grouped) == 2
    for actual, expected_values in zip(grouped, expected):
        np.testing.assert_allclose(actual, expected_values, rtol=0.0, atol=1e-12)
    assert values.dtype == np.float64
    np.testing.assert_allclose(values, expected[0] + expected[1], rtol=0.0, atol=1e-12)


def test_jump_lengths_presorted_helper_accepts_arrow():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import jump_lengths_presorted

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ends = np.array([3, 5], dtype=np.uintp)

    arrow_starts, arrow_ends, arrow_values = jump_lengths_presorted(
        pl.Series(lats).to_arrow(), pl.Series(lngs).to_arrow(), ends
    )
    numpy_starts, numpy_ends, numpy_values = jump_lengths_presorted(lats, lngs, ends)
    numpy_values = _arrow_to_numpy(numpy_values)
    result_arrow = _grouped_from_offsets(arrow_starts, arrow_ends, _arrow_to_numpy(arrow_values))
    result_numpy = _grouped_from_offsets(numpy_starts, numpy_ends, numpy_values)

    for actual, expected in zip(result_arrow, result_numpy):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert hasattr(arrow_values, "__arrow_c_array__")
    np.testing.assert_allclose(_arrow_to_numpy(arrow_values), numpy_values, rtol=0.0, atol=1e-12)


def test_jump_lengths_indexed_helper_groups_by_offsets():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    from fastmob._core import jump_lengths_indexed, jump_lengths_km

    lats = np.array([10.0, 0.0, 10.0, 0.0, 0.0, 10.0], dtype=np.float64)
    lngs = np.array([4.0, 3.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64)
    indices = np.array([3, 4, 1, 2, 5, 0], dtype=np.uintp)
    ends = np.array([3, 6], dtype=np.uintp)

    starts, value_ends, values = jump_lengths_indexed(lats, lngs, indices, ends)
    values = _arrow_to_numpy(values)

    assert starts.tolist() == [0, 2]
    assert value_ends.tolist() == [2, 4]
    grouped = _grouped_from_offsets(starts, value_ends, values)
    expected = [
        jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]),
        jump_lengths_km([10.0, 10.0, 10.0], [0.0, 2.0, 4.0]),
    ]
    for actual, expected_values in zip(grouped, expected):
        np.testing.assert_allclose(actual, expected_values, rtol=0.0, atol=1e-12)


def test_jump_lengths_indexed_helper_accepts_arrow():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import jump_lengths_indexed

    lats = np.array([10.0, 0.0, 10.0, 0.0, 0.0, 10.0], dtype=np.float64)
    lngs = np.array([4.0, 3.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64)
    indices = np.array([3, 4, 1, 2, 5, 0], dtype=np.uintp)
    ends = np.array([3, 6], dtype=np.uintp)

    arrow_starts, arrow_ends, arrow_values = jump_lengths_indexed(pa.array(lats), pa.array(lngs), indices, ends)
    numpy_starts, numpy_ends, numpy_values = jump_lengths_indexed(lats, lngs, indices, ends)

    np.testing.assert_array_equal(arrow_starts, numpy_starts)
    np.testing.assert_array_equal(arrow_ends, numpy_ends)
    np.testing.assert_allclose(_arrow_to_numpy(arrow_values), _arrow_to_numpy(numpy_values), rtol=0.0, atol=1e-12)


def test_jump_lengths_indexed_helper_handles_empty_and_single_point_groups():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    from fastmob._core import jump_lengths_indexed

    lats = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    indices = np.array([0, 1, 2], dtype=np.uintp)
    ends = np.array([0, 1, 3], dtype=np.uintp)

    starts, value_ends, values = jump_lengths_indexed(lats, lngs, indices, ends)

    assert starts.tolist() == [0, 0, 0]
    assert value_ends.tolist() == [0, 0, 1]
    assert len(values) == 1


def test_jump_lengths_indexed_helper_filters_null_and_non_finite_coords():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import jump_lengths_indexed, jump_lengths_km

    lats = pa.array([0.0, None, 0.0, float("nan"), 0.0])
    lngs = pa.array([0.0, 5.0, 1.0, 7.0, 3.0])
    indices = np.array([0, 1, 2, 3, 4], dtype=np.uintp)
    ends = np.array([5], dtype=np.uintp)

    starts, value_ends, values = jump_lengths_indexed(lats, lngs, indices, ends)

    assert starts.tolist() == [0]
    assert value_ends.tolist() == [2]
    np.testing.assert_allclose(
        _arrow_to_numpy(values),
        jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_jump_lengths_presorted_helper_validation_errors():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    from fastmob._core import jump_lengths_presorted

    arr = np.array([0.0, 1.0], dtype=np.float64)
    ends = np.array([1], dtype=np.uintp)
    with pytest.raises(ValueError, match="same length"):
        jump_lengths_presorted(arr, arr[:1], ends)
    with pytest.raises(ValueError, match="range end"):
        jump_lengths_presorted(arr, arr, np.array([3], dtype=np.uintp))
    with pytest.raises(ValueError, match="monotonically"):
        jump_lengths_presorted(arr, arr, np.array([2, 1], dtype=np.uintp))


def test_jump_lengths_indexed_helper_validation_errors():
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    from fastmob._core import jump_lengths_indexed

    arr = np.array([0.0, 1.0], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.uintp)
    ends = np.array([2], dtype=np.uintp)
    with pytest.raises(ValueError, match="same length"):
        jump_lengths_indexed(arr, arr[:1], indices, ends)
    with pytest.raises(ValueError, match="monotonically"):
        jump_lengths_indexed(arr, arr, indices, np.array([2, 1], dtype=np.uintp))
    with pytest.raises(ValueError, match="index must be within"):
        jump_lengths_indexed(arr, arr, np.array([0, 2], dtype=np.uintp), ends)
    with pytest.raises(ValueError, match="range end"):
        jump_lengths_indexed(arr, arr, indices, np.array([3], dtype=np.uintp))


@pytest.mark.skmob
def test_jump_lengths_matches_skmob(comparison_skmob):
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )
    import pandas as pd
    from fastmob.measures.individual.jump_lengths import jump_lengths as fastmob_jl
    from skmob.measures.individual import jump_lengths as skmob_jl

    skmob_result = skmob_jl(comparison_skmob, show_progress=False, merge=False)
    fastmob_input = pd.DataFrame(comparison_skmob).copy()
    fastmob_result = fastmob_jl(fastmob_input, merge=False)

    baseline = _normalize_result(skmob_result)
    candidate = _normalize_result(fastmob_result)

    assert set(baseline.keys()) == set(candidate.keys())
    for uid in baseline:
        left = baseline[uid]
        right = candidate[uid]
        assert left.shape == right.shape, f"Shape mismatch for uid={uid}"
        # skmob uses its Python gislib Haversine implementation; fastmob uses
        # Rust geo::Haversine. Individual long jumps can differ by metres.
        assert np.allclose(left, right, rtol=2e-5, atol=1e-2), f"Value mismatch for uid={uid}"


def test_jump_lengths_matches_cached_reference(comparison_skmob_reference):
    """jump_lengths matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("fastmob._core", reason="Build the fastmob extension first (maturin develop)")
    from fastmob.measures.individual.jump_lengths import jump_lengths as fastmob_jl

    ref = comparison_skmob_reference
    skmob_result = ref.result("jump_lengths")
    fastmob_result = fastmob_jl(ref.input_df, merge=False)

    baseline = _normalize_result(skmob_result)
    candidate = _normalize_result(fastmob_result)

    assert set(baseline.keys()) == set(candidate.keys())
    for uid in baseline:
        left = baseline[uid]
        right = candidate[uid]
        assert left.shape == right.shape, f"Shape mismatch for uid={uid}"
        assert np.allclose(left, right, rtol=2e-5, atol=1e-2), f"Value mismatch for uid={uid}"
