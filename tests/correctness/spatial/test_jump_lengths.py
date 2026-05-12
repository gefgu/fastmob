"""Correctness tests for skmob2/measures/jump_lengths.py."""

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
    try:
        return values.to_numpy(zero_copy_only=False)
    except TypeError:
        return values.to_numpy()


def _grouped_from_offsets(starts, ends, values) -> list[np.ndarray]:
    values = np.asarray(values, dtype=float)
    return [values[int(start) : int(end)] for start, end in zip(starts, ends)]


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


def test_jump_lengths_merge_returns_numpy_array(synthetic_tdf):
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.spatial.jump_lengths import jump_lengths

    result = jump_lengths(synthetic_tdf, merge=True)
    assert isinstance(result, np.ndarray), "pandas merge=True should return a NumPy array"
    assert result.dtype == np.float64
    # 3 users × 4 jumps each = 12
    assert len(result) == 12


def test_jump_lengths_sorts_by_user_time_without_full_dataframe_ordering():
    """Interleaved input rows are ordered chronologically within each user."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    import pandas as pd
    from skmob2._core import jump_lengths_km
    from skmob2.measures.spatial.jump_lengths import jump_lengths

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
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    from skmob2._core import jump_lengths_km
    from skmob2.measures.spatial.jump_lengths import jump_lengths

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
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    import pandas as pd
    from skmob2._core import jump_lengths_km
    from skmob2.measures.spatial.jump_lengths import jump_lengths

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


def test_jump_lengths_no_uid_column_sorts_by_time():
    """Without uid, the whole trajectory is still ordered by timestamp."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    import pandas as pd
    from skmob2._core import jump_lengths_km
    from skmob2.measures.spatial.jump_lengths import jump_lengths

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


def test_jump_lengths_polars_merge_returns_arrow_array(synthetic_tdf_polars):
    """Polars-backed merge=True keeps the flat result on the Arrow path."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2.measures.spatial.jump_lengths import jump_lengths

    result = jump_lengths(synthetic_tdf_polars, merge=True)
    assert isinstance(result, pa.Array)
    assert result.type == pa.float64()
    assert len(result) == 12


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


def test_jump_lengths_numpy_helper_groups_by_ranges():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    from skmob2._core import jump_lengths_flat_numpy, jump_lengths_km, jump_lengths_numpy

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ranges = [(0, 3), (3, 5)]

    value_starts, value_ends, values = jump_lengths_numpy(lats, lngs, ranges)
    grouped = _grouped_from_offsets(value_starts, value_ends, values)
    expected = [jump_lengths_km(lats[start:end].tolist(), lngs[start:end].tolist()) for start, end in ranges]

    assert len(grouped) == 2
    for actual, expected_values in zip(grouped, expected):
        np.testing.assert_allclose(actual, expected_values, rtol=0.0, atol=1e-12)
    flat = jump_lengths_flat_numpy(lats, lngs, ranges)
    assert isinstance(flat, np.ndarray)
    assert flat.dtype == np.float64
    np.testing.assert_allclose(flat, expected[0] + expected[1], rtol=0.0, atol=1e-12)


def test_jump_lengths_arrow_helper_matches_numpy_helper():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import jump_lengths_arrow, jump_lengths_flat_arrow, jump_lengths_flat_numpy, jump_lengths_numpy

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ranges = [(0, 3), (3, 5)]

    arrow_starts, arrow_ends, arrow_values = jump_lengths_arrow(pl.Series(lats).to_arrow(), pl.Series(lngs).to_arrow(), ranges)
    numpy_starts, numpy_ends, numpy_values = jump_lengths_numpy(lats, lngs, ranges)
    result_arrow = _grouped_from_offsets(arrow_starts, arrow_ends, _arrow_to_numpy(arrow_values))
    result_numpy = _grouped_from_offsets(numpy_starts, numpy_ends, numpy_values)
    flat_arrow = jump_lengths_flat_arrow(pl.Series(lats).to_arrow(), pl.Series(lngs).to_arrow(), ranges)
    flat_numpy = jump_lengths_flat_numpy(lats, lngs, ranges)

    for actual, expected in zip(result_arrow, result_numpy):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert hasattr(flat_arrow, "__arrow_c_array__")
    np.testing.assert_allclose(_arrow_to_numpy(flat_arrow), flat_numpy, rtol=0.0, atol=1e-12)


def test_jump_lengths_time_ordered_numpy_helper_groups_and_sorts_by_time():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    from skmob2._core import jump_lengths_km, jump_lengths_time_ordered_numpy

    uids = np.array([2, 1, 2, 1, 1, 2], dtype=np.int64)
    timestamps = np.array([2.0, 2.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    lats = np.array([10.0, 0.0, 10.0, 0.0, 0.0, 10.0], dtype=np.float64)
    lngs = np.array([4.0, 3.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64)

    indices, starts, ends, value_starts, value_ends, values = jump_lengths_time_ordered_numpy(uids, timestamps, lats, lngs)

    assert isinstance(indices, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert indices.tolist() == [3, 4, 1, 2, 5, 0]
    assert starts.tolist() == [0, 3]
    assert ends.tolist() == [3, 6]
    expected = [
        jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]),
        jump_lengths_km([10.0, 10.0, 10.0], [0.0, 2.0, 4.0]),
    ]
    grouped = _grouped_from_offsets(value_starts, value_ends, values)
    for actual, expected_values in zip(grouped, expected):
        np.testing.assert_allclose(actual, expected_values, rtol=0.0, atol=1e-12)


def test_jump_lengths_time_ordered_flat_numpy_helper():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    from skmob2._core import jump_lengths_km, jump_lengths_time_ordered_flat_numpy

    uids = np.array([2, 1, 2, 1], dtype=np.int64)
    timestamps = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    lats = np.array([10.0, 0.0, 10.0, 0.0], dtype=np.float64)
    lngs = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float64)

    indices, starts, ends, flat = jump_lengths_time_ordered_flat_numpy(uids, timestamps, lats, lngs)

    assert isinstance(indices, np.ndarray)
    assert isinstance(flat, np.ndarray)
    assert flat.dtype == np.float64
    assert indices.tolist() == [3, 1, 2, 0]
    assert starts.tolist() == [0, 2]
    assert ends.tolist() == [2, 4]
    expected = jump_lengths_km([0.0, 0.0], [0.0, 1.0]) + jump_lengths_km([10.0, 10.0], [0.0, 2.0])
    np.testing.assert_allclose(flat, expected, rtol=0.0, atol=1e-12)


def test_jump_lengths_time_ordered_single_numpy_helper():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    from skmob2._core import jump_lengths_km, jump_lengths_time_ordered_single_numpy

    timestamps = np.array([2.0, 0.0, 1.0], dtype=np.float64)
    lats = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    lngs = np.array([3.0, 0.0, 1.0], dtype=np.float64)

    value_starts, value_ends, values = jump_lengths_time_ordered_single_numpy(timestamps, lats, lngs)
    (result,) = _grouped_from_offsets(value_starts, value_ends, values)
    np.testing.assert_allclose(result, jump_lengths_km([0.0, 0.0, 0.0], [0.0, 1.0, 3.0]), rtol=0.0, atol=1e-12)


def test_jump_lengths_time_ordered_arrow_helper_supports_strings():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import jump_lengths_time_ordered_arrow

    indices, starts, ends, value_starts, value_ends, values = jump_lengths_time_ordered_arrow(
        pa.array(["b", "a", "b", "a"]),
        pa.array([1.0, 1.0, 0.0, 0.0]),
        pa.array([10.0, 0.0, 10.0, 0.0]),
        pa.array([2.0, 1.0, 0.0, 0.0]),
    )

    assert isinstance(indices, np.ndarray)
    assert indices.tolist() == [3, 1, 2, 0]
    assert starts.tolist() == [0, 2]
    assert ends.tolist() == [2, 4]
    grouped = _grouped_from_offsets(value_starts, value_ends, _arrow_to_numpy(values))
    assert len(grouped) == 2


def test_jump_lengths_time_ordered_flat_arrow_helper_supports_strings():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import jump_lengths_km, jump_lengths_time_ordered_flat_arrow

    indices, starts, ends, flat = jump_lengths_time_ordered_flat_arrow(
        pa.array(["b", "a", "b", "a"]),
        pa.array([1.0, 1.0, 0.0, 0.0]),
        pa.array([10.0, 0.0, 10.0, 0.0]),
        pa.array([2.0, 1.0, 0.0, 0.0]),
    )

    assert isinstance(indices, np.ndarray)
    assert hasattr(flat, "__arrow_c_array__")
    assert indices.tolist() == [3, 1, 2, 0]
    assert starts.tolist() == [0, 2]
    assert ends.tolist() == [2, 4]
    expected = jump_lengths_km([0.0, 0.0], [0.0, 1.0]) + jump_lengths_km([10.0, 10.0], [0.0, 2.0])
    np.testing.assert_allclose(_arrow_to_numpy(flat), expected, rtol=0.0, atol=1e-12)


def test_jump_lengths_numpy_helper_validation_errors():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    from skmob2._core import jump_lengths_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        jump_lengths_numpy(arr, arr[:1], [(0, 1)])
    with pytest.raises(ValueError, match="range end"):
        jump_lengths_numpy(arr, arr, [(0, 3)])


def test_jump_lengths_time_ordered_numpy_helper_validation_errors():
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    from skmob2._core import jump_lengths_time_ordered_single_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        jump_lengths_time_ordered_single_numpy(arr[:1], arr, arr)


@pytest.mark.skmob
def test_jump_lengths_matches_skmob(comparison_skmob):
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    import pandas as pd
    from skmob.measures.individual import jump_lengths as skmob_jl
    from skmob2.measures.spatial.jump_lengths import jump_lengths as skmob2_jl

    skmob_result = skmob_jl(comparison_skmob, show_progress=False, merge=False)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
    skmob2_result = skmob2_jl(skmob2_input, merge=False)

    baseline = _normalize_result(skmob_result)
    candidate = _normalize_result(skmob2_result)

    assert set(baseline.keys()) == set(candidate.keys())
    for uid in baseline:
        left = baseline[uid]
        right = candidate[uid]
        assert left.shape == right.shape, f"Shape mismatch for uid={uid}"
        # skmob uses its Python gislib Haversine implementation; skmob2 uses
        # Rust geo::Haversine. Individual long jumps can differ by metres.
        assert np.allclose(left, right, rtol=2e-5, atol=1e-2), f"Value mismatch for uid={uid}"


def test_jump_lengths_matches_cached_reference(comparison_skmob_reference):
    """jump_lengths matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first (maturin develop)")
    from skmob2.measures.spatial.jump_lengths import jump_lengths as skmob2_jl

    ref = comparison_skmob_reference
    skmob_result = ref.result("jump_lengths")
    skmob2_result = skmob2_jl(ref.input_df, merge=False)

    baseline = _normalize_result(skmob_result)
    candidate = _normalize_result(skmob2_result)

    assert set(baseline.keys()) == set(candidate.keys())
    for uid in baseline:
        left = baseline[uid]
        right = candidate[uid]
        assert left.shape == right.shape, f"Shape mismatch for uid={uid}"
        assert np.allclose(left, right, rtol=2e-5, atol=1e-2), f"Value mismatch for uid={uid}"
