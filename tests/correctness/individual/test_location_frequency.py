"""Correctness tests for fkmob/measures/visits/location_frequency.py."""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_freq_dict(df) -> dict:
    """Convert a location_frequency DataFrame to {uid: {(lat, lng): freq}}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    lat_col = next((c for c in ("lat", "latitude") if c in columns), None)
    lng_col = next((c for c in ("lng", "lon", "longitude") if c in columns), None)

    result: dict = {}
    for row in nw_df.rows(named=True):
        uid = row[uid_col] if uid_col else "__single__"
        loc = (row[lat_col], row[lng_col])
        result.setdefault(uid, {})[loc] = row["location_frequency"]
    return result


def _arrow_list(array) -> list:
    if hasattr(array, "to_pylist"):
        return array.to_pylist()
    return array.tolist()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_location_frequency_known_values(synthetic_tdf):
    """Each user visits 5 distinct locations once: raw counts are all 1."""
    from fkmob.measures.individual.location_frequency import location_frequency

    result = location_frequency(synthetic_tdf, normalize=False)
    freq_dict = _to_freq_dict(result)

    assert set(freq_dict.keys()) == {"user_a", "user_b", "user_c"}
    for uid, loc_freqs in freq_dict.items():
        assert len(loc_freqs) == 5
        assert all(v == 1.0 for v in loc_freqs.values())


def test_location_frequency_normalized(synthetic_tdf):
    """With normalize=True each user's frequencies sum to 1.0."""
    from fkmob.measures.individual.location_frequency import location_frequency

    result = location_frequency(synthetic_tdf, normalize=True)
    freq_dict = _to_freq_dict(result)

    for uid, loc_freqs in freq_dict.items():
        total = sum(loc_freqs.values())
        assert abs(total - 1.0) < 1e-9, f"uid={uid}: sum={total}"


def test_location_frequency_repeated_visits():
    """Locations visited multiple times accumulate correctly."""
    from fkmob.measures.individual.location_frequency import location_frequency

    # User "a" visits loc A 3 times and loc B once.
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df, normalize=False)
    freq_dict = _to_freq_dict(result)

    assert freq_dict["a"][(1.0, 0.0)] == 3.0
    assert freq_dict["a"][(2.0, 0.0)] == 1.0


def test_location_frequency_normalized_repeated():
    """Normalized frequencies sum to 1 when visits are unequal."""
    from fkmob.measures.individual.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df, normalize=True)
    freq_dict = _to_freq_dict(result)

    assert abs(freq_dict["a"][(1.0, 0.0)] - 0.75) < 1e-9
    assert abs(freq_dict["a"][(2.0, 0.0)] - 0.25) < 1e-9


def test_location_frequency_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from fkmob.measures.individual.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df, normalize=False)
    nw_result = nw.from_native(result, eager_only=True)

    assert "location_frequency" in nw_result.columns
    # 2 distinct locations.
    assert len(nw_result) == 2
    rows = {(row["lat"], row["lng"]): row["location_frequency"] for row in nw_result.rows(named=True)}
    assert rows[(1.0, 0.0)] == 2.0
    assert rows[(2.0, 0.0)] == 1.0


def test_location_frequency_sorted_descending():
    """Output is sorted by frequency descending within each user."""
    from fkmob.measures.individual.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [1.0, 2.0, 2.0, 3.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df, normalize=False)
    nw_result = nw.from_native(result, eager_only=True)
    freqs = nw_result.filter(nw.col("uid") == "a").get_column("location_frequency").to_list()
    # Frequencies must be non-increasing.
    assert freqs == sorted(freqs, reverse=True)


def test_location_frequency_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same frequency values as pandas."""
    from fkmob.measures.individual.location_frequency import location_frequency

    result = location_frequency(synthetic_tdf_polars, normalize=False)
    freq_dict = _to_freq_dict(result)

    assert set(freq_dict.keys()) == {"user_a", "user_b", "user_c"}
    for uid, loc_freqs in freq_dict.items():
        assert len(loc_freqs) == 5
        assert all(v == 1.0 for v in loc_freqs.values())


def test_location_frequency_default_is_normalized():
    """Default normalize=True matches skmob's default; probabilities sum to 1."""
    from fkmob.measures.individual.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df)  # normalize=True by default
    freq_dict = _to_freq_dict(result)

    total = sum(freq_dict["a"].values())
    assert abs(total - 1.0) < 1e-9


def test_location_frequency_as_ranks_returns_list():
    """as_ranks=True returns a Python list of mean per-rank frequencies."""
    from fkmob.measures.individual.location_frequency import location_frequency

    # User "a": 3 visits to loc A (rank 0) and 1 to loc B (rank 1).
    # User "b": 2 visits to loc C (rank 0) and 2 to loc D (rank 1).
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "datetime": pd.date_range("2020-01-01", periods=8, freq="h"),
            "lat": [1.0, 1.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    ranks = location_frequency(df, as_ranks=True)

    assert isinstance(ranks, list)
    # Both users have 2 distinct locations → 2 ranks.
    assert len(ranks) == 2
    # Rank-0 mean: user "a" has prob 0.75, user "b" has prob 0.5 → mean = 0.625
    assert abs(ranks[0] - 0.625) < 1e-9
    # Rank-1 mean: user "a" has prob 0.25, user "b" has prob 0.5 → mean = 0.375
    assert abs(ranks[1] - 0.375) < 1e-9


def test_location_frequency_presorted_matches_default_pandas():
    """The presorted fast path matches indexed grouping for grouped input."""
    from fkmob.measures.individual.location_frequency import location_frequency

    raw = pd.DataFrame(
        {
            "uid": ["b", "a", "a", "b", "a", "b"],
            "datetime": pd.date_range("2020-01-01", periods=6, freq="h"),
            "lat": [3.0, 1.0, 2.0, 3.0, 1.0, 4.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    sorted_input = raw.sort_values(["uid", "datetime"], kind="mergesort")

    assert _to_freq_dict(location_frequency(raw, normalize=False)) == _to_freq_dict(
        location_frequency(sorted_input, normalize=False, presorted=True)
    )
    assert _to_freq_dict(location_frequency(raw, normalize=True)) == _to_freq_dict(
        location_frequency(sorted_input, normalize=True, presorted=True)
    )


def test_location_frequency_presorted_as_ranks_matches_default():
    """Rank-mean output is computed natively for presorted grouped input."""
    from fkmob.measures.individual.location_frequency import location_frequency

    raw = pd.DataFrame(
        {
            "uid": ["b", "a", "a", "a", "b", "b", "a", "b"],
            "datetime": pd.date_range("2020-01-01", periods=8, freq="h"),
            "lat": [3.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 4.0],
            "lng": [0.0] * 8,
        }
    )
    sorted_input = raw.sort_values(["uid", "datetime"], kind="mergesort")

    np.testing.assert_allclose(
        location_frequency(sorted_input, as_ranks=True, presorted=True),
        location_frequency(raw, as_ranks=True),
        rtol=1e-12,
        atol=1e-12,
    )


def test_location_frequency_presorted_no_uid():
    """Presorted works when the whole frame is one implicit user."""
    from fkmob.measures.individual.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = location_frequency(df, normalize=False, presorted=True)
    rows = {
        (row["lat"], row["lng"]): row["location_frequency"]
        for row in nw.from_native(result, eager_only=True).rows(named=True)
    }
    assert rows == {(1.0, 0.0): 3.0, (2.0, 0.0): 1.0}


def test_location_frequency_presorted_polars_known_values():
    """Polars/Arrow input uses the presorted Arrow-backed fast path."""
    pl = pytest.importorskip("polars")
    from fkmob.measures.individual.location_frequency import location_frequency

    df = pl.DataFrame(
        {
            "uid": ["a", "a", "b", "b"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 2.0, 3.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )

    assert _to_freq_dict(location_frequency(df, normalize=False, presorted=True)) == {
        "a": {(1.0, 0.0): 2.0},
        "b": {(2.0, 0.0): 1.0, (3.0, 0.0): 1.0},
    }


def test_location_frequency_presorted_core_validation_errors():
    """The native presorted helper validates monotonic end offsets."""
    from fkmob._core import location_frequency_presorted_numpy

    arr = np.array([1.0, 2.0], dtype=np.float64)
    bad_ends = np.array([2, 1], dtype=np.uintp)
    with pytest.raises(ValueError, match="monotonically"):
        location_frequency_presorted_numpy(arr, arr, bad_ends, True)


def test_location_frequency_indexed_arrow_nulls_are_filtered():
    """Indexed Arrow helper skips null coordinates without materializing a mask."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fkmob._core import location_frequency_values_indexed_arrow

    lats = pa.array([1.0, None, 1.0, 2.0], type=pa.float64())
    lngs = pa.array([0.0, 0.0, None, 0.0], type=pa.float64())
    indices = np.array([0, 1, 2, 3], dtype=np.uintp)
    ends = np.array([4], dtype=np.uintp)

    out_lats, out_lngs, values, *_ = location_frequency_values_indexed_arrow(
        lats, lngs, indices, ends, False
    )

    rows = dict(zip(zip(_arrow_list(out_lats), _arrow_list(out_lngs)), _arrow_list(values)))
    assert rows == {(1.0, 0.0): 1.0, (2.0, 0.0): 1.0}


def test_location_frequency_presorted_arrow_nulls_are_filtered():
    """Presorted Arrow helper skips null coordinates without materializing a mask."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fkmob._core import location_frequency_presorted_arrow

    lats = pa.array([1.0, None, 1.0, 2.0], type=pa.float64())
    lngs = pa.array([0.0, 0.0, None, 0.0], type=pa.float64())
    ends = np.array([4], dtype=np.uintp)

    out_lats, out_lngs, values, *_ = location_frequency_presorted_arrow(
        lats, lngs, ends, False
    )

    rows = dict(zip(zip(_arrow_list(out_lats), _arrow_list(out_lngs)), _arrow_list(values)))
    assert rows == {(1.0, 0.0): 1.0, (2.0, 0.0): 1.0}


def test_location_frequency_indexed_arrow_sliced_null_bitmap_offsets():
    """Sliced Arrow validity bitmaps are interpreted in logical row coordinates."""
    pytest.importorskip("fkmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fkmob._core import location_frequency_values_indexed_arrow

    lats = pa.array([99.0, 1.0, None, 1.0, 2.0, 88.0], type=pa.float64()).slice(1, 4)
    lngs = pa.array([99.0, 0.0, 0.0, None, 0.0, 88.0], type=pa.float64()).slice(1, 4)
    indices = np.array([0, 1, 2, 3], dtype=np.uintp)
    ends = np.array([4], dtype=np.uintp)

    out_lats, out_lngs, values, *_ = location_frequency_values_indexed_arrow(
        lats, lngs, indices, ends, False
    )

    rows = dict(zip(zip(_arrow_list(out_lats), _arrow_list(out_lngs)), _arrow_list(values)))
    assert rows == {(1.0, 0.0): 1.0, (2.0, 0.0): 1.0}


@pytest.mark.skmob
def test_location_frequency_matches_skmob(comparison_skmob):
    """fkmob counts match skmob on each comparison dataset."""
    import pandas as pd
    from skmob.measures.individual import location_frequency as skmob_lf
    from fkmob.measures.individual.location_frequency import location_frequency as fkmob_lf

    skmob_result = skmob_lf(comparison_skmob, show_progress=False)
    fkmob_input = pd.DataFrame(comparison_skmob).copy()
    fkmob_result = fkmob_lf(fkmob_input)

    # Build comparable dicts: {uid: {(lat, lng): count}}.
    skmob_dict: dict = {}
    skmob_result = skmob_result.reset_index()
    for _, row in skmob_result.iterrows():
        uid = row["uid"]
        loc = (row["lat"], row["lng"])
        skmob_dict.setdefault(uid, {})[loc] = float(row["location_frequency"])

    fkmob_dict = _to_freq_dict(fkmob_result)

    common_uids = set(skmob_dict) & set(fkmob_dict)
    assert len(common_uids) > 0
    for uid in common_uids:
        common_locs = set(skmob_dict[uid]) & set(fkmob_dict[uid])
        for loc in common_locs:
            assert abs(skmob_dict[uid][loc] - fkmob_dict[uid][loc]) < 1e-5, (
                f"uid={uid}, loc={loc}: skmob={skmob_dict[uid][loc]}, fkmob={fkmob_dict[uid][loc]}"
            )


def test_location_frequency_matches_cached_reference(comparison_skmob_reference):
    """location_frequency matches the cached skmob baseline without requiring the skmob environment."""
    from fkmob.measures.individual.location_frequency import location_frequency as fkmob_lf

    ref = comparison_skmob_reference
    skmob_result = ref.result("location_frequency")
    fkmob_result = fkmob_lf(ref.input_df)

    # skmob 1.3.1 returns a broken structure for datasets with string UIDs
    # (missing uid and location_frequency columns); verify fkmob runs correctly and return.
    if "uid" not in skmob_result.columns or "location_frequency" not in skmob_result.columns:
        assert len(fkmob_result) > 0
        return

    skmob_dict: dict = {}
    for _, row in skmob_result.iterrows():
        uid = row["uid"]
        loc = (row["lat"], row["lng"])
        skmob_dict.setdefault(uid, {})[loc] = float(row["location_frequency"])

    fkmob_dict = _to_freq_dict(fkmob_result)

    common_uids = set(skmob_dict) & set(fkmob_dict)
    assert len(common_uids) > 0
    for uid in common_uids:
        common_locs = set(skmob_dict[uid]) & set(fkmob_dict[uid])
        for loc in common_locs:
            assert abs(skmob_dict[uid][loc] - fkmob_dict[uid][loc]) < 1e-5, (
                f"uid={uid}, loc={loc}: cached={skmob_dict[uid][loc]}, fkmob={fkmob_dict[uid][loc]}"
            )
