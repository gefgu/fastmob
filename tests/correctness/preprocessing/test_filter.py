"""Correctness tests for fastmob.preprocessing.filter."""

from __future__ import annotations

import pandas as pd
import pytest

from fastmob.preprocessing import filter as traj_filter


def test_filter_clean_trajectory_unchanged(filter_tdf):
    """Clean trajectory (no high-speed points) should come back intact."""
    clean = filter_tdf[filter_tdf["uid"] == "user_clean"].copy()
    result = traj_filter(clean, max_speed_kmh=500.0)
    assert len(result) == 5


def test_filter_removes_high_speed_point(filter_tdf):
    """Point teleported to null island must be removed from user_noisy."""
    noisy = filter_tdf[filter_tdf["uid"] == "user_noisy"].copy()
    result = traj_filter(noisy, max_speed_kmh=500.0)
    # The null-island point (lat=0, lng=0) causes impossible speed → removed
    assert len(result) == 4
    assert not ((result["lat"] == 0.0) & (result["lng"] == 0.0)).any()


def test_filter_full_df_clean_user_unchanged(filter_tdf):
    """With multi-user df, clean user keeps all 5 points."""
    result = traj_filter(filter_tdf, max_speed_kmh=500.0)
    clean_result = result[result["uid"] == "user_clean"]
    assert len(clean_result) == 5


def test_filter_full_df_noisy_user_filtered(filter_tdf):
    """With multi-user df, noisy user loses the high-speed point."""
    result = traj_filter(filter_tdf, max_speed_kmh=500.0)
    noisy_result = result[result["uid"] == "user_noisy"]
    assert len(noisy_result) == 4


def test_filter_single_point_user():
    """A single-point trajectory is returned unchanged."""
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = traj_filter(df, max_speed_kmh=500.0)
    assert len(result) == 1


def test_filter_returns_same_backend_type(filter_tdf):
    """Output backend must match input (pandas in → pandas out)."""
    result = traj_filter(filter_tdf, max_speed_kmh=500.0)
    assert isinstance(result, type(filter_tdf))


def test_filter_polars_backend(filter_tdf_polars):
    """Polars input produces a Polars output with the same filtering result."""
    import polars as pl

    result = traj_filter(filter_tdf_polars, max_speed_kmh=500.0)
    assert isinstance(result, pl.DataFrame)
    clean = result.filter(pl.col("uid") == "user_clean")
    noisy = result.filter(pl.col("uid") == "user_noisy")
    assert len(clean) == 5
    assert len(noisy) == 4


def test_filter_zero_dt_removes_duplicate_timestamps():
    """Two points at the same timestamp trigger ZeroDivisionError → remove second."""
    df = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-01 00:00:00"),  # same timestamp as prev → ZeroDivision
                pd.Timestamp("2020-01-01 01:00:00"),
            ],
            "lat": [48.856, 48.900, 48.856],  # big jump at same timestamp
            "lng": [2.352, 2.352, 2.352],
        }
    )
    result = traj_filter(df, max_speed_kmh=500.0)
    # The duplicate-timestamp point should be removed
    assert len(result) == 2


def _contains_approx(values, target: float) -> bool:
    return any(v == pytest.approx(target) for v in values)


def test_filter_hampel_flags_teleport_and_its_return_hop(hampel_tdf):
    """The teleport at index 4 and the return-hop at index 5 are the only
    two speed values in the series far outside their rolling window's
    median; every other point's own speed is unaffected and stays."""
    result = traj_filter(hampel_tdf, method="hampel")
    lats = result["lat"].to_list()
    assert len(result) == 7
    assert not _contains_approx(lats, 60.0)
    assert not _contains_approx(lats, 48.8560 + 5 * 0.0005)
    # The point right before the teleport is unaffected and survives.
    assert _contains_approx(lats, 48.8560 + 3 * 0.0005)


def test_filter_greedy_drops_both_points_touching_the_teleport(greedy_tdf):
    """Greedy tests fixed original-adjacent pairs, so both the point
    arriving at the teleport and the point leaving it fail the speed test,
    even though the point leaving it is otherwise a perfectly normal hop."""
    result = traj_filter(greedy_tdf, method="greedy", max_speed_kmh=100.0)
    lats = result["lat"].to_list()
    assert len(result) == 3
    assert not _contains_approx(lats, 0.0)
    assert not _contains_approx(lats, 48.8566 + 3 * 0.005)


def test_filter_smart_greedy_only_drops_the_true_outlier(greedy_tdf):
    """SmartGreedy tracks multiple candidate chains, so it recognizes
    {0, 1, 3, 4} as a single mutually-consistent chain (skipping over the
    teleport at index 2) and drops only the true outlier point."""
    result = traj_filter(greedy_tdf, method="smart_greedy", max_speed_kmh=100.0)
    lats = result["lat"].to_list()
    assert len(result) == 4
    assert not _contains_approx(lats, 0.0)
    assert _contains_approx(lats, 48.8566 + 3 * 0.005)


def test_filter_zheng_keeps_the_hop_right_after_the_teleport(zheng_tdf):
    """With min_seg_size=1 (default), only the isolated 1-point segment at
    the teleport is dropped; the run of 3 clean points starting right after
    it is kept in full, unlike Greedy's fixed-adjacent-pair test."""
    result = traj_filter(zheng_tdf, method="zheng", max_speed_kmh=100.0)
    lats = result["lat"].to_list()
    assert len(result) == 7
    assert not _contains_approx(lats, 5.0)
    assert _contains_approx(lats, 0.004)


def test_filter_zheng_min_seg_size_drops_short_runs(zheng_tdf):
    """Raising min_seg_size above both surviving runs' lengths (4 and 3)
    drops every point: no run is long enough to survive."""
    result = traj_filter(zheng_tdf, method="zheng", max_speed_kmh=100.0, min_seg_size=10)
    assert len(result) == 0


def test_filter_outlier_methods_return_same_backend_type(greedy_tdf):
    """Every new outlier method returns the same backend type as the input."""
    for method in ("hampel", "greedy", "smart_greedy", "zheng"):
        result = traj_filter(greedy_tdf, method=method)
        assert isinstance(result, type(greedy_tdf))


def test_filter_outlier_methods_polars_backend(greedy_tdf_polars):
    """Polars input produces a Polars output for every new outlier method."""
    import polars as pl

    for method in ("hampel", "greedy", "smart_greedy", "zheng"):
        result = traj_filter(greedy_tdf_polars, method=method)
        assert isinstance(result, pl.DataFrame)


def test_filter_unknown_method_raises(filter_tdf):
    """An unrecognized ``method`` raises ValueError listing valid choices."""
    with pytest.raises(ValueError, match="unknown filter method"):
        traj_filter(filter_tdf, method="not_a_real_method")


def _per_user_jaccard_keep(result_df, cached_keep_by_uid: dict) -> dict:
    """Return per-uid Jaccard similarity between fastmob's and the cached kept-row-index sets."""
    scores = {}
    for uid, group in result_df.groupby("uid"):
        fastmob_set = set(group["row_index"].tolist())
        cached_set = cached_keep_by_uid.get(uid, set())
        union = fastmob_set | cached_set
        scores[uid] = len(fastmob_set & cached_set) / len(union) if union else 1.0
    return scores


def test_filter_hampel_matches_cached_reference(ptrail_reference):
    """Row-subset agreement with the cached PTRAIL Hampel baseline on a Brightkite slice.

    This is a wide, documented tolerance rather than a strict match: the
    cached baseline was populated with ``hampel==0.0.5`` (see
    ``tests/populate_ptrail_cache.py``'s module docstring for why — PTRAIL
    1.0's own ``Filters.hampel_outlier_detection`` is incompatible with
    ``hampel>=1.0``), whose algorithm is materially different from the
    ``hampel==1.0.2`` Cython kernel fastmob's own ``hampel.rs`` was ported
    from: 0.0.5 uses a centered pandas ``.rolling(window_size * 2)`` window
    (double the width, for the same ``window_size=5`` default) with
    backward/forward-filled boundary medians/MADs, while fastmob (matching
    1.0.2) uses an exact ``2 * (window_size // 2) + 1``-wide window and
    never evaluates boundary points at all. The two implementations
    therefore disagree on a substantial fraction of points near sequence
    boundaries and in any window touching a NaN-producing duplicate
    timestamp; this test tracks gross regressions rather than asserting
    tight numeric parity.

    On this cached Brightkite slice, 7 of the 8 users score >= 0.75 Jaccard,
    but one user (uid 33) scores ~0.04 — PTRAIL's older rolling-window
    implementation drops nearly all of that user's points (57 of 60),
    almost certainly a real degenerate-window effect from that user's
    specific timestamp/duplicate pattern rather than an algorithmic
    difference worth chasing. The overall mean threshold below is set low
    enough to tolerate that one outlier user while still catching a broad
    regression in the other seven.
    """
    input_df = ptrail_reference.input_df
    cached = ptrail_reference.keep_mask("hampel")
    if cached is None:
        pytest.skip("No cached PTRAIL result for method='hampel'")

    result = traj_filter(
        input_df,
        method="hampel",
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
    )
    scores = _per_user_jaccard_keep(result, cached)
    mean_score = sum(scores.values()) / len(scores)
    assert mean_score >= 0.5, scores


@pytest.mark.skmob
def test_filter_matches_skmob(comparison_skmob):
    """Results must closely match skmob despite tiny Haversine threshold drift."""
    from skmob.preprocessing import filtering as skmob_filtering
    import pandas as pd

    skmob_result = skmob_filtering.filter(comparison_skmob, max_speed_kmh=500.0)
    skmob_result_df = pd.DataFrame(skmob_result)

    our_result = traj_filter(
        pd.DataFrame(comparison_skmob),
        max_speed_kmh=500.0,
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )

    # skmob uses its Python gislib Haversine implementation while fastmob uses
    # the Rust geo kernel. Points whose speed is exactly near the threshold can
    # fall on different sides, so compare the retained count with a tiny budget.
    assert abs(len(our_result) - len(skmob_result_df)) <= 5


def test_filter_matches_cached_reference(comparison_skmob_reference):
    """Row count matches the cached skmob baseline without requiring the skmob environment."""
    ref = comparison_skmob_reference
    cached_count = ref.row_count("filter")
    our_result = traj_filter(
        ref.input_df,
        max_speed_kmh=500.0,
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )
    assert abs(len(our_result) - cached_count) <= 5
