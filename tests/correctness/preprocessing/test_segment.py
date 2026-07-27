"""Correctness tests for fastmob.preprocessing.segment.

Each hand-crafted fixture in ``conftest.py`` (``segment_tdf`` /
``segment_tdf_polars``) isolates one documented behavior per algorithm; see
the fixture docstring comments there for the exact geometry/timing choices.

fastmob's segmentation has a hard row-preservation contract: every method
adds a ``segment_id`` column without ever dropping rows. This is a
deliberate, documented semantic difference from MovingPandas'
``trajectory_splitter.py`` splitters, which drop rows outright (a
``min_length`` cutoff, or every row strictly inside a detected stop /
non-moving run / observation gap). Where MovingPandas would drop those rows,
fastmob instead folds them forward into a neighboring segment (or, for
``stop``, gives the stop its own bracketed segment) — see
``test_segment_short_non_moving_run_is_folded_forward_not_dropped`` below,
and each method's Rust kernel docstring, for the exact adaptation.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.preprocessing import segment


def _user(df, uid: str):
    if hasattr(df, "loc"):
        return df[df["uid"] == uid].reset_index(drop=True)
    import polars as pl

    return df.filter(pl.col("uid") == uid)


def test_observation_gap_a_two_hour_gap_splits_into_two_segments(segment_tdf):
    """A 2-hour gap (above a 10-minute gap_s threshold) between points 2 and
    3 starts a new segment there; every other point stays in segment 0."""
    user = _user(segment_tdf, "gap_case")
    result = segment(user, method="observation_gap", gap_s=600.0)
    assert result["segment_id"].tolist() == [0, 0, 0, 1, 1]


def test_observation_gap_row_count_is_preserved(segment_tdf):
    """segment() never drops rows, unlike MovingPandas' ObservationGapSplitter."""
    user = _user(segment_tdf, "gap_case")
    result = segment(user, method="observation_gap", gap_s=600.0)
    assert len(result) == len(user)


def test_value_change_creates_three_segments(segment_tdf):
    """poi values [a, a, b, b, c] change twice → 3 segments."""
    user = _user(segment_tdf, "value_case")
    result = segment(user, method="value_change", col_name="poi")
    assert result["segment_id"].tolist() == [0, 0, 1, 1, 2]


def test_temporal_hour_mode_matches_value_change_on_truncated_hour(segment_tdf):
    """temporal(mode='hour') buckets by truncated hour; 5 points 1 minute
    apart all share the same hour bucket → a single segment (0 minutes
    through 4 minutes never crosses an hour boundary)."""
    user = _user(segment_tdf, "value_case")
    result = segment(user, method="temporal", mode="hour")
    assert result["segment_id"].tolist() == [0, 0, 0, 0, 0]


def test_temporal_with_a_null_datetime_row_does_not_raise():
    """A null datetime anywhere in the trajectory must not crash the
    bucket-id builder on pandas: casting a NaT-derived NaN straight to
    Int64 raises there (unlike Polars' nullable Int64), discovered via a
    real Brightkite benchmark run where the raw dataset's few unparseable
    timestamps (`errors="coerce"` -> NaT) slipped into a large slice.
    """
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 4,
            "datetime": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:30:00", None, "2020-01-01 02:00:00"], errors="coerce"
            ),
            "lat": [10.0, 10.0, 10.0, 10.0],
            "lng": [10.0, 10.0, 10.0, 10.0],
        }
    )
    result = segment(df, method="temporal", mode="hour")
    assert len(result) == len(df)


def test_angle_change_a_ninety_degree_turn_starts_a_new_segment(segment_tdf):
    """0->1 heads due north, 1->2 heads due east: a 90 deg change, well
    above the default min_angle=45, starts a new segment at the turn point."""
    user = _user(segment_tdf, "angle_case")
    result = segment(user, method="angle_change")
    assert result["segment_id"].tolist() == [0, 0, 1]


def test_speed_a_long_parked_run_brackets_a_new_segment(segment_tdf):
    """Points 2,3 are parked (~0 km/h, below a speed_kmh=5.0 "moving" floor)
    for 300s, at the default duration_s=300 threshold: they get their own
    bracketing segment between the two moving legs."""
    user = _user(segment_tdf, "speed_case")
    result = segment(user, method="speed", speed_kmh=5.0)
    assert result["segment_id"].tolist() == [0, 0, 1, 1, 2, 2]


def test_segment_short_non_moving_run_is_folded_forward_not_dropped(segment_tdf):
    """A single parked point (60s, well under the 300s duration_s threshold)
    does not start a new segment at all — it is folded into the surrounding
    segment. MovingPandas' SpeedSplitter would instead drop every
    "non-moving" row outright (filtering them out before running
    ObservationGapSplitter on what remains); fastmob's row-preservation
    contract means the row survives, merged forward, rather than
    disappearing."""
    user = _user(segment_tdf, "short_parked_case")
    result = segment(user, method="speed", speed_kmh=5.0)
    assert result["segment_id"].tolist() == [0, 0, 0, 0]
    assert len(result) == len(user)


def test_stop_is_bracketed_by_two_moving_segments(segment_tdf):
    """A detected 40-minute stop (above the default 20-minute threshold)
    gets its own segment, bracketed by the moving segment before it and the
    moving segment after it."""
    user = _user(segment_tdf, "stop_case")
    result = segment(user, method="stop")
    assert result["segment_id"].tolist() == [0, 0, 1, 1, 1, 1, 2]


def test_segment_single_point_user():
    """Single-point user gets segment_id 0 (nothing to split)."""
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = segment(df, method="observation_gap", gap_s=60.0)
    assert result["segment_id"].tolist() == [0]


def test_segment_multiuser_segment_ids_restart_at_zero_per_user(segment_tdf):
    """Every uid's segment_id sequence restarts at 0, independent of other users."""
    result = segment(segment_tdf, method="value_change", col_name="poi")
    for uid, group in result.groupby("uid"):
        assert group["segment_id"].min() == 0, uid


def test_segment_returns_same_backend_type(segment_tdf):
    """Pandas input -> pandas output."""
    result = segment(segment_tdf, method="value_change", col_name="poi")
    assert isinstance(result, type(segment_tdf))


def test_segment_polars_backend(segment_tdf_polars):
    """Polars input -> Polars output, row count preserved, segment_id present."""
    import polars as pl

    result = segment(segment_tdf_polars, method="observation_gap", gap_s=600.0)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(segment_tdf_polars)
    gap_case = result.filter(pl.col("uid") == "gap_case").sort("datetime")
    assert gap_case["segment_id"].to_list() == [0, 0, 0, 1, 1]


def test_segment_unknown_method_raises(segment_tdf):
    """An unrecognized ``method`` raises ValueError listing valid choices."""
    with pytest.raises(ValueError, match="unknown segment method"):
        segment(segment_tdf, method="not_a_real_method")


def test_segment_value_change_requires_col_name(segment_tdf):
    """``method='value_change'`` without ``col_name`` raises ValueError."""
    user = _user(segment_tdf, "value_case")
    with pytest.raises(ValueError, match="col_name"):
        segment(user, method="value_change")


def test_segment_unknown_temporal_mode_raises(segment_tdf):
    """An unrecognized ``mode`` for ``method='temporal'`` raises ValueError."""
    user = _user(segment_tdf, "value_case")
    with pytest.raises(ValueError, match="unknown temporal mode"):
        segment(user, method="temporal", mode="fortnight")


def _rand_index(row_ids: list, partition_a: dict, partition_b: dict) -> float:
    """Fraction of row-index pairs where two segment-id partitions agree on same/different-segment.

    A partition-equality-flavored similarity score (a co-membership Rand
    index) rather than strict set equality: two partitions can disagree on
    exact segment boundaries while still mostly agreeing on which rows
    travel together, which is what this measures. ``1.0`` means the two
    partitions are identical (up to relabeling); lower scores indicate
    genuine structural disagreement.

    @usedBy `test_segment_matches_cached_movingpandas_reference_partitions`.
    """
    n = len(row_ids)
    if n < 2:
        return 1.0
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = partition_a[row_ids[i]] == partition_a[row_ids[j]]
            same_b = partition_b[row_ids[i]] == partition_b[row_ids[j]]
            if same_a == same_b:
                agree += 1
            total += 1
    return agree / total


def _mean_rand_index_over_kept_rows(result_df, cached: dict) -> float:
    """Mean per-uid Rand index between fastmob's segment_id and cached partitions.

    MovingPandas' splitters drop rows outright (a `min_length` cutoff, a
    detected stop's interior, an entire "non-moving"/gap run) — those rows
    never appear in ``cached``. fastmob never drops rows, so the comparison
    is restricted to exactly the row-index universe MovingPandas kept per
    user (``kept_rows``), isolating the actual grouping-structure question
    from that documented, deliberate row-preservation difference — the same
    spirit as `test_simplify.py`'s Jaccard-over-kept-row-index comparison
    for the row-dropping `simplify()` methods.

    @usedBy `test_segment_matches_cached_movingpandas_reference_partitions`.
    """
    scores = []
    for uid, group in result_df.groupby("uid"):
        cached_partition = cached.get(uid)
        if cached_partition is None:
            continue
        kept_rows = sorted(set().union(*cached_partition)) if cached_partition else []
        if not kept_rows:
            continue
        cached_seg = {row: seg_id for seg_id, rows in enumerate(cached_partition) for row in rows}
        fastmob_seg = dict(zip(group["row_index"].tolist(), group["segment_id"].tolist()))
        scores.append(_rand_index(kept_rows, cached_seg, fastmob_seg))
    return sum(scores) / len(scores) if scores else 1.0


@pytest.mark.parametrize(
    ("method", "kwargs", "min_mean_rand_index"),
    [
        # observation_gap/angle_change/value_change/temporal are exact or
        # near-exact ports of MovingPandas' own grouping-key logic (see
        # tests/populate_movingpandas_cache.py's module docstring) and match
        # (Rand index == 1.0) on every cached user.
        ("observation_gap", {"gap_s": 3600.0}, 0.99),
        ("angle_change", {"min_angle": 45.0, "min_speed_kmh": 0.0}, 0.99),
        ("value_change", {"col_name": "location_id"}, 0.99),
        ("temporal", {"mode": "day"}, 0.99),
        # speed and stop are compared with a documented, wide tolerance
        # (tracks gross regressions rather than asserting tight parity):
        # speed's "moving"/non-moving classification interacts differently
        # with MovingPandas' post-filter ObservationGapSplitter delegation
        # than fastmob's single-pass non-moving-run-duration criterion on
        # this sparse, low-frequency check-in dataset (mean Rand index
        # ~0.27 on the cached Brightkite slice); stop uses a genuinely
        # different stop-detection algorithm (fastmob's anchor-point
        # expanding window vs MovingPandas' point-cluster diameter), so a
        # `stop_radius_km`-to-`max_diameter` unit conversion is only an
        # approximate correspondence, not the same criterion (mean Rand
        # index ~0.29).
        ("speed", {"speed_kmh": 0.0, "duration_s": 300.0}, 0.2),
        ("stop", {"stop_radius_km": 0.2, "minutes_for_a_stop": 20.0}, 0.2),
    ],
)
def test_segment_matches_cached_movingpandas_reference_partitions(
    movingpandas_reference, method, kwargs, min_mean_rand_index
):
    """Partition agreement with the cached MovingPandas baseline on a Brightkite slice.

    Compared by **partition equality** (groupby(segment_id) -> set of
    row-index frozensets — here summarized as a co-membership Rand index,
    see `_rand_index`), not exact segment_id values, since segment numbering
    is not guaranteed to match between libraries — only the partition
    structure is (see the project plan's "Cache schema per capability
    area" section) — and restricted to MovingPandas' own kept-row universe
    per user (see `_mean_rand_index_over_kept_rows`).
    """
    cached = movingpandas_reference.segment_partitions(method)
    if cached is None:
        pytest.skip(f"No cached MovingPandas segment result for method={method!r}")

    input_df = movingpandas_reference.input_df
    result = segment(
        input_df,
        method=method,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        **kwargs,
    )
    result = result.assign(row_index=input_df["row_index"].to_numpy())
    mean_rand_index = _mean_rand_index_over_kept_rows(result, cached)
    assert mean_rand_index >= min_mean_rand_index, mean_rand_index


@pytest.mark.skip(reason="movetk segmentation deferred, see plan doc")
def test_movetk_segmentation_placeholder():
    """Placeholder for MoveTK's Monotone/Model-based/Brownian-bridge segmentation.

    All three are explicitly out of scope for this phase (see the project
    plan's "What ships vs. what's deferred" table): Monotone is a generic
    criterion-family rather than one algorithm, Model-based needs a real
    change-point/model-selection framework, and Brownian-bridge is a full
    probabilistic movement model. MovingPandas' 6 splitters (all shipped
    here) already fully satisfy "a menu of segmentation criteria", so this
    is a documented deferral, not an oversight. When implemented, this test
    should assert that each MoveTK strategy produces a `segment_id` column
    whose partitions match a real MoveTK C++ driver run on the same
    Brightkite slice used elsewhere in this file.
    """
    raise NotImplementedError
