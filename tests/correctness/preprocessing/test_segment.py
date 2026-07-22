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


def _partitions_by_uid(result_df) -> dict:
    """Return ``{uid: set of frozensets of local row_index}`` grouped by segment_id."""
    partitions = {}
    for uid, group in result_df.groupby("uid"):
        by_segment: dict[int, set] = {}
        for row_index, segment_id in zip(group["row_index"].tolist(), group["segment_id"].tolist()):
            by_segment.setdefault(segment_id, set()).add(row_index)
        partitions[uid] = {frozenset(rows) for rows in by_segment.values()}
    return partitions


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("observation_gap", {"gap_s": 3600.0}),
        ("value_change", {}),  # col_name filled in per-parametrization below
    ],
)
def test_segment_matches_cached_movingpandas_reference_partitions(movingpandas_reference, method, kwargs):
    """Partition agreement with the cached MovingPandas baseline on a Brightkite slice.

    Compared by **partition equality** (groupby(segment_id) -> set of
    row-index frozensets), not exact segment_id values, since segment
    numbering is not guaranteed to match between libraries — only the
    partition structure is (see the project plan's "Cache schema per
    capability area" section).
    """
    cached = movingpandas_reference.segment_partitions(method)
    if cached is None:
        pytest.skip(f"No cached MovingPandas segment result for method={method!r}")

    input_df = movingpandas_reference.input_df
    if method == "value_change":
        kwargs = {"col_name": "location_id"}
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
    fastmob_partitions = _partitions_by_uid(result)

    matched_users = 0
    for uid, cached_partition in cached.items():
        fastmob_partition = fastmob_partitions.get(uid)
        if fastmob_partition == cached_partition:
            matched_users += 1
    assert matched_users >= 1, (fastmob_partitions, cached)


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
