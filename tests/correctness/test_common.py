"""Correctness tests for skmob2/measures/_common.py."""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _pick_existing_column
# ---------------------------------------------------------------------------


class TestPickExistingColumn:
    def test_returns_first_match(self):
        from skmob2.measures._common import _pick_existing_column

        assert _pick_existing_column(["lat", "latitude"], ["lat", "latitude"]) == "lat"

    def test_skips_absent_candidates(self):
        from skmob2.measures._common import _pick_existing_column

        assert _pick_existing_column(["latitude"], ["lat", "latitude"]) == "latitude"

    def test_returns_none_when_no_match(self):
        from skmob2.measures._common import _pick_existing_column

        assert _pick_existing_column(["x", "y"], ["lat", "latitude"]) is None

    def test_empty_columns(self):
        from skmob2.measures._common import _pick_existing_column

        assert _pick_existing_column([], ["lat"]) is None

    def test_empty_candidates(self):
        from skmob2.measures._common import _pick_existing_column

        assert _pick_existing_column(["lat"], []) is None


# ---------------------------------------------------------------------------
# _detect_trajectory_columns
# ---------------------------------------------------------------------------


class TestDetectTrajectoryColumns:
    def _make_nw_df(self, columns: list[str]):
        import narwhals as nw

        data = {col: [] for col in columns}
        return nw.from_native(pd.DataFrame(data), eager_only=True)

    def test_detects_standard_columns(self):
        from skmob2.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat", "lng", "uid"])
        dt, lat, lng, uid = _detect_trajectory_columns(nw_df)
        assert dt == "datetime"
        assert lat == "lat"
        assert lng == "lng"
        assert uid == "uid"

    def test_uid_is_none_when_absent(self):
        from skmob2.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat", "lng"])
        dt, lat, lng, uid = _detect_trajectory_columns(nw_df)
        assert uid is None

    def test_explicit_override_used(self):
        from skmob2.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["ts", "y", "x", "person"])
        dt, lat, lng, uid = _detect_trajectory_columns(
            nw_df,
            datetime_col="ts",
            lat_col="y",
            lng_col="x",
            uid_col="person",
        )
        assert dt == "ts"
        assert lat == "y"
        assert lng == "x"
        assert uid == "person"

    def test_raises_on_missing_required_column(self):
        from skmob2.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat"])  # missing lng
        with pytest.raises(ValueError, match="longitude"):
            _detect_trajectory_columns(nw_df)

    def test_error_message_lists_candidates(self):
        from skmob2.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat"])
        with pytest.raises(ValueError, match="lng"):
            _detect_trajectory_columns(nw_df)

    def test_detects_alternative_column_names(self):
        from skmob2.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["check-in_time", "latitude", "longitude", "user"])
        dt, lat, lng, uid = _detect_trajectory_columns(nw_df)
        assert dt == "check-in_time"
        assert lat == "latitude"
        assert lng == "longitude"
        assert uid == "user"


# ---------------------------------------------------------------------------
# _prepare_trajectory
# ---------------------------------------------------------------------------


class TestPrepareTrajectory:
    def _make_df(self):
        return pd.DataFrame(
            {
                "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
                "lat": [0.0, None, 2.0, 3.0],
                "lng": [10.0, 11.0, None, 13.0],
                "uid": ["u1", "u1", "u1", "u1"],
            }
        )

    def test_returns_five_tuple(self):
        from skmob2.measures._common import _prepare_trajectory

        result = _prepare_trajectory(self._make_df())
        assert len(result) == 5

    def test_drops_nulls_in_required_columns(self):
        from skmob2.measures._common import _prepare_trajectory

        df, *_ = _prepare_trajectory(self._make_df())
        # 2 rows have nulls in lat or lng — only 2 rows should survive
        assert len(df) == 2

    def test_lat_lng_cast_to_float64(self):
        import narwhals as nw
        from skmob2.measures._common import _prepare_trajectory

        df_in = pd.DataFrame(
            {
                "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
                "lat": [1, 2],  # integers
                "lng": [10, 20],
            }
        )
        df, _, lat_col, lng_col, _ = _prepare_trajectory(df_in)
        assert df.get_column(lat_col).dtype == nw.Float64
        assert df.get_column(lng_col).dtype == nw.Float64

    def test_sorted_by_uid_then_datetime(self):
        from skmob2.measures._common import _prepare_trajectory

        df_in = pd.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01", "2020-01-01"]),
                "lat": [0.0, 1.0, 2.0, 3.0],
                "lng": [0.0, 1.0, 2.0, 3.0],
            }
        )
        df, _, _, _, uid_col = _prepare_trajectory(df_in)
        uids = df.get_column(uid_col).to_list()
        # All "a" rows come before all "b" rows
        a_indices = [i for i, u in enumerate(uids) if u == "a"]
        b_indices = [i for i, u in enumerate(uids) if u == "b"]
        assert max(a_indices) < min(b_indices)

    def test_sort_false_preserves_row_order_after_dropping_nulls(self):
        import narwhals as nw
        from skmob2.measures._common import _prepare_trajectory

        df_in = pd.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01", "2020-01-01"]),
                "lat": [0, 1, None, 3],
                "lng": [10, 11, 12, 13],
            }
        )
        df, _, lat_col, lng_col, uid_col = _prepare_trajectory(df_in, sort=False)

        assert df.get_column(uid_col).to_list() == ["b", "a", "a"]
        assert df.get_column(lat_col).to_list() == [0.0, 1.0, 3.0]
        assert df.get_column(lat_col).dtype == nw.Float64
        assert df.get_column(lng_col).dtype == nw.Float64

    def test_raises_on_missing_column(self):
        from skmob2.measures._common import _prepare_trajectory

        df_in = pd.DataFrame({"datetime": [], "lat": []})
        with pytest.raises(ValueError, match="longitude"):
            _prepare_trajectory(df_in)


# ---------------------------------------------------------------------------
# _build_user_ranges
# ---------------------------------------------------------------------------


class TestBuildUserRanges:
    def test_builds_ranges_from_contiguous_uid_groups(self):
        import narwhals as nw
        from skmob2.measures._common import _build_user_ranges

        df = nw.from_native(pd.DataFrame({"uid": ["a", "a", "b", "b", "b", "c"]}), eager_only=True)

        uid_values, ranges = _build_user_ranges(df, "uid")

        assert uid_values == ["a", "b", "c"]
        assert ranges == [(0, 2), (2, 5), (5, 6)]

    def test_builds_empty_ranges_for_empty_uid_dataframe(self):
        import narwhals as nw
        from skmob2.measures._common import _build_user_ranges

        df = nw.from_native(pd.DataFrame({"uid": []}), eager_only=True)

        assert _build_user_ranges(df, "uid") == ([], [])

    def test_builds_single_range_without_uid_column(self):
        import narwhals as nw
        from skmob2.measures._common import _build_user_ranges

        df = nw.from_native(pd.DataFrame({"lat": [1.0, 2.0, 3.0]}), eager_only=True)

        assert _build_user_ranges(df, None) == ([None], [(0, 3)])


# ---------------------------------------------------------------------------
# _shannon_entropy
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_equal_counts_two_items(self):
        """Two items with equal counts -> 1 bit of entropy."""
        from skmob2.measures._common import _shannon_entropy

        assert abs(_shannon_entropy([1, 1]) - 1.0) < 1e-12

    def test_single_item(self):
        """One item -> 0 bits (no uncertainty)."""
        from skmob2.measures._common import _shannon_entropy

        assert _shannon_entropy([5]) == 0.0

    def test_empty_list(self):
        """Empty list -> 0.0 without error."""
        from skmob2.measures._common import _shannon_entropy

        assert _shannon_entropy([]) == 0.0

    def test_zero_total(self):
        """All-zero counts -> 0.0 without division error."""
        from skmob2.measures._common import _shannon_entropy

        assert _shannon_entropy([0, 0, 0]) == 0.0

    def test_uniform_five_items(self):
        """Five equal-count items -> log2(5) bits."""
        from skmob2.measures._common import _shannon_entropy
        import math

        result = _shannon_entropy([1, 1, 1, 1, 1])
        assert abs(result - math.log2(5)) < 1e-12

    def test_skewed_distribution(self):
        """p=0.75, p=0.25 -> hand-computed value."""
        from skmob2.measures._common import _shannon_entropy
        import math

        p1, p2 = 0.75, 0.25
        expected = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        result = _shannon_entropy([3, 1])
        assert abs(result - expected) < 1e-12

    def test_zeros_ignored(self):
        """Zero-count items do not affect entropy (0*log(0) = 0)."""
        from skmob2.measures._common import _shannon_entropy

        # [1, 1] and [1, 1, 0] should give the same entropy.
        assert abs(_shannon_entropy([1, 1]) - _shannon_entropy([1, 1, 0])) < 1e-12
