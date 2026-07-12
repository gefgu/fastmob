"""Correctness tests for fastmob/measures/_common.py."""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# backend kernel dispatch
# ---------------------------------------------------------------------------


class TestBackendKernelDispatch:
    def test_pandas_uses_numpy_kernel_path(self):
        import narwhals as nw
        from fastmob.measures._common import _use_arrow_kernel_path

        df = pd.DataFrame({"lat": [1.0], "lng": [2.0]})
        nw_df = nw.from_native(df, eager_only=True)

        assert _use_arrow_kernel_path(nw_df) is False

    def test_polars_uses_arrow_kernel_path(self):
        pl = pytest.importorskip("polars", reason="Polars not installed")
        import narwhals as nw
        from fastmob.measures._common import _use_arrow_kernel_path

        df = pl.DataFrame({"lat": [1.0], "lng": [2.0]})
        nw_df = nw.from_native(df, eager_only=True)

        assert _use_arrow_kernel_path(nw_df) is True


# ---------------------------------------------------------------------------
# _pick_existing_column
# ---------------------------------------------------------------------------


class TestPickExistingColumn:
    def test_returns_first_match(self):
        from fastmob.measures._common import _pick_existing_column

        assert _pick_existing_column(["lat", "latitude"], ["lat", "latitude"]) == "lat"

    def test_skips_absent_candidates(self):
        from fastmob.measures._common import _pick_existing_column

        assert _pick_existing_column(["latitude"], ["lat", "latitude"]) == "latitude"

    def test_returns_none_when_no_match(self):
        from fastmob.measures._common import _pick_existing_column

        assert _pick_existing_column(["x", "y"], ["lat", "latitude"]) is None

    def test_empty_columns(self):
        from fastmob.measures._common import _pick_existing_column

        assert _pick_existing_column([], ["lat"]) is None

    def test_empty_candidates(self):
        from fastmob.measures._common import _pick_existing_column

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
        from fastmob.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat", "lng", "uid"])
        dt, lat, lng, uid = _detect_trajectory_columns(nw_df)
        assert dt == "datetime"
        assert lat == "lat"
        assert lng == "lng"
        assert uid == "uid"

    def test_uid_is_none_when_absent(self):
        from fastmob.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat", "lng"])
        dt, lat, lng, uid = _detect_trajectory_columns(nw_df)
        assert uid is None

    def test_explicit_override_used(self):
        from fastmob.measures._common import _detect_trajectory_columns

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
        from fastmob.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat"])  # missing lng
        with pytest.raises(ValueError, match="longitude"):
            _detect_trajectory_columns(nw_df)

    def test_error_message_lists_candidates(self):
        from fastmob.measures._common import _detect_trajectory_columns

        nw_df = self._make_nw_df(["datetime", "lat"])
        with pytest.raises(ValueError, match="lng"):
            _detect_trajectory_columns(nw_df)

    def test_detects_alternative_column_names(self):
        from fastmob.measures._common import _detect_trajectory_columns

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

    def _prepare(self, df_in, **kwargs):
        import narwhals as nw
        from fastmob.measures._common import _detect_trajectory_columns, _prepare_trajectory

        nw_df = nw.from_native(df_in, eager_only=True)
        datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(nw_df)
        df = _prepare_trajectory(
            nw_df,
            datetime_col=datetime_col,
            lat_col=lat_col,
            lng_col=lng_col,
            uid_col=uid_col,
            **kwargs,
        )
        return df, datetime_col, lat_col, lng_col, uid_col

    def test_returns_dataframe(self):
        import narwhals as nw
        from fastmob.measures._common import _prepare_trajectory

        nw_df = nw.from_native(self._make_df(), eager_only=True)
        result = _prepare_trajectory(
            nw_df,
            datetime_col="datetime",
            lat_col="lat",
            lng_col="lng",
            uid_col="uid",
        )
        assert isinstance(result, nw.DataFrame)

    def test_drops_nulls_in_required_columns(self):
        df, *_ = self._prepare(self._make_df())
        # 2 rows have nulls in lat or lng — only 2 rows should survive
        assert len(df) == 2

    def test_lat_lng_cast_to_float64(self):
        import narwhals as nw

        df_in = pd.DataFrame(
            {
                "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
                "lat": [1, 2],  # integers
                "lng": [10, 20],
            }
        )
        df, _, lat_col, lng_col, _ = self._prepare(df_in)
        assert df.get_column(lat_col).dtype == nw.Float64
        assert df.get_column(lng_col).dtype == nw.Float64

    def test_datetime_strings_are_converted(self):
        import narwhals as nw

        df_in = pd.DataFrame(
            {
                "datetime": ["2020-01-01 01:00:00", "2020-01-01 00:00:00"],
                "lat": [1.0, 2.0],
                "lng": [10.0, 20.0],
            }
        )
        df, datetime_col, *_ = self._prepare(df_in)
        assert df.get_column(datetime_col).dtype == nw.Datetime
        assert df.get_column("lat").to_list() == [2.0, 1.0]

    def test_sorted_by_uid_then_datetime(self):
        from fastmob.measures._common import _ROW_ORDER_COL

        df_in = pd.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01", "2020-01-01"]),
                "lat": [0.0, 1.0, 2.0, 3.0],
                "lng": [0.0, 1.0, 2.0, 3.0],
            }
        )
        df, _, _, _, uid_col = self._prepare(df_in)
        assert _ROW_ORDER_COL not in df.columns
        uids = df.get_column(uid_col).to_list()
        # All "a" rows come before all "b" rows
        a_indices = [i for i, u in enumerate(uids) if u == "a"]
        b_indices = [i for i, u in enumerate(uids) if u == "b"]
        assert max(a_indices) < min(b_indices)

    def test_sort_false_preserves_row_order_after_dropping_nulls(self):
        import narwhals as nw
        from fastmob.measures._common import _ROW_ORDER_COL

        df_in = pd.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01", "2020-01-01"]),
                "lat": [0, 1, None, 3],
                "lng": [10, 11, 12, 13],
            }
        )
        df, _, lat_col, lng_col, uid_col = self._prepare(df_in, sort=False)

        assert _ROW_ORDER_COL not in df.columns
        assert df.get_column(uid_col).to_list() == ["b", "a", "a"]
        assert df.get_column(lat_col).to_list() == [0.0, 1.0, 3.0]
        assert df.get_column(lat_col).dtype == nw.Float64
        assert df.get_column(lng_col).dtype == nw.Float64

    def test_drop_nulls_false_preserves_null_coordinate_rows(self):
        df, *_ = self._prepare(self._make_df(), sort=False, drop_nulls=False)
        assert len(df) == 4

    def test_detect_raises_on_missing_column_before_prepare(self):
        import narwhals as nw
        from fastmob.measures._common import _detect_trajectory_columns

        df_in = pd.DataFrame({"datetime": [], "lat": []})
        nw_df = nw.from_native(df_in, eager_only=True)
        with pytest.raises(ValueError, match="longitude"):
            _detect_trajectory_columns(nw_df)


class TestBuildTimeOrderedUserRanges:
    def test_pandas_string_uids_use_numpy_rust_path(self, capsys):
        import narwhals as nw
        import numpy as np
        from fastmob.measures._common import _build_time_ordered_user_ranges, _extract_timestamps_s

        df = pd.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": pd.to_datetime(
                    [
                        "2020-01-01 01:00:00",
                        "2020-01-01 02:00:00",
                        "2020-01-01 00:00:00",
                        "2020-01-01 00:30:00",
                    ]
                ),
                "lat": [0.0, 0.0, 0.0, 0.0],
                "lng": [0.0, 0.0, 0.0, 0.0],
            }
        )
        nw_df = nw.from_native(df, eager_only=True)
        timestamps = _extract_timestamps_s(nw_df, "datetime")

        uid_values, indices, ends = _build_time_ordered_user_ranges(
            nw_df,
            "uid",
            "datetime",
            timestamps.to_numpy(),
        )

        captured = capsys.readouterr()
        assert "Indexing@fallback" not in captured.out
        assert uid_values == ["b", "a"]
        assert np.asarray(indices).tolist() == [2, 0, 3, 1]
        assert np.asarray(ends).tolist() == [2, 4]

    def test_polars_string_uids_use_first_seen_arrow_rust_path(self):
        import narwhals as nw
        import numpy as np
        from fastmob.measures._common import _build_time_ordered_user_ranges, _extract_timestamps_s

        pl = pytest.importorskip("polars", reason="Polars not installed")
        df = pl.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": [
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:30:00",
                ],
                "lat": [0.0, 0.0, 0.0, 0.0],
                "lng": [0.0, 0.0, 0.0, 0.0],
            }
        ).with_columns(pl.col("datetime").str.to_datetime())
        nw_df = nw.from_native(df, eager_only=True)
        timestamps = _extract_timestamps_s(nw_df, "datetime")

        uid_values, indices, ends = _build_time_ordered_user_ranges(
            nw_df,
            "uid",
            "datetime",
            timestamps.to_arrow(),
        )

        assert uid_values == ["b", "a"]
        assert np.asarray(indices).tolist() == [2, 0, 3, 1]
        assert np.asarray(ends).tolist() == [2, 4]

    def test_fallback_preserves_first_seen_user_order(self, monkeypatch):
        import narwhals as nw
        import numpy as np
        import fastmob.measures._common as common
        from fastmob.measures._common import _build_time_ordered_user_ranges, _extract_timestamps_s

        df = pd.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": pd.to_datetime(
                    [
                        "2020-01-01 01:00:00",
                        "2020-01-01 02:00:00",
                        "2020-01-01 00:00:00",
                        "2020-01-01 00:30:00",
                    ]
                ),
            }
        )
        nw_df = nw.from_native(df, eager_only=True)
        timestamps = _extract_timestamps_s(nw_df, "datetime")

        def unsupported(*args):
            raise ValueError("unsupported dtype")

        monkeypatch.setitem(
            common._TIME_ORDERED_USER_RANGES_DISPATCHER.dispatch["numpy"],
            "time_ordered_indices",
            unsupported,
        )

        uid_values, indices, ends = _build_time_ordered_user_ranges(
            nw_df,
            "uid",
            "datetime",
            timestamps.to_numpy(),
        )

        assert uid_values == ["b", "a"]
        assert np.asarray(indices).tolist() == [2, 0, 3, 1]
        assert np.asarray(ends).tolist() == [2, 4]


class TestBuildIndexedUserRangesFast:
    def test_pandas_string_uids_use_first_seen_numpy_rust_path(self):
        import narwhals as nw
        import numpy as np
        from fastmob.measures._common import _build_indexed_user_ranges_fast

        df = nw.from_native(
            pd.DataFrame({"uid": ["b", "a", "b", "c", "a", "c"]}),
            eager_only=True,
        )

        uid_values, indices, ends = _build_indexed_user_ranges_fast(df, "uid")

        assert uid_values == ["b", "a", "c"]
        assert np.asarray(indices).tolist() == [0, 2, 1, 4, 3, 5]
        assert np.asarray(ends).tolist() == [2, 4, 6]

    def test_polars_string_uids_use_first_seen_arrow_rust_path(self):
        import narwhals as nw
        import numpy as np
        from fastmob.measures._common import _build_indexed_user_ranges_fast

        pl = pytest.importorskip("polars", reason="Polars not installed")
        df = nw.from_native(
            pl.DataFrame({"uid": ["b", "a", "b", "c", "a", "c"]}),
            eager_only=True,
        )

        uid_values, indices, ends = _build_indexed_user_ranges_fast(df, "uid")

        assert uid_values == ["b", "a", "c"]
        assert np.asarray(indices).tolist() == [0, 2, 1, 4, 3, 5]
        assert np.asarray(ends).tolist() == [2, 4, 6]


# ---------------------------------------------------------------------------
# _build_user_ranges
# ---------------------------------------------------------------------------


class TestBuildUserRanges:
    def test_builds_ranges_from_contiguous_uid_groups(self):
        import narwhals as nw
        from fastmob.measures._common import _build_user_ranges

        df = nw.from_native(pd.DataFrame({"uid": ["a", "a", "b", "b", "b", "c"]}), eager_only=True)

        uid_values, ranges = _build_user_ranges(df, "uid")

        assert uid_values == ["a", "b", "c"]
        assert ranges == [(0, 2), (2, 5), (5, 6)]

    def test_builds_empty_ranges_for_empty_uid_dataframe(self):
        import narwhals as nw
        from fastmob.measures._common import _build_user_ranges

        df = nw.from_native(pd.DataFrame({"uid": []}), eager_only=True)

        assert _build_user_ranges(df, "uid") == ([], [])

    def test_builds_single_range_without_uid_column(self):
        import narwhals as nw
        from fastmob.measures._common import _build_user_ranges

        df = nw.from_native(pd.DataFrame({"lat": [1.0, 2.0, 3.0]}), eager_only=True)

        assert _build_user_ranges(df, None) == ([None], [(0, 3)])


# ---------------------------------------------------------------------------
# _build_presorted_user_ends
# ---------------------------------------------------------------------------


class TestBuildPresortedUserEnds:
    def test_builds_ends_from_pandas_contiguous_uid_groups(self):
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pd.DataFrame({"uid": ["a", "a", "b", "b", "b", "c"]}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values == ["a", "b", "c"]
        assert ends.tolist() == [2, 5, 6]

    def test_builds_empty_ends_for_empty_uid_dataframe(self):
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pd.DataFrame({"uid": []}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values == []
        assert ends.tolist() == []

    def test_builds_single_end_without_uid_column(self):
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pd.DataFrame({"lat": [1.0, 2.0, 3.0]}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, None)

        assert uid_values is None
        assert ends.tolist() == [3]

    def test_builds_ends_from_polars_contiguous_uid_groups(self):
        pl = pytest.importorskip("polars", reason="Polars not installed")
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pl.DataFrame({"uid": ["a", "a", "b", "b", "c"]}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values == ["a", "b", "c"]
        assert ends.tolist() == [2, 4, 5]


# ---------------------------------------------------------------------------
# _shannon_entropy
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_equal_counts_two_items(self):
        """Two items with equal counts -> 1 bit of entropy."""
        from fastmob.measures._common import _shannon_entropy

        assert abs(_shannon_entropy([1, 1]) - 1.0) < 1e-12

    def test_single_item(self):
        """One item -> 0 bits (no uncertainty)."""
        from fastmob.measures._common import _shannon_entropy

        assert _shannon_entropy([5]) == 0.0

    def test_empty_list(self):
        """Empty list -> 0.0 without error."""
        from fastmob.measures._common import _shannon_entropy

        assert _shannon_entropy([]) == 0.0

    def test_zero_total(self):
        """All-zero counts -> 0.0 without division error."""
        from fastmob.measures._common import _shannon_entropy

        assert _shannon_entropy([0, 0, 0]) == 0.0

    def test_uniform_five_items(self):
        """Five equal-count items -> log2(5) bits."""
        from fastmob.measures._common import _shannon_entropy
        import math

        result = _shannon_entropy([1, 1, 1, 1, 1])
        assert abs(result - math.log2(5)) < 1e-12

    def test_skewed_distribution(self):
        """p=0.75, p=0.25 -> hand-computed value."""
        from fastmob.measures._common import _shannon_entropy
        import math

        p1, p2 = 0.75, 0.25
        expected = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        result = _shannon_entropy([3, 1])
        assert abs(result - expected) < 1e-12

    def test_zeros_ignored(self):
        """Zero-count items do not affect entropy (0*log(0) = 0)."""
        from fastmob.measures._common import _shannon_entropy

        # [1, 1] and [1, 1, 0] should give the same entropy.
        assert abs(_shannon_entropy([1, 1]) - _shannon_entropy([1, 1, 0])) < 1e-12
