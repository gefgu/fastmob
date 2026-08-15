"""Correctness tests for shared fastmob utility helpers."""

from __future__ import annotations

import math
import random

import pandas as pd
import pytest


class TestArrowCoercion:
    def test_converts_pandas_backed_narwhals_series(self):
        import narwhals as nw
        import pyarrow as pa
        from fastmob.utils._common import _as_arrow

        values = _as_arrow(nw.from_native(pd.DataFrame({"x": [1, 2]}), eager_only=True).get_column("x"))
        assert isinstance(values, pa.Array)
        assert values.to_pylist() == [1, 2]

    def test_preserves_chunked_arrow_input(self):
        import pyarrow as pa
        from fastmob.utils._common import _as_arrow

        values = pa.chunked_array([["a"], ["b"]])
        assert _as_arrow(values) is values


class TestArrowTimestampOrdering:
    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_time_ordering_accepts_native_arrow_timestamp_units(self, unit):
        import pyarrow as pa
        from fastmob._core import time_ordered_user_indices

        timestamps = pa.array([2, 1, 1], type=pa.timestamp(unit))
        indices, ends = time_ordered_user_indices(None, timestamps)

        assert indices.to_pylist() == [1, 2, 0]
        assert ends.to_pylist() == [3]

    def test_time_ordering_places_null_timestamps_before_values(self):
        import pyarrow as pa
        from fastmob._core import time_ordered_user_indices

        timestamps = pa.array([2, None, 1], type=pa.timestamp("us"))
        indices, ends = time_ordered_user_indices(None, timestamps)

        assert indices.to_pylist() == [1, 2, 0]
        assert ends.to_pylist() == [3]


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


class TestArrowFactorization:
    @staticmethod
    def _reference(values, sort):
        """Reference the documented first-seen/null-last factorization contract."""

        def key(value):
            if value is None:
                return (2, None)
            if isinstance(value, float) and math.isnan(value):
                return (1, None)
            return (0, value)

        representatives = []
        codes = []
        seen = []
        for index, value in enumerate(values):
            value_key = key(value)
            try:
                code = seen.index(value_key)
            except ValueError:
                code = len(seen)
                seen.append(value_key)
                representatives.append(index)
            codes.append(code)
        if not sort:
            return codes, representatives
        order = sorted(range(len(seen)), key=lambda code: seen[code])
        ranks = [0] * len(order)
        for rank, code in enumerate(order):
            ranks[code] = rank
        return [ranks[code] for code in codes], [representatives[code] for code in order]

    def test_preserves_first_seen_order_and_groups_nulls(self):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        codes, representatives = factorize_arrow(pa.array(["b", None, "a", "b", None]))

        assert pa.array(codes).to_pylist() == [0, 1, 2, 0, 1]
        assert pa.array(representatives).to_pylist() == [0, 1, 2]

    def test_codes_dtype_is_uint32_and_representatives_uint64(self):
        # Dense categorical codes stay narrow at the Arrow boundary; row
        # representatives remain UInt64 because they are positional indices.
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        codes, representatives = factorize_arrow(pa.array(["b", None, "a", "b", None]))

        assert pa.array(codes).type == pa.uint32()
        assert pa.array(representatives).type == pa.uint64()

    def test_sorted_values_put_null_last(self):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        codes, representatives = factorize_arrow(pa.array([2, None, 1, 2]), True)

        assert pa.array(codes).to_pylist() == [1, 2, 0, 1]
        assert pa.array(representatives).to_pylist() == [2, 0, 1]

    def test_dense_nullable_integers_preserve_first_seen_order(self):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        codes, representatives = factorize_arrow(pa.array([3, None, 2, 3, None]))

        assert pa.array(codes).to_pylist() == [0, 1, 2, 0, 1]
        assert pa.array(representatives).to_pylist() == [0, 1, 2]

    def test_dense_nullable_integers_sort_with_null_last(self):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        codes, representatives = factorize_arrow(pa.array([3, None, 2, 3, None]), True)

        assert pa.array(codes).to_pylist() == [1, 2, 0, 1, 2]
        assert pa.array(representatives).to_pylist() == [2, 0, 1]

    def test_boolean_fast_path_preserves_both_orderings(self):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        values = pa.array([True, None, False, True, None])
        codes, representatives = factorize_arrow(values)
        assert pa.array(codes).to_pylist() == [0, 1, 2, 0, 1]
        assert pa.array(representatives).to_pylist() == [0, 1, 2]

        codes, representatives = factorize_arrow(values, True)
        assert pa.array(codes).to_pylist() == [1, 2, 0, 1, 2]
        assert pa.array(representatives).to_pylist() == [2, 0, 1]

    @pytest.mark.parametrize("sort", [False, True])
    @pytest.mark.parametrize(
        "values",
        [[], [None, None], [7, None, 7], [3, 1, 3, None, 2], [0.0, -0.0, float("nan"), None, float("nan")]],
    )
    def test_matches_reference_for_empty_null_and_float_edge_cases(self, values, sort):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        array = (
            pa.array(values, type=pa.float64())
            if not values or all(value is None for value in values)
            else pa.array(values)
        )
        codes, representatives = factorize_arrow(array, sort)
        expected_codes, expected_representatives = self._reference(values, sort)
        assert pa.array(codes).to_pylist() == expected_codes
        assert pa.array(representatives).to_pylist() == expected_representatives

    @pytest.mark.parametrize("sort", [False, True])
    def test_matches_reference_for_seeded_nullable_integers(self, sort):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        rng = random.Random(20260802)
        values = [None if rng.randrange(11) == 0 else rng.randrange(-7, 8) for _ in range(512)]
        codes, representatives = factorize_arrow(pa.array(values), sort)
        expected_codes, expected_representatives = self._reference(values, sort)
        assert pa.array(codes).to_pylist() == expected_codes
        assert pa.array(representatives).to_pylist() == expected_representatives

    @pytest.mark.parametrize("sort", [False, True])
    def test_utf8_dictionary_fast_path_matches_decoded_values(self, sort):
        import pyarrow as pa
        from fastmob._core import factorize_arrow

        dictionary = pa.DictionaryArray.from_arrays(
            pa.array([1, None, 0, 1, 2, 0], type=pa.int8()),
            pa.array(["b", "a", "a"]),
        )
        codes, representatives = factorize_arrow(dictionary, sort)
        expected_codes, expected_representatives = self._reference(dictionary.to_pylist(), sort)
        assert pa.array(codes).to_pylist() == expected_codes
        assert pa.array(representatives).to_pylist() == expected_representatives

    def test_rejects_numpy_input(self):
        import numpy as np
        from fastmob._core import factorize_arrow

        with pytest.raises(ValueError, match="expected Arrow array"):
            factorize_arrow(np.array([1, 2, 1]))


class TestMotifPurposeEncoding:
    def test_encodes_nullable_arrow_chunks_in_first_seen_order(self):
        import pyarrow as pa
        from fastmob._core import encode_motif_purposes

        codes, home_code, unmatched_code = encode_motif_purposes(
            pa.chunked_array([["WORK", None, "HOME"], ["WORK", None]])
        )

        assert pa.array(codes).type == pa.uint16()
        assert pa.array(codes).to_pylist() == [0, 1, 2, 0, 1]
        assert home_code == 2
        assert unmatched_code == 3

    def test_accepts_polars_string_view_stream(self):
        import pyarrow as pa
        from fastmob._core import encode_motif_purposes

        pl = pytest.importorskip("polars", reason="Polars not installed")
        codes, home_code, unmatched_code = encode_motif_purposes(pl.Series(["HOME", "WORK", "HOME"]))

        assert pa.array(codes).to_pylist() == [0, 1, 0]
        assert home_code == 0
        assert unmatched_code == 2


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
        _dt, _lat, _lng, uid = _detect_trajectory_columns(nw_df)
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
    def test_pandas_string_uids_use_arrow_rust_path(self):
        import narwhals as nw
        from fastmob.measures._common import _build_indexed_user_ranges, _extract_timestamp_arrow

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
        timestamps = _extract_timestamp_arrow(nw_df, "datetime")

        uid_values, indices, ends = _build_indexed_user_ranges(
            nw_df,
            "uid",
            timestamps,
        )

        assert uid_values.to_pylist() == ["b", "a"]
        assert indices.to_pylist() == [2, 0, 3, 1]
        assert ends.to_pylist() == [2, 4]

    def test_polars_string_uids_use_first_seen_arrow_rust_path(self):
        import narwhals as nw
        from fastmob.measures._common import _build_indexed_user_ranges, _extract_timestamp_arrow

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
        timestamps = _extract_timestamp_arrow(nw_df, "datetime")

        uid_values, indices, ends = _build_indexed_user_ranges(
            nw_df,
            "uid",
            timestamps,
        )

        assert uid_values.to_pylist() == ["b", "a"]
        assert indices.to_pylist() == [2, 0, 3, 1]
        assert ends.to_pylist() == [2, 4]

    def test_equal_timestamps_keep_original_row_order(self):
        import narwhals as nw
        from fastmob.measures._common import _build_indexed_user_ranges, _extract_timestamp_arrow

        df = pd.DataFrame(
            {
                "uid": ["b", "a", "b", "a"],
                "datetime": pd.to_datetime(["2020-01-01 00:00:00"] * 4),
            }
        )
        nw_df = nw.from_native(df, eager_only=True)
        timestamps = _extract_timestamp_arrow(nw_df, "datetime")

        uid_values, indices, ends = _build_indexed_user_ranges(nw_df, "uid", timestamps)

        assert uid_values.to_pylist() == ["b", "a"]
        assert indices.to_pylist() == [0, 2, 1, 3]
        assert ends.to_pylist() == [2, 4]


class TestBuildIndexedUserRangesFast:
    def test_pandas_string_uids_use_first_seen_numpy_rust_path(self):
        import narwhals as nw
        from fastmob.measures._common import _build_indexed_user_ranges

        df = nw.from_native(
            pd.DataFrame({"uid": ["b", "a", "b", "c", "a", "c"]}),
            eager_only=True,
        )

        uid_values, indices, ends = _build_indexed_user_ranges(df, "uid")

        assert uid_values.to_pylist() == ["b", "a", "c"]
        assert indices.to_pylist() == [0, 2, 1, 4, 3, 5]
        assert ends.to_pylist() == [2, 4, 6]

    def test_polars_string_uids_use_first_seen_arrow_rust_path(self):
        import narwhals as nw
        from fastmob.measures._common import _build_indexed_user_ranges

        pl = pytest.importorskip("polars", reason="Polars not installed")
        df = nw.from_native(
            pl.DataFrame({"uid": ["b", "a", "b", "c", "a", "c"]}),
            eager_only=True,
        )

        uid_values, indices, ends = _build_indexed_user_ranges(df, "uid")

        assert uid_values.to_pylist() == ["b", "a", "c"]
        assert indices.to_pylist() == [0, 2, 1, 4, 3, 5]
        assert ends.to_pylist() == [2, 4, 6]


# ---------------------------------------------------------------------------
# _build_presorted_user_ends
# ---------------------------------------------------------------------------


class TestBuildPresortedUserEnds:
    def test_builds_ends_from_pandas_contiguous_uid_groups(self):
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pd.DataFrame({"uid": ["a", "a", "b", "b", "b", "c"]}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values.to_pylist() == ["a", "b", "c"]
        assert ends.to_pylist() == [2, 5, 6]

    def test_builds_ends_from_pandas_uint64_uid_groups(self):
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(
            pd.DataFrame({"uid": pd.Series([1, 1, 3, 3, 9], dtype="uint64")}),
            eager_only=True,
        )

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values.to_pylist() == [1, 3, 9]
        assert ends.to_pylist() == [2, 4, 5]

    def test_builds_empty_ends_for_empty_uid_dataframe(self):
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pd.DataFrame({"uid": []}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values.to_pylist() == []
        assert ends.to_pylist() == []

    def test_builds_single_end_without_uid_column(self):
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pd.DataFrame({"lat": [1.0, 2.0, 3.0]}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, None)

        assert uid_values is None
        assert ends.to_pylist() == [3]

    def test_builds_ends_from_polars_contiguous_uid_groups(self):
        pl = pytest.importorskip("polars", reason="Polars not installed")
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(pl.DataFrame({"uid": ["a", "a", "b", "b", "c"]}), eager_only=True)

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values.to_pylist() == ["a", "b", "c"]
        assert ends.to_pylist() == [2, 4, 5]

    def test_builds_ends_from_polars_uint64_uid_groups(self):
        pl = pytest.importorskip("polars", reason="Polars not installed")
        import narwhals as nw
        from fastmob.measures._common import _build_presorted_user_ends

        df = nw.from_native(
            pl.DataFrame({"uid": [1, 1, 3, 3, 9]}, schema={"uid": pl.UInt64}),
            eager_only=True,
        )

        uid_values, ends = _build_presorted_user_ends(df, "uid")

        assert uid_values.to_pylist() == [1, 3, 9]
        assert ends.to_pylist() == [2, 4, 5]
