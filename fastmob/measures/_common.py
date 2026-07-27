from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import narwhals as nw
import numpy as np

from fastmob.core.dispatch import TrajectoryDispatcher

_ROW_ORDER_COL = "__fastmob_row_order__"
_USER_RANGE_START_COL = "__fastmob_user_range_start__"

# ---------------------------------------------------------------------------
# Authoritative candidate lists for column auto-detection.
# All measure files should import from here rather than defining their own.
# ---------------------------------------------------------------------------

DATETIME_CANDIDATES: list[str] = ["datetime", "check-in_time", "timestamp", "time"]
LAT_CANDIDATES: list[str] = ["lat", "latitude"]
LNG_CANDIDATES: list[str] = ["lng", "lon", "longitude", "long"]
UID_CANDIDATES: list[str] = ["uid", "user", "user_id"]

ACTIVITY_CANDIDATES: list[str] = ["purpose", "activity", "act", "location_type"]
TIMESTAMP_CANDIDATES: list[str] = ["start_timestamp", "timestamp", "datetime"]
DAY_CANDIDATES: list[str] = ["day_of_week", "day", "weekday"]
USER_ID_CANDIDATES: list[str] = ["user_id", "uid", "agent_id", "user"]
LOCATION_CANDIDATES: list[str] = ["location_id", "area", "venueId", "location"]
DURATION_CANDIDATES: list[str] = ["duration_steps", "duration_minutes", "duration"]
PURPOSE_CANDIDATES: list[str] = ["purpose", "activity", "location_type"]
LOCATION_TYPE_CANDIDATES: list[str] = ["location_type", "purpose", "activity"]
ORIGIN_CANDIDATES: list[str] = ["origin_area", "Origin_Area", "area_o", "ORIGIN_AREA"]
DEST_CANDIDATES: list[str] = [
    "destination_area",
    "Dest_Area",
    "area_d",
    "DESTINATION_AREA",
]


def _shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy in bits for a list of event counts.

    Parameters
    ----------
    counts:
        A list of non-negative integer counts (e.g. visit counts per location).
        Zero-count items are ignored (contribute 0 to entropy, matching the
        information-theoretic convention ``0 * log2(0) = 0``).

    Returns
    -------
    float
        Shannon entropy in bits.  Returns 0.0 when the total count is 0 or
        when all probability mass is concentrated on a single item.

    Examples
    --------
    >>> _shannon_entropy([1, 1])   # two equal-probability items -> 1 bit
    1.0
    >>> _shannon_entropy([1, 0])   # one item -> 0 bits
    0.0
    >>> _shannon_entropy([])
    0.0
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _pick_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in ``columns``, or None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def is_arrow_backed(nw_df: nw.DataFrame) -> bool:
    """Return True when a Narwhals DataFrame is backed by an Arrow object."""
    native = nw_df.to_native()
    return hasattr(native, "to_arrow") and hasattr(native, "dtypes")


def _is_polars_backed(nw_df: nw.DataFrame) -> bool:
    """Return True when a Narwhals DataFrame is backed by a Polars object."""
    implementation = getattr(nw_df, "implementation", None)
    if implementation is not None and hasattr(implementation, "is_polars"):
        return bool(implementation.is_polars())

    native = nw_df.to_native()
    return hasattr(native, "lazy")


def _use_arrow_kernel_path(nw_df: nw.DataFrame) -> bool:
    """Return True when Rust kernels should consume Arrow buffers directly.

    Pandas-backed Narwhals series expose ``to_arrow()``, but that converts
    pandas/NumPy storage into Arrow buffers and can add large temporary
    allocations.  Use Arrow kernels only for backends whose native storage is
    already Arrow-oriented.
    """
    implementation = getattr(nw_df, "implementation", None)
    if implementation is not None:
        if hasattr(implementation, "is_polars") and implementation.is_polars():
            return True
        if hasattr(implementation, "is_pyarrow") and implementation.is_pyarrow():
            return True
    return False


def _is_pandas_backed(nw_df: nw.DataFrame) -> bool:
    """Return True when a Narwhals DataFrame is backed by pandas."""
    native = nw_df.to_native()
    return hasattr(native, "iloc") and hasattr(native, "dtypes")


def _is_pyarrow_backed(nw_df: nw.DataFrame) -> bool:
    """Return True when a Narwhals DataFrame is backed by PyArrow."""
    implementation = getattr(nw_df, "implementation", None)
    if implementation is not None and getattr(implementation, "value", None) == "pyarrow":
        return True

    native = nw_df.to_native()
    return hasattr(native, "column") and hasattr(native, "schema")


def _empty_like(nw_df: nw.DataFrame, columns: list[str]) -> Any:
    """Build an empty native DataFrame using the same backend as ``nw_df``."""
    return nw.from_dict({col: [] for col in columns}, backend=nw_df.implementation).to_native()


def _dispatch_kernel(
    numpy_kernel: Callable,
    arrow_kernel: Callable,
    series: Iterable[nw.Series],
    *extra_args,
    use_arrow: bool,
    convert_arrow_result: bool = True,
) -> Any:
    """Call the NumPy or Arrow kernel after converting input series.

    When ``use_arrow`` is True the input series are converted via ``to_arrow``
    and the (possibly pyo3-arrow-wrapped) result is unwrapped through
    :func:`_arrow_result_values` unless ``convert_arrow_result=False`` (used
    when the result is fed back into another Arrow kernel).
    """
    if use_arrow:
        result = arrow_kernel(*(s.to_arrow() for s in series), *extra_args)
        return _arrow_result_values(result) if convert_arrow_result else result
    return numpy_kernel(*(s.to_numpy() for s in series), *extra_args)


def _dispatch_pair_kernel(
    numpy_kernel: Callable,
    arrow_kernel: Callable,
    series: Iterable[nw.Series],
    *extra_args,
    use_arrow: bool,
    convert_arrow_result: bool = True,
) -> tuple[Any, Any]:
    """Like :func:`_dispatch_kernel` but for kernels returning a 2-tuple."""
    if use_arrow:
        first, second = arrow_kernel(*(s.to_arrow() for s in series), *extra_args)
        if convert_arrow_result:
            return _arrow_result_values(first), _arrow_result_values(second)
        return first, second
    return numpy_kernel(*(s.to_numpy() for s in series), *extra_args)


def _with_datetime_column(df: nw.DataFrame, column: str) -> nw.DataFrame:
    """Ensure a trajectory datetime column has a Narwhals datetime dtype."""
    try:
        return df.with_columns(nw.col(column).cast(nw.Datetime).alias(column))
    except Exception:  # noqa: BLE001
        return df.with_columns(nw.col(column).str.to_datetime().alias(column))


def _as_index_array(values: Any) -> np.ndarray:
    """Return unsigned pointer-sized indexes for Rust indexed kernels."""
    return np.asarray(values, dtype=np.uintp)


def _ranges_to_starts_ends(
    ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Python range tuples to NumPy start/end arrays."""
    starts = np.fromiter((start for start, _ in ranges), dtype=np.uintp, count=len(ranges))
    ends = np.fromiter((end for _, end in ranges), dtype=np.uintp, count=len(ranges))
    return starts, ends


def _ranges_to_ends(ranges: list[tuple[int, int]]) -> np.ndarray:
    """Convert Python range tuples to a cumulative end-boundary array."""
    return np.fromiter((end for _, end in ranges), dtype=np.uintp, count=len(ranges))


def _starts_from_ends(ends: Any) -> np.ndarray:
    """Derive range starts from a cumulative end-boundary array."""
    ends = np.asarray(ends, dtype=np.uintp)
    starts = np.empty_like(ends)
    if len(ends) == 0:
        return starts
    starts[0] = 0
    starts[1:] = ends[:-1]
    return starts


def _ranges_from_ends(ends: Any) -> list[tuple[int, int]]:
    """Convert cumulative end boundaries to Python half-open ranges."""
    starts = _starts_from_ends(ends)
    ends = np.asarray(ends, dtype=np.uintp)
    return list(zip(starts.tolist(), ends.tolist()))


def _arrow_result_values(values: Any) -> Any:
    """Return a PyArrow value object when a pyo3-arrow wrapper is returned."""
    if hasattr(values, "to_pyarrow"):
        return values.to_pyarrow()
    return values


def _filter_result_values(values: Any, keep: Any, *, dtype: Any = None) -> Any:
    """Filter NumPy or Arrow-like result values with a boolean keep mask."""
    values = _arrow_result_values(values)
    if hasattr(values, "__arrow_c_array__"):
        import pyarrow as pa
        import pyarrow.compute as pc

        return pc.filter(pa.array(values), pa.array(keep))
    return np.asarray(values, dtype=dtype)[keep]


def _result_scalar(values: Any) -> float:
    """Extract a single Python float from a length-1 NumPy/Arrow result."""
    if hasattr(values, "to_numpy"):
        return float(values.to_numpy()[0])
    return float(np.asarray(values)[0])


def _to_native(values_dict: dict[str, Any], df: nw.DataFrame) -> Any:
    """Build a backend-matching result dataframe from a column dict."""
    columns = {}
    for name, values in values_dict.items():
        try:
            is_list_array = str(values.type).startswith("list<")
        except Exception:  # noqa: BLE001
            is_list_array = False
        if hasattr(values, "__arrow_c_array__") and hasattr(values, "to_numpy") and not is_list_array:
            try:
                values = values.to_numpy(zero_copy_only=False)
            except TypeError:
                values = values.to_numpy()
        columns[name] = values
    return nw.from_dict(columns, backend=df.implementation).to_native()


def _take_uid_values(uid_values: list | None, user_indices: Any) -> Any:
    """Return one UID label per flat output row using vectorized positional take."""
    if uid_values is None:
        return None
    return np.asarray(uid_values, dtype=object).take(np.asarray(user_indices, dtype=np.uintp))


def _indexed_group_indices(uids: Any, num_groups: int) -> Any:
    from fastmob._core import indexed_user_indices

    return indexed_user_indices(uids, num_groups)


def _time_ordered_user_indices_from_ndarray(uids: Any, timestamps: Any, num_groups: int | None = None) -> Any:
    from fastmob._core import time_ordered_user_indices

    if uids is None:
        return time_ordered_user_indices(None, timestamps)
    return time_ordered_user_indices(uids, timestamps, num_groups)


def _time_ordered_user_indices_from_c_array(uids: Any, timestamps: Any, num_groups: int | None = None) -> Any:
    from fastmob._core import time_ordered_user_indices

    if uids is None:
        return time_ordered_user_indices(None, timestamps)
    return time_ordered_user_indices(uids, timestamps, num_groups)


def _presorted_user_starts_ends_numpy(uids: Any) -> tuple[Any, Any]:
    from fastmob._core import presorted_user_starts_ends_numpy

    return presorted_user_starts_ends_numpy(uids)


def _presorted_user_starts_ends_arrow(uids: Any) -> tuple[Any, Any]:
    from fastmob._core import presorted_user_starts_ends_arrow

    return presorted_user_starts_ends_arrow(uids)


_TIME_ORDERED_USER_RANGES_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={"time_ordered_indices": _time_ordered_user_indices_from_c_array},
    numpy_ops={"time_ordered_indices": _time_ordered_user_indices_from_ndarray},
)

_PRESORTED_USER_ENDS_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={"starts_ends": _presorted_user_starts_ends_arrow},
    numpy_ops={"starts_ends": _presorted_user_starts_ends_numpy},
)


def _uint64_series(df: nw.DataFrame, values: Any) -> nw.Series:
    return nw.new_series(
        "__fastmob_uid_codes__",
        values,
        dtype=nw.UInt64,
        backend=df.implementation,
    )


def _factorize_numpy_values_uint64(values: Any, *, sort: bool) -> tuple[np.ndarray, int]:
    import pandas as pd

    codes, uniques = pd.factorize(values, sort=sort, use_na_sentinel=False)
    return np.asarray(codes, dtype=np.uint64), len(uniques)


def _factorize_polars_uids_uint64(df: nw.DataFrame, uid_col: str, *, sort: bool) -> tuple[Any, int]:
    import polars as pl

    native = df.to_native()
    unique_values = native.get_column(uid_col).unique(maintain_order=not sort)
    if sort:
        unique_values = unique_values.sort(nulls_last=True)
        replacement_codes = pl.Series(
            "__fastmob_uid_codes__",
            np.arange(len(unique_values), dtype=np.uint64),
        )
        codes = native.select(
            pl.col(uid_col)
            .replace_strict(
                unique_values,
                replacement_codes,
                return_dtype=pl.UInt64,
            )
            .alias("__code__")
        ).get_column("__code__")
        return codes, len(unique_values)

    raw_codes = native.select(
        pl.col(uid_col).cast(pl.Utf8).cast(pl.Categorical).to_physical().alias("__code__")
    ).get_column("__code__")
    max_code = raw_codes.max()
    fill_value = 0 if max_code is None else max_code + 1
    return raw_codes.fill_null(fill_value).cast(pl.UInt64), len(unique_values)


def _factorize_pyarrow_uids_uint64(df: nw.DataFrame, uid_col: str, *, sort: bool) -> tuple[Any, int]:
    import pyarrow as pa
    import pyarrow.compute as pc

    values = df.get_column(uid_col).to_arrow()
    if sort:
        unique_values = pc.unique(values)
        sorted_values = pc.take(unique_values, pc.sort_indices(unique_values))
        codes = pc.index_in(values, value_set=sorted_values)
        num_groups = len(sorted_values)
    else:
        encoded = pc.dictionary_encode(values)
        codes = pc.fill_null(
            encoded.indices,
            pa.scalar(len(encoded.dictionary), type=encoded.indices.type),
        )
        num_groups = len(encoded.dictionary) + int(encoded.indices.null_count > 0)
    return pc.cast(codes, pa.uint64()), num_groups


def _factorize_uids_uint64(
    df: nw.DataFrame,
    uid_col: str | None,
    *,
    sort: bool = False,
) -> tuple[nw.Series, int] | None:
    """Return dense UInt64 UID codes and group count while preserving original UID labels separately."""
    if uid_col is None:
        return None

    if _is_pandas_backed(df):
        native = df.to_native()
        codes, num_groups = _factorize_numpy_values_uint64(native[uid_col], sort=sort)
        return _uint64_series(df, codes), num_groups

    if _is_polars_backed(df):
        codes, num_groups = _factorize_polars_uids_uint64(df, uid_col, sort=sort)
        return _uint64_series(df, codes), num_groups

    if _is_pyarrow_backed(df):
        codes, num_groups = _factorize_pyarrow_uids_uint64(df, uid_col, sort=sort)
        return _uint64_series(df, codes), num_groups

    codes, num_groups = _factorize_numpy_values_uint64(df.get_column(uid_col).to_numpy(), sort=sort)
    return _uint64_series(df, codes), num_groups


def _extract_timestamps_ms(df: nw.DataFrame, datetime_col: str) -> nw.Series:
    """Return a Float64 Narwhals Series of millisecond Unix timestamps."""
    return df.with_columns(nw.col(datetime_col).dt.timestamp("ms").cast(nw.Float64).alias("__ts_ms__")).get_column(
        "__ts_ms__"
    )


def _extract_timestamps_s(df: nw.DataFrame, datetime_col: str) -> nw.Series:
    """Return a Float64 Narwhals Series of second-resolution Unix timestamps.

    Goes through the millisecond path so backends with nanosecond or
    microsecond storage produce identical values.
    """
    return df.with_columns((nw.col(datetime_col).dt.timestamp("ms") / 1000.0).alias("__ts_s__")).get_column("__ts_s__")


def _timestamps_s_to_datetime_ns(values: Any) -> np.ndarray:
    """Convert flat seconds-since-epoch values (NumPy or Arrow) to a naive ``datetime64[ns]`` NumPy array.

    Inverse of :func:`_extract_timestamps_s`. No existing measure needs to
    re-emit an expanded/reconstructed datetime column, so this is new;
    mirrors ``fastmob/preprocessing/_stay_locations.py``'s private
    ``_seconds_to_naive_utc`` helper.
    """
    values = _arrow_result_values(values)
    if hasattr(values, "to_numpy"):
        try:
            values = values.to_numpy(zero_copy_only=False)
        except TypeError:
            values = values.to_numpy()
    return (np.asarray(values, dtype="float64") * 1e9).astype("int64").view("datetime64[ns]")


def _extract_hours(df: nw.DataFrame, datetime_col: str) -> tuple[nw.DataFrame, nw.Series]:
    """Add and return an ``__hour__`` Float64 column derived from datetime."""
    df = df.with_columns(nw.col(datetime_col).dt.hour().cast(nw.Float64).alias("__hour__"))
    return df, df.get_column("__hour__")


def _uid_values_from_index_ranges(
    uids: nw.Series,
    indices: Any,
    ends: Any,
    *,
    use_arrow: bool,
) -> list:
    """Extract one UID label per indexed range without scanning all rows in Python."""
    starts = _starts_from_ends(ends)
    first_row_indices = np.asarray(indices, dtype=np.uintp)[np.asarray(starts, dtype=np.uintp)]
    if use_arrow:
        import pyarrow.compute as pc

        uid_arrow = uids.to_arrow()
        return pc.take(uid_arrow, first_row_indices).to_pylist()

    uid_values = uids.to_numpy()
    return uid_values[first_row_indices].tolist()


def _build_time_ordered_user_ranges(
    df: nw.DataFrame,
    uid_col: str | None,
    datetime_col: str,
    timestamps_data: Any,
    *,
    row_index_col: str = "__fastmob_time_order_row_index__",
) -> tuple[list | None, Any, np.ndarray]:
    """Build stable time-ordered indexes/ranges without sorting the full dataframe."""
    ops = _TIME_ORDERED_USER_RANGES_DISPATCHER.get_ops(df)
    use_arrow = _TIME_ORDERED_USER_RANGES_DISPATCHER.get_backend_key(df) == "arrow"

    if uid_col is None:
        indices, ends = ops["time_ordered_indices"](None, timestamps_data)
        return None, _as_index_array(indices), _as_index_array(ends)

    uids = df.get_column(uid_col)
    uid_codes, num_groups = _factorize_uids_uint64(df, uid_col, sort=False)
    try:
        indices, ends = ops["time_ordered_indices"](ops["extract_data"](uid_codes), timestamps_data, num_groups)
        indices = _as_index_array(indices)
        ends = _as_index_array(ends)
        uid_values = _uid_values_from_index_ranges(uids, indices, ends, use_arrow=use_arrow)
        return uid_values, indices, ends
    except ValueError as exc:
        if "unsupported" not in str(exc):
            raise

    index_df = (
        df.select([uid_col, datetime_col])
        .with_columns(uid_codes)
        .with_row_index(row_index_col)
        .sort("__fastmob_uid_codes__", datetime_col, row_index_col)
    )
    uid_values, ranges = _build_user_ranges(index_df, uid_col)
    indices = [int(idx) for idx in index_df.get_column(row_index_col).to_list()]
    ends = _ranges_to_ends(ranges)
    return uid_values, _as_index_array(indices), ends


def _build_indexed_user_ranges_fast(
    df: nw.DataFrame,
    uid_col: str | None,
) -> tuple[list | None, Any, np.ndarray]:
    """Build grouped row indexes using Rust for supported UID dtypes."""
    use_arrow = _is_polars_backed(df) or _is_pyarrow_backed(df)

    if uid_col is None:
        indices = np.arange(len(df), dtype=np.uintp)
        ends = np.array([] if len(df) == 0 else [len(df)], dtype=np.uintp)
        return None, indices, ends

    uids = df.get_column(uid_col)
    uid_codes, num_groups = _factorize_uids_uint64(df, uid_col, sort=False)
    try:
        uid_code_data = uid_codes.to_arrow() if use_arrow else uid_codes.to_numpy()
        indices, ends = _indexed_group_indices(uid_code_data, num_groups)
        uid_values = _uid_values_from_index_ranges(uids, indices, ends, use_arrow=use_arrow)
        return uid_values, _as_index_array(indices), _as_index_array(ends)
    except ValueError as exc:
        if "unsupported" not in str(exc):
            raise

    uid_values, indices, ranges = _build_indexed_user_ranges(df, uid_col)
    ends = _ranges_to_ends(ranges)
    return uid_values, _as_index_array(indices), ends


def _detect_trajectory_columns(
    nw_df: nw.DataFrame,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    *,
    cast_float_coordinates: bool = False,
) -> Any:
    """Auto-detect trajectory column names from a Narwhals DataFrame.

    Accepts explicit overrides for any column; auto-detects the rest from the
    authoritative candidate lists defined in this module.

    Parameters
    ----------
    nw_df:
        A Narwhals DataFrame (already wrapped via ``nw.from_native``).
    datetime_col:
        Explicit datetime column name; auto-detected when None.
    lat_col:
        Explicit latitude column name; auto-detected when None.
    lng_col:
        Explicit longitude column name; auto-detected when None.
    uid_col:
        Explicit user-ID column name; auto-detected when None. When no
        user-ID column is found the whole frame is treated as one user.
    cast_float_coordinates:
        When True, cast latitude and longitude columns to Float64 and return
        the possibly updated DataFrame before the column names.

    Returns
    -------
    tuple
        By default, ``(datetime_col, lat_col, lng_col, uid_col)`` where
        ``uid_col`` may be None if no user-ID column exists. With
        ``cast_float_coordinates=True``, returns
        ``(nw_df, datetime_col, lat_col, lng_col, uid_col)``.

    Raises
    ------
    ValueError
        When a required column (datetime, lat, or lng) cannot be found.

    Examples
    --------
    >>> import pandas as pd, narwhals as nw
    >>> df = pd.DataFrame({"datetime": [], "lat": [], "lng": [], "uid": []})
    >>> nw_df = nw.from_native(df, eager_only=True)
    >>> _detect_trajectory_columns(nw_df)
    ('datetime', 'lat', 'lng', 'uid')
    """
    columns = nw_df.columns

    if datetime_col is None:
        datetime_col = _pick_existing_column(columns, DATETIME_CANDIDATES)
    if lat_col is None:
        lat_col = _pick_existing_column(columns, LAT_CANDIDATES)
    if lng_col is None:
        lng_col = _pick_existing_column(columns, LNG_CANDIDATES)
    if uid_col is None:
        uid_col = _pick_existing_column(columns, UID_CANDIDATES)

    missing = [
        name
        for name, col in zip(
            ["datetime", "latitude", "longitude"],
            [datetime_col, lat_col, lng_col],
        )
        if col is None
    ]
    if missing:
        hints = {
            "datetime": DATETIME_CANDIDATES,
            "latitude": LAT_CANDIDATES,
            "longitude": LNG_CANDIDATES,
        }
        details = "; ".join(f"'{m}' (looked for: {hints[m]})" for m in missing)
        raise ValueError(
            f"Could not find required column(s): {details}. "
            f"Available columns: {columns}. "
            f"Pass the column name(s) explicitly."
        )

    if cast_float_coordinates:
        schema = nw_df.schema
        if schema[lat_col] != nw.Float64 or schema[lng_col] != nw.Float64:
            nw_df = nw_df.with_columns(
                nw.col(lat_col).cast(nw.Float64),
                nw.col(lng_col).cast(nw.Float64),
            )
        return nw_df, datetime_col, lat_col, lng_col, uid_col

    return datetime_col, lat_col, lng_col, uid_col


def _prepare_trajectory(
    nw_df: nw.DataFrame,
    *,
    datetime_col: str,
    lat_col: str,
    lng_col: str,
    uid_col: str | None = None,
    sort: bool = True,
    drop_nulls: bool = True,
) -> nw.DataFrame:
    """Optionally sort and cast a trajectory into a clean DataFrame.

    This is the standard preprocessing pipeline shared by all trajectory-based
    measures (jump lengths, radius of gyration, etc.).  It:

    1. Drops nulls in the required coordinate/datetime columns.
    2. Optionally sorts by ``[uid, datetime]`` (with a stable row-order
       tiebreaker) to ensure chronological order within each user.
    3. Casts lat/lng to ``Float64``.

    Parameters
    ----------
    nw_df:
        Narwhals trajectory dataframe.
    datetime_col, lat_col, lng_col, uid_col:
        Resolved column names. Use ``_detect_trajectory_columns`` before
        calling this function.
    sort:
        Whether to sort by user and datetime. When False, rows keep their
        input order after null rows are dropped.
    drop_nulls:
        Whether to drop rows with null datetime/lat/lng values before returning.

    Returns
    -------
    nw.DataFrame
        The cleaned Narwhals DataFrame.

    Examples
    --------
    >>> import pandas as pd, narwhals as nw
    >>> df = pd.DataFrame({
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ...     "uid": ["u1", "u1", "u1"],
    ... })
    >>> nw_df = nw.from_native(df, eager_only=True)
    >>> clean_df = _prepare_trajectory(nw_df, datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid")
    """
    if sort:
        nw_df = nw_df.with_row_index(_ROW_ORDER_COL)

    nw_df = _with_datetime_column(nw_df, datetime_col)

    if drop_nulls:
        nw_df = nw_df.drop_nulls(subset=[datetime_col, lat_col, lng_col])
    if sort:
        sort_cols = [uid_col, datetime_col, _ROW_ORDER_COL] if uid_col else [datetime_col, _ROW_ORDER_COL]
        nw_df = nw_df.sort(*sort_cols)

    nw_df = nw_df.with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )
    if sort:
        nw_df = nw_df.drop(_ROW_ORDER_COL)

    return nw_df


def _build_user_ranges(df: nw.DataFrame, uid_col: str | None) -> tuple[list, list[tuple[int, int]]]:
    """Split a uid-sorted DataFrame into per-user (uid_value, index_range) pairs.

    Returns
    -------
    tuple[list, list[tuple[int, int]]]
        ``(uid_values, ranges)`` where ``ranges[i]`` is the half-open row
        interval ``[start, end)`` for ``uid_values[i]``.  When ``uid_col`` is
        None the whole frame is treated as one user and returns
        ``([None], [(0, len(df))])``.
    """
    n = len(df)
    if uid_col is None:
        return [None], [(0, n)]
    if n == 0:
        return [], []

    starts_df = (
        df.select([uid_col])
        .with_row_index(_USER_RANGE_START_COL)
        .filter((nw.col(uid_col) != nw.col(uid_col).shift(1)).fill_null(True))
    )
    starts = starts_df.get_column(_USER_RANGE_START_COL).to_list()
    uid_values = starts_df.get_column(uid_col).to_list()
    ends = starts[1:] + [n]
    ranges = list(zip(starts, ends))

    return uid_values, ranges


def _build_presorted_user_ends(df: nw.DataFrame, uid_col: str | None) -> tuple[list | None, np.ndarray]:
    """Build contiguous user group end indices for data already grouped by user."""
    n = len(df)
    if uid_col is None:
        return None, np.array([n], dtype=np.uintp)

    if n == 0:
        return [], np.array([], dtype=np.uintp)

    try:
        ops = _PRESORTED_USER_ENDS_DISPATCHER.get_ops(df)
        uid_series = df.get_column(uid_col)
        starts, ends = ops["starts_ends"](ops["extract_data"](uid_series))
        starts = np.asarray(starts, dtype=np.uintp)
        ends = np.asarray(ends, dtype=np.uintp)
        uid_values = uid_series.to_numpy()[starts].tolist()
        return uid_values, ends
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        pass

    if _is_pandas_backed(df):
        native = df.to_native()
        uid_series = native[uid_col]
        boundaries = uid_series.ne(uid_series.shift(1)).fillna(True).to_numpy(dtype=bool, copy=False)
        starts_array = np.flatnonzero(boundaries)
        ends = np.empty(len(starts_array), dtype=np.uintp)
        if len(starts_array) > 1:
            ends[:-1] = starts_array[1:]
        ends[-1] = n
        uid_values = uid_series.iloc[starts_array].tolist()
        return uid_values, ends

    starts_df = (
        df.select([uid_col])
        .with_row_index(_USER_RANGE_START_COL)
        .filter((nw.col(uid_col) != nw.col(uid_col).shift(1)).fill_null(True))
    )
    starts = starts_df.get_column(_USER_RANGE_START_COL).to_list()
    uid_values = starts_df.get_column(uid_col).to_list()
    ends = np.asarray(starts[1:] + [n], dtype=np.uintp)
    return uid_values, ends


def _build_presorted_user_ranges(df: nw.DataFrame, uid_col: str | None) -> tuple[list | None, list[tuple[int, int]]]:
    """Build contiguous user ranges for data already grouped by user."""
    uid_values, ends = _build_presorted_user_ends(df, uid_col)
    return uid_values, _ranges_from_ends(ends)


def _build_indexed_user_ranges(
    df: nw.DataFrame,
    uid_col: str,
    *,
    row_index_col: str = "__fastmob_indexed_row_index__",
) -> tuple[list, list[int], list[tuple[int, int]]]:
    """Build stable row indices and user ranges without reordering the input frame.

    This is the reusable version of the radius-of-gyration indexed grouping
    pattern.  It sorts a narrow ``uid + row_index`` projection, preserving
    stable row-order ties, then returns indices into the original frame plus
    contiguous ranges over that index vector.
    """
    n = len(df)
    if n == 0:
        return [], [], []

    index_df = df.select([uid_col]).with_row_index(row_index_col).sort(uid_col, row_index_col)
    uid_values, ranges = _build_user_ranges(index_df, uid_col)
    indices = [int(idx) for idx in index_df.get_column(row_index_col).to_list()]
    return uid_values, indices, ranges


def _value_offsets_from_index_ranges(
    index_starts: Any,
    index_ends: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert row-index ranges to value-array offsets for non-ordered results.

    For a group of n rows, the kernel produces n-1 jump values.  This converts
    the per-user row-index spans into half-open slices into the flat output array.
    """
    starts = np.asarray(index_starts, dtype=np.uintp)
    ends = np.asarray(index_ends, dtype=np.uintp)
    lengths = np.maximum(ends - starts, 1) - 1
    value_ends = np.cumsum(lengths, dtype=np.uintp)
    value_starts = value_ends - lengths
    return value_starts, value_ends


def _grouped_numpy_values(
    starts: Any,
    ends: Any,
    values: Any,
    *,
    value_offsets: bool = False,
) -> list[np.ndarray]:
    """Group a flat NumPy values array into per-user sub-arrays.

    When ``value_offsets=True``, *starts*/*ends* are already value-space offsets.
    When ``False``, they are row-index ranges converted via
    :func:`_value_offsets_from_index_ranges`.
    """
    if value_offsets:
        starts = np.asarray(starts, dtype=np.uintp)
        ends = np.asarray(ends, dtype=np.uintp)
    else:
        starts, ends = _value_offsets_from_index_ranges(starts, ends)
    values = np.asarray(values, dtype=np.float64)
    return [values[int(s) : int(e)] for s, e in zip(starts, ends)]


def _arrow_flat_result_values(values: Any) -> Any:
    """Unwrap a pyo3-arrow wrapper and materialise into a concrete ``pa.Array``."""
    values = _arrow_result_values(values)
    if hasattr(values, "__arrow_c_array__"):
        import pyarrow as pa

        return pa.array(values)
    return values


def _grouped_arrow_values(
    starts: Any,
    ends: Any,
    values: Any,
    *,
    value_offsets: bool = False,
) -> Any:
    """Group a flat Arrow values array into a ``pa.ListArray`` of per-user sub-arrays."""
    import pyarrow as pa

    if value_offsets:
        starts = np.asarray(starts, dtype=np.uintp)
        ends = np.asarray(ends, dtype=np.uintp)
    else:
        starts, ends = _value_offsets_from_index_ranges(starts, ends)
    offsets = np.empty(len(starts) + 1, dtype=np.int32)
    offsets[:-1] = starts
    offsets[-1] = ends[-1] if len(ends) else 0
    return pa.ListArray.from_arrays(offsets, _arrow_flat_result_values(values))
