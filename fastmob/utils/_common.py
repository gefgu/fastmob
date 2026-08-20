from __future__ import annotations

import os
import time
from collections.abc import Iterable
from importlib import import_module
from typing import Any

import narwhals as nw
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

_ROW_ORDER_COL = "__fastmob_row_order__"


def require_optional(module_name: str, extra: str):
    """Import an optional dependency or give its fastmob installation hint."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"{module_name} is required: pip install fastmob[{extra}]") from exc


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


def _pick_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in ``columns``, or None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _detect_required_column(
    df: nw.DataFrame,
    explicit: str | None,
    candidates: list[str],
) -> str:
    """Return an explicit or auto-detected column, raising when none is available."""
    column = explicit or _pick_existing_column(df.columns, candidates)
    if column is None:
        raise ValueError(
            f"Could not detect a required column. Tried: {candidates}. "
            f"Available columns: {df.columns}. Pass the column name explicitly."
        )
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found. Available columns: {df.columns}.")
    return column


def _empty_like(nw_df: nw.DataFrame, columns: list[str]) -> Any:
    """Build an empty native DataFrame using the same backend as ``nw_df``."""
    return nw.from_dict({col: [] for col in columns}, backend=nw_df.implementation).to_native()


def _with_datetime_column(df: nw.DataFrame, column: str) -> nw.DataFrame:
    """Ensure a trajectory datetime column has a Narwhals datetime dtype."""
    try:
        return df.with_columns(nw.col(column).cast(nw.Datetime).alias(column))
    except Exception:  # noqa: BLE001
        return df.with_columns(nw.col(column).str.to_datetime().alias(column))


def _strip_time_zone(df: nw.DataFrame, column: str) -> nw.DataFrame:
    """Drop timezone metadata while preserving each timestamp's local clock time."""
    dtype = df.schema[column]
    if isinstance(dtype, nw.Datetime) and dtype.time_zone is not None:
        return df.with_columns(nw.col(column).dt.replace_time_zone(None))
    return df


def _as_arrow(values: Any) -> pa.Array | pa.ChunkedArray:
    """Return a concrete Arrow array from a Narwhals, pyo3-arrow, or Arrow value.

    This is the only conversion used at Python/Rust boundaries.  It deliberately
    accepts pandas-backed Narwhals series too: the project uses Arrow buffers
    consistently for kernel inputs, regardless of the caller's dataframe
    backend.
    """
    if hasattr(values, "to_arrow"):
        values = values.to_arrow()
    if hasattr(values, "to_pyarrow"):
        values = values.to_pyarrow()
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return values
    return pa.array(values)


def _finite_arrow_array(values: Any) -> pa.Array | pa.ChunkedArray:
    """Cast to a float64 Arrow array and drop non-finite entries."""
    arr = pc.cast(_as_arrow(values), pa.float64())
    return pc.filter(arr, pc.is_finite(arr))


def _values_to_list(values: Any) -> list[Any]:
    """Materialize Arrow/NumPy/pandas-like values as a Python list."""
    values = _as_arrow(values)
    if hasattr(values, "to_pylist"):
        return values.to_pylist()
    if hasattr(values, "to_list"):
        return values.to_list()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _null_sentinel_to_none(values: list[Any], sentinel: Any) -> list[Any]:
    """Convert sentinel values back to None in a materialized Python list."""
    return [None if value == sentinel else value for value in values]


def _list_column_from_offsets(flat_values: Any, offsets: Any) -> list[list[Any]]:
    """Build list-column values from a flat values buffer and offset buffer."""
    flat_list = _values_to_list(flat_values)
    offset_list = _values_to_list(offsets)
    return [flat_list[start:end] for start, end in zip(offset_list, offset_list[1:])]


def _filter_result_values(values: Any, keep: Any, *, dtype: Any = None) -> Any:
    """Filter NumPy or Arrow-like result values with a boolean keep mask."""
    values = _as_arrow(values)
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


def _narwhals_safe_value(values: Any) -> Any:
    """Downgrade a raw (non-list) Arrow array to NumPy so narwhals can build any backend from it.

    ``nw.from_dict`` doesn't accept a bare ``pyarrow.Array`` as a column value for every
    backend (e.g. pandas) -- only list-typed Arrow arrays (grouped per-user results) are passed
    through untouched, since those are already handled as list columns downstream.
    """
    try:
        is_list_array = str(values.type).startswith("list<")
    except Exception:  # noqa: BLE001
        is_list_array = False
    if hasattr(values, "__arrow_c_array__") and hasattr(values, "to_numpy") and not is_list_array:
        try:
            return values.to_numpy(zero_copy_only=False)
        except TypeError:
            return values.to_numpy()
    return values


def _to_native(values_dict: dict[str, Any], df: nw.DataFrame) -> Any:
    """Build a backend-matching result dataframe from a column dict."""
    arrays = {name: _as_arrow(values) for name, values in values_dict.items()}
    return nw.from_arrow(pa.table(arrays), backend=df.implementation).to_native()


def _take_uid_values(uid_values: Any | None, user_indices: Any) -> Any:
    """Return one UID label per flat output row using vectorized positional take."""
    if uid_values is None:
        return None
    return pc.take(_as_arrow(uid_values), _as_arrow(user_indices))


def _uint32_series(df: nw.DataFrame, values: Any) -> nw.Series:
    return nw.from_arrow(
        pa.table({"__fastmob_uid_codes__": pa.array(_as_arrow(values), type=pa.uint32())}),
        backend=df.implementation,
    ).get_column("__fastmob_uid_codes__")


def _factorize_arrow_values(values: Any, *, sort: bool) -> tuple[Any, Any]:
    """Dense-code an Arrow array's values.

    The Rust kernel (``fastmob-py/src/measures/individual/factorization.rs``)
    automatically dispatches to its parallel implementation once the input
    clears an internal row-count threshold -- there is no caller-facing
    parallel switch to set here.
    """
    from fastmob._core import factorize_arrow

    values = _as_arrow(values)
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    raw_codes, raw_representatives = factorize_arrow(values, sort)
    return _as_arrow(raw_codes), _as_arrow(raw_representatives)


def _joint_factorize_arrow_values(*values: Any) -> tuple[pa.Array, ...]:
    """Factorize Arrow-compatible columns with one shared codebook.

    Nulls retain their validity bitmap so callers can pass the resulting codes
    directly to a kernel that skips missing values.
    """
    arrays = []
    for value in values:
        array = _as_arrow(value)
        arrays.append(array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array)
    if len({array.type for array in arrays}) != 1:
        raise ValueError("jointly factorized columns must use the same Arrow data type")
    codes, _ = _factorize_arrow_values(pa.concat_arrays(arrays), sort=False)
    offset = 0
    results = []
    for array in arrays:
        encoded = codes.slice(offset, len(array))
        results.append(pc.if_else(pc.is_valid(array), encoded, pa.scalar(None, type=pa.uint32())))
        offset += len(array)
    return tuple(results)


def _factorize_uids_uint32(
    df: nw.DataFrame,
    uid_col: str | None,
    *,
    sort: bool = False,
) -> tuple[nw.Series, int] | None:
    """Return dense UInt32 UID codes and group count while preserving original UID labels separately."""
    if uid_col is None:
        return None

    codes, representatives = _factorize_arrow_values(df.get_column(uid_col), sort=sort)
    return _uint32_series(df, codes), len(representatives)


_NULL_TIMESTAMP_SENTINEL_MS = -(2**63)  # i64::MIN; never a real Unix-ms timestamp


def _extract_timestamps(df: nw.DataFrame, datetime_col: str) -> nw.Series:
    """Return Int64 Unix milliseconds from a Narwhals datetime column.

    Rust kernels take millisecond ``i64`` timestamps directly; a kernel that
    strictly needs elapsed seconds derives them from milliseconds itself
    rather than Python doing that conversion.

    A null datetime row becomes ``_NULL_TIMESTAMP_SENTINEL_MS`` rather than a
    null Int64 cell: plain (non-nullable) pandas/NumPy int64 cannot represent
    a null at all -- unlike Polars/Arrow's nullable Int64 -- so every backend
    needs a concrete placeholder here (mirrors ``_build_bucket_ids``'s own
    ``fill_null(0)``). The Rust ms->seconds conversion
    (`run_indexed_timed_coordinate_arrow_ms` et al.) maps this sentinel back
    to ``f64::NAN``, so the existing `is_finite()` null-row exclusion
    downstream is unaffected.
    """
    values = nw.col(datetime_col).dt.timestamp("ms").fill_null(_NULL_TIMESTAMP_SENTINEL_MS).cast(nw.Int64)
    return df.with_columns(values.alias("__fastmob_timestamp__")).get_column("__fastmob_timestamp__")


def _extract_timestamp_arrow(df: nw.DataFrame, datetime_col: str) -> pa.Array | pa.ChunkedArray:
    """Return the native Arrow timestamp column without changing its unit."""
    return _as_arrow(df.get_column(datetime_col).to_arrow())


def _timestamps_ms_to_datetime_ns(values: Any) -> np.ndarray:
    """Convert flat milliseconds-since-epoch values (NumPy or Arrow) to a naive ``datetime64[ns]`` NumPy array.

    Inverse of :func:`_extract_timestamps`, shared by every measure that reconstructs a
    datetime column from Rust kernel output (e.g. ``fastmob/preprocessing/_stay_locations.py``,
    ``fastmob/trajectory/_interpolate.py``).
    """
    values = _as_arrow(values)
    if hasattr(values, "to_numpy"):
        try:
            values = values.to_numpy(zero_copy_only=False)
        except TypeError:
            values = values.to_numpy()
    return (np.asarray(values, dtype="int64") * 1_000_000).astype("int64").view("datetime64[ns]")


def _extract_hours(df: nw.DataFrame, datetime_col: str) -> tuple[nw.DataFrame, nw.Series]:
    """Add and return an ``__hour__`` Float64 column derived from datetime.

    Computed via a millisecond-epoch cast plus integer arithmetic
    (``(ms // 3_600_000) % 24``) instead of the ``.dt.hour()`` accessor:
    pandas' tz-aware ``.dt.hour`` goes through a per-element localization
    path that is far slower than ``.dt.timestamp("ms")`` on the same
    column (profiled at ~74ms vs ~38ms combined on a 4M-row tz-aware
    Brightkite column -- roughly 2x), for an identical wall-clock hour.
    Timezone metadata must be stripped first (keeping the wall-clock value,
    via ``_strip_time_zone``): ``.dt.timestamp()`` reports the true UTC
    instant for tz-aware values, not the localized wall-clock time
    ``.dt.hour()`` reports, so computing straight from the tz-aware column
    would silently shift every hour bucket by the zone's UTC offset.
    """
    naive = _strip_time_zone(df, datetime_col)
    hour_values = naive.with_columns(
        (nw.col(datetime_col).dt.timestamp("ms") // 3_600_000 % 24).cast(nw.Float64).alias("__hour__")
    ).get_column("__hour__")
    df = df.with_columns(hour_values)
    return df, df.get_column("__hour__")


def _build_indexed_user_ranges(
    df: nw.DataFrame,
    uid_col: str | None,
    timestamps: Any | None = None,
    uid_values: Any | None = None,
) -> tuple[Any | None, Any, Any]:
    """Build stable grouped indices, optionally ordered by timestamp within each user.

    `uid_values` lets a caller pass an Arrow uid column it already holds; see
    `_build_presorted_user_ends`.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    from fastmob._core import indexed_user_indices, single_user_indices, time_ordered_user_indices

    profile = os.environ.get("FASTMOB_PROFILE_JUMP_LENGTHS") == "1"

    if uid_col is None:
        if timestamps is None:
            raw_indices, raw_ends = single_user_indices(len(df))
        else:
            raw_indices, raw_ends = time_ordered_user_indices(None, timestamps)
        return None, pa.array(_as_arrow(raw_indices)), pa.array(_as_arrow(raw_ends))

    if uid_values is None:
        uid_values = pa.array(df.get_column(uid_col).to_arrow())
    factorize_started = time.perf_counter()
    codes, representatives = _factorize_arrow_values(uid_values, sort=False)
    if profile:
        print(
            f"[jump_lengths] user-ID factorization: "
            f"{time.perf_counter() - factorize_started:.6f}s "
            f"({len(representatives):,} groups)",
            flush=True,
        )
    ordering_started = time.perf_counter()
    if timestamps is None:
        raw_indices, raw_ends = indexed_user_indices(codes, len(representatives))
    else:
        raw_indices, raw_ends = time_ordered_user_indices(codes, timestamps, len(representatives))
    if profile:
        print(
            f"[jump_lengths] user/time ordering and range construction: {time.perf_counter() - ordering_started:.6f}s",
            flush=True,
        )
    labels_started = time.perf_counter()
    labels = pc.take(uid_values, representatives)
    if profile:
        print(
            f"[jump_lengths] representative label extraction: {time.perf_counter() - labels_started:.6f}s",
            flush=True,
        )
    return labels, pa.array(_as_arrow(raw_indices)), pa.array(_as_arrow(raw_ends))


def _build_user_ranges_auto(
    df: nw.DataFrame,
    uid_col: str | None,
    timestamps: Any | None = None,
    *,
    coordinates: tuple[Any, Any] | None = None,
) -> tuple[Any | None, Any | None, Any]:
    """Build user ranges, taking the contiguous fast path when the data allows.

    Same contract as `_build_indexed_user_ranges`, except that the returned
    `indices` is ``None`` when the rows already satisfy the `presorted` kernels'
    preconditions -- the caller should then use its contiguous kernel with
    `ends` alone.

    Pass `timestamps` when the measure is order-dependent (jump lengths and
    friends): the fast path then additionally requires non-decreasing time
    within each user.  Order-independent measures (radius of gyration) leave it
    ``None`` and only require grouping.

    `coordinates` is the ``(latitudes, longitudes)`` pair the caller is about to
    hand its kernel.  The contiguous kernels sum coordinates with no per-row
    validity test, whereas the indexed ones skip invalid rows, so a frame with a
    null or NaN coordinate must stay on the indexed path to keep the same
    answer. Callers whose kernels do not read coordinates leave it ``None``;
    in that case only grouping and optional timestamp ordering are checked.

    The detection is deliberately built out of the work the *fast* path needs
    anyway, not the slow one.  `_build_presorted_user_ends`'s single
    `run_end_encode` scan already yields both the run boundaries the contiguous
    kernels want and, as its values, each run's uid -- so contiguity reduces to
    "no uid leads two runs", over an array with one entry per run.  Dense-coding
    the uid column instead would answer the same question, but measured ~23ms
    per 4M rows against ~8ms here, and would be wasted whenever the answer is
    yes.  The cost of guessing wrong is that one `run_end_encode` scan, after
    which this falls back to `_build_indexed_user_ranges` unchanged.

    The ordering checks run before the coordinate one even though the coordinate
    one is simpler: row order is what actually varies between frames, so testing
    it first is what keeps a frame that was never going to qualify from paying
    for a scan of both coordinate columns.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    from fastmob._core import (
        coordinates_all_finite,
        single_user_indices,
        time_ordered_user_indices,
        validate_timestamps_sorted,
        validate_timestamps_within_ends,
    )

    def _coordinates_usable() -> bool:
        return coordinates is None or coordinates_all_finite(*coordinates)

    n = len(df)
    if uid_col is None:
        ordered = timestamps is None or validate_timestamps_sorted(timestamps)
        if ordered and _coordinates_usable():
            return None, None, pa.array([] if n == 0 else [n], type=pa.uint64())
        if timestamps is None:
            raw_indices, raw_ends = single_user_indices(n)
        else:
            raw_indices, raw_ends = time_ordered_user_indices(None, timestamps)
        return None, pa.array(_as_arrow(raw_indices)), pa.array(_as_arrow(raw_ends))

    # Materialized once and shared with whichever path wins, so guessing wrong
    # does not convert the uid column twice.
    uid_values = pa.array(df.get_column(uid_col).to_arrow())
    labels, ends = _build_presorted_user_ends(df, uid_col, uid_values)
    grouped = len(pc.unique(labels)) == len(labels)
    if grouped and (timestamps is None or validate_timestamps_within_ends(timestamps, ends)) and _coordinates_usable():
        return labels, None, ends

    return _build_indexed_user_ranges(df, uid_col, timestamps, uid_values)


def _rechunk_kernel_columns(nw_df: nw.DataFrame, columns: Iterable[str | None]) -> nw.DataFrame:
    """Consolidate the columns a Rust kernel will read into one chunk each.

    Polars parses and sorts in parallel and leaves one chunk per split -- 641
    straight out of `read_csv`, 40 after a `sort` -- because rechunking is a
    copy most polars pipelines never need.  The kernels do need it: they take a
    single contiguous slice, so `.to_arrow()` consolidates the column on every
    call.  Doing it here instead, once, is worth roughly 2x: four separate
    `.to_arrow()` calls measured 17.7ms on a 4M-row frame against 8.9ms for one
    `rechunk` of the same four columns, because polars rechunks columns
    concurrently while four Series calls run one after another.  Extraction
    afterwards is a zero-copy view (0.07ms for all four).

    Only the columns the kernels read are touched, so a wide frame does not pay
    to consolidate columns nobody looks at, and a frame that is already
    contiguous returns untouched.

    This is the one place in `fastmob/` that reaches past Narwhals to a specific
    backend, because Narwhals has no `rechunk` and chunking is not part of its
    model.  It is a memory-layout normalization, not a computation: no result
    depends on which branch runs.
    """
    if nw_df.implementation.value != "polars":
        return nw_df
    native = nw_df.to_native()
    wanted = [name for name in dict.fromkeys(columns) if name is not None and name in native.columns]
    chunked = [name for name in wanted if native[name].n_chunks() > 1]
    if not chunked:
        return nw_df
    # One frame-level `rechunk`, not one per column: polars consolidates the
    # columns of a frame concurrently, whereas `Series.rechunk()` in a Python
    # loop is serial and costs exactly what the serial `.to_arrow()` calls it
    # replaces (17.8ms either way on four 4M-row columns, against 8.8ms here).
    # Selecting first keeps a wide frame from consolidating columns no kernel
    # reads; splicing the results back is metadata only.
    consolidated = native.select(chunked).rechunk()
    return nw.from_native(
        native.with_columns([consolidated[name] for name in chunked]),
        eager_only=True,
    )


def _detect_trajectory_columns(
    nw_df: nw.DataFrame,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    *,
    cast_float_coordinates: bool = False,
    require_datetime: bool = True,
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
    require_datetime:
        When False, datetime is resolved when present but is not required.

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
            [datetime_col if require_datetime else "", lat_col, lng_col],
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
        # Returning the frame means a kernel is about to read these columns.
        nw_df = _rechunk_kernel_columns(nw_df, (lat_col, lng_col, uid_col))
        return nw_df, datetime_col, lat_col, lng_col, uid_col

    return datetime_col, lat_col, lng_col, uid_col


def _trajectory_input(
    traj: Any,
    *,
    datetime_col: str | None,
    lat_col: str | None,
    lng_col: str | None,
    uid_col: str | None,
) -> tuple[nw.DataFrame, str, str, str, str | None]:
    """Normalize a TrajDataFrame or dataframe-like trajectory input."""
    native = getattr(traj, "df", traj)
    datetime_col = datetime_col or getattr(traj, "datetime_col", None)
    lat_col = lat_col or getattr(traj, "lat_col", None)
    lng_col = lng_col or getattr(traj, "lng_col", None)
    uid_col = uid_col or getattr(traj, "uid_col", None)
    df = nw.from_native(native, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df, datetime_col=datetime_col, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col
    )
    df = _with_datetime_column(df, datetime_col)
    if df.schema[lat_col] != nw.Float64 or df.schema[lng_col] != nw.Float64:
        df = df.with_columns(nw.col(lat_col).cast(nw.Float64), nw.col(lng_col).cast(nw.Float64))
    required = [datetime_col, lat_col, lng_col] + ([uid_col] if uid_col is not None else [])
    return df.drop_nulls(subset=required), datetime_col, lat_col, lng_col, uid_col


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


def _build_presorted_user_ends(
    df: nw.DataFrame,
    uid_col: str | None,
    uid_values: Any | None = None,
) -> tuple[Any | None, Any]:
    """Build contiguous user group end indices for data already grouped by user.

    `uid_values` lets a caller that has already materialized the uid column as
    an Arrow array hand it over rather than have it converted again; converting
    a 4M-row column measured ~6ms.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    from fastmob._core import value_run_boundaries

    n = len(df)
    if uid_col is None:
        return None, pa.array([] if n == 0 else [n], type=pa.uint64())

    if uid_values is None:
        uid_values = pa.array(df.get_column(uid_col).to_arrow())
    if n == 0:
        return uid_values.slice(0, 0), pa.array([], type=pa.uint64())

    # Comparing adjacent elements of a flat primitive buffer in Rust measured
    # ~15x cheaper than `run_end_encode` at 4M rows.  It declines anything whose
    # runs are not simply "adjacent values compare equal" -- strings,
    # dictionaries, a column with nulls -- which then takes the Arrow route.
    boundaries = value_run_boundaries(uid_values)
    if boundaries is not None:
        starts, ends = boundaries
        return pc.take(uid_values, pa.array(starts)), pa.array(ends)

    encoded = pc.run_end_encode(uid_values)
    return encoded.values, pc.cast(encoded.run_ends, pa.uint64())


def _build_presorted_indices_and_ends(df: nw.DataFrame, uid_col: str | None) -> tuple[Any | None, Any, Any]:
    """Cheap presorted-mode drop-in for `_build_indexed_user_ranges`.

    `_build_indexed_user_ranges` always pays for a hash-based factorize plus
    the ``indexed_user_indices``/``time_ordered_user_indices`` Rust kernels --
    work that makes sense when the caller's ordering can't be trusted, but is
    pure waste when the frame is already grouped by user and, within each
    user, already in the desired order: the "sorted indices" the kernel needs
    are then just the identity permutation, and the only real work left is
    finding each user's contiguous run boundary, which
    `_build_presorted_user_ends` already does via a single `pyarrow`
    `run_end_encode` scan -- no hashing, no Rust factorize call.

    ``cluster()`` gets the same saving a different way: it physically sorts
    the frame via ``_prepare_trajectory(sort=True)`` and then calls
    `_build_presorted_user_ends` directly, since its kernel needs only
    boundaries and no indices array. This helper carries that same
    "boundaries are cheap" idea to the indices-based, non-physically-resorted
    call sites, which still need an explicit ``sorted_indices`` array to hand
    the kernel.

    The remaining caller is `fastmob.preprocessing._filter`, whose ``filter``
    and outlier-detection entry points take an internal ``is_sorted`` flag.
    Measures that detect orderedness rather than being told about it go
    through `_build_user_ranges_auto` instead, which returns ``None`` indices
    on the contiguous path and lets the caller substitute
    ``single_user_indices`` only if its kernel needs them.
    """
    from fastmob._core import single_user_indices

    labels, ends = _build_presorted_user_ends(df, uid_col)
    identity_indices, _ = single_user_indices(len(df))
    return labels, pa.array(_as_arrow(identity_indices)), ends


def _value_offsets_from_index_ranges(
    index_starts: Any,
    index_ends: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert row-index ranges to value-array offsets for non-ordered results.

    For a group of n rows, the kernel produces n-1 jump values.  This converts
    the per-user row-index spans into half-open slices into the flat output array.
    """
    starts = np.asarray(index_starts, dtype=np.uint64)
    ends = np.asarray(index_ends, dtype=np.uint64)
    lengths = np.maximum(ends - starts, 1) - 1
    value_ends = np.cumsum(lengths, dtype=np.uint64)
    value_starts = value_ends - lengths
    return value_starts, value_ends


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
        starts = np.asarray(starts, dtype=np.uint64)
        ends = np.asarray(ends, dtype=np.uint64)
    else:
        starts, ends = _value_offsets_from_index_ranges(starts, ends)
    # PyArrow's ListArray constructor accepts int64 offsets; semantic row and
    # value offsets remain UInt64 throughout this path.
    offsets = np.empty(len(starts) + 1, dtype=np.int64)
    offsets[:-1] = starts
    offsets[-1] = ends[-1] if len(ends) else 0
    return pa.ListArray.from_arrays(offsets, _as_arrow(values))
