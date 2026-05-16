from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Iterable

import narwhals as nw
import numpy as np

_ROW_ORDER_COL = "__skmob2_row_order__"
_USER_RANGE_START_COL = "__skmob2_user_range_start__"

# ---------------------------------------------------------------------------
# Authoritative candidate lists for column auto-detection.
# All measure files should import from here rather than defining their own.
# ---------------------------------------------------------------------------

DATETIME_CANDIDATES: list[str] = ["datetime", "check-in_time", "timestamp", "time"]
LAT_CANDIDATES: list[str] = ["lat", "latitude"]
LNG_CANDIDATES: list[str] = ["lng", "lon", "longitude"]
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
DEST_CANDIDATES: list[str] = ["destination_area", "Dest_Area", "area_d", "DESTINATION_AREA"]


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


def _is_polars_backed(nw_df: nw.DataFrame) -> bool:
    """Return True when a Narwhals DataFrame is backed by a Polars object."""
    native = nw_df.to_native()
    return hasattr(native, "lazy")


def _is_pandas_backed(nw_df: nw.DataFrame) -> bool:
    """Return True when a Narwhals DataFrame is backed by pandas."""
    native = nw_df.to_native()
    return hasattr(native, "iloc") and hasattr(native, "dtypes")


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
    except Exception:
        return df.with_columns(nw.col(column).str.to_datetime().alias(column))


def _as_index_array(values: Any) -> np.ndarray:
    """Return unsigned pointer-sized indexes for Rust indexed kernels."""
    return np.asarray(values, dtype=np.uintp)


def _ranges_to_starts_ends(ranges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert Python range tuples to NumPy start/end arrays."""
    starts = np.fromiter((start for start, _ in ranges), dtype=np.uintp, count=len(ranges))
    ends = np.fromiter((end for _, end in ranges), dtype=np.uintp, count=len(ranges))
    return starts, ends


def _arrow_result_values(values: Any) -> Any:
    """Return a PyArrow value object when a pyo3-arrow wrapper is returned."""
    if hasattr(values, "to_pyarrow"):
        return values.to_pyarrow()
    return values


def _result_scalar(values: Any) -> float:
    """Extract a single Python float from a length-1 NumPy/Arrow result."""
    if hasattr(values, "to_numpy"):
        return float(values.to_numpy()[0])
    return float(np.asarray(values)[0])


def _to_native(values_dict: dict[str, Any], df: nw.DataFrame) -> Any:
    """Build a backend-matching result dataframe from a column dict."""
    return nw.from_dict(values_dict, backend=df.implementation).to_native()


def _extract_timestamps_ms(df: nw.DataFrame, datetime_col: str) -> nw.Series:
    """Return a Float64 Narwhals Series of millisecond Unix timestamps."""
    return df.with_columns(
        nw.col(datetime_col).dt.timestamp("ms").cast(nw.Float64).alias("__ts_ms__")
    ).get_column("__ts_ms__")


def _extract_timestamps_s(df: nw.DataFrame, datetime_col: str) -> nw.Series:
    """Return a Float64 Narwhals Series of second-resolution Unix timestamps.

    Goes through the millisecond path so backends with nanosecond or
    microsecond storage produce identical values.
    """
    return df.with_columns(
        (nw.col(datetime_col).dt.timestamp("ms") / 1000.0).alias("__ts_s__")
    ).get_column("__ts_s__")


def _extract_hours(df: nw.DataFrame, datetime_col: str) -> tuple[nw.DataFrame, nw.Series]:
    """Add and return an ``__hour__`` Float64 column derived from datetime."""
    df = df.with_columns(nw.col(datetime_col).dt.hour().cast(nw.Float64).alias("__hour__"))
    return df, df.get_column("__hour__")


def _uid_values_from_index_ranges(
    uids: nw.Series,
    indices: Any,
    starts: Any,
    *,
    use_arrow: bool,
) -> list:
    """Extract one UID label per indexed range without scanning all rows in Python."""
    if use_arrow:
        uid_arrow = uids.to_arrow()
        return [uid_arrow[int(indices[int(start)])].as_py() for start in starts]

    uid_values = uids.to_numpy()
    return uid_values[np.asarray(indices, dtype=np.uintp)[np.asarray(starts, dtype=np.uintp)]].tolist()


def _build_time_ordered_user_ranges(
    df: nw.DataFrame,
    uid_col: str | None,
    datetime_col: str,
    timestamps: nw.Series,
    *,
    use_arrow: bool,
    row_index_col: str = "__skmob2_time_order_row_index__",
) -> tuple[list | None, Any, np.ndarray, np.ndarray]:
    """Build stable time-ordered indexes/ranges without sorting the full dataframe."""
    from skmob2._core import (
        time_ordered_user_indices_arrow,
        time_ordered_user_indices_numpy,
    )

    if uid_col is None:
        if use_arrow:
            indices, starts, ends = time_ordered_user_indices_arrow(None, timestamps.to_arrow())
        else:
            indices, starts, ends = time_ordered_user_indices_numpy(None, timestamps.to_numpy())
        return None, _as_index_array(indices), _as_index_array(starts), _as_index_array(ends)

    uids = df.get_column(uid_col)
    try:
        if use_arrow:
            indices, starts, ends = time_ordered_user_indices_arrow(uids.to_arrow(), timestamps.to_arrow())
        else:
            indices, starts, ends = time_ordered_user_indices_numpy(uids.to_numpy(), timestamps.to_numpy())
        indices = _as_index_array(indices)
        starts = _as_index_array(starts)
        ends = _as_index_array(ends)
        uid_values = _uid_values_from_index_ranges(uids, indices, starts, use_arrow=use_arrow)
        return uid_values, indices, starts, ends
    except ValueError as exc:
        if "unsupported" not in str(exc):
            raise

    index_df = (
        df.select([uid_col, datetime_col])
        .with_row_index(row_index_col)
        .sort(uid_col, datetime_col, row_index_col)
    )
    uid_values, ranges = _build_user_ranges(index_df, uid_col)
    indices = [int(idx) for idx in index_df.get_column(row_index_col).to_list()]
    starts, ends = _ranges_to_starts_ends(ranges)
    return uid_values, _as_index_array(indices), starts, ends


def _build_indexed_user_ranges_fast(
    df: nw.DataFrame,
    uid_col: str | None,
    *,
    use_arrow: bool,
    row_index_col: str = "__skmob2_fast_row_index__",
) -> tuple[list | None, Any, np.ndarray, np.ndarray]:
    """Build grouped row indexes using Rust for supported UID dtypes."""
    from skmob2._core import radius_of_gyration_user_indices_arrow, radius_of_gyration_user_indices_numpy

    if uid_col is None:
        indices = np.arange(len(df), dtype=np.uintp)
        starts = np.array([0], dtype=np.uintp)
        ends = np.array([len(df)], dtype=np.uintp)
        return None, indices, starts, ends

    uids = df.get_column(uid_col)
    try:
        if use_arrow:
            indices, starts, ends = radius_of_gyration_user_indices_arrow(uids.to_arrow())
        else:
            indices, starts, ends = radius_of_gyration_user_indices_numpy(uids.to_numpy())
        uid_values = _uid_values_from_index_ranges(uids, indices, starts, use_arrow=use_arrow)
        return uid_values, _as_index_array(indices), _as_index_array(starts), _as_index_array(ends)
    except ValueError as exc:
        if "unsupported" not in str(exc):
            raise

    uid_values, indices, ranges = _build_indexed_user_ranges(df, uid_col, row_index_col=row_index_col)
    starts, ends = _ranges_to_starts_ends(ranges)
    return uid_values, _as_index_array(indices), starts, ends


def _detect_trajectory_columns(
    nw_df: nw.DataFrame,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> tuple[str, str, str, str | None]:
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

    Returns
    -------
    tuple[str, str, str, str | None]
        ``(datetime_col, lat_col, lng_col, uid_col)`` where ``uid_col`` may
        be None if no user-ID column exists.

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

    df = nw_df.drop_nulls(subset=[datetime_col, lat_col, lng_col]) if drop_nulls else nw_df
    if sort:
        sort_cols = [uid_col, datetime_col, _ROW_ORDER_COL] if uid_col else [datetime_col, _ROW_ORDER_COL]
        df = df.sort(*sort_cols)

    df = df.with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )
    if sort:
        df = df.drop(_ROW_ORDER_COL)

    return df


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


def _build_indexed_user_ranges(
    df: nw.DataFrame,
    uid_col: str,
    *,
    row_index_col: str = "__skmob2_indexed_row_index__",
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
