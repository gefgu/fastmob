from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob._core import RadiusOfGyrationStream
from fastmob.core.base import unwrap_native
from fastmob.utils._common import (
    _as_arrow,
    _detect_trajectory_columns,
    _filter_result_values,
    _result_scalar,
    _to_native,
)

DEFAULT_CHUNK_SIZE = 250_000


def _restore_uid_type(labels: pa.Array, uid_type: pa.DataType) -> pa.Array:
    """Cast the stream's canonical label array (int64/uint64, float64, bool,
    large_string, large_binary) back to the input's user-ID type."""
    if labels.type == uid_type:
        return labels
    if pa.types.is_temporal(uid_type):
        storage = pa.int32() if uid_type.bit_width == 32 else pa.int64()
        return labels.cast(storage).view(uid_type)
    return labels.cast(uid_type)


def radius_of_gyration_lazy(
    traj: Any,
    *,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
):
    """Compute the radius of gyration (km) per user by streaming a Polars LazyFrame.

    Streaming counterpart of [`radius_of_gyration`][fastmob.radius_of_gyration]
    for sources that do not fit in memory, such as ``pl.scan_parquet`` or
    ``pl.scan_csv``.  The lazy plan is executed in batches, and memory grows
    with the number of users rather than the number of rows.

    Radius of gyration needs each user's centre of mass before it can sum
    squared distances to it, so the plan is executed **twice**: the first pass
    accumulates per-user coordinate sums, the second the squared haversine
    distances to each centre.  The plan must therefore produce the same rows
    on both executions (a deterministic scan does); a mismatch raises
    ``ValueError``.  Only the latitude, longitude, and user-ID columns are read.

    Results equal `radius_of_gyration` on the collected frame up to
    floating-point summation order, with users in first-appearance order.

    Parameters
    ----------
    traj : polars.LazyFrame or narwhals.LazyFrame
        Polars-backed lazy trajectory data.  Must have columns for datetime,
        latitude, and longitude.  A user-ID column is optional; when absent
        the whole frame is treated as a single individual.
    chunk_size : int or None, optional
        Rows per streamed batch, passed to ``LazyFrame.collect_batches``.
        Defaults to 250,000 rows.  Each batch is one parallel Rust call, so
        tiny batches pay per-call overhead, while very large ones make Polars
        spend time concatenating its internal morsels (on a 30 GB CSV, 100k-250k
        rows ran in ~17 s, 1M rows in ~28 s).  ``None`` lets Polars pick, which
        yields batches of only a few thousand rows.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    lat_col : str or None, optional
        Explicit latitude column name.  Auto-detected when None.
    lng_col : str or None, optional
        Explicit longitude column name.  Auto-detected when None.
    uid_col : str or None, optional
        Explicit user-ID column name.  Auto-detected when None.

    Returns
    -------
    polars.DataFrame
        One row per user with columns ``[uid_col, "radius_of_gyration"]``.

    Raises
    ------
    TypeError
        If ``traj`` is not a Polars-backed lazy frame.

    Examples
    --------
    >>> import polars as pl
    >>> from fastmob import radius_of_gyration_lazy
    >>> lf = pl.LazyFrame(
    ...     {
    ...         "uid": [1, 1, 2, 2],
    ...         "datetime": [0, 1, 0, 1],
    ...         "lat": [45.0, 45.1, 10.0, 10.0],
    ...         "lng": [9.0, 9.0, 20.0, 20.2],
    ...     }
    ... )
    >>> print(radius_of_gyration_lazy(lf, chunk_size=1).with_columns(pl.col("radius_of_gyration").round(3)))
    shape: (2, 2)
    ┌─────┬────────────────────┐
    │ uid ┆ radius_of_gyration │
    │ --- ┆ ---                │
    │ i64 ┆ f64                │
    ╞═════╪════════════════════╡
    │ 1   ┆ 5.56               │
    │ 2   ┆ 10.951             │
    └─────┴────────────────────┘

    See Also
    --------
    radius_of_gyration : Eager version for in-memory dataframes.
    """
    lf = nw.from_native(unwrap_native(traj))
    if not isinstance(lf, nw.LazyFrame) or not lf.implementation.is_polars():
        raise TypeError(
            "radius_of_gyration_lazy expects a Polars LazyFrame; use radius_of_gyration for eager dataframes"
        )

    # A zero-row collect (pushed down by polars) resolves the schema once, for
    # column detection here and for the output backend and uid dtype below.
    empty = lf.head(0).collect()
    _, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        empty,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    projection = [
        *([nw.col(uid_col)] if uid_col is not None else []),
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    ]
    projected = lf.select(*projection).to_native()
    result_frame = empty.select(*projection)

    def batches(*, maintain_order: bool) -> Iterator[tuple[Any, Any, Any]]:
        # Columns go to Rust as native Series over the Arrow C stream interface:
        # zero-copy and chunk-preserving, where `.to_arrow()` would rechunk and
        # copy string user IDs out of Polars' string-view layout.
        for native in projected.collect_batches(chunk_size=chunk_size, maintain_order=maintain_order):
            if not len(native):
                continue
            batch = nw.from_native(native, eager_only=True)
            uids = batch.get_column(uid_col).to_native() if uid_col is not None else None
            yield uids, batch.get_column(lat_col).to_native(), batch.get_column(lng_col).to_native()

    stream = RadiusOfGyrationStream()
    # Pass 1 keeps batch order: user slots, and so output rows, follow first appearance.
    for columns in batches(maintain_order=True):
        stream.accumulate_moments(*columns)
    stream.finalize_centers()
    for columns in batches(maintain_order=False):
        stream.accumulate_sq(*columns)
    raw_labels, raw_values, raw_validity = stream.finalize()

    keep = raw_validity
    filtered_count = keep.size - int(keep.sum())
    if filtered_count:
        noun = "user" if filtered_count == 1 else "users"
        warnings.warn(
            f"radius_of_gyration filtered out {filtered_count} {noun} with no valid coordinate rows",
            RuntimeWarning,
            stacklevel=2,
        )

    rog_values = _as_arrow(raw_values)
    if uid_col is None:
        if not keep.any():
            return _to_native({"radius_of_gyration": [0.0]}, result_frame)
        return _to_native(
            {"radius_of_gyration": [_result_scalar(_filter_result_values(rog_values, keep))]},
            result_frame,
        )

    uid_type = result_frame.get_column(uid_col).to_arrow().type
    uid_values = _restore_uid_type(pa.array(_as_arrow(raw_labels)), uid_type)
    return _to_native(
        {
            uid_col: pc.filter(uid_values, pa.array(keep)),
            "radius_of_gyration": _filter_result_values(rog_values, keep),
        },
        result_frame,
    )
