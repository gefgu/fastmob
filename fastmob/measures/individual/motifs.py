"""Daily home-anchored mobility motifs."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import narwhals as nw

from fastmob import _core
from fastmob.utils._common import (
    DURATION_CANDIDATES,
    LOCATION_CANDIDATES,
    PURPOSE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _arrow_result_values,
    _pick_existing_column,
)

_END_TIMESTAMP_CANDIDATES = ["end_timestamp", "end_time"]
_ROW_INDEX = "__fastmob_motif_row_index__"
_ORDER_INDEX = "__fastmob_motif_order_index__"


def _detect_visit_columns(
    df: nw.DataFrame,
    *,
    uid_col: str | None,
    location_col: str | None,
    purpose_col: str | None,
    datetime_col: str | None,
    end_datetime_col: str | None,
    duration_col: str | None,
) -> tuple[str, str, str, str, str, str | None]:
    """Resolve the visit schema in one place, matching other measures."""
    columns = df.columns
    uid_col = uid_col or _pick_existing_column(columns, USER_ID_CANDIDATES) or "agent_id"
    location_col = location_col or _pick_existing_column(columns, LOCATION_CANDIDATES) or "location_id"
    purpose_col = purpose_col or _pick_existing_column(columns, PURPOSE_CANDIDATES) or "purpose"
    datetime_col = datetime_col or _pick_existing_column(columns, TIMESTAMP_CANDIDATES) or "start_timestamp"
    end_datetime_col = (
        end_datetime_col or _pick_existing_column(columns, _END_TIMESTAMP_CANDIDATES) or "end_timestamp"
    )
    if duration_col is None:
        duration_col = _pick_existing_column(columns, DURATION_CANDIDATES)
    elif duration_col not in columns:
        raise ValueError(f"Duration column {duration_col!r} does not exist.")
    return uid_col, location_col, purpose_col, datetime_col, end_datetime_col, duration_col


def _strip_time_zone(df: nw.DataFrame, column: str) -> nw.DataFrame:
    dtype = df.schema[column]
    if isinstance(dtype, nw.Datetime) and dtype.time_zone is not None:
        return df.with_columns(nw.col(column).dt.replace_time_zone(None))
    return df


def _encode_locations(df: nw.DataFrame, column: str) -> Any:
    """Use integer IDs directly; factorize other exact location types."""
    import pyarrow as pa
    import pyarrow.compute as pc

    series = df.get_column(column)
    dtype = df.schema[column]
    if dtype.is_integer() and series.null_count() == 0:
        values = series.to_arrow()
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        return pc.cast(values, pa.uint64(), safe=False)
    values = series.to_arrow()
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    encoded = pc.dictionary_encode(values)
    codes = encoded.indices
    if codes.null_count:
        codes = pc.fill_null(codes, len(encoded.dictionary))
    return pc.cast(codes, pa.uint64())


def _user_ranges(df: nw.DataFrame, uid_col: str, datetime_col: str | None) -> tuple[Any, Any, Any]:
    """Return Arrow user labels, optional row ordering, and cumulative ends."""
    import pyarrow as pa

    columns = [uid_col] if datetime_col is None else [uid_col, datetime_col]
    ordered = df.select(columns).with_row_index(_ROW_INDEX)
    if datetime_col is not None:
        ordered = ordered.sort(uid_col, datetime_col, _ROW_INDEX)

    starts = (
        ordered.with_row_index(_ORDER_INDEX)
        .filter((nw.col(uid_col) != nw.col(uid_col).shift(1)).fill_null(True))
        .select([uid_col, _ORDER_INDEX])
    )
    labels = starts.get_column(uid_col).to_arrow()
    start_values = starts.get_column(_ORDER_INDEX).cast(nw.UInt64).to_arrow()
    if isinstance(labels, pa.ChunkedArray):
        labels = labels.combine_chunks()
    if isinstance(start_values, pa.ChunkedArray):
        start_values = start_values.combine_chunks()
    ends = (
        pa.concat_arrays([start_values.slice(1), pa.array([len(df)], type=pa.uint64())])
        if len(start_values)
        else pa.array([], type=pa.uint64())
    )
    indices = None
    if datetime_col is not None:
        indices = ordered.get_column(_ROW_INDEX).cast(nw.UInt64).to_arrow()
        if isinstance(indices, pa.ChunkedArray):
            indices = indices.combine_chunks()
    return labels, indices, ends


def daily_motifs(
    visits: Any,
    *,
    uid_col: str | None = None,
    location_col: str | None = None,
    purpose_col: str | None = None,
    datetime_col: str | None = None,
    end_datetime_col: str | None = None,
    duration_col: str | None = None,
    presorted: bool = False,
) -> Any:
    """Compute one home-anchored mobility motif per user and day.

    The returned dataframe matches the input backend and contains the resolved
    user-ID column, ``date``, and ``motif_id``. Set ``presorted=True`` only when
    rows are already grouped by user and chronological within each user.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    profiling = os.environ.get("FASTMOB_PROFILE_MOTIFS") is not None
    total_started = time.perf_counter()
    stage_started = total_started
    timings: dict[str, float] = {}

    def finish_stage(name: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        timings[name] = now - stage_started
        stage_started = now

    df = nw.from_native(visits, eager_only=True)
    backend = df.implementation
    uid_col, location_col, purpose_col, datetime_col, end_datetime_col, duration_col = (
        _detect_visit_columns(
            df,
            uid_col=uid_col,
            location_col=location_col,
            purpose_col=purpose_col,
            datetime_col=datetime_col,
            end_datetime_col=end_datetime_col,
            duration_col=duration_col,
        )
    )
    uid_dtype = df.schema[uid_col]
    df = _strip_time_zone(df, datetime_col)
    df = _strip_time_zone(df, end_datetime_col)
    df = df.with_columns(
        nw.col(datetime_col).cast(nw.Datetime("us")).alias(datetime_col),
        nw.col(end_datetime_col).cast(nw.Datetime("us")).alias(end_datetime_col),
        nw.col(purpose_col).cast(nw.String).alias(purpose_col),
    )
    finish_stage("detect_and_cast")

    if presorted:
        uid_labels, indices, ends = _user_ranges(df, uid_col, None)
    else:
        uid_labels, indices, ends = _user_ranges(df, uid_col, datetime_col)
    finish_stage("order_users")

    location_values = _encode_locations(df, location_col)
    finish_stage("encode_nodes")

    durations = None
    if duration_col is not None:
        duration_series = df.get_column(duration_col).cast(nw.Float64).fill_null(0.0)
        durations = duration_series.to_arrow()

    purpose_data = df.get_column(purpose_col).to_arrow()
    if isinstance(purpose_data, pa.ChunkedArray):
        purpose_data = purpose_data.combine_chunks()

    kernel_args = (
        location_values,
        purpose_data,
        df.get_column(datetime_col).to_arrow(),
        df.get_column(end_datetime_col).to_arrow(),
        durations,
    )
    if presorted:
        raw_users, raw_dates, raw_motifs = _core.daily_motifs_presorted(
            *kernel_args,
            ends,
        )
    else:
        raw_users, raw_dates, raw_motifs = _core.daily_motifs_indexed(
            *kernel_args,
            indices,
            ends,
        )
    finish_stage("rust_kernel")

    user_indices = pa.array(_arrow_result_values(raw_users))
    date_ids = pa.array(_arrow_result_values(raw_dates))
    motif_ids = pa.array(_arrow_result_values(raw_motifs))
    user_ids = pc.take(uid_labels, user_indices)
    result = (
        nw.from_arrow(
            pa.table(
                {
                    uid_col: user_ids,
                    "__date_id": date_ids,
                    "motif_id": motif_ids,
                }
            ),
            backend=backend,
        )
        .with_columns(
            nw.col(uid_col).cast(uid_dtype),
            nw.col("__date_id").cast(nw.Int32).cast(nw.Date).alias("date"),
            nw.col("motif_id").cast(nw.Int64),
        )
        .select([uid_col, "date", "motif_id"])
        .to_native()
    )
    finish_stage("assemble")
    if profiling:
        detail = " ".join(f"{name}={value:.6f}s" for name, value in timings.items())
        print(
            f"fastmob motif python: {detail} total={time.perf_counter() - total_started:.6f}s",
            file=sys.stderr,
            flush=True,
        )
    return result


def motif_distribution(daily: Any, motif_id_col: str = "motif_id") -> Any:
    """Aggregate daily motif IDs into count and percentage columns."""
    df = nw.from_native(daily, eager_only=True)
    if motif_id_col not in df.columns:
        raise ValueError(f"Motif ID column {motif_id_col!r} does not exist.")
    total = len(df)
    if total == 0:
        import pyarrow as pa

        return (
            nw.from_arrow(
                pa.table(
                    {
                        motif_id_col: pa.array([], type=pa.int64()),
                        "count": pa.array([], type=pa.int64()),
                        "percentage": pa.array([], type=pa.float64()),
                    }
                ),
                backend=df.implementation,
            )
            .to_native()
        )
    return (
        df.group_by(motif_id_col)
        .agg(nw.len().alias("count"))
        .sort(motif_id_col)
        .with_columns(
            nw.col("count").cast(nw.Int64),
            (nw.col("count") / total * 100.0).alias("percentage"),
        )
        .to_native()
    )
