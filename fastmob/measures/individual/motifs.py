"""Daily home-anchored mobility motifs."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob import _core
from fastmob.utils._common import (
    _auto_presorted,
    DURATION_CANDIDATES,
    LOCATION_CANDIDATES,
    PURPOSE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _as_arrow,
    _build_indexed_user_ranges,
    _build_presorted_user_ends,
    _extract_timestamp_arrow,
    _factorize_arrow_values,
    _pick_existing_column,
    _strip_time_zone,
)

_END_TIMESTAMP_CANDIDATES = ["end_timestamp", "end_time"]


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
    uid_col = uid_col or _pick_existing_column(columns, USER_ID_CANDIDATES)
    location_col = location_col or _pick_existing_column(columns, LOCATION_CANDIDATES)
    purpose_col = purpose_col or _pick_existing_column(columns, PURPOSE_CANDIDATES)
    datetime_col = datetime_col or _pick_existing_column(columns, TIMESTAMP_CANDIDATES)
    end_datetime_col = end_datetime_col or _pick_existing_column(columns, _END_TIMESTAMP_CANDIDATES)

    # Unlike _detect_interval_columns, daily_motifs's result-assembly path
    # (pc.take(uid_labels, ...), df.schema[uid_col], pa.table({uid_col: ...}))
    # does not support uid_col=None end-to-end, so every one of these raises
    # instead of falling back to a guessed column name or "single individual" mode.
    if uid_col is None:
        raise ValueError(f"Could not find a user-ID column; checked {USER_ID_CANDIDATES}")
    if location_col is None:
        raise ValueError(f"Could not find a location column; checked {LOCATION_CANDIDATES}")
    if purpose_col is None:
        raise ValueError(f"Could not find a purpose column; checked {PURPOSE_CANDIDATES}")
    if datetime_col is None:
        raise ValueError(f"Could not find a start-timestamp column; checked {TIMESTAMP_CANDIDATES}")
    if end_datetime_col is None:
        raise ValueError(f"Could not find an end-timestamp column; checked {_END_TIMESTAMP_CANDIDATES}")

    if duration_col is None:
        duration_col = _pick_existing_column(columns, DURATION_CANDIDATES)
    elif duration_col not in columns:
        raise ValueError(f"Duration column {duration_col!r} does not exist.")
    return uid_col, location_col, purpose_col, datetime_col, end_datetime_col, duration_col


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
        return pc.cast(values, pa.uint32(), safe=False)
    values = series.to_arrow()
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    codes, _ = _factorize_arrow_values(values, sort=False)
    return codes


def _encode_locations_pair(visits_series: nw.Series, locations_series: nw.Series, dtype: Any) -> tuple[Any, Any]:
    """Encode two location-id columns with one consistent codebook.

    Used by :func:`daily_motifs_from_staypoints` so a location_id value maps
    to the same code on both the visits (Staypoints) side and the locations
    (Locations) side. Integer ids (the real trackintel case -- ``location_id`` from
    ``generate_user_locations`` is always an integer cluster id) are cast
    independently on each side; raw integer values are inherently
    consistent between tables, no shared factorization state needed. Other
    types are concatenated and factorized once so the same raw value maps
    to the same code on both sides.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    visits_values = _arrow_array(visits_series)
    locations_values = _arrow_array(locations_series)
    if dtype.is_integer() and visits_series.null_count() == 0 and locations_series.null_count() == 0:
        return (
            pc.cast(visits_values, pa.uint32(), safe=False),
            pc.cast(locations_values, pa.uint32(), safe=False),
        )
    if visits_values.type != locations_values.type:
        visits_values = pc.cast(visits_values, pa.string())
        locations_values = pc.cast(locations_values, pa.string())
    n_visits = len(visits_values)
    n_locations = len(locations_values)
    combined = pa.concat_arrays([visits_values, locations_values])
    codes, _ = _factorize_arrow_values(combined, sort=False)
    return codes.slice(0, n_visits), codes.slice(n_visits, n_locations)


def _arrow_array(series: nw.Series) -> Any:
    """Return one contiguous Arrow array for a motif input column."""
    import pyarrow as pa

    values = series.to_arrow()
    return values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values


def daily_motifs(
    visits: Any,
    *,
    uid_col: str | None = None,
    location_col: str | None = None,
    purpose_col: str | None = None,
    datetime_col: str | None = None,
    end_datetime_col: str | None = None,
    duration_col: str | None = None,
) -> Any:
    """Compute one home-anchored mobility motif per user and day.

    The returned dataframe matches the input backend and contains the resolved
    user-ID column, ``date``, and ``motif_id``. Row ordering is detected
    automatically before selecting the native kernel.
    """
    import pyarrow as pa

    df = nw.from_native(visits, eager_only=True)
    backend = df.implementation
    uid_col, location_col, purpose_col, datetime_col, end_datetime_col, duration_col = _detect_visit_columns(
        df,
        uid_col=uid_col,
        location_col=location_col,
        purpose_col=purpose_col,
        datetime_col=datetime_col,
        end_datetime_col=end_datetime_col,
        duration_col=duration_col,
    )
    uid_dtype = df.schema[uid_col]
    df = _strip_time_zone(df, datetime_col)
    df = _strip_time_zone(df, end_datetime_col)
    df = df.with_columns(
        nw.col(datetime_col).cast(nw.Datetime("us")).alias(datetime_col),
        nw.col(end_datetime_col).cast(nw.Datetime("us")).alias(end_datetime_col),
        nw.col(purpose_col).cast(nw.String).alias(purpose_col),
    )

    timestamps = _extract_timestamp_arrow(df, datetime_col)
    presorted = _auto_presorted(df, uid_col, timestamps)
    if presorted:
        uid_labels, ends = _build_presorted_user_ends(df, uid_col)
        indices = None
    else:
        uid_labels, indices, ends = _build_indexed_user_ranges(df, uid_col, timestamps)

    location_values = _encode_locations(df, location_col)

    purpose_series = df.get_column(purpose_col)
    purpose_source = purpose_series.to_native()
    if not hasattr(purpose_source, "__arrow_c_stream__"):
        purpose_source = pa.chunked_array([_arrow_array(purpose_series)])
    raw_purpose_codes, home_purpose_code, _unmatched_purpose_code = _core.encode_motif_purposes(purpose_source)
    purpose_codes = pa.array(_as_arrow(raw_purpose_codes))

    start_timestamps = _arrow_array(df.get_column(datetime_col))
    end_timestamps = _arrow_array(df.get_column(end_datetime_col))
    durations = None
    if duration_col is not None:
        duration_series = df.get_column(duration_col).cast(nw.Float64).fill_null(0.0)
        durations = _arrow_array(duration_series)

    if presorted:
        raw_users, raw_dates, raw_motifs = _core.daily_motifs_presorted(
            location_values,
            purpose_codes,
            start_timestamps,
            end_timestamps,
            durations,
            ends,
            home_purpose_code,
        )
    else:
        raw_users, raw_dates, raw_motifs = _core.daily_motifs_indexed(
            location_values,
            purpose_codes,
            start_timestamps,
            end_timestamps,
            durations,
            indices,
            ends,
            home_purpose_code,
        )

    return _assemble_motif_result(uid_col, uid_dtype, uid_labels, backend, raw_users, raw_dates, raw_motifs)


def _assemble_motif_result(
    uid_col: str,
    uid_dtype: Any,
    uid_labels: Any,
    backend: Any,
    raw_users: Any,
    raw_dates: Any,
    raw_motifs: Any,
) -> Any:
    """Build the ``[uid_col, "date", "motif_id"]`` result shared by :func:`daily_motifs`
    and :func:`daily_motifs_from_staypoints`.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    user_indices = pa.array(_as_arrow(raw_users))
    date_ids = pa.array(_as_arrow(raw_dates))
    motif_ids = pa.array(_as_arrow(raw_motifs))
    user_ids = pc.take(uid_labels, user_indices)
    return (
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


def daily_motifs_from_staypoints(
    staypoints: Any,
    locations: Any,
) -> Any:
    """Compute daily motifs directly from a ``Staypoints``/``Locations`` pair.

    Projects each ``Locations`` row's ``purpose`` (as produced by
    ``Locations.identify``) onto its ``Staypoints`` rows, uppercased
    (``identify_locations`` emits lowercase ``"home"``/``"work"``/``"other"``,
    while the motif purpose encoder anchors on an exact-match ``"HOME"``).

    Builds a small ``(user_idx, location_code) -> purpose_code`` lookup from
    ``locations`` (one row per ``(uid, location_id)``) and passes it straight
    to a Rust kernel that resolves each visit row's purpose lazily during the
    per-user scan, rather than materializing a full-length purpose column
    alongside every staypoint row; see ``fastmob-core``'s
    ``compute_daily_motifs_indexed_joined``.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    uid_col = staypoints.uid_col
    if uid_col is None:
        raise ValueError("generate_daily_motifs requires staypoints to have a uid column")
    if locations.scope != "user":
        raise ValueError("generate_daily_motifs requires user-scoped Locations; purposes are user-specific")

    sp_nw = nw.from_native(staypoints.df, eager_only=True)
    if "location_id" not in sp_nw.columns:
        raise ValueError("generate_daily_motifs requires staypoints to have a 'location_id' column")

    loc_nw = nw.from_native(locations.df, eager_only=True)
    if uid_col not in loc_nw.columns:
        raise ValueError(f"generate_daily_motifs requires locations to have a {uid_col!r} column")

    started_at_col = staypoints.started_at_col
    finished_at_col = staypoints.finished_at_col
    sp_nw = _strip_time_zone(sp_nw, started_at_col)
    sp_nw = _strip_time_zone(sp_nw, finished_at_col)
    sp_nw = sp_nw.with_columns(
        nw.col(started_at_col).cast(nw.Datetime("us")),
        nw.col(finished_at_col).cast(nw.Datetime("us")),
    )
    uid_dtype = sp_nw.schema[uid_col]
    backend = sp_nw.implementation

    presorted = _auto_presorted(sp_nw, uid_col)

    if presorted:
        uid_labels, ends = _build_presorted_user_ends(sp_nw, uid_col)
        indices = None
    else:
        timestamps = _extract_timestamp_arrow(sp_nw, started_at_col)
        uid_labels, indices, ends = _build_indexed_user_ranges(sp_nw, uid_col, timestamps)

    location_dtype = sp_nw.schema["location_id"]
    location_id_col = locations.location_id_col
    visit_location_codes, lookup_location_codes = _encode_locations_pair(
        sp_nw.get_column("location_id"), loc_nw.get_column(location_id_col), location_dtype
    )

    purpose_series = loc_nw.get_column("purpose").cast(nw.String).str.to_uppercase()
    purpose_source = purpose_series.to_native()
    if not hasattr(purpose_source, "__arrow_c_stream__"):
        purpose_source = pa.chunked_array([_arrow_array(purpose_series)])
    raw_lookup_purpose_codes, home_purpose_code, unmatched_purpose_code = _core.encode_motif_purposes(purpose_source)
    lookup_purpose_codes = pa.array(_as_arrow(raw_lookup_purpose_codes))

    loc_uid_values = pa.array(loc_nw.get_column(uid_col).to_arrow())
    matched_user_idx = pc.index_in(loc_uid_values, value_set=uid_labels)
    keep_mask = pc.is_valid(matched_user_idx)
    lookup_user_idx = pc.cast(pc.filter(matched_user_idx, keep_mask), pa.uint32())
    lookup_location_code = pc.filter(lookup_location_codes, keep_mask)
    lookup_purpose_code = pc.filter(lookup_purpose_codes, keep_mask)

    start_timestamps = _arrow_array(sp_nw.get_column(started_at_col))
    end_timestamps = _arrow_array(sp_nw.get_column(finished_at_col))
    durations = None
    duration_col = _pick_existing_column(sp_nw.columns, DURATION_CANDIDATES)
    if duration_col is not None:
        duration_series = sp_nw.get_column(duration_col).cast(nw.Float64).fill_null(0.0)
        durations = _arrow_array(duration_series)

    if presorted:
        raw_users, raw_dates, raw_motifs = _core.daily_motifs_presorted_joined(
            visit_location_codes,
            start_timestamps,
            end_timestamps,
            durations,
            lookup_user_idx,
            lookup_location_code,
            lookup_purpose_code,
            ends,
            home_purpose_code,
            unmatched_purpose_code,
        )
    else:
        raw_users, raw_dates, raw_motifs = _core.daily_motifs_indexed_joined(
            visit_location_codes,
            start_timestamps,
            end_timestamps,
            durations,
            lookup_user_idx,
            lookup_location_code,
            lookup_purpose_code,
            indices,
            ends,
            home_purpose_code,
            unmatched_purpose_code,
        )

    return _assemble_motif_result(uid_col, uid_dtype, uid_labels, backend, raw_users, raw_dates, raw_motifs)


def motif_distribution(daily: Any, motif_id_col: str = "motif_id") -> Any:
    """Aggregate daily motif IDs into count and percentage columns."""
    df = nw.from_native(daily, eager_only=True)
    if motif_id_col not in df.columns:
        raise ValueError(f"Motif ID column {motif_id_col!r} does not exist.")
    total = len(df)
    if total == 0:
        import pyarrow as pa

        return nw.from_arrow(
            pa.table(
                {
                    motif_id_col: pa.array([], type=pa.int64()),
                    "count": pa.array([], type=pa.int64()),
                    "percentage": pa.array([], type=pa.float64()),
                }
            ),
            backend=df.implementation,
        ).to_native()
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
