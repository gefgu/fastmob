"""Shared execution adapter for Rust-backed privacy risk functions."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import privacy_assess_risk
from fastmob.utils._common import (
    _build_indexed_user_ranges,
    _build_presorted_user_ends,
    _detect_trajectory_columns,
    _extract_timestamps,
    _factorize_arrow_values,
)

_NO_ROW = (1 << 64) - 1


def _target_indices(df: nw.DataFrame, uid_col: str | None, uid_values: Any, targets: Any) -> Any:
    import pyarrow as pa

    if uid_col is None:
        return pa.array([0] if len(df) else [], type=pa.uint64())
    if targets is None:
        target_uids = df.select(uid_col).unique().sort(uid_col).get_column(uid_col).to_list()
    elif isinstance(targets, list):
        target_uids = targets
    else:
        target_uids = nw.from_native(targets, eager_only=True).get_column(uid_col).unique().sort().to_list()
    positions = {value: index for index, value in enumerate(uid_values.to_pylist())}
    return pa.array([positions[value] for value in target_uids if value in positions], type=pa.uint64())


def _time_keys(df: nw.DataFrame, datetime_col: str, precision: str) -> Any:
    formats = {"year": "%Y", "month": "%Y%m", "day": "%Y%m%d", "hour": "%Y%m%d%H", "minute": "%Y%m%d%H%M", "second": "%Y%m%d%H%M%S"}
    normalized = precision.lower()
    if normalized not in formats:
        raise ValueError("time_precision must be one of: Year, Month, Day, Hour, Minute, Second")
    values = df.with_columns(nw.col(datetime_col).dt.to_string(formats[normalized]).alias("__privacy_time_key__")).get_column("__privacy_time_key__")
    codes, _ = _factorize_arrow_values(values, sort=False)
    return codes


def _uids(uid_values: Any, indices: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    if uid_values is None:
        return [1] * len(indices)
    return pc.take(uid_values, pa.array(indices, type=pa.uint64()))


def _result(
    df: nw.DataFrame,
    result: Any,
    *,
    uid_values: Any,
    uid_col: str | None,
    lat_col: str,
    lng_col: str,
    datetime_col: str | None,
    force_instances: bool,
) -> Any:
    output_uid = uid_col or "uid"
    if not force_instances:
        return nw.from_dict(
            {output_uid: _uids(uid_values, result.user_indices), "risk": result.risks},
            backend=df.implementation,
        ).sort(output_uid).to_native()

    columns: dict[str, Any] = {
        lat_col: result.force_lats,
        lng_col: result.force_lngs,
    }
    if datetime_col is not None:
        values = df.get_column(datetime_col).to_list()
        columns[datetime_col] = [None if int(index) == _NO_ROW else values[int(index)] for index in result.row_indices]
    columns.update(
        {
            output_uid: _uids(uid_values, result.force_user_indices),
            "instance": result.instances,
            "instance_elem": result.elems,
            "prob": result.probs,
        }
    )
    return nw.from_dict(columns, backend=df.implementation).sort([output_uid, "instance", "instance_elem"]).to_native()


def assess_risk(
    traj: Any,
    knowledge_length: int,
    *,
    attack: str,
    targets: Any = None,
    force_instances: bool = False,
    tolerance: float = 0.0,
    time_precision: str | None = None,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
    require_datetime: bool = False,
    h3_resolution: int = 12,
    locations: Any | None = None,
    location_id_col: str = "location_id",
) -> Any:
    if knowledge_length < 1:
        raise ValueError("knowledge_length must be greater than zero")
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be in the interval [0.0, 1.0]")
    if isinstance(h3_resolution, bool) or not isinstance(h3_resolution, int) or not 0 <= h3_resolution <= 15:
        raise ValueError("h3_resolution must be an integer between 0 and 15")
    df = nw.from_native(traj, eager_only=True)
    df, datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        cast_float_coordinates=True,
        require_datetime=require_datetime,
    )
    location_ids = None
    if locations is not None:
        if locations.scope != "global":
            raise ValueError("privacy risk assessment requires global Locations")
        if location_id_col not in df.columns:
            raise ValueError(f"Input data is missing global location-ID column {location_id_col!r}")
        known = set(nw.from_native(locations.df, eager_only=True).get_column(locations.location_id_col).to_list())
        present = set(df.get_column(location_id_col).drop_nulls().to_list())
        if present - known:
            raise ValueError("Input contains location IDs absent from the global Locations catalogue")
        location_ids, _ = _factorize_arrow_values(df.get_column(location_id_col), sort=False)
        if force_instances:
            raise ValueError("force_instances is not supported with externally assigned global locations")
    time_keys = _time_keys(df, datetime_col, time_precision) if time_precision is not None else None
    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        indices = None
    else:
        timestamps = _extract_timestamps(df, datetime_col) if attack == "sequence" else None
        uid_values, indices, ends = _build_indexed_user_ranges(df, uid_col, timestamps)
    result = privacy_assess_risk(
        df.get_column(lat_col).to_arrow(),
        df.get_column(lng_col).to_arrow(),
        ends,
        _target_indices(df, uid_col, uid_values, targets),
        attack,
        knowledge_length,
        time_keys=time_keys,
        indices=indices,
        tolerance=tolerance,
        force_instances=force_instances,
        h3_resolution=h3_resolution,
        location_ids=location_ids,
    )
    return _result(
        df,
        result,
        uid_values=uid_values,
        uid_col=uid_col,
        lat_col=lat_col,
        lng_col=lng_col,
        datetime_col=datetime_col if require_datetime else None,
        force_instances=force_instances,
    )
