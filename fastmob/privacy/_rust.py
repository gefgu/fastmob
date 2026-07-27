from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from fastmob._core import (
    privacy_assess_risk_indexed,
    privacy_assess_risk_presorted,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _build_indexed_user_ranges_fast,
    _build_presorted_user_ends,
    _build_time_ordered_user_ranges,
    _extract_timestamps_ms,
)

from ._constants import DATETIME, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE, PRIVACY_RISK, PROBABILITY, TEMP, UID
from ._dataframe import _as_frame, _with_date_time_precision

LOCATION = 0
SEQUENCE = 1
TIME = 2
UNIQUE_LOCATION = 3
FREQUENCY = 4
PROBABILITY_ATTACK = 5
PROPORTION = 6
HOME_WORK = 7

_NO_ROW = np.iinfo(np.uintp).max

_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _target_user_indices(df: nw.DataFrame, uid_values: list[Any] | None, targets: Any) -> np.ndarray:
    if uid_values is None:
        return np.array([0], dtype=np.uintp) if len(df) else np.array([], dtype=np.uintp)

    if targets is None:
        target_uids = df.select(UID).unique().sort(UID).get_column(UID).to_list()
    elif isinstance(targets, list):
        target_uids = (
            nw.from_dict({UID: targets}, backend=df.implementation).unique().sort(UID).get_column(UID).to_list()
        )
    else:
        target_uids = _as_frame(targets).select(UID).unique().sort(UID).get_column(UID).to_list()

    uid_to_index = {uid: idx for idx, uid in enumerate(uid_values)}
    return np.asarray([uid_to_index[uid] for uid in target_uids if uid in uid_to_index], dtype=np.uintp)


def _time_keys(df: nw.DataFrame, precision: str | None) -> np.ndarray | None:
    if precision is None:
        return None

    values = _with_date_time_precision(df, DATETIME, TEMP, precision).get_column(TEMP).to_list()
    codes: dict[Any, int] = {}
    out = np.empty(len(values), dtype=np.uint64)
    for idx, value in enumerate(values):
        code = codes.setdefault(value, len(codes))
        out[idx] = code
    return out


def _datetime_values(df: nw.DataFrame, row_indices: np.ndarray, include_datetime: bool) -> list[Any]:
    if not include_datetime:
        return [None] * len(row_indices)
    values = df.get_column(DATETIME).to_list()
    return [None if int(idx) == _NO_ROW else values[int(idx)] for idx in row_indices]


def _normal_result(df: nw.DataFrame, uid_values: list[Any] | None, user_indices: Any, risks: Any) -> Any:
    user_indices = np.asarray(user_indices, dtype=np.uintp)
    risks = np.asarray(risks, dtype=np.float64)
    if uid_values is None:
        values = [1] * len(risks)
    else:
        uid_array = np.asarray(uid_values, dtype=object)
        values = uid_array.take(user_indices).tolist()
    return nw.from_dict({UID: values, PRIVACY_RISK: risks}, backend=df.implementation).sort(UID).to_native()


def _force_result(
    df: nw.DataFrame,
    uid_values: list[Any] | None,
    include_datetime: bool,
    lats: Any,
    lngs: Any,
    row_indices: Any,
    user_indices: Any,
    instances: Any,
    elems: Any,
    probs: Any,
) -> Any:
    row_indices = np.asarray(row_indices, dtype=np.uintp)
    user_indices = np.asarray(user_indices, dtype=np.uintp)
    if uid_values is None:
        uids = [1] * len(user_indices)
    else:
        uid_array = np.asarray(uid_values, dtype=object)
        uids = uid_array.take(user_indices).tolist()
    out = nw.from_dict(
        {
            LATITUDE: np.asarray(lats, dtype=np.float64),
            LONGITUDE: np.asarray(lngs, dtype=np.float64),
            DATETIME: _datetime_values(df, row_indices, include_datetime),
            UID: uids,
            INSTANCE: np.asarray(instances, dtype=np.int64),
            INSTANCE_ELEMENT: np.asarray(elems, dtype=np.int64),
            PROBABILITY: np.asarray(probs, dtype=np.float64),
        },
        backend=df.implementation,
    )
    return out.sort([UID, INSTANCE, INSTANCE_ELEMENT]).to_native()


def assess_risk_rust(
    traj: Any,
    *,
    attack_kind: int,
    knowledge_length: int,
    tolerance: float = 0.0,
    targets: Any = None,
    force_instances: bool = False,
    presorted: bool = False,
    time_precision: str | None = None,
    include_datetime: bool = False,
) -> Any:
    df = _as_frame(traj)
    if df.schema[LATITUDE] != nw.Float64 or df.schema[LONGITUDE] != nw.Float64:
        df = df.with_columns(nw.col(LATITUDE).cast(nw.Float64), nw.col(LONGITUDE).cast(nw.Float64))

    lats = df.get_column(LATITUDE).to_arrow()
    lngs = df.get_column(LONGITUDE).to_arrow()
    time_keys = _time_keys(df, time_precision)

    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, UID)
        target_indices = _target_user_indices(df, uid_values, targets)
        result = privacy_assess_risk_presorted(
            lats,
            lngs,
            time_keys,
            ends,
            target_indices,
            attack_kind,
            knowledge_length,
            tolerance,
            force_instances,
        )
    else:
        if attack_kind == SEQUENCE:
            timestamps = _extract_timestamps_ms(df, DATETIME)
            uid_values, indices, ends = _build_time_ordered_user_ranges(
                df,
                UID,
                DATETIME,
                _TIMESTAMP_EXTRACTOR.get_ops(df)["extract_data"](timestamps),
            )
        else:
            uid_values, indices, ends = _build_indexed_user_ranges_fast(df, UID)
        target_indices = _target_user_indices(df, uid_values, targets)
        result = privacy_assess_risk_indexed(
            lats,
            lngs,
            time_keys,
            indices,
            ends,
            target_indices,
            attack_kind,
            knowledge_length,
            tolerance,
            force_instances,
        )

    user_indices, risks, force_lats, force_lngs, row_indices, force_user_indices, instances, elems, probs = result
    if force_instances:
        return _force_result(
            df,
            uid_values,
            include_datetime,
            force_lats,
            force_lngs,
            row_indices,
            force_user_indices,
            instances,
            elems,
            probs,
        )
    return _normal_result(df, uid_values, user_indices, risks)


__all__ = [
    "FREQUENCY",
    "HOME_WORK",
    "LOCATION",
    "PROBABILITY_ATTACK",
    "PROPORTION",
    "SEQUENCE",
    "TIME",
    "UNIQUE_LOCATION",
    "assess_risk_rust",
]
