"""Base machinery for privacy risk attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import narwhals as nw

from ._constants import DATETIME, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE, PRIVACY_RISK, PROBABILITY, UID
from ._dataframe import _as_frame

_CANDIDATE_UID = "__candidate_uid__"
_CANDIDATE_POS = "__candidate_pos__"
_COMBO_IDX = "__combo_idx__"
_EFFECTIVE_LENGTH = "__effective_length__"
_MATCH_COUNT = "__match_count__"
_POS = "__pos__"
_ROW_NR = "__row_nr__"
_TARGET_UID = "__target_uid__"
_USER_LENGTH = "__user_length__"


def _empty_frame(reference: nw.DataFrame, columns: list[str]) -> nw.DataFrame:
    return nw.from_dict({column: [] for column in columns}, backend=reference.implementation)


class Attack(ABC):
    """Abstract base class for privacy attacks.

    Parameters
    ----------
    knowledge_length:
        Number of observations known by the attacker. Values greater than a
        user's trajectory length are capped to that user's available rows.

    Raises
    ------
    ValueError
        If ``knowledge_length`` is less than 1.
    """

    def __init__(self, knowledge_length: int):
        self.knowledge_length = knowledge_length

    @property
    def knowledge_length(self) -> int:
        """Number of trajectory observations known by the attacker."""
        return self._knowledge_length

    @knowledge_length.setter
    def knowledge_length(self, val: int) -> None:
        if val < 1:
            raise ValueError("Parameter knowledge_length should not be less than 1")
        self._knowledge_length = val

    def _all_risks(
        self,
        traj: Any,
        targets: Any = None,
        force_instances: bool = False,
        show_progress: bool = False,
    ) -> Any:
        del show_progress
        df = _as_frame(traj)
        target_uids = self._target_uids(df, targets)
        instance_values = self._instance_values(df)
        candidate_values = self._candidate_values(df)
        instances = self._generate_instances(instance_values, target_uids)

        if len(instances) == 0:
            return self._empty_result(df, force_instances)

        match_counts = self._match_counts(candidate_values, instances)
        probs = match_counts.with_columns((1.0 / nw.col(_MATCH_COUNT)).alias(PROBABILITY))

        if force_instances:
            return self._force_instance_output(df, instances, probs).to_native()

        result = (
            probs.group_by(_TARGET_UID)
            .agg(nw.col(PROBABILITY).max().alias(PRIVACY_RISK))
            .rename({_TARGET_UID: UID})
            .sort(UID)
            .select([UID, PRIVACY_RISK])
        )
        return result.to_native()

    def _target_uids(self, df: nw.DataFrame, targets: Any) -> nw.DataFrame:
        if targets is None:
            return df.select(UID).unique().sort(UID)
        if isinstance(targets, list):
            return nw.from_dict({UID: targets}, backend=df.implementation).unique().sort(UID)
        return _as_frame(targets).select(UID).unique().sort(UID)

    def _instance_values(self, df: nw.DataFrame) -> nw.DataFrame:
        return self._with_user_position(df.select(self._instance_columns()))

    def _candidate_values(self, df: nw.DataFrame) -> nw.DataFrame:
        return self._with_user_position(df.select(self._candidate_columns())).rename(
            {UID: _CANDIDATE_UID, _POS: _CANDIDATE_POS}
        )

    def _with_user_position(self, df: nw.DataFrame) -> nw.DataFrame:
        return (
            df.with_row_index(_ROW_NR)
            .with_columns(
                nw.col(_ROW_NR).rank(method="ordinal").over(UID).cast(nw.Int64).alias(_POS),
                nw.len().over(UID).alias(_USER_LENGTH),
            )
            .drop(_ROW_NR)
        )

    def _instance_columns(self) -> list[str]:
        columns = [UID, LATITUDE, LONGITUDE]
        if self._has_datetime:
            columns.append(DATETIME)
        columns.extend(self._metric_columns())
        return columns

    def _candidate_columns(self) -> list[str]:
        return self._instance_columns()

    @property
    def _has_datetime(self) -> bool:
        return True

    def _metric_columns(self) -> list[str]:
        return []

    def _match_columns(self) -> list[str]:
        return [LATITUDE, LONGITUDE]

    def _generate_instances(self, values: nw.DataFrame, target_uids: nw.DataFrame) -> nw.DataFrame:
        target_values = (
            values.join(target_uids, on=UID, how="semi")
            .with_columns(
                nw.when(nw.col(_USER_LENGTH) < self.knowledge_length)
                .then(nw.col(_USER_LENGTH))
                .otherwise(self.knowledge_length)
                .cast(nw.Int64)
                .alias(_EFFECTIVE_LENGTH)
            )
        )
        if len(target_values) == 0:
            return self._empty_instances(values)

        lengths = [int(length) for length in sorted(target_values.select(_EFFECTIVE_LENGTH).unique().get_column(_EFFECTIVE_LENGTH).to_list())]
        frames = [self._generate_instances_of_length(target_values.filter(nw.col(_EFFECTIVE_LENGTH) == length), length) for length in lengths]
        frames = [frame for frame in frames if len(frame) > 0]
        if not frames:
            return self._empty_instances(values)
        return nw.concat(frames, how="vertical").sort([_TARGET_UID, INSTANCE, INSTANCE_ELEMENT])

    def _generate_instances_of_length(self, values: nw.DataFrame, length: int) -> nw.DataFrame:
        source = values.rename({UID: _TARGET_UID})
        if length == 1:
            wide = source.select([_TARGET_UID, nw.col(_POS).alias(f"{_POS}_1"), *self._wide_value_exprs(1)])
        else:
            wide = source.select([_TARGET_UID, nw.col(_POS).alias(f"{_POS}_1"), *self._wide_value_exprs(1)])
            for idx in range(2, length + 1):
                right = source.select([_TARGET_UID, nw.col(_POS).alias(f"{_POS}_{idx}"), *self._wide_value_exprs(idx)])
                wide = wide.join(right, on=_TARGET_UID, how="inner").filter(nw.col(f"{_POS}_{idx}") > nw.col(f"{_POS}_{idx - 1}"))

        sort_cols = [_TARGET_UID, *[f"{_POS}_{idx}" for idx in range(1, length + 1)]]
        wide = (
            wide.sort(sort_cols)
            .with_row_index(_COMBO_IDX)
            .with_columns(nw.col(_COMBO_IDX).rank(method="ordinal").over(_TARGET_UID).cast(nw.Int64).alias(INSTANCE))
            .drop(_COMBO_IDX)
        )
        return self._wide_instances_to_long(wide, length)

    def _wide_value_exprs(self, idx: int) -> list[Any]:
        return [nw.col(column).alias(f"{column}_{idx}") for column in self._value_columns()]

    def _wide_instances_to_long(self, wide: nw.DataFrame, length: int) -> nw.DataFrame:
        frames = []
        for idx in range(1, length + 1):
            frames.append(
                wide.select(
                    [
                        _TARGET_UID,
                        INSTANCE,
                        nw.lit(idx).alias(INSTANCE_ELEMENT),
                        *[nw.col(f"{column}_{idx}").alias(column) for column in self._value_columns()],
                    ]
                )
            )
        return nw.concat(frames, how="vertical").sort([_TARGET_UID, INSTANCE, INSTANCE_ELEMENT])

    def _value_columns(self) -> list[str]:
        return [column for column in self._instance_columns() if column != UID]

    def _empty_instances(self, reference: nw.DataFrame) -> nw.DataFrame:
        return _empty_frame(reference, [_TARGET_UID, INSTANCE, INSTANCE_ELEMENT, *self._value_columns()])

    def _empty_result(self, reference: nw.DataFrame, force_instances: bool) -> Any:
        if force_instances:
            return nw.from_dict(
                {
                    LATITUDE: [],
                    LONGITUDE: [],
                    DATETIME: [],
                    UID: [],
                    INSTANCE: [],
                    INSTANCE_ELEMENT: [],
                    PROBABILITY: [],
                },
                backend=reference.implementation,
            ).to_native()
        return nw.from_dict({UID: [], PRIVACY_RISK: []}, backend=reference.implementation).to_native()

    def _force_instance_output(self, reference: nw.DataFrame, instances: nw.DataFrame, probs: nw.DataFrame) -> nw.DataFrame:
        out = instances.join(probs, on=[_TARGET_UID, INSTANCE], how="left")
        if DATETIME not in out.columns:
            out = out.with_columns(nw.lit(None).alias(DATETIME))
        return (
            out.rename({_TARGET_UID: UID})
            .select([LATITUDE, LONGITUDE, DATETIME, UID, INSTANCE, INSTANCE_ELEMENT, PROBABILITY])
            .sort([UID, INSTANCE, INSTANCE_ELEMENT])
        )

    def _required_key_count(self, instances: nw.DataFrame, keys: list[str]) -> nw.DataFrame:
        return (
            instances.select([_TARGET_UID, INSTANCE, *keys])
            .unique()
            .group_by([_TARGET_UID, INSTANCE])
            .agg(nw.len().alias("__required_keys__"))
        )

    def _candidate_key_count(self, candidates: nw.DataFrame, keys: list[str]) -> nw.DataFrame:
        return (
            candidates.select([_CANDIDATE_UID, *keys])
            .unique()
            .group_by(_CANDIDATE_UID)
            .agg(nw.len().alias("__candidate_keys__"))
        )

    def _set_match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame, keys: list[str]) -> nw.DataFrame:
        required = self._required_key_count(instances, keys)
        matched = (
            instances.select([_TARGET_UID, INSTANCE, *keys])
            .unique()
            .join(candidates.select([_CANDIDATE_UID, *keys]).unique(), on=keys, how="inner")
            .group_by([_TARGET_UID, INSTANCE, _CANDIDATE_UID])
            .agg(nw.len().alias("__matched_keys__"))
            .join(required, on=[_TARGET_UID, INSTANCE], how="inner")
            .filter(nw.col("__matched_keys__") == nw.col("__required_keys__"))
        )
        return self._count_candidates(matched)

    def _multiset_match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame, keys: list[str]) -> nw.DataFrame:
        required_counts = (
            instances.group_by([_TARGET_UID, INSTANCE, *keys])
            .agg(nw.len().alias("__required_count__"))
        )
        required_keys = required_counts.group_by([_TARGET_UID, INSTANCE]).agg(nw.len().alias("__required_keys__"))
        candidate_counts = (
            candidates.group_by([_CANDIDATE_UID, *keys])
            .agg(nw.len().alias("__candidate_count__"))
        )
        matched = (
            required_counts.join(candidate_counts, on=keys, how="inner")
            .filter(nw.col("__candidate_count__") >= nw.col("__required_count__"))
            .group_by([_TARGET_UID, INSTANCE, _CANDIDATE_UID])
            .agg(nw.len().alias("__matched_keys__"))
            .join(required_keys, on=[_TARGET_UID, INSTANCE], how="inner")
            .filter(nw.col("__matched_keys__") == nw.col("__required_keys__"))
        )
        return self._count_candidates(matched)

    def _count_candidates(self, matched: nw.DataFrame) -> nw.DataFrame:
        return (
            matched.group_by([_TARGET_UID, INSTANCE])
            .agg(nw.len().alias(_MATCH_COUNT))
            .select([_TARGET_UID, INSTANCE, _MATCH_COUNT])
        )

    @abstractmethod
    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        pass


__all__ = ["Attack"]
