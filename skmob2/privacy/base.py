"""Base machinery for privacy risk attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any

from ._constants import DATETIME, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE, PRIVACY_RISK, PROBABILITY, UID
from ._dataframe import _backend, _records, _rows_by_uid, _to_native, _uid_values


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
        backend = _backend(traj)
        rows = _records(traj)

        if targets is None:
            target_uids = {row[UID] for row in rows}
        elif isinstance(targets, list):
            target_uids = set(targets)
        else:
            target_uids = _uid_values(targets)

        grouped = _rows_by_uid(rows)
        target_grouped = {uid: grouped[uid] for uid in sorted(grouped) if uid in target_uids}
        prepared_group_counts = self._prepare_group_counts(grouped.values())

        if force_instances:
            out = {
                LATITUDE: [],
                LONGITUDE: [],
                DATETIME: [],
                UID: [],
                INSTANCE: [],
                INSTANCE_ELEMENT: [],
                PROBABILITY: [],
            }
            for single_rows in target_grouped.values():
                inst_result = self._risk(single_rows, prepared_group_counts, force_instances=True)
                for key in out:
                    out[key].extend(inst_result[key])
            return _to_native(out, backend)

        uids: list[Any] = []
        risks: list[float] = []
        for uid, single_rows in target_grouped.items():
            uids.append(uid)
            risks.append(self._risk(single_rows, prepared_group_counts, force_instances=False))
        return _to_native({UID: uids, PRIVACY_RISK: risks}, backend)

    def _generate_instances(self, single_traj: Any):
        rows = _records(single_traj)
        size = len(rows)
        length = size if self.knowledge_length > size else self.knowledge_length
        return combinations(rows, length)

    def _prepare_group(self, single_traj: Any) -> Any:
        return single_traj

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return None

    def _prepare_group_counts(self, groups: Any) -> list[tuple[Any, int]]:
        grouped: list[tuple[Any, int]] = []
        keyed: dict[Any, list[Any]] = {}
        for group in groups:
            prepared = self._prepare_group(group)
            key = self._prepared_group_key(prepared)
            if key is None:
                grouped.append((prepared, 1))
                continue
            existing = keyed.get(key)
            if existing is None:
                keyed[key] = [prepared, 1]
            else:
                existing[1] += 1
        grouped.extend((prepared, count) for prepared, count in keyed.values())
        return grouped

    def _prepare_instance(self, instance: Any) -> Any:
        return instance

    def _match_prepared(self, prepared_group: Any, prepared_instance: Any) -> int:
        return self._match(prepared_group, prepared_instance)

    def _risk(self, single_traj: Any, all_groups: list[tuple[Any, int]], force_instances: bool = False) -> Any:
        instances = self._generate_instances(single_traj)
        risk = 0.0

        if force_instances:
            inst_data = {
                LATITUDE: [],
                LONGITUDE: [],
                DATETIME: [],
                UID: [],
                INSTANCE: [],
                INSTANCE_ELEMENT: [],
                PROBABILITY: [],
            }
            inst_id = 1
            for instance in instances:
                prepared_instance = self._prepare_instance(instance)
                matches = sum(count for group, count in all_groups if self._match_prepared(group, prepared_instance))
                prob = 1.0 / matches
                for elem_count, elem in enumerate(instance, start=1):
                    values = list(elem.values())
                    inst_data[LATITUDE].append(values[0])
                    inst_data[LONGITUDE].append(values[1])
                    inst_data[DATETIME].append(values[2])
                    inst_data[UID].append(values[3])
                    inst_data[INSTANCE].append(inst_id)
                    inst_data[INSTANCE_ELEMENT].append(elem_count)
                    inst_data[PROBABILITY].append(prob)
                inst_id += 1
            return inst_data

        for instance in instances:
            prepared_instance = self._prepare_instance(instance)
            matches = sum(count for group, count in all_groups if self._match_prepared(group, prepared_instance))
            prob = 1.0 / matches
            if prob > risk:
                risk = prob
            if risk == 1.0:
                break
        return risk

    @abstractmethod
    def assess_risk(
        self,
        traj: Any,
        targets: Any = None,
        force_instances: bool = False,
        show_progress: bool = False,
    ) -> Any:
        """Assess privacy risk for users in a trajectory DataFrame.

        Parameters
        ----------
        traj:
            Trajectory DataFrame with ``uid``, ``lat``, ``lng``, and
            ``datetime`` columns. Any Narwhals-compatible eager DataFrame is
            accepted.
        targets:
            Optional user selection. Pass None for all users, a list of user
            IDs, or a DataFrame whose ``uid`` values identify the users to
            assess.
        force_instances:
            If False, return one maximum risk per user. If True, return one
            row per known observation in every generated instance with that
            instance's re-identification probability.
        show_progress:
            Accepted for compatibility; currently ignored.

        Returns
        -------
        DataFrame
            If ``force_instances`` is False, columns are ``["uid", "risk"]``.
            If True, columns are ``["lat", "lng", "datetime", "uid",
            "instance", "instance_elem", "prob"]``. The returned backend
            matches the input backend.
        """
        pass

    @abstractmethod
    def _match(self, single_traj: Any, instance: Any) -> int:
        pass


__all__ = ["Attack"]
