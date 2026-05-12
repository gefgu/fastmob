"""Location-based privacy attacks."""

from __future__ import annotations

from typing import Any

from ._constants import DATETIME, LATITUDE, LONGITUDE, PRECISION_LEVELS, TEMP, UID
from ._dataframe import _as_frame, _date_time_precision, _group_counts, _records, _records_like, _to_native
from .base import Attack


class LocationAttack(Attack):
    """Assess risk from a set of visited locations.

    The attacker knows up to ``knowledge_length`` location observations for a
    user. Matching ignores visit order and time, but preserves repeated
    occurrences of the same latitude/longitude pair.
    """

    def __init__(self, knowledge_length: int):
        super().__init__(knowledge_length)

    def assess_risk(
        self,
        traj: Any,
        targets: Any = None,
        force_instances: bool = False,
        show_progress: bool = False,
    ) -> Any:
        """Assess privacy risk for users in a trajectory DataFrame.

        Parameters are the same as :meth:`skmob2.privacy.base.Attack.assess_risk`.
        """
        sorted_traj = _as_frame(traj).sort([UID, DATETIME]).to_native()
        return self._all_risks(sorted_traj, targets, force_instances, show_progress)

    def _match(self, single_traj: Any, instance: Any) -> int:
        rows = _records(single_traj)
        inst_rows = _records_like(instance, list(rows[0]) if rows else [])
        return self._match_prepared(self._prepare_group(rows), self._prepare_instance(inst_rows))

    def _prepare_group(self, single_traj: Any) -> Any:
        return _group_counts(_records(single_traj), [LATITUDE, LONGITUDE])

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return tuple(sorted(prepared_group.items()))

    def _prepare_instance(self, instance: Any) -> Any:
        return _group_counts(_records_like(instance, []), [LATITUDE, LONGITUDE])

    def _match_prepared(self, locs: Any, inst: Any) -> int:
        for key, inst_count in inst.items():
            if locs.get(key, 0) < inst_count:
                return 0
        return 1


class LocationSequenceAttack(Attack):
    """Assess risk from a chronologically ordered sequence of locations.

    The attacker knows up to ``knowledge_length`` locations and their relative
    order. Matching ignores timestamps but requires the known locations to
    appear as an ordered subsequence of a user's trajectory.
    """

    def __init__(self, knowledge_length: int):
        super().__init__(knowledge_length)

    def assess_risk(
        self,
        traj: Any,
        targets: Any = None,
        force_instances: bool = False,
        show_progress: bool = False,
    ) -> Any:
        """Assess privacy risk for users in a trajectory DataFrame.

        Parameters are the same as :meth:`skmob2.privacy.base.Attack.assess_risk`.
        """
        sorted_traj = _as_frame(traj).sort([UID, DATETIME]).to_native()
        return self._all_risks(sorted_traj, targets, force_instances, show_progress)

    def _match(self, single_traj: Any, instance: Any) -> int:
        rows = _records(single_traj)
        inst = _records_like(instance, list(rows[0]) if rows else [])
        return self._match_prepared(self._prepare_group(rows), self._prepare_instance(inst))

    def _prepare_group(self, single_traj: Any) -> Any:
        return [(row[LATITUDE], row[LONGITUDE]) for row in _records(single_traj)]

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return tuple(prepared_group)

    def _prepare_instance(self, instance: Any) -> Any:
        return [(row[LATITUDE], row[LONGITUDE]) for row in _records_like(instance, [])]

    def _match_prepared(self, rows: Any, inst: Any) -> int:
        if not inst:
            return 1
        inst_idx = 0
        for row in rows:
            current = inst[inst_idx]
            if current == row:
                inst_idx += 1
                if inst_idx == len(inst):
                    return 1
        return 0


class LocationTimeAttack(Attack):
    """Assess risk from locations observed at a selected time precision.

    The attacker knows up to ``knowledge_length`` location/time observations.
    Matching ignores order but requires latitude, longitude, and the datetime
    value truncated to ``time_precision`` to match.

    Parameters
    ----------
    knowledge_length:
        Number of observations known by the attacker.
    time_precision:
        Datetime precision used for matching. Valid values are ``"Year"``,
        ``"Month"``, ``"Day"``, ``"Hour"``, ``"Minute"``, ``"Second"``, and
        their lowercase forms. The default is ``"Hour"``.

    Raises
    ------
    ValueError
        If ``knowledge_length`` is less than 1 or ``time_precision`` is not a
        supported value.
    """

    def __init__(self, knowledge_length: int, time_precision: str = "Hour"):
        self.time_precision = time_precision
        super().__init__(knowledge_length)

    @property
    def time_precision(self) -> str:
        """Datetime precision used to match known observations."""
        return self._time_precision

    @time_precision.setter
    def time_precision(self, val: str) -> None:
        if val not in PRECISION_LEVELS:
            raise ValueError("Possible time precisions are: Year, Month, Day, Hour, Minute, Second")
        self._time_precision = val

    def assess_risk(
        self,
        traj: Any,
        targets: Any = None,
        force_instances: bool = False,
        show_progress: bool = False,
    ) -> Any:
        """Assess privacy risk for users in a trajectory DataFrame.

        Parameters are the same as :meth:`skmob2.privacy.base.Attack.assess_risk`.
        """
        sorted_df = _as_frame(traj).sort([UID, DATETIME])
        rows = []
        for row in sorted_df.rows(named=True):
            updated = dict(row)
            updated[TEMP] = _date_time_precision(updated[DATETIME], self.time_precision)
            rows.append(updated)
        columns = sorted_df.columns + [TEMP]
        transformed = {column: [row[column] for row in rows] for column in columns}
        return self._all_risks(
            _to_native(transformed, sorted_df.implementation), targets, force_instances, show_progress
        )

    def _match(self, single_traj: Any, instance: Any) -> int:
        rows = _records(single_traj)
        inst_rows = _records_like(instance, list(rows[0]) if rows else [])
        return self._match_prepared(self._prepare_group(rows), self._prepare_instance(inst_rows))

    def _prepare_group(self, single_traj: Any) -> Any:
        return _group_counts(_records(single_traj), [LATITUDE, LONGITUDE, TEMP])

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return tuple(sorted(prepared_group.items()))

    def _prepare_instance(self, instance: Any) -> Any:
        return _group_counts(_records_like(instance, []), [LATITUDE, LONGITUDE, TEMP])

    def _match_prepared(self, locs: Any, inst: Any) -> int:
        for key, inst_count in inst.items():
            if locs.get(key, 0) < inst_count:
                return 0
        return 1


__all__ = ["LocationAttack", "LocationSequenceAttack", "LocationTimeAttack"]
