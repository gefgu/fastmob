"""Frequency-based privacy attacks."""

from __future__ import annotations

from typing import Any

from ._constants import FREQUENCY, LATITUDE, LONGITUDE, PROBABILITY
from ._dataframe import _frequency_vector, _probability_vector, _records, _records_like
from .base import Attack


class UniqueLocationAttack(Attack):
    """Assess risk from the set of unique locations visited by each user.

    The attacker knows up to ``knowledge_length`` distinct locations. Matching
    ignores the number of visits, visit order, and timestamps.
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
        return self._all_risks(_frequency_vector(traj), targets, force_instances, show_progress)

    def _match(self, single_traj: Any, instance: Any) -> int:
        rows = _records(single_traj)
        inst_rows = _records_like(instance, list(rows[0]) if rows else [])
        return self._match_prepared(self._prepare_group(rows), self._prepare_instance(inst_rows))

    def _prepare_group(self, single_traj: Any) -> Any:
        return {(row[LATITUDE], row[LONGITUDE]) for row in _records(single_traj)}

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return tuple(sorted(prepared_group))

    def _prepare_instance(self, instance: Any) -> Any:
        return [(row[LATITUDE], row[LONGITUDE]) for row in _records_like(instance, [])]

    def _match_prepared(self, locs: Any, inst_locs: Any) -> int:
        return int(all(loc in locs for loc in inst_locs))


class LocationFrequencyAttack(Attack):
    """Assess risk from locations and visit frequencies.

    The attacker knows up to ``knowledge_length`` locations and their visit
    counts. A known frequency matches a user's frequency when it falls within
    ``tolerance`` of the user's value.

    Parameters
    ----------
    knowledge_length:
        Number of observations known by the attacker.
    tolerance:
        Relative frequency tolerance in ``[0.0, 1.0]``. The default is 0.0.

    Raises
    ------
    ValueError
        If ``knowledge_length`` is less than 1 or ``tolerance`` is outside
        ``[0.0, 1.0]``.
    """

    def __init__(self, knowledge_length: int, tolerance: float = 0.0):
        self.tolerance = tolerance
        super().__init__(knowledge_length)

    @property
    def tolerance(self) -> float:
        """Relative tolerance used when comparing known values."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val: float) -> None:
        if val > 1.0 or val < 0.0:
            raise ValueError("Tolerance should be in the interval [0.0,1.0]")
        self._tolerance = val

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
        return self._all_risks(_frequency_vector(traj), targets, force_instances, show_progress)

    def _match(self, single_traj: Any, instance: Any) -> int:
        rows = _records(single_traj)
        inst_rows = _records_like(instance, list(rows[0]) if rows else [])
        return self._match_prepared(self._prepare_group(rows), self._prepare_instance(inst_rows))

    def _prepare_group(self, single_traj: Any) -> Any:
        return {(row[LATITUDE], row[LONGITUDE]): row[FREQUENCY] for row in _records(single_traj)}

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return tuple(sorted(prepared_group.items()))

    def _prepare_instance(self, instance: Any) -> Any:
        return [((row[LATITUDE], row[LONGITUDE]), row[FREQUENCY]) for row in _records_like(instance, [])]

    def _match_prepared(self, locs: Any, instance: Any) -> int:
        for loc, inst_freq in instance:
            freq = locs.get(loc)
            if freq is None:
                return 0
            lower = freq - (freq * self.tolerance)
            upper = freq + (freq * self.tolerance)
            if not lower <= inst_freq <= upper:
                return 0
        return 1


class LocationProbabilityAttack(Attack):
    """Assess risk from locations and visit probabilities.

    The attacker knows up to ``knowledge_length`` locations and their visit
    probabilities within a user's trajectory. A known probability matches a
    user's probability when it falls within ``tolerance`` of the user's value.

    Parameters
    ----------
    knowledge_length:
        Number of observations known by the attacker.
    tolerance:
        Relative probability tolerance in ``[0.0, 1.0]``. The default is 0.0.

    Raises
    ------
    ValueError
        If ``knowledge_length`` is less than 1 or ``tolerance`` is outside
        ``[0.0, 1.0]``.
    """

    def __init__(self, knowledge_length: int, tolerance: float = 0.0):
        self.tolerance = tolerance
        super().__init__(knowledge_length)

    @property
    def tolerance(self) -> float:
        """Relative tolerance used when comparing known values."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val: float) -> None:
        if val > 1.0 or val < 0.0:
            raise ValueError("Tolerance should be in the interval [0.0,1.0]")
        self._tolerance = val

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
        return self._all_risks(_probability_vector(traj), targets, force_instances, show_progress)

    def _match(self, single_traj: Any, instance: Any) -> int:
        rows = _records(single_traj)
        inst_rows = _records_like(instance, list(rows[0]) if rows else [])
        return self._match_prepared(self._prepare_group(rows), self._prepare_instance(inst_rows))

    def _prepare_group(self, single_traj: Any) -> Any:
        return {(row[LATITUDE], row[LONGITUDE]): row[PROBABILITY] for row in _records(single_traj)}

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return tuple(sorted(prepared_group.items()))

    def _prepare_instance(self, instance: Any) -> Any:
        return [((row[LATITUDE], row[LONGITUDE]), row[PROBABILITY]) for row in _records_like(instance, [])]

    def _match_prepared(self, locs: Any, instance: Any) -> int:
        for loc, inst_prob in instance:
            prob = locs.get(loc)
            if prob is None:
                return 0
            lower = prob - (prob * self.tolerance)
            upper = prob + (prob * self.tolerance)
            if not lower <= inst_prob <= upper:
                return 0
        return 1


class LocationProportionAttack(LocationFrequencyAttack):
    """Assess risk from locations and relative visit-frequency proportions.

    The attacker knows locations and frequencies, but matching compares the
    proportions of those frequencies relative to the maximum known frequency.
    The inherited ``tolerance`` parameter controls proportional matching.
    """

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
        return self._all_risks(_frequency_vector(traj), targets, force_instances, show_progress)

    def _match(self, single_traj: Any, instance: Any) -> int:
        single_rows = _records(single_traj)
        inst_rows = _records_like(instance, list(single_rows[0]) if single_rows else [])
        return self._match_prepared(self._prepare_group(single_rows), self._prepare_instance(inst_rows))

    def _match_prepared(self, locs: Any, instance: Any) -> int:
        if not instance:
            return 1
        matched_freqs = []
        for loc, _inst_freq in instance:
            freq = locs.get(loc)
            if freq is None:
                return 0
            matched_freqs.append(freq)
        max_single = max(matched_freqs) if matched_freqs else 0
        max_inst = max(inst_freq for _loc, inst_freq in instance)
        for (_loc, inst_freq), freq in zip(instance, matched_freqs):
            if max_single == 0 or max_inst == 0:
                return 0
            proportion = freq / max_single
            inst_proportion = inst_freq / max_inst
            lower = proportion - (proportion * self.tolerance)
            upper = proportion + (proportion * self.tolerance)
            if not lower <= inst_proportion <= upper:
                return 0
        return 1


class HomeWorkAttack(Attack):
    """Assess risk from each user's two most frequent locations.

    This attack uses the first two rows of each user's frequency vector as the
    known instance, matching users whose top-two locations contain the same
    latitude/longitude pairs.
    """

    def __init__(self, knowledge_length: int = 1):
        super().__init__(knowledge_length)

    def _generate_instances(self, single_traj: Any):
        return [_records(single_traj)[:2]]

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
        return self._all_risks(_frequency_vector(traj), targets, force_instances, show_progress)

    def _match(self, single_traj: Any, instance: Any) -> int:
        rows = _records(single_traj)
        inst_rows = _records_like(instance, list(rows[0]) if rows else [])
        return self._match_prepared(self._prepare_group(rows), self._prepare_instance(inst_rows))

    def _prepare_group(self, single_traj: Any) -> Any:
        return {(row[LATITUDE], row[LONGITUDE]) for row in _records(single_traj)[:2]}

    def _prepared_group_key(self, prepared_group: Any) -> Any:
        return tuple(sorted(prepared_group))

    def _prepare_instance(self, instance: Any) -> Any:
        return [(row[LATITUDE], row[LONGITUDE]) for row in _records_like(instance, [])]

    def _match_prepared(self, top_two: Any, inst_locs: Any) -> int:
        return int(all(loc in top_two for loc in inst_locs))


__all__ = [
    "UniqueLocationAttack",
    "LocationFrequencyAttack",
    "LocationProbabilityAttack",
    "LocationProportionAttack",
    "HomeWorkAttack",
]
