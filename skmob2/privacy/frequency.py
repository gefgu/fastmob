"""Frequency-based privacy attacks."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from ._constants import FREQUENCY, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE, PROBABILITY, UID
from ._dataframe import _frequency_vector, _probability_vector
from .base import (
    _CANDIDATE_UID,
    _POS,
    _TARGET_UID,
    Attack,
)


class UniqueLocationAttack(Attack):
    """Assess risk from the set of unique locations visited by each user.

    The attacker knows up to ``knowledge_length`` distinct locations. Matching
    ignores the number of visits, visit order, and timestamps.
    """

    def __init__(self, knowledge_length: int):
        super().__init__(knowledge_length)

    @property
    def _has_datetime(self) -> bool:
        return False

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

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._set_match_counts(candidates, instances, [LATITUDE, LONGITUDE])


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
    def _has_datetime(self) -> bool:
        return False

    def _metric_columns(self) -> list[str]:
        return [FREQUENCY]

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

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._metric_tolerance_match_counts(candidates, instances, FREQUENCY)

    def _metric_tolerance_match_counts(
        self,
        candidates: nw.DataFrame,
        instances: nw.DataFrame,
        metric: str,
    ) -> nw.DataFrame:
        candidate_metric = f"__candidate_{metric}__"
        required = instances.group_by([_TARGET_UID, INSTANCE]).agg(nw.len().alias("__required_rows__"))
        matched = (
            instances.join(
                candidates.rename({metric: candidate_metric}),
                on=[LATITUDE, LONGITUDE],
                how="inner",
            )
            .filter(
                (nw.col(metric) >= nw.col(candidate_metric) * (1.0 - self.tolerance))
                & (nw.col(metric) <= nw.col(candidate_metric) * (1.0 + self.tolerance))
            )
            .group_by([_TARGET_UID, INSTANCE, _CANDIDATE_UID])
            .agg(nw.len().alias("__matched_rows__"))
            .join(required, on=[_TARGET_UID, INSTANCE], how="inner")
            .filter(nw.col("__matched_rows__") == nw.col("__required_rows__"))
        )
        return self._count_candidates(matched)


class LocationProbabilityAttack(LocationFrequencyAttack):
    """Assess risk from locations and visit probabilities.

    The attacker knows up to ``knowledge_length`` locations and their visit
    probabilities within a user's trajectory. A known probability matches a
    user's probability when it falls within ``tolerance`` of the user's value.
    """

    def _metric_columns(self) -> list[str]:
        return [PROBABILITY]

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

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._metric_tolerance_match_counts(candidates, instances, PROBABILITY)


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

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        candidate_freq = "__candidate_freq__"
        group_cols = [_TARGET_UID, INSTANCE, _CANDIDATE_UID]
        required = instances.group_by([_TARGET_UID, INSTANCE]).agg(nw.len().alias("__required_rows__"))
        joined = instances.join(
            candidates.rename({FREQUENCY: candidate_freq}),
            on=[LATITUDE, LONGITUDE],
            how="inner",
        ).with_columns(
            nw.col(FREQUENCY).max().over([_TARGET_UID, INSTANCE]).alias("__max_instance_freq__"),
            nw.col(candidate_freq).max().over(group_cols).alias("__max_candidate_freq__"),
        )
        matched = (
            joined.filter(nw.col("__max_instance_freq__") > 0)
            .filter(nw.col("__max_candidate_freq__") > 0)
            .with_columns(
                (nw.col(FREQUENCY) / nw.col("__max_instance_freq__")).alias("__instance_prop__"),
                (nw.col(candidate_freq) / nw.col("__max_candidate_freq__")).alias("__candidate_prop__"),
            )
            .filter(
                (nw.col("__instance_prop__") >= nw.col("__candidate_prop__") * (1.0 - self.tolerance))
                & (nw.col("__instance_prop__") <= nw.col("__candidate_prop__") * (1.0 + self.tolerance))
            )
            .group_by(group_cols)
            .agg(nw.len().alias("__matched_rows__"))
            .join(required, on=[_TARGET_UID, INSTANCE], how="inner")
            .filter(nw.col("__matched_rows__") == nw.col("__required_rows__"))
        )
        return self._count_candidates(matched)


class HomeWorkAttack(UniqueLocationAttack):
    """Assess risk from each user's first two frequency-vector locations."""

    def __init__(self, knowledge_length: int = 1):
        super().__init__(knowledge_length)

    def _instance_values(self, df: nw.DataFrame) -> nw.DataFrame:
        return self._top_two_values(df)

    def _candidate_values(self, df: nw.DataFrame) -> nw.DataFrame:
        return self._top_two_values(df).rename({UID: _CANDIDATE_UID})

    def _top_two_values(self, df: nw.DataFrame) -> nw.DataFrame:
        return self._with_user_position(df.select([UID, LATITUDE, LONGITUDE])).filter(nw.col(_POS) <= 2)

    def _generate_instances(self, values: nw.DataFrame, target_uids: nw.DataFrame) -> nw.DataFrame:
        target_values = values.join(target_uids, on=UID, how="semi")
        if len(target_values) == 0:
            return self._empty_instances(values)
        return (
            target_values.rename({UID: _TARGET_UID, _POS: INSTANCE_ELEMENT})
            .with_columns(nw.lit(1).alias(INSTANCE))
            .select([_TARGET_UID, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE])
            .sort([_TARGET_UID, INSTANCE_ELEMENT])
        )

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


__all__ = [
    "UniqueLocationAttack",
    "LocationFrequencyAttack",
    "LocationProbabilityAttack",
    "LocationProportionAttack",
    "HomeWorkAttack",
]
