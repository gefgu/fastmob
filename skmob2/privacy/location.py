"""Location-based privacy attacks."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from ._constants import DATETIME, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE, PRECISION_LEVELS, TEMP, UID
from ._dataframe import _as_frame, _with_date_time_precision
from .base import _CANDIDATE_POS, _CANDIDATE_UID, _TARGET_UID, Attack


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
        sorted_traj = _as_frame(traj).sort([UID, DATETIME])
        return self._all_risks(sorted_traj, targets, force_instances, show_progress)

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._multiset_match_counts(candidates, instances, [LATITUDE, LONGITUDE])


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
        sorted_traj = _as_frame(traj).sort([UID, DATETIME])
        return self._all_risks(sorted_traj, targets, force_instances, show_progress)

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        instance_lengths = instances.group_by([_TARGET_UID, INSTANCE]).agg(
            nw.col(INSTANCE_ELEMENT).max().alias("__instance_length__")
        )
        lengths = [int(length) for length in sorted(instance_lengths.select("__instance_length__").unique().get_column("__instance_length__").to_list())]
        if not lengths:
            return self._count_candidates(instances.select([_TARGET_UID, INSTANCE, _CANDIDATE_UID]))

        candidate_locs = candidates.select([_CANDIDATE_UID, _CANDIDATE_POS, LATITUDE, LONGITUDE])
        matched_frames = []
        for length in lengths:
            length_instances = instances.join(
                instance_lengths.filter(nw.col("__instance_length__") == length).drop("__instance_length__"),
                on=[_TARGET_UID, INSTANCE],
                how="inner",
            )
            state = (
                length_instances.filter(nw.col(INSTANCE_ELEMENT) == 1)
                .select([_TARGET_UID, INSTANCE, LATITUDE, LONGITUDE])
                .join(candidate_locs, on=[LATITUDE, LONGITUDE], how="inner")
                .group_by([_TARGET_UID, INSTANCE, _CANDIDATE_UID])
                .agg(nw.col(_CANDIDATE_POS).min().alias("__matched_pos__"))
            )
            for elem in range(2, length + 1):
                required = length_instances.filter(nw.col(INSTANCE_ELEMENT) == elem).select(
                    [_TARGET_UID, INSTANCE, LATITUDE, LONGITUDE]
                )
                state = (
                    state.join(required, on=[_TARGET_UID, INSTANCE], how="inner")
                    .join(candidate_locs, on=[_CANDIDATE_UID, LATITUDE, LONGITUDE], how="inner")
                    .filter(nw.col(_CANDIDATE_POS) > nw.col("__matched_pos__"))
                    .group_by([_TARGET_UID, INSTANCE, _CANDIDATE_UID])
                    .agg(nw.col(_CANDIDATE_POS).min().alias("__matched_pos__"))
                )
            matched_frames.append(state)

        return self._count_candidates(nw.concat(matched_frames, how="vertical"))


class LocationTimeAttack(LocationAttack):
    """Assess risk from locations observed at a selected time precision.

    The attacker knows up to ``knowledge_length`` location/time observations.
    Matching ignores order but requires latitude, longitude, and the datetime
    value truncated to ``time_precision`` to match.
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

    def _instance_columns(self) -> list[str]:
        return [UID, LATITUDE, LONGITUDE, DATETIME, TEMP]

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
        transformed = _with_date_time_precision(sorted_df, DATETIME, TEMP, self.time_precision)
        return self._all_risks(transformed, targets, force_instances, show_progress)

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._multiset_match_counts(candidates, instances, [LATITUDE, LONGITUDE, TEMP])


__all__ = ["LocationAttack", "LocationSequenceAttack", "LocationTimeAttack"]
