"""Location-based privacy attacks."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from ._constants import DATETIME, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE, PRECISION_LEVELS, TEMP, UID
from .base import _CANDIDATE_POS, _CANDIDATE_UID, _TARGET_UID, Attack
from ._rust import LOCATION, SEQUENCE, TIME, assess_risk_rust


class LocationAttack(Attack):
    """Location Attack: assess re-identification risk from visited locations.

    The attacker knows the coordinates of up to ``knowledge_length`` location
    observations for a target user [TIST2018]_ [MOB2018]_.  Matching is
    multiset-based: the instance locations must appear in the candidate
    trajectory with at least the same frequency, regardless of visit order or
    timestamp.

    Parameters
    ----------
    knowledge_length : int
        Number of location observations known by the attacker.

    Attributes
    ----------
    knowledge_length : int
        Number of trajectory observations known by the attacker.

    Examples
    --------
    >>> import pandas as pd
    >>> from fkmob.privacy.attacks import LocationAttack
    >>> traj = pd.DataFrame(
    ...     {
    ...         "uid": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
    ...         "lat": [43.843, 43.544, 43.708, 43.779, 43.843, 43.708, 43.843, 43.544,
    ...                 43.544, 43.708, 43.843, 43.779, 43.708, 43.544, 43.779, 43.708,
    ...                 43.779, 43.843, 43.843, 43.544],
    ...         "lng": [10.508, 10.326, 10.404, 11.246, 10.508, 10.404, 10.508, 10.326,
    ...                 10.326, 10.404, 10.508, 11.246, 10.404, 10.326, 11.246, 10.404,
    ...                 11.246, 10.508, 10.508, 10.326],
    ...         "datetime": pd.to_datetime([
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-03 10:34", "2011-02-04 10:34",
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-04 10:34", "2011-02-04 11:34",
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-04 10:34", "2011-02-04 11:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34", "2011-02-04 12:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34", "2011-02-05 12:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34",
    ...         ]),
    ...     }
    ... )
    >>> at = LocationAttack(knowledge_length=2)
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid      risk
       1  0.333333
       2  1.000000
       3  0.333333
       4  0.333333
       5  0.333333
       6  0.250000

    References
    ----------
    - [TIST2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2017)
      A Data Mining Approach to Assess Privacy Risk in Human Mobility Data.
      ACM Trans. Intell. Syst. Technol. 9(3), Article 31.
      https://doi.org/10.1145/3106774
    - [MOB2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2018)
      Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129.
    """

    def __init__(self, knowledge_length: int):
        super().__init__(knowledge_length)

    def assess_risk(
        self,
        traj: Any,
        targets: Any = None,
        force_instances: bool = False,
        show_progress: bool = False,
        *,
        presorted: bool = False,
    ) -> Any:
        """Assess privacy risk for each user in the trajectory.

        Parameters
        ----------
        traj : DataFrame-like
            Trajectory dataframe; any Narwhals-compatible eager backend
            (pandas, polars, …). Must have ``uid``, ``lat``, ``lng``, and
            ``datetime`` columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fkmob. Default: False.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            When ``force_instances=False``: one row per user with columns
            ``["uid", "risk"]``.
            When ``force_instances=True``: one row per instance element with
            columns ``["lat", "lng", "datetime", "uid", "instance",
            "instance_elem", "prob"]``.
            The returned backend matches the input backend.
        """
        del show_progress
        return assess_risk_rust(
            traj,
            attack_kind=LOCATION,
            knowledge_length=self.knowledge_length,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
            include_datetime=True,
        )

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._multiset_match_counts(candidates, instances, [LATITUDE, LONGITUDE])


class LocationSequenceAttack(Attack):
    """Location Sequence Attack: assess risk from an ordered location sequence.

    The attacker knows the coordinates of up to ``knowledge_length`` locations
    and their relative temporal order [TIST2018]_ [MOB2018]_.  Matching
    requires the known locations to appear as an ordered subsequence of the
    candidate trajectory — timestamps are not used, only the visit order.

    Parameters
    ----------
    knowledge_length : int
        Number of ordered location observations known by the attacker.

    Attributes
    ----------
    knowledge_length : int
        Number of trajectory observations known by the attacker.

    Examples
    --------
    >>> import pandas as pd
    >>> from fkmob.privacy.attacks import LocationSequenceAttack
    >>> traj = pd.DataFrame(
    ...     {
    ...         "uid": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
    ...         "lat": [43.843, 43.544, 43.708, 43.779, 43.843, 43.708, 43.843, 43.544,
    ...                 43.544, 43.708, 43.843, 43.779, 43.708, 43.544, 43.779, 43.708,
    ...                 43.779, 43.843, 43.843, 43.544],
    ...         "lng": [10.508, 10.326, 10.404, 11.246, 10.508, 10.404, 10.508, 10.326,
    ...                 10.326, 10.404, 10.508, 11.246, 10.404, 10.326, 11.246, 10.404,
    ...                 11.246, 10.508, 10.508, 10.326],
    ...         "datetime": pd.to_datetime([
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-03 10:34", "2011-02-04 10:34",
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-04 10:34", "2011-02-04 11:34",
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-04 10:34", "2011-02-04 11:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34", "2011-02-04 12:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34", "2011-02-05 12:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34",
    ...         ]),
    ...     }
    ... )
    >>> at = LocationSequenceAttack(knowledge_length=2)
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid      risk
       1  0.500000
       2  1.000000
       3  1.000000
       4  0.500000
       5  1.000000
       6  0.333333

    References
    ----------
    - [TIST2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2017)
      A Data Mining Approach to Assess Privacy Risk in Human Mobility Data.
      ACM Trans. Intell. Syst. Technol. 9(3), Article 31.
      https://doi.org/10.1145/3106774
    - [MOB2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2018)
      Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129.
    """

    def __init__(self, knowledge_length: int):
        super().__init__(knowledge_length)

    def assess_risk(
        self,
        traj: Any,
        targets: Any = None,
        force_instances: bool = False,
        show_progress: bool = False,
        *,
        presorted: bool = False,
    ) -> Any:
        """Assess privacy risk for each user in the trajectory.

        Parameters
        ----------
        traj : DataFrame-like
            Trajectory dataframe; any Narwhals-compatible eager backend
            (pandas, polars, …). Must have ``uid``, ``lat``, ``lng``, and
            ``datetime`` columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fkmob. Default: False.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            When ``force_instances=False``: one row per user with columns
            ``["uid", "risk"]``.
            When ``force_instances=True``: one row per instance element with
            columns ``["lat", "lng", "datetime", "uid", "instance",
            "instance_elem", "prob"]``.
            The returned backend matches the input backend.
        """
        del show_progress
        return assess_risk_rust(
            traj,
            attack_kind=SEQUENCE,
            knowledge_length=self.knowledge_length,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
            include_datetime=True,
        )

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
    """Location Time Attack: assess risk from locations and timestamps.

    The attacker knows the coordinates and timestamps (truncated to
    ``time_precision``) of up to ``knowledge_length`` observations
    [TIST2018]_ [MOB2018]_.  Matching requires latitude, longitude, and the
    truncated datetime to coincide, regardless of visit order.

    Parameters
    ----------
    knowledge_length : int
        Number of location/time observations known by the attacker.
    time_precision : str, optional
        Datetime component to use when comparing timestamps.  One of
        ``"Year"``, ``"Month"``, ``"Day"``, ``"Hour"``, ``"Minute"``,
        ``"Second"``. Default: ``"Hour"``.

    Attributes
    ----------
    knowledge_length : int
        Number of trajectory observations known by the attacker.
    time_precision : str
        Datetime precision used to match known observations.

    Examples
    --------
    >>> import pandas as pd
    >>> from fkmob.privacy.attacks import LocationTimeAttack
    >>> traj = pd.DataFrame(
    ...     {
    ...         "uid": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
    ...         "lat": [43.843, 43.544, 43.708, 43.779, 43.843, 43.708, 43.843, 43.544,
    ...                 43.544, 43.708, 43.843, 43.779, 43.708, 43.544, 43.779, 43.708,
    ...                 43.779, 43.843, 43.843, 43.544],
    ...         "lng": [10.508, 10.326, 10.404, 11.246, 10.508, 10.404, 10.508, 10.326,
    ...                 10.326, 10.404, 10.508, 11.246, 10.404, 10.326, 11.246, 10.404,
    ...                 11.246, 10.508, 10.508, 10.326],
    ...         "datetime": pd.to_datetime([
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-03 10:34", "2011-02-04 10:34",
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-04 10:34", "2011-02-04 11:34",
    ...             "2011-02-03 08:34", "2011-02-03 09:34", "2011-02-04 10:34", "2011-02-04 11:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34", "2011-02-04 12:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34", "2011-02-05 12:34",
    ...             "2011-02-04 10:34", "2011-02-04 11:34",
    ...         ]),
    ...     }
    ... )
    >>> at = LocationTimeAttack(knowledge_length=2, time_precision="Hour")
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid  risk
       1   1.0
       2   1.0
       3   1.0
       4   1.0
       5   1.0
       6   0.5
    >>> at.time_precision = "Month"
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid      risk
       1  0.333333
       2  1.000000
       3  0.333333
       4  0.333333
       5  0.333333
       6  0.250000

    References
    ----------
    - [TIST2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2017)
      A Data Mining Approach to Assess Privacy Risk in Human Mobility Data.
      ACM Trans. Intell. Syst. Technol. 9(3), Article 31.
      https://doi.org/10.1145/3106774
    - [MOB2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2018)
      Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129.
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
        *,
        presorted: bool = False,
    ) -> Any:
        """Assess privacy risk for each user in the trajectory.

        Parameters
        ----------
        traj : DataFrame-like
            Trajectory dataframe; any Narwhals-compatible eager backend
            (pandas, polars, …). Must have ``uid``, ``lat``, ``lng``, and
            ``datetime`` columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fkmob. Default: False.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            When ``force_instances=False``: one row per user with columns
            ``["uid", "risk"]``.
            When ``force_instances=True``: one row per instance element with
            columns ``["lat", "lng", "datetime", "uid", "instance",
            "instance_elem", "prob"]``.
            The returned backend matches the input backend.
        """
        del show_progress
        return assess_risk_rust(
            traj,
            attack_kind=TIME,
            knowledge_length=self.knowledge_length,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
            time_precision=self.time_precision,
            include_datetime=True,
        )

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._multiset_match_counts(candidates, instances, [LATITUDE, LONGITUDE, TEMP])


__all__ = ["LocationAttack", "LocationSequenceAttack", "LocationTimeAttack"]
