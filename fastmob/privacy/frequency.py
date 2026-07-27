"""Frequency-based privacy attacks."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from ._constants import FREQUENCY, INSTANCE, INSTANCE_ELEMENT, LATITUDE, LONGITUDE, PROBABILITY, UID
from ._rust import FREQUENCY as FREQUENCY_ATTACK
from ._rust import HOME_WORK, PROBABILITY_ATTACK, PROPORTION, UNIQUE_LOCATION, assess_risk_rust
from .base import (
    _CANDIDATE_UID,
    _POS,
    _TARGET_UID,
    Attack,
)


class UniqueLocationAttack(Attack):
    """Unique Location Attack: assess risk from the set of distinct locations.

    The attacker knows up to ``knowledge_length`` distinct (lat, lng) locations
    visited by a target user [TIST2018]_ [MOB2018]_. Matching is set-based:
    an instance matches a candidate if every instance location appears in the
    candidate's frequency vector at least once. Visit counts, order, and
    timestamps are not used.

    Parameters
    ----------
    knowledge_length : int
        Number of distinct locations known by the attacker.

    Attributes
    ----------
    knowledge_length : int
        Number of trajectory observations known by the attacker.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.privacy.attacks import UniqueLocationAttack
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
    >>> at = UniqueLocationAttack(knowledge_length=2)
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid      risk
       1  0.333333
       2  0.250000
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

    @property
    def _has_datetime(self) -> bool:
        return False

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
            (pandas, polars, …). Must have ``uid``, ``lat``, and ``lng``
            columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fastmob. Default: False.

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
            attack_kind=UNIQUE_LOCATION,
            knowledge_length=self.knowledge_length,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
        )

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._set_match_counts(candidates, instances, [LATITUDE, LONGITUDE])


class LocationFrequencyAttack(Attack):
    """Location Frequency Attack: assess risk from locations and visit counts.

    The attacker knows up to ``knowledge_length`` distinct locations and the
    number of times each was visited [TIST2018]_ [MOB2018]_. Matching compares
    the known frequency against the candidate's frequency vector within a
    relative ``tolerance``: a frequency :math:`f_k` matches :math:`f_c` when

    .. math::

        f_c(1 - \\text{tolerance}) \\leq f_k \\leq f_c(1 + \\text{tolerance})

    Parameters
    ----------
    knowledge_length : int
        Number of location observations known by the attacker.
    tolerance : float, optional
        Relative tolerance used when comparing known frequencies to candidate
        frequencies. Must be in ``[0.0, 1.0]``. Default: 0.0 (exact match).

    Attributes
    ----------
    knowledge_length : int
        Number of trajectory observations known by the attacker.
    tolerance : float
        Relative tolerance for frequency matching.

    Raises
    ------
    ValueError
        If ``knowledge_length`` is less than 1 or ``tolerance`` is outside
        ``[0.0, 1.0]``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.privacy.attacks import LocationFrequencyAttack
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
    >>> at = LocationFrequencyAttack(knowledge_length=2)
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid      risk
       1  0.333333
       2  1.000000
       3  0.333333
       4  0.333333
       5  0.333333
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
        *,
        presorted: bool = False,
    ) -> Any:
        """Assess privacy risk for each user in the trajectory.

        Parameters
        ----------
        traj : DataFrame-like
            Trajectory dataframe; any Narwhals-compatible eager backend
            (pandas, polars, …). Must have ``uid``, ``lat``, and ``lng``
            columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fastmob. Default: False.

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
            attack_kind=FREQUENCY_ATTACK,
            knowledge_length=self.knowledge_length,
            tolerance=self.tolerance,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
        )

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
    """Location Probability Attack: assess risk from locations and visit probabilities.

    The attacker knows up to ``knowledge_length`` distinct locations and the
    probability (relative frequency) of visiting each one [TIST2018]_
    [MOB2018]_. Matching compares known probabilities against the candidate's
    probability vector within a relative ``tolerance``.  Inherits
    ``knowledge_length`` and ``tolerance`` from :class:`LocationFrequencyAttack`.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.privacy.attacks import LocationProbabilityAttack
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
    >>> at = LocationProbabilityAttack(knowledge_length=2)
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid  risk
       1   0.5
       2   1.0
       3   0.5
       4   1.0
       5   1.0
       6   1.0

    References
    ----------
    - [TIST2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2017)
      A Data Mining Approach to Assess Privacy Risk in Human Mobility Data.
      ACM Trans. Intell. Syst. Technol. 9(3), Article 31.
      https://doi.org/10.1145/3106774
    - [MOB2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2018)
      Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129.
    """

    def _metric_columns(self) -> list[str]:
        return [PROBABILITY]

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
            (pandas, polars, …). Must have ``uid``, ``lat``, and ``lng``
            columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fastmob. Default: False.

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
            attack_kind=PROBABILITY_ATTACK,
            knowledge_length=self.knowledge_length,
            tolerance=self.tolerance,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
        )

    def _match_counts(self, candidates: nw.DataFrame, instances: nw.DataFrame) -> nw.DataFrame:
        return self._metric_tolerance_match_counts(candidates, instances, PROBABILITY)


class LocationProportionAttack(LocationFrequencyAttack):
    """Location Proportion Attack: assess risk from locations and frequency proportions.

    The attacker knows up to ``knowledge_length`` distinct locations and the
    relative proportions between their visit frequencies [TIST2018]_ [MOB2018]_.
    Matching normalises both the instance and candidate frequencies by their
    respective maximums before comparing within ``tolerance``.  Inherits
    ``knowledge_length`` and ``tolerance`` from :class:`LocationFrequencyAttack`.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.privacy.attacks import LocationProportionAttack
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
    >>> at = LocationProportionAttack(knowledge_length=2)
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid      risk
       1  0.333333
       2  1.000000
       3  0.333333
       4  0.333333
       5  0.333333
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
            (pandas, polars, …). Must have ``uid``, ``lat``, and ``lng``
            columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fastmob. Default: False.

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
            attack_kind=PROPORTION,
            knowledge_length=self.knowledge_length,
            tolerance=self.tolerance,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
        )

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
    """Home-Work Attack: assess risk from the two most-visited locations.

    A special case of :class:`UniqueLocationAttack` where the attacker always
    uses the two most-visited locations as background knowledge, regardless of
    ``knowledge_length``. This models the scenario where an attacker knows a
    user's home and workplace [TIST2018]_ [MOB2018]_.

    Parameters
    ----------
    knowledge_length : int, optional
        Kept for API compatibility; always uses exactly two locations.
        Default: 1.

    Attributes
    ----------
    knowledge_length : int
        Number of trajectory observations known by the attacker.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.privacy.attacks import HomeWorkAttack
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
    >>> at = HomeWorkAttack()
    >>> print(at.assess_risk(traj).to_string(index=False))
     uid  risk
       1  0.25
       2  0.25
       3  0.25
       4  0.25
       5  1.00
       6  1.00

    References
    ----------
    - [TIST2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2017)
      A Data Mining Approach to Assess Privacy Risk in Human Mobility Data.
      ACM Trans. Intell. Syst. Technol. 9(3), Article 31.
      https://doi.org/10.1145/3106774
    - [MOB2018] Pellungrini, R., Pappalardo, L., Pratesi, F. & Monreale, A. (2018)
      Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129.
    """

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
        *,
        presorted: bool = False,
    ) -> Any:
        """Assess privacy risk for each user in the trajectory.

        Parameters
        ----------
        traj : DataFrame-like
            Trajectory dataframe; any Narwhals-compatible eager backend
            (pandas, polars, …). Must have ``uid``, ``lat``, and ``lng``
            columns (auto-detected by name).
        targets : DataFrame-like or list of int, optional
            Subset of user IDs to assess. When None (default), risk is
            computed for every user in ``traj``.
        force_instances : bool, optional
            When True, return one row per background-knowledge instance
            element with its re-identification probability instead of the
            per-user maximum. Default: False.
        show_progress : bool, optional
            Accepted for API compatibility with skmob; has no effect in
            fastmob. Default: False.

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
            attack_kind=HOME_WORK,
            knowledge_length=self.knowledge_length,
            targets=targets,
            force_instances=force_instances,
            presorted=presorted,
        )


__all__ = [
    "HomeWorkAttack",
    "LocationFrequencyAttack",
    "LocationProbabilityAttack",
    "LocationProportionAttack",
    "UniqueLocationAttack",
]
