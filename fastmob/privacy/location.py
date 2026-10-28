"""Location-based privacy risk measures."""

from __future__ import annotations

from typing import Any

from ._engine import assess_risk


def location_risk(
    traj: Any,
    knowledge_length: int,
    *,
    targets: Any = None,
    force_instances: bool = False,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    h3_resolution: int = 12,
    locations: Any = None,
    location_id_col: str = "location_id",
) -> Any:
    """Assess re-identification risk from visited locations.

    The attacker knows ``knowledge_length`` places a person visited, without knowing how often or when,
    and looks for the users who visited all of them. The candidate knowledge sets are enumerated over
    visit records: a place visited three times is three records, so it appears as three instances.
    Use ``unique_location_risk`` to enumerate distinct places instead.

    Parameters
    ----------
    traj : dataframe
        Trajectory table (any eager dataframe) with latitude and longitude columns.
        A user column is required: the risk is computed for each user against all the others.
    knowledge_length : int
        How many places of a person's history the attacker knows (at least 1).
    targets : array-like, optional
        Users to assess; ``None`` assesses every user.
    force_instances : bool, optional
        If ``False`` (default) return one row per user with ``risk``, the highest probability over all
        possible attacker knowledge sets of this size. If ``True`` return one row per knowledge
        instance with its probability ``prob``.
    lat_col : str, optional
        Column names; auto-detected when ``None``.
    lng_col : str, optional
        Column names; auto-detected when ``None``.
    uid_col : str, optional
        Column names; auto-detected when ``None``.
    h3_resolution : int, optional
        H3 resolution (0 to 15) used to turn coordinates into places. Coarser cells (lower values)
        make people harder to tell apart. Default 12 (about 10 m). Ignored when ``locations`` is given.
    locations : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its location ids, to use
        your own places instead of H3 cells.
    location_id_col : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its location ids, to use
        your own places instead of H3 cells.

    Returns
    -------
    DataFrame
        With ``force_instances=False``: columns ``uid`` and ``risk`` (a probability between 0 and 1;
        1 means some knowledge set of this size matches only that user). With ``force_instances=True``:
        one row per knowledge instance with ``uid``, ``instance``, the known element(s) and ``prob``. Known
        places are given as ``lat`` and ``lng`` of the H3 cell centre.
        The result is in the input's dataframe type.

    Notes
    -----
    The risk is relative to the other users in ``traj``: with few users almost any knowledge identifies
    a person. Risk is 1 divided by the number of users whose history is compatible with the knowledge.
    """
    return assess_risk(
        traj,
        knowledge_length,
        attack="location",
        targets=targets,
        force_instances=force_instances,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


def location_sequence_risk(
    traj: Any,
    knowledge_length: int,
    *,
    targets: Any = None,
    force_instances: bool = False,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    h3_resolution: int = 12,
    locations: Any = None,
    location_id_col: str = "location_id",
) -> Any:
    """Assess re-identification risk from ordered location sequences.

    The attacker knows ``knowledge_length`` consecutive places in the order the person visited them.

    Parameters
    ----------
    traj : dataframe
        Trajectory table (any eager dataframe) with latitude and longitude columns, and a datetime column.
        A user column is required: the risk is computed for each user against all the others.
    knowledge_length : int
        How many consecutive places of a person's history the attacker knows (at least 1).
    targets : array-like, optional
        Users to assess; ``None`` assesses every user.
    force_instances : bool, optional
        If ``False`` (default) return one row per user with ``risk``, the highest probability over all
        possible attacker knowledge sets of this size. If ``True`` return one row per knowledge
        instance with its probability ``prob``.
    lat_col : str, optional
        Column names; auto-detected when ``None``.
    lng_col : str, optional
        Column names; auto-detected when ``None``.
    uid_col : str, optional
        Column names; auto-detected when ``None``.
    h3_resolution : int, optional
        H3 resolution (0 to 15) used to turn coordinates into places. Coarser cells (lower values)
        make people harder to tell apart. Default 12 (about 10 m). Ignored when ``locations`` is given.
    locations : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its location ids, to use
        your own places instead of H3 cells.
    location_id_col : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its location ids, to use
        your own places instead of H3 cells.

    Returns
    -------
    DataFrame
        With ``force_instances=False``: columns ``uid`` and ``risk`` (a probability between 0 and 1;
        1 means some knowledge set of this size matches only that user). With ``force_instances=True``:
        one row per knowledge instance with ``uid``, ``instance``, the known element(s) and ``prob``. Known
        places are given as ``lat`` and ``lng`` of the H3 cell centre.
        The result is in the input's dataframe type.

    Notes
    -----
    The risk is relative to the other users in ``traj``: with few users almost any knowledge identifies
    a person. Risk is 1 divided by the number of users whose history is compatible with the knowledge.
    """
    return assess_risk(
        traj,
        knowledge_length,
        attack="sequence",
        targets=targets,
        force_instances=force_instances,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        require_datetime=True,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


def location_time_risk(
    traj: Any,
    knowledge_length: int,
    *,
    time_precision: str = "Hour",
    targets: Any = None,
    force_instances: bool = False,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    h3_resolution: int = 12,
    locations: Any = None,
    location_id_col: str = "location_id",
) -> Any:
    """Assess re-identification risk from locations and the time of the visit.

    The attacker knows ``knowledge_length`` (place, time) pairs, with time given at ``time_precision``.

    Parameters
    ----------
    traj : dataframe
        Trajectory table (any eager dataframe) with latitude and longitude columns, and a datetime column.
        A user column is required: the risk is computed for each user against all the others.
    knowledge_length : int
        How many (place, time) pairs of a person's history the attacker knows (at least 1).
    time_precision : str, optional
        Time resolution of the known visits, for example ``"Hour"`` or ``"Day"``. Default ``"Hour"``.
    targets : array-like, optional
        Users to assess; ``None`` assesses every user.
    force_instances : bool, optional
        If ``False`` (default) return one row per user with ``risk``, the highest probability over all
        possible attacker knowledge sets of this size. If ``True`` return one row per knowledge
        instance with its probability ``prob``.
    lat_col : str, optional
        Column names; auto-detected when ``None``.
    lng_col : str, optional
        Column names; auto-detected when ``None``.
    uid_col : str, optional
        Column names; auto-detected when ``None``.
    h3_resolution : int, optional
        H3 resolution (0 to 15) used to turn coordinates into places. Coarser cells (lower values)
        make people harder to tell apart. Default 12 (about 10 m). Ignored when ``locations`` is given.
    locations : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its location ids, to use
        your own places instead of H3 cells.
    location_id_col : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its location ids, to use
        your own places instead of H3 cells.

    Returns
    -------
    DataFrame
        With ``force_instances=False``: columns ``uid`` and ``risk`` (a probability between 0 and 1;
        1 means some knowledge set of this size matches only that user). With ``force_instances=True``:
        one row per knowledge instance with ``uid``, ``instance``, the known element(s) and ``prob``. Known
        places are given as ``lat`` and ``lng`` of the H3 cell centre.
        The result is in the input's dataframe type.

    Notes
    -----
    The risk is relative to the other users in ``traj``: with few users almost any knowledge identifies
    a person. Risk is 1 divided by the number of users whose history is compatible with the knowledge.
    """
    return assess_risk(
        traj,
        knowledge_length,
        attack="time",
        targets=targets,
        force_instances=force_instances,
        time_precision=time_precision,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        require_datetime=True,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


__all__ = ["location_risk", "location_sequence_risk", "location_time_risk"]
