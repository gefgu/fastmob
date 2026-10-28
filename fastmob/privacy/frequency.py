"""Frequency-based privacy risk measures."""

from __future__ import annotations

from typing import Any

from ._engine import assess_risk


def unique_location_risk(
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
    """Assess re-identification risk from each user's distinct locations.

    The attacker knows ``knowledge_length`` places, enumerated over the set of distinct places a user
    visited, so a place visited several times is one instance. This is the difference from
    ``location_risk``, which enumerates visit records. For ``knowledge_length=1`` the two give the same
    per-user risk; for larger values ``location_risk`` can be higher, because it also lets the attacker
    know the same place twice.

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
        attack="unique_location",
        targets=targets,
        force_instances=force_instances,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


def location_frequency_risk(
    traj: Any,
    knowledge_length: int,
    *,
    tolerance: float = 0.0,
    targets: Any = None,
    force_instances: bool = False,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    h3_resolution: int = 12,
    locations: Any = None,
    location_id_col: str = "location_id",
) -> Any:
    """Assess re-identification risk from locations and their visit counts.

    The attacker knows ``knowledge_length`` places and how many times the person visited each.

    Parameters
    ----------
    traj : dataframe
        Trajectory table (any eager dataframe) with latitude and longitude columns.
        A user column is required: the risk is computed for each user against all the others.
    knowledge_length : int
        How many places with their visit counts of a person's history the attacker knows (at least 1).
    tolerance : float, optional
        Allowed relative error (0 to 1) on the known visit count. Default 0 (exact).
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
        attack="frequency",
        tolerance=tolerance,
        targets=targets,
        force_instances=force_instances,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


def location_probability_risk(
    traj: Any,
    knowledge_length: int,
    *,
    tolerance: float = 0.0,
    targets: Any = None,
    force_instances: bool = False,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    h3_resolution: int = 12,
    locations: Any = None,
    location_id_col: str = "location_id",
) -> Any:
    """Assess re-identification risk from locations and visit probabilities.

    The attacker knows ``knowledge_length`` places and the probability of visiting each (visits to that
    place divided by all visits).

    Parameters
    ----------
    traj : dataframe
        Trajectory table (any eager dataframe) with latitude and longitude columns.
        A user column is required: the risk is computed for each user against all the others.
    knowledge_length : int
        How many places with their visit probabilities of a person's history the attacker knows (at least 1).
    tolerance : float, optional
        Allowed relative error (0 to 1) on the known probability. Default 0 (exact).
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
        attack="probability",
        tolerance=tolerance,
        targets=targets,
        force_instances=force_instances,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


def location_proportion_risk(
    traj: Any,
    knowledge_length: int,
    *,
    tolerance: float = 0.0,
    targets: Any = None,
    force_instances: bool = False,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    h3_resolution: int = 12,
    locations: Any = None,
    location_id_col: str = "location_id",
) -> Any:
    """Assess re-identification risk from locations and their share of the most visited place.

    The attacker knows ``knowledge_length`` places and, for each, the number of visits as a proportion of
    the visits to the person's most visited place (1 for the top place, 0.5 for a place visited half as
    often). ``tolerance`` allows a relative error on that proportion.

    Parameters
    ----------
    traj : dataframe
        Trajectory table (any eager dataframe) with latitude and longitude columns.
        A user column is required: the risk is computed for each user against all the others.
    knowledge_length : int
        How many places with their relative visit frequency of a person's history the attacker knows (at least 1).
    tolerance : float, optional
        Allowed relative error (0 to 1) on the known proportion. Default 0 (exact).
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
        attack="proportion",
        tolerance=tolerance,
        targets=targets,
        force_instances=force_instances,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


def home_work_risk(
    traj: Any,
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
    """Assess re-identification risk from a person's two most visited places.

    The attacker knows the two places a person visits most often (typically home and work), without
    knowing which is which, and looks for the users with the same two top places.

    Parameters
    ----------
    traj : dataframe
        Trajectory table (any eager dataframe) with latitude, longitude and a user column.
    targets : array-like, optional
        Users to assess; ``None`` assesses every user.
    force_instances : bool, optional
        If ``True``, return the knowledge instance of each user with its probability ``prob`` instead of
        ``risk``.
    lat_col : str, optional
        Column names; auto-detected when ``None``.
    lng_col : str, optional
        Column names; auto-detected when ``None``.
    uid_col : str, optional
        Column names; auto-detected when ``None``.
    h3_resolution : int, optional
        H3 resolution (0 to 15) that turns coordinates into places. Coarser cells make people harder to
        tell apart. Default 12.
    locations : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its ids, to use your own places.
    location_id_col : optional
        A global ``Locations`` catalogue and the column of ``traj`` holding its ids, to use your own places.

    Returns
    -------
    DataFrame
        Columns ``uid`` and ``risk``, a probability between 0 and 1 (1 means no other user shares the two
        top places), in the input's dataframe type.
    """
    return assess_risk(
        traj,
        1,
        attack="home_work",
        targets=targets,
        force_instances=force_instances,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        h3_resolution=h3_resolution,
        locations=locations,
        location_id_col=location_id_col,
    )


__all__ = [
    "home_work_risk",
    "location_frequency_risk",
    "location_probability_risk",
    "location_proportion_risk",
    "unique_location_risk",
]
