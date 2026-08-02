"""Frequency-based privacy risk measures."""

from __future__ import annotations

from typing import Any

from ._engine import assess_risk


def unique_location_risk(traj: Any, knowledge_length: int, *, targets: Any = None, force_instances: bool = False, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False) -> Any:
    """Assess re-identification risk from each user's distinct locations."""
    return assess_risk(traj, knowledge_length, attack="unique_location", targets=targets, force_instances=force_instances, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted)


def location_frequency_risk(traj: Any, knowledge_length: int, *, tolerance: float = 0.0, targets: Any = None, force_instances: bool = False, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False) -> Any:
    """Assess re-identification risk from locations and visit frequencies."""
    return assess_risk(traj, knowledge_length, attack="frequency", tolerance=tolerance, targets=targets, force_instances=force_instances, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted)


def location_probability_risk(traj: Any, knowledge_length: int, *, tolerance: float = 0.0, targets: Any = None, force_instances: bool = False, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False) -> Any:
    """Assess re-identification risk from locations and visit probabilities."""
    return assess_risk(traj, knowledge_length, attack="probability", tolerance=tolerance, targets=targets, force_instances=force_instances, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted)


def location_proportion_risk(traj: Any, knowledge_length: int, *, tolerance: float = 0.0, targets: Any = None, force_instances: bool = False, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False) -> Any:
    """Assess re-identification risk from locations and relative visit frequencies."""
    return assess_risk(traj, knowledge_length, attack="proportion", tolerance=tolerance, targets=targets, force_instances=force_instances, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted)


def home_work_risk(traj: Any, *, targets: Any = None, force_instances: bool = False, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False) -> Any:
    """Assess re-identification risk from the two most frequent locations."""
    return assess_risk(traj, 1, attack="home_work", targets=targets, force_instances=force_instances, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted)


__all__ = ["home_work_risk", "location_frequency_risk", "location_probability_risk", "location_proportion_risk", "unique_location_risk"]
