"""Location-based privacy risk measures."""

from __future__ import annotations

from typing import Any

from ._engine import assess_risk


def location_risk(traj: Any, knowledge_length: int, *, targets: Any = None, force_instances: bool = False, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False, h3_resolution: int = 12) -> Any:
    """Assess re-identification risk from visited locations."""
    return assess_risk(traj, knowledge_length, attack="location", targets=targets, force_instances=force_instances, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted, h3_resolution=h3_resolution)


def location_sequence_risk(traj: Any, knowledge_length: int, *, targets: Any = None, force_instances: bool = False, datetime_col: str | None = None, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False, h3_resolution: int = 12) -> Any:
    """Assess re-identification risk from ordered location sequences."""
    return assess_risk(traj, knowledge_length, attack="sequence", targets=targets, force_instances=force_instances, datetime_col=datetime_col, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted, require_datetime=True, h3_resolution=h3_resolution)


def location_time_risk(traj: Any, knowledge_length: int, *, time_precision: str = "Hour", targets: Any = None, force_instances: bool = False, datetime_col: str | None = None, lat_col: str | None = None, lng_col: str | None = None, uid_col: str | None = None, presorted: bool = False, h3_resolution: int = 12) -> Any:
    """Assess re-identification risk from locations at a time precision."""
    return assess_risk(traj, knowledge_length, attack="time", targets=targets, force_instances=force_instances, time_precision=time_precision, datetime_col=datetime_col, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col, presorted=presorted, require_datetime=True, h3_resolution=h3_resolution)


__all__ = ["location_risk", "location_sequence_risk", "location_time_risk"]
