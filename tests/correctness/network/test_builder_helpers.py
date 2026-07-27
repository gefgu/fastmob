"""Unit tests for fastmob.network.builder's pure-logic helpers (no network needed)."""

from __future__ import annotations

import pytest
from fastmob._core import haversine_m_batch
from fastmob.network.builder import _direction_for_pair, _is_missing_or_empty, _speed_kmh_for_pair
from fastmob.network.speeds import DEFAULT_SPEED_KMH_BY_CLASS


def test_is_missing_or_empty_none():
    assert _is_missing_or_empty(None) is True


def test_is_missing_or_empty_nan():
    assert _is_missing_or_empty(float("nan")) is True


def test_is_missing_or_empty_empty_list():
    assert _is_missing_or_empty([]) is True


def test_is_missing_or_empty_nonempty_list():
    assert _is_missing_or_empty([{"max_speed": {"value": 50}}]) is False


def test_speed_kmh_for_pair_falls_back_to_class_default():
    speed = _speed_kmh_for_pair(None, "residential", 0.0, 0.5)
    assert speed == DEFAULT_SPEED_KMH_BY_CLASS["residential"]


def test_speed_kmh_for_pair_uses_whole_segment_speed_limit():
    speed_limits = [{"max_speed": {"value": 50, "unit": "km/h"}}]
    speed = _speed_kmh_for_pair(speed_limits, "residential", 0.0, 0.5)
    assert speed == 50.0


def test_speed_kmh_for_pair_converts_mph():
    speed_limits = [{"max_speed": {"value": 30, "unit": "mph"}}]
    speed = _speed_kmh_for_pair(speed_limits, "residential", 0.0, 0.5)
    assert speed == pytest.approx(30 * 1.609344)


def test_speed_kmh_for_pair_uses_between_range_matching_midpoint():
    speed_limits = [
        {"max_speed": {"value": 30, "unit": "km/h"}, "between": [0.0, 0.4]},
        {"max_speed": {"value": 90, "unit": "km/h"}, "between": [0.4, 1.0]},
    ]
    speed = _speed_kmh_for_pair(speed_limits, "residential", 0.5, 0.9)  # mid = 0.7
    assert speed == 90.0


def test_direction_for_pair_no_restrictions_is_both():
    assert _direction_for_pair(None) == "both"
    assert _direction_for_pair([]) == "both"


def test_direction_for_pair_forward_denied_means_backward_only():
    restrictions = [{"access_type": "denied", "when": {"heading": "forward"}}]
    assert _direction_for_pair(restrictions) == "backward"


def test_direction_for_pair_both_denied_means_not_drivable():
    restrictions = [
        {"access_type": "denied", "when": {"heading": "forward"}},
        {"access_type": "denied", "when": {"heading": "backward"}},
    ]
    assert _direction_for_pair(restrictions) is None


def test_direction_for_pair_foot_only_restriction_ignored():
    restrictions = [{"access_type": "denied", "when": {"heading": "forward", "mode": ["foot"]}}]
    assert _direction_for_pair(restrictions) == "both"


def test_direction_for_pair_unconditional_denial_means_not_drivable():
    restrictions = [{"access_type": "denied", "when": {}}]
    assert _direction_for_pair(restrictions) is None


def test_haversine_m_batch_known_distance():
    import numpy as np

    # ~1 degree of latitude is ~111.19 km
    d = haversine_m_batch(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert d[0] == pytest.approx(111195.0, rel=0.01)
