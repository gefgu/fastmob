from __future__ import annotations

import numpy as np

_EARTH_RADIUS_M = 6371000.0


def haversine_m_batch(lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance (metres) between two arrays of points.

    Kept distinct from :func:`fastmob.models._common.haversine_km` (a scalar,
    kilometre function used by the gravity/radiation models) -- different
    unit, different array-vs-scalar contract; a clearly distinct name avoids
    a collision rather than forcing an awkward merge.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
