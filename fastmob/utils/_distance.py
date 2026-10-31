"""Straight-line distances between geographic coordinates."""

import numpy as np

from fastmob._core import haversine_m_batch


def haversine(lat1, lng1, lat2, lng2, *, unit="m", output="numpy"):
    """Compute elementwise great-circle distances between coordinate pairs.

    Parameters
    ----------
    lat1 : array-like
        Origin latitudes in degrees, as a one-dimensional numeric array or Series.
    lng1 : array-like
        Origin longitudes in degrees.
    lat2 : array-like
        Destination latitudes in degrees.
    lng2 : array-like
        Destination longitudes in degrees. All inputs must have the same length.
    unit : {"m", "km"}, optional
        Output distance unit. Defaults to metres (``"m"``).
    output : {"numpy", "pandas", "arrow"}, optional
        Output container. Defaults to a NumPy array. The pandas option returns a
        Series with a default index; results are positional, not index-aligned.

    Returns
    -------
    numpy.ndarray, pandas.Series, or pyarrow.Array
        One distance per input row. NaN coordinates produce NaN distances.

    Notes
    -----
    Uses the spherical Haversine formula and a parallel native batch kernel.
    Results are straight-line surface distances, not road-network distances.

    Examples
    --------
    >>> from fastmob.utils import haversine
    >>> distances = haversine([48.8566], [2.3522], [51.5074], [-0.1278], unit="km")
    >>> round(distances[0])
    344
    """
    if unit not in ("m", "km"):
        raise ValueError("unit must be 'm' or 'km'")
    if output not in ("numpy", "pandas", "arrow"):
        raise ValueError("output must be 'numpy', 'pandas', or 'arrow'")
    coordinates = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in (lat1, lng1, lat2, lng2))
    if any(value.ndim != 1 for value in coordinates):
        raise ValueError("coordinates must be one-dimensional")
    if len({value.size for value in coordinates}) != 1:
        raise ValueError("lat1, lng1, lat2, and lng2 must have the same length")
    result = haversine_m_batch(*coordinates)
    if unit == "km":
        result /= 1000.0
    if output == "pandas":
        import pandas as pd

        return pd.Series(result, name="distance")
    if output == "arrow":
        import pyarrow as pa

        return pa.array(result)
    return result
