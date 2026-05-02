from __future__ import annotations

import math
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd

LATITUDE = "lat"
LONGITUDE = "lng"
DATETIME = "datetime"
UID = "uid"
TILE_ID = "tile_id"
RELEVANCE = "relevance"
TOT_OUTFLOW = "tot_outflow"
ORIGIN = "origin"
DESTINATION = "destination"
FLOW = "flow"
CLUSTER = "cluster"

EARTH_RADIUS_KM = 6371.01


def to_pandas_frame(df: Any) -> pd.DataFrame:
    """Return a pandas DataFrame copy for pandas, GeoPandas, or Narwhals inputs."""
    if isinstance(df, pd.DataFrame):
        return df.copy()
    try:
        return nw.from_native(df, eager_only=True).to_native().copy()
    except Exception:
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        raise


def haversine_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(float(origin[0])), math.radians(float(origin[1]))
    lat2, lon2 = math.radians(float(destination[0])), math.radians(float(destination[1]))
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    ds = 2.0 * math.asin(
        math.sqrt(math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2)
    )
    return EARTH_RADIUS_KM * ds


def geometry_centroid_lat_lng(geom: Any) -> tuple[float, float]:
    centroid = geom.centroid if hasattr(geom, "centroid") else geom
    return float(centroid.y), float(centroid.x)


def tessellation_lat_lngs(spatial_tessellation: Any) -> np.ndarray:
    df = to_pandas_frame(spatial_tessellation)
    if "geometry" in df.columns:
        return np.asarray([geometry_centroid_lat_lng(geom) for geom in df["geometry"].values], dtype=float)
    if LATITUDE in df.columns and LONGITUDE in df.columns:
        return df[[LATITUDE, LONGITUDE]].to_numpy(dtype=float)
    if "latitude" in df.columns and "longitude" in df.columns:
        return df[["latitude", "longitude"]].to_numpy(dtype=float)
    if "lat" in df.columns and "lon" in df.columns:
        return df[["lat", "lon"]].to_numpy(dtype=float)
    raise ValueError("spatial_tessellation must include a geometry column or latitude/longitude columns.")


class FlowDataFrame(pd.DataFrame):
    _metadata = ["tessellation", "tile_id"]

    @property
    def _constructor(self):
        return FlowDataFrame

    def to_matrix(self):
        if self.empty:
            return np.zeros((0, 0))
        if getattr(self, "tessellation", None) is not None and getattr(self, "tile_id", None) in self.tessellation:
            tile_ids = list(self.tessellation[self.tile_id].values)
        else:
            tile_ids = sorted(set(self[ORIGIN].tolist()) | set(self[DESTINATION].tolist()))
        index = {tile_id: i for i, tile_id in enumerate(tile_ids)}
        matrix = np.zeros((len(tile_ids), len(tile_ids)), dtype=float)
        for _, row in self.iterrows():
            matrix[index[row[ORIGIN]], index[row[DESTINATION]]] = row[FLOW]
        return matrix


def flow_dataframe(
    rows: list[list[Any]],
    columns: tuple[str, str, str] = (ORIGIN, DESTINATION, FLOW),
    *,
    tessellation: Any | None = None,
    tile_id: str = TILE_ID,
) -> pd.DataFrame:
    df = FlowDataFrame(rows, columns=list(columns))
    df.tessellation = tessellation
    df.tile_id = tile_id
    return df


class TrajDataFrame(pd.DataFrame):
    _metadata = ["parameters"]

    @property
    def _constructor(self):
        return TrajDataFrame

    def sort_by_uid_and_datetime(self):
        return self.sort_values([UID, DATETIME]).reset_index(drop=True)


def trajectory_dataframe(rows: list[tuple[Any, float, float, Any]], parameters: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[UID, LATITUDE, LONGITUDE, DATETIME])
    if df.empty:
        out = TrajDataFrame(columns=[UID, DATETIME, LATITUDE, LONGITUDE])
    else:
        out = TrajDataFrame(
            df.sort_values([UID, DATETIME]).reset_index(drop=True)[[UID, DATETIME, LATITUDE, LONGITUDE]]
        )
    out.parameters = parameters
    return out


def require_optional(module_name: str, extra: str = "generation"):
    try:
        module = __import__(module_name)
    except ImportError as exc:
        raise ImportError(f"{module_name} is required: pip install skmob2[{extra}]") from exc
    return module
