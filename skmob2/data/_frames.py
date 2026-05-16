from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LATITUDE = "lat"
LONGITUDE = "lng"
DATETIME = "datetime"
UID = "uid"
TID = "tid"
TILE_ID = "tile_id"
ORIGIN = "origin"
DESTINATION = "destination"
FLOW = "flow"
DEFAULT_CRS = {"init": "epsg:4326"}


class TrajDataFrame(pd.DataFrame):
    """Pandas-backed trajectory dataframe for dataset compatibility."""

    _metadata = ["_parameters", "_crs", "_info"]

    @property
    def _constructor(self):
        return TrajDataFrame

    def __init__(
        self,
        data=None,
        latitude: int | str = LATITUDE,
        longitude: int | str = LONGITUDE,
        datetime: int | str = DATETIME,
        user_id: int | str = UID,
        trajectory_id: int | str = TID,
        timestamp: bool = False,
        crs: dict | None = DEFAULT_CRS,
        parameters: dict | None = None,
        **kwargs: Any,
    ):
        self._parameters = {} if parameters is None else parameters
        self._crs = DEFAULT_CRS if crs is None else crs
        self._info = None

        if isinstance(data, pd.DataFrame):
            frame = data.copy().rename(
                columns={
                    latitude: LATITUDE,
                    longitude: LONGITUDE,
                    datetime: DATETIME,
                    user_id: UID,
                    trajectory_id: TID,
                }
            )
            super().__init__(frame, **kwargs)
        elif isinstance(data, dict):
            frame = pd.DataFrame.from_dict(data).rename(
                columns={
                    latitude: LATITUDE,
                    longitude: LONGITUDE,
                    datetime: DATETIME,
                    user_id: UID,
                    trajectory_id: TID,
                }
            )
            super().__init__(frame, **kwargs)
        elif isinstance(data, (list, np.ndarray)) and len(data) > 0:
            mapping = {
                latitude: LATITUDE,
                longitude: LONGITUDE,
                datetime: DATETIME,
                user_id: UID,
                trajectory_id: TID,
            }
            columns = [mapping.get(idx, idx) for idx in range(len(data[0]))]
            super().__init__(data, columns=columns, **kwargs)
        else:
            super().__init__(data, **kwargs)

        if not isinstance(self._parameters, dict):
            raise AttributeError("parameters must be a dictionary.")
        if not isinstance(self._crs, dict):
            raise TypeError("crs must be a dict type.")
        if self._has_traj_columns():
            self._set_traj(timestamp=timestamp, inplace=True)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        self._crs = value

    def _has_traj_columns(self) -> bool:
        return DATETIME in self and LATITUDE in self and LONGITUDE in self

    def _set_traj(self, timestamp: bool = False, inplace: bool = False):
        frame = self if inplace else self.copy()
        if timestamp:
            frame[DATETIME] = pd.to_datetime(frame[DATETIME], unit="s")
        elif not pd.api.types.is_datetime64_any_dtype(frame[DATETIME]):
            frame[DATETIME] = pd.to_datetime(frame[DATETIME])
        frame[LATITUDE] = frame[LATITUDE].astype(float)
        frame[LONGITUDE] = frame[LONGITUDE].astype(float)
        frame.parameters = self._parameters
        frame.crs = self._crs
        if not inplace:
            return frame

    def to_flowdataframe(self, tessellation, self_loops: bool = True):
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("geopandas is required for flow datasets: pip install skmob2[data]") from exc

        frame = self.sort_values([UID, DATETIME], kind="mergesort").reset_index(drop=True)
        points = gpd.GeoDataFrame(
            frame.copy(),
            geometry=gpd.points_from_xy(frame[LONGITUDE], frame[LATITUDE]),
            crs="EPSG:4326",
        )
        tess = tessellation
        if getattr(tess, "crs", None) is None:
            tess = tess.set_crs("EPSG:4326")
        if points.crs != tess.crs:
            points = points.to_crs(tess.crs)

        joined = gpd.sjoin(points, tess[[TILE_ID, "geometry"]], how="left", predicate="within")
        joined[DESTINATION] = joined[TILE_ID].shift(-1)
        joined["next_uid"] = joined[UID].shift(-1)
        flow = joined[joined[UID] == joined["next_uid"]].dropna(subset=[TILE_ID, DESTINATION])
        flow = flow.groupby([TILE_ID, DESTINATION], dropna=True).size().reset_index(name=FLOW)
        flow = flow.rename(columns={TILE_ID: ORIGIN})
        if not self_loops:
            flow = flow[flow[ORIGIN] != flow[DESTINATION]]
        return FlowDataFrame(flow, tessellation=tessellation)


class FlowDataFrame(pd.DataFrame):
    """Pandas-backed flow dataframe for dataset compatibility."""

    _metadata = ["tessellation", "tile_id", "_parameters", "_info"]

    @property
    def _constructor(self):
        return FlowDataFrame

    def __init__(
        self,
        data=None,
        origin: str = ORIGIN,
        destination: str = DESTINATION,
        flow: str = FLOW,
        tile_id: str = TILE_ID,
        tessellation=None,
        parameters: dict | None = None,
        **kwargs: Any,
    ):
        self.tessellation = tessellation
        self.tile_id = tile_id
        self._parameters = {} if parameters is None else parameters
        self._info = None
        if isinstance(data, pd.DataFrame):
            data = data.rename(columns={origin: ORIGIN, destination: DESTINATION, flow: FLOW})
        super().__init__(data, **kwargs)
        if ORIGIN in self:
            self[ORIGIN] = self[ORIGIN].astype(str)
        if DESTINATION in self:
            self[DESTINATION] = self[DESTINATION].astype(str)

    def to_matrix(self):
        if self.empty:
            return np.zeros((0, 0))
        if getattr(self, "tessellation", None) is not None and getattr(self, "tile_id", None) in self.tessellation:
            tile_ids = list(self.tessellation[self.tile_id].astype(str).values)
        else:
            tile_ids = sorted(set(self[ORIGIN].astype(str).tolist()) | set(self[DESTINATION].astype(str).tolist()))
        index = {tile_id: i for i, tile_id in enumerate(tile_ids)}
        matrix = np.zeros((len(tile_ids), len(tile_ids)), dtype=float)
        for _, row in self.iterrows():
            matrix[index[str(row[ORIGIN])], index[str(row[DESTINATION])]] = row[FLOW]
        return matrix
