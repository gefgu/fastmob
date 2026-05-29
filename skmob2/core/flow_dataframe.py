from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd

from skmob2.core.base import BaseDataFrame

TILE_ID = "tile_id"
ORIGIN = "origin"
DESTINATION = "destination"
FLOW = "flow"


class FlowDataFrame(BaseDataFrame):
    def __init__(
        self,
        df=None,
        origin: str = ORIGIN,
        destination: str = DESTINATION,
        flow: str = FLOW,
        tile_id: str = TILE_ID,
        tessellation: Any | None = None,
        parameters: dict | None = None,
        **kwargs,
    ):
        if isinstance(df, FlowDataFrame):
            super().__init__(df.df)
            self.tessellation = getattr(df, "tessellation", tessellation)
            self.tile_id = getattr(df, "tile_id", tile_id)
            self.parameters = getattr(df, "parameters", {})
            self._info = getattr(df, "_info", None)
            return

        self.tessellation = tessellation
        self.tile_id = tile_id
        self.parameters = {} if parameters is None else parameters
        self._info = None

        frame = self._coerce_frame(df, kwargs)
        frame = self._rename_columns(frame, {origin: ORIGIN, destination: DESTINATION, flow: FLOW})
        super().__init__(frame)

    @staticmethod
    def _coerce_frame(df, kwargs):
        if df is None:
            return pd.DataFrame(**kwargs)
        if isinstance(df, pd.DataFrame):
            return df.copy()
        if isinstance(df, dict):
            return pd.DataFrame.from_dict(df, **kwargs)
        if isinstance(df, (list, np.ndarray)):
            return pd.DataFrame(df, **kwargs)
        return df

    @staticmethod
    def _rename_columns(df, mapping):
        mapping = {source: target for source, target in mapping.items() if source != target}
        if not mapping:
            return df
        if isinstance(df, pd.DataFrame):
            return df.rename(columns=mapping)
        try:
            return nw.from_native(df, eager_only=True).rename(mapping).to_native()
        except Exception:
            return df

    def to_matrix(self):
        if len(self.df) == 0:
            return np.zeros((0, 0))

        frame = self.df if isinstance(self.df, pd.DataFrame) else nw.from_native(self.df, eager_only=True).to_native()
        if self.tessellation is not None and self.tile_id in self.tessellation:
            tile_ids = [self._key(value) for value in self.tessellation[self.tile_id].values]
        else:
            tile_ids = sorted({self._key(value) for value in frame[ORIGIN]} | {self._key(value) for value in frame[DESTINATION]})

        index = {tile_id: i for i, tile_id in enumerate(tile_ids)}
        matrix = np.zeros((len(tile_ids), len(tile_ids)), dtype=float)
        for _, row in frame.iterrows():
            matrix[index[self._key(row[ORIGIN])], index[self._key(row[DESTINATION])]] = row[FLOW]
        return matrix

    @staticmethod
    def _key(value):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
