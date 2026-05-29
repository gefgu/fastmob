import narwhals as nw
import numpy as np
import pandas as pd

from skmob2.core.base import BaseDataFrame
from skmob2.measures._common import _detect_trajectory_columns, _prepare_trajectory
from skmob2.measures.individual import jump_lengths

LATITUDE = "lat"
LONGITUDE = "lng"
DATETIME = "datetime"
UID = "uid"
TID = "tid"
TILE_ID = "tile_id"
DEFAULT_CRS = {"init": "epsg:4326"}


class TrajDataFrame(BaseDataFrame):
    def __init__(
        self,
        df,
        sort=False,
        timestamp=False,
        datetime_col=None,
        lat_col=None,
        lng_col=None,
        uid_col=None,
        latitude=None,
        longitude=None,
        datetime=None,
        user_id=None,
        trajectory_id=None,
        crs=None,
        parameters=None,
        **kwargs,
    ):
        if isinstance(df, TrajDataFrame):
            super().__init__(df.df)
            self.sorted = df.sorted
            self.datetime_col = df.datetime_col
            self.lat_col = df.lat_col
            self.lng_col = df.lng_col
            self.uid_col = df.uid_col
            self.trajectory_id_col = getattr(df, "trajectory_id_col", TID)
            self.crs = getattr(df, "crs", DEFAULT_CRS)
            self.parameters = getattr(df, "parameters", {})
            self._info = getattr(df, "_info", None)
            return

        latitude = LATITUDE if latitude is None else latitude
        longitude = LONGITUDE if longitude is None else longitude
        datetime = DATETIME if datetime is None else datetime
        user_id = UID if user_id is None else user_id
        trajectory_id = TID if trajectory_id is None else trajectory_id
        self.crs = DEFAULT_CRS if crs is None else crs
        self.parameters = {} if parameters is None else parameters
        self._info = None

        if latitude != LATITUDE or longitude != LONGITUDE or datetime != DATETIME or user_id != UID or trajectory_id != TID:
            df = self._rename_columns(
                df,
                {
                    latitude: LATITUDE,
                    longitude: LONGITUDE,
                    datetime: DATETIME,
                    user_id: UID,
                    trajectory_id: TID,
                },
            )

        super().__init__(df)

        self.sorted = False
        datetime_col = DATETIME if datetime_col is None and DATETIME in self.df else datetime_col
        lat_col = LATITUDE if lat_col is None and LATITUDE in self.df else lat_col
        lng_col = LONGITUDE if lng_col is None and LONGITUDE in self.df else lng_col
        uid_col = UID if uid_col is None and UID in self.df else uid_col
        self.trajectory_id_col = TID if TID in self.df else None
        self.datetime_col, self.lat_col, self.lng_col, self.uid_col = _detect_trajectory_columns(
            self.df, datetime_col, lat_col, lng_col, uid_col
        )

        if timestamp and self.datetime_col is not None:
            # 1. Wrap the native dataframe (pandas/polars) into a Narwhals frame
            nw_df = nw.from_native(self.df)

            # 2. Check the schema to see if the column is already a Datetime type
            col_dtype = nw_df.schema[self.datetime_col]

            if not isinstance(col_dtype, nw.Datetime):
                # 3. Use Narwhals expressions to convert the string to datetime
                # and unwrap it back to the native library format
                if "polars" in str(type(self.df)).lower():
                    import polars as pl

                    # Fix for Polars: parse with timezone handling built into the expression
                    native_pl_df = nw_df.to_native()
                    self.df = native_pl_df.with_columns(pl.col(self.datetime_col).str.to_datetime(time_zone="UTC"))
                else:
                    # 3. Fallback for Pandas, Modin, CuDF, etc. via Narwhals
                    self.df = nw_df.with_columns(nw.col(self.datetime_col).str.to_datetime()).to_native()

        if sort:
            nw_df = nw.from_native(self.df, eager_only=True)
            self.df = _prepare_trajectory(
                nw_df,
                datetime_col=self.datetime_col,
                lat_col=self.lat_col,
                lng_col=self.lng_col,
                uid_col=self.uid_col,
            ).to_native()
            self.sorted = True

    @staticmethod
    def _rename_columns(df, mapping):
        mapping = {source: target for source, target in mapping.items() if source != target}
        if not mapping:
            return df
        if isinstance(df, pd.DataFrame):
            return df.copy().rename(columns=mapping)
        if isinstance(df, dict):
            return pd.DataFrame.from_dict(df).rename(columns=mapping)
        if isinstance(df, (list, np.ndarray)) and len(df) > 0:
            columns = [mapping.get(idx, idx) for idx in range(len(df[0]))]
            return pd.DataFrame(df, columns=columns)
        try:
            return nw.from_native(df, eager_only=True).rename(mapping).to_native()
        except Exception:
            return df

    def jump_lengths(self, merge=False):
        return jump_lengths(
            self.df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            sorted=self.sorted,
            merge=merge,
        )

    def radius_of_gyration(self):
        from skmob2.measures.individual import radius_of_gyration

        return radius_of_gyration(
            self.df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            # sorted=self.sorted,
            # merge=merge,
        )

    # Apply compress function from the skmob2 library to the TrajDataFrame class
    def compress(self, spatial_radius_km=0.2, inplace=False):
        from skmob2.preprocessing.compress import compress

        if inplace:
            self.df = compress(
                self.df,
                spatial_radius_km=spatial_radius_km,
                datetime_col=self.datetime_col,
                lat_col=self.lat_col,
                lng_col=self.lng_col,
                uid_col=self.uid_col,
            )

            return self

        compressed_df = compress(
            self.df,
            spatial_radius_km=spatial_radius_km,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

        return TrajDataFrame(
            compressed_df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

    def stay_locations(self, inplace=False, **kwargs):
        from skmob2.preprocessing.stay_locations import stay_locations

        if inplace:
            self.df = stay_locations(
                self.df,
                datetime_col=self.datetime_col,
                lat_col=self.lat_col,
                lng_col=self.lng_col,
                uid_col=self.uid_col,
                **kwargs,
            )

            return self

        stay_locations_df = stay_locations(
            self.df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            **kwargs,
        )

        return TrajDataFrame(
            stay_locations_df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

    def to_flowdataframe(self, tessellation, self_loops=True):
        from skmob2.core import FlowDataFrame

        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("geopandas is required for flow datasets: pip install skmob2[data]") from exc

        frame = self.df
        if not isinstance(frame, pd.DataFrame):
            frame = nw.from_native(frame, eager_only=True).to_native()
        frame = frame.sort_values([self.uid_col, self.datetime_col], kind="mergesort").reset_index(drop=True)
        points = gpd.GeoDataFrame(
            frame.copy(),
            geometry=gpd.points_from_xy(frame[self.lng_col], frame[self.lat_col]),
            crs="EPSG:4326",
        )
        tess = tessellation
        if getattr(tess, "crs", None) is None:
            tess = tess.set_crs("EPSG:4326")
        if points.crs != tess.crs:
            points = points.to_crs(tess.crs)

        joined = gpd.sjoin(points, tess[[TILE_ID, "geometry"]], how="left", predicate="within")
        joined["destination"] = joined[TILE_ID].shift(-1)
        joined["next_uid"] = joined[self.uid_col].shift(-1)
        flow = joined[joined[self.uid_col] == joined["next_uid"]].dropna(subset=[TILE_ID, "destination"])
        flow = flow.groupby([TILE_ID, "destination"], dropna=True).size().reset_index(name="flow")
        flow = flow.rename(columns={TILE_ID: "origin"})
        if not self_loops:
            flow = flow[flow["origin"] != flow["destination"]]
        return FlowDataFrame(flow, tessellation=tessellation)
