from skmob2.core.base import BaseDataFrame
from skmob2.measures._common import _detect_trajectory_columns, _prepare_trajectory
from skmob2.measures.spatial import jump_lengths
import narwhals as nw


class TrajDataFrame(BaseDataFrame):
    def __init__(
        self,
        df,
        sort=False,
        timestamp=True,
        datetime_col=None,
        lat_col=None,
        lng_col=None,
        uid_col=None,
        **kwargs,
    ):
        if isinstance(df, TrajDataFrame):
            # If the input is already a TrajDataFrame, we can skip the column detection and preparation steps
            super().__init__(df.df, **kwargs)
            return

        super().__init__(df, **kwargs)

        self.sorted = False
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
        from skmob2.measures.spatial import radius_of_gyration

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
