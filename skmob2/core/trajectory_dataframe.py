from skmob2.core.base import BaseDataFrame
from skmob2.measures._common import _detect_trajectory_columns, _prepare_trajectory
from skmob2.measures.spatial import jump_lengths
import narwhals as nw


class TrajDataFrame(BaseDataFrame):
    def __init__(self, df, sort=False, timestamp=True, **kwargs):
        super().__init__(df, **kwargs)

        self.sorted = False
        datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(self.df)
        self.datetime_col = datetime_col
        self.lat_col = lat_col
        self.lng_col = lng_col
        self.uid_col = uid_col

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
                    self.df = native_pl_df.with_columns(
                        pl.col(self.datetime_col).str.to_datetime(time_zone="UTC")
                    )
                else:
                    # 3. Fallback for Pandas, Modin, CuDF, etc. via Narwhals
                    self.df = nw_df.with_columns(
                        nw.col(self.datetime_col).str.to_datetime()
                    ).to_native()

        if sort:
            nw_df = nw.from_native(self.df, eager_only=True)
            datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(nw_df)
            self.df = _prepare_trajectory(
                nw_df,
                datetime_col=datetime_col,
                lat_col=lat_col,
                lng_col=lng_col,
                uid_col=uid_col,
            ).to_native()
            self.sorted = True

    def jump_lengths(self):
        return jump_lengths(
            self.df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            sorted=self.sorted,
        )
