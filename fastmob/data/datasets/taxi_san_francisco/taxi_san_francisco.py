from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from fastmob.data._frames import TrajDataFrame
from fastmob.data.load import DatasetBuilder
from fastmob.utils import _arrow_io

_SEVEN_HOURS = pa.scalar(7 * 3600, type=pa.duration("s"))


class taxi_san_francisco(DatasetBuilder):
    def prepare(self, f_names):
        fs = [path for path in f_names if "new_abboip.txt" in path]

        raw_data = _arrow_io.read_delimited(
            fs[0],
            delimiter=self.dataset_info["sep"],
            encoding=self.dataset_info["encoding"],
            header=False,
            column_names=["latitude", "longitude", "occupancy", "time"],
        )
        # Epoch seconds -> UTC timestamp, then shift by the same fixed -7h
        # offset the original pandas date_parser applied.
        timestamps = pc.subtract(pc.cast(raw_data["time"], pa.timestamp("s")), _SEVEN_HOURS)
        raw_data = raw_data.set_column(raw_data.column_names.index("time"), "time", timestamps)
        raw_data = raw_data.append_column("user_id", pa.array(["abboip"] * raw_data.num_rows))
        return TrajDataFrame(raw_data, latitude="latitude", longitude="longitude", user_id="user_id", datetime="time")
