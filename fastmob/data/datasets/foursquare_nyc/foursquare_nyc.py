from __future__ import annotations

import pyarrow.compute as pc

from fastmob.data._frames import TrajDataFrame
from fastmob.data.load import DatasetBuilder
from fastmob.utils import _arrow_io

_COLUMN_NAMES = [
    "user_id",
    "venue_id",
    "venue_category_id",
    "venue_category_name",
    "latitude",
    "longitude",
    "tz_offset",
    "datetime",
]
_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"


class foursquare_nyc(DatasetBuilder):
    def prepare(self, f_names):
        fs = [path for path in f_names if "TSMC2014_NYC.txt" in path]
        raw_data = _arrow_io.read_delimited(
            fs[0],
            delimiter=self.dataset_info["sep"],
            encoding=self.dataset_info["encoding"],
            header=False,
            column_names=_COLUMN_NAMES,
        )
        raw_data = raw_data.set_column(
            raw_data.column_names.index("datetime"),
            "datetime",
            pc.strptime(raw_data["datetime"], format=_TIMESTAMP_FORMAT, unit="us"),
        )
        return TrajDataFrame(
            raw_data, latitude="latitude", longitude="longitude", user_id="user_id", datetime="datetime"
        )
