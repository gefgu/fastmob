from __future__ import annotations

from datetime import timedelta

import pandas as pd

from fkmob.data._frames import TrajDataFrame
from fkmob.data.load import DatasetBuilder


class taxi_san_francisco(DatasetBuilder):
    def prepare(self, f_names):
        fs = [path for path in f_names if "new_abboip.txt" in path]

        def mydateparser(value):
            return pd.to_datetime(value, unit="s") + timedelta(minutes=-7 * 60)

        raw_data = pd.read_csv(
            fs[0],
            sep=self.dataset_info["sep"],
            encoding=self.dataset_info["encoding"],
            names=["latitude", "longitude", "occupancy", "time"],
            parse_dates=["time"],
            date_parser=mydateparser,
            header=None,
        )
        raw_data["user_id"] = ["abboip"] * len(raw_data)
        return TrajDataFrame(raw_data, latitude="latitude", longitude="longitude", user_id="user_id", datetime="time")
