from __future__ import annotations

import pandas as pd

from fkmob.data._frames import TrajDataFrame
from fkmob.data.load import DatasetBuilder


class foursquare_nyc(DatasetBuilder):
    def prepare(self, f_names):
        fs = [path for path in f_names if "TSMC2014_NYC.txt" in path]
        raw_data = pd.read_csv(
            fs[0],
            sep=self.dataset_info["sep"],
            encoding=self.dataset_info["encoding"],
            header=None,
        )
        return TrajDataFrame(raw_data, latitude=4, longitude=5, user_id=0, datetime=7)
