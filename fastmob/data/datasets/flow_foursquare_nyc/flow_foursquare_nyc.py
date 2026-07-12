from __future__ import annotations

import pandas as pd

from fastmob.data._frames import TrajDataFrame
from fastmob.data.load import DatasetBuilder
from fastmob.tessellation import tilers


class flow_foursquare_nyc(DatasetBuilder):
    def prepare(self, f_names):
        fs = [path for path in f_names if "TSMC2014_NYC.txt" in path]
        raw_data = pd.read_csv(
            fs[0],
            sep=self.dataset_info["sep"],
            encoding=self.dataset_info["encoding"],
            header=None,
        )
        tdf_dataset = TrajDataFrame(raw_data, latitude=4, longitude=5, user_id=0, datetime=7)
        tessellation = tilers.tiler.get("squared", base_shape="New York City", meters=4000)
        return tdf_dataset.to_flowdataframe(tessellation=tessellation, self_loops=True)
