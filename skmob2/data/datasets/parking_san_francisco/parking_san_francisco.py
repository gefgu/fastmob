from __future__ import annotations

import pandas as pd

from skmob2.data.load import DatasetBuilder


class parking_san_francisco(DatasetBuilder):
    def prepare(self, f_names):
        fs = [path for path in f_names if path.endswith(".csv")]
        return pd.read_csv(fs[0], sep=self.dataset_info["sep"])
