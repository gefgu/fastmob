from __future__ import annotations

from fastmob.data.load import DatasetBuilder
from fastmob.utils import _arrow_io


class parking_san_francisco(DatasetBuilder):
    """Parking meters in San Francisco.

    ``prepare()`` returns a raw ``pyarrow.Table`` (this loader does not wrap
    its output in `TrajDataFrame`/`FlowDataFrame`).
    """

    def prepare(self, f_names):
        fs = [path for path in f_names if path.endswith(".csv")]
        return _arrow_io.read_delimited(fs[0], delimiter=self.dataset_info["sep"])
