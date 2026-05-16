from __future__ import annotations

from skmob2.data.load import DatasetBuilder


class nyc_boundaries(DatasetBuilder):
    def prepare(self, f_names):
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("geopandas is required to load shape datasets: pip install skmob2[data]") from exc

        fs = [path for path in f_names if path.endswith(".shp")]
        return gpd.read_file(fs[0])
