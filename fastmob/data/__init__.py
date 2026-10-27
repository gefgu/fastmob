"""Dataset loaders compatible with the original scikit-mobility data API."""

from .brightkite import BRIGHTKITE_URL as BRIGHTKITE_SAMPLE
from .brightkite import load_brightkite
from .geolife import load_geolife
from .load import DatasetBuilder, get_dataset_info, list_datasets, load_dataset

__all__ = [
    "BRIGHTKITE_SAMPLE",
    "DatasetBuilder",
    "get_dataset_info",
    "list_datasets",
    "load_brightkite",
    "load_dataset",
    "load_geolife",
]
