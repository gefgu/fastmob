"""Dataset loaders compatible with the original scikit-mobility data API."""

from .load import DatasetBuilder, get_dataset_info, list_datasets, load_dataset

BRIGHTKITE_SAMPLE = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"

__all__ = ["BRIGHTKITE_SAMPLE", "DatasetBuilder", "get_dataset_info", "list_datasets", "load_dataset"]
