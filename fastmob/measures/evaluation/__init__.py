"""fastmob.measures.evaluation — generic comparison metrics.

For grouped/dataframe-level comparisons, use ``BaseDataFrame.compare_to()``
(inherited by ``Staypoints``, ``Trips``, ``Triplegs``, ``Locations``,
``TrajDataFrame``, ``FlowDataFrame``, ``Tours``) instead of hand-coding a
grouping around these primitives.
"""

from .compare import ComparisonResult, compare_to
from .cpc import common_part_of_commuters, common_part_of_commuters_distance, common_part_of_links
from .information import information_gain, kullback_leibler_divergence
from .metrics import jensen_shannon_divergence, wasserstein_distance
from .regression import max_error, mse, nrmse, r_squared, rmse
from .spatial import stvd_emd

__all__ = [
    "ComparisonResult",
    "common_part_of_commuters",
    "common_part_of_commuters_distance",
    "common_part_of_links",
    "compare_to",
    "information_gain",
    "jensen_shannon_divergence",
    "kullback_leibler_divergence",
    "max_error",
    "mse",
    "nrmse",
    "r_squared",
    "rmse",
    "stvd_emd",
    "wasserstein_distance",
]
