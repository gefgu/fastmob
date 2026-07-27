"""fastmob.measures.evaluation — distribution comparison and evaluation metrics."""

from .activity import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix_jensen_shannon_divergence,
    motif_distribution_jensen_shannon_divergence,
)
from .distribution import (
    column_distribution_jensen_shannon_divergence,
    column_distribution_wasserstein_distance,
    visits_per_user_jensen_shannon_divergence,
    visits_per_user_wasserstein_distance,
)
from .evaluation import (
    common_part_of_commuters,
    common_part_of_commuters_distance,
    common_part_of_links,
    information_gain,
    kullback_leibler_divergence,
    max_error,
    mse,
    nrmse,
    pearson_correlation,
    r_squared,
    rmse,
    spearman_correlation,
)
from .metrics import (
    histogram_jensen_shannon_divergence,
    jensen_shannon_divergence,
    matrix_jensen_shannon_divergence,
    time_bin_matrix_jensen_shannon_divergence,
    wasserstein_distance,
)
from .spatial import (
    dwell_time_wasserstein_distance,
    od_matrix_common_part_of_commuters,
    profile_metric_wasserstein_distance,
    radius_of_gyration_wasserstein_distance,
    stvd_emd,
    trajectory_common_part_of_commuters,
    trajectory_common_part_of_commuters_multi,
)

__all__ = [
    "activity_distribution_jensen_shannon_divergence",
    "activity_transition_matrix_jensen_shannon_divergence",
    "column_distribution_jensen_shannon_divergence",
    "column_distribution_wasserstein_distance",
    "common_part_of_commuters",
    "common_part_of_commuters_distance",
    "common_part_of_links",
    "dwell_time_wasserstein_distance",
    "histogram_jensen_shannon_divergence",
    "information_gain",
    "jensen_shannon_divergence",
    "kullback_leibler_divergence",
    "matrix_jensen_shannon_divergence",
    "max_error",
    "motif_distribution_jensen_shannon_divergence",
    "mse",
    "nrmse",
    "od_matrix_common_part_of_commuters",
    "pearson_correlation",
    "profile_metric_wasserstein_distance",
    "r_squared",
    "radius_of_gyration_wasserstein_distance",
    "rmse",
    "spearman_correlation",
    "stvd_emd",
    "time_bin_matrix_jensen_shannon_divergence",
    "trajectory_common_part_of_commuters",
    "trajectory_common_part_of_commuters_multi",
    "visits_per_user_jensen_shannon_divergence",
    "visits_per_user_wasserstein_distance",
    "wasserstein_distance",
]
