"""fkmob.measures.evaluation — distribution comparison and evaluation metrics."""

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
    "jensen_shannon_divergence",
    "matrix_jensen_shannon_divergence",
    "time_bin_matrix_jensen_shannon_divergence",
    "histogram_jensen_shannon_divergence",
    "wasserstein_distance",
    "column_distribution_jensen_shannon_divergence",
    "column_distribution_wasserstein_distance",
    "visits_per_user_jensen_shannon_divergence",
    "visits_per_user_wasserstein_distance",
    "od_matrix_common_part_of_commuters",
    "profile_metric_wasserstein_distance",
    "radius_of_gyration_wasserstein_distance",
    "dwell_time_wasserstein_distance",
    "stvd_emd",
    "trajectory_common_part_of_commuters",
    "trajectory_common_part_of_commuters_multi",
    "activity_distribution_jensen_shannon_divergence",
    "activity_transition_matrix_jensen_shannon_divergence",
    "motif_distribution_jensen_shannon_divergence",
    "common_part_of_commuters",
    "common_part_of_links",
    "common_part_of_commuters_distance",
    "r_squared",
    "mse",
    "rmse",
    "nrmse",
    "max_error",
    "information_gain",
    "kullback_leibler_divergence",
    "pearson_correlation",
    "spearman_correlation",
]
