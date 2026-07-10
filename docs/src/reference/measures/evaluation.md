# Evaluation Measures

| API | Description |
| --- | --- |
| [`common_part_of_commuters`](#fkmob.measures.evaluation.common_part_of_commuters) | Return the common part of commuters (CPC) between two flow arrays. |
| [`common_part_of_links`](#fkmob.measures.evaluation.common_part_of_links) | Return the common part of links (CPL) between two flow arrays. |
| [`common_part_of_commuters_distance`](#fkmob.measures.evaluation.common_part_of_commuters_distance) | Return the common part of commuters by distance (CPCD). |
| [`r_squared`](#fkmob.measures.evaluation.r_squared) | Return the coefficient of determination R-squared. |
| [`rmse`](#fkmob.measures.evaluation.rmse) | Return the root mean squared error between true and predicted values. |
| [`nrmse`](#fkmob.measures.evaluation.nrmse) | Return the normalized root mean squared error (RMSE / sum(true)). |
| [`information_gain`](#fkmob.measures.evaluation.information_gain) | Return the information gain of true over predicted values. |
| [`pearson_correlation`](#fkmob.measures.evaluation.pearson_correlation) | Return the Pearson correlation coefficient and its two-tailed p-value. |
| [`spearman_correlation`](#fkmob.measures.evaluation.spearman_correlation) | Return the Spearman rank-order correlation coefficient and its p-value. |
| [`kullback_leibler_divergence`](#fkmob.measures.evaluation.kullback_leibler_divergence) | Return the Kullback-Leibler divergence between true and predicted values. |
| [`max_error`](#fkmob.measures.evaluation.max_error) | Return the maximum signed error max(true_i - pred_i). |
| [`mse`](#fkmob.measures.evaluation.mse) | Return the mean squared error between true and predicted values. |
| [`jensen_shannon_divergence`](#fkmob.measures.evaluation.jensen_shannon_divergence) | Return Jensen-Shannon divergence between two distributions. |
| [`matrix_jensen_shannon_divergence`](#fkmob.measures.evaluation.matrix_jensen_shannon_divergence) | Return Jensen-Shannon divergence between two category matrices. |
| [`time_bin_matrix_jensen_shannon_divergence`](#fkmob.measures.evaluation.time_bin_matrix_jensen_shannon_divergence) | Return mean per-column Jensen-Shannon divergence for time-bin matrices. |
| [`histogram_jensen_shannon_divergence`](#fkmob.measures.evaluation.histogram_jensen_shannon_divergence) | Bin two value arrays and return Jensen-Shannon divergence. |
| [`wasserstein_distance`](#fkmob.measures.evaluation.wasserstein_distance) | Return Rust-backed 1D Wasserstein distance between empirical samples. |
| [`column_distribution_jensen_shannon_divergence`](#fkmob.measures.evaluation.column_distribution_jensen_shannon_divergence) | Compare grouped numeric column distributions with Jensen-Shannon divergence. |
| [`column_distribution_wasserstein_distance`](#fkmob.measures.evaluation.column_distribution_wasserstein_distance) | Compare grouped numeric column distributions with Rust-backed Wasserstein distance. |
| [`visits_per_user_jensen_shannon_divergence`](#fkmob.measures.evaluation.visits_per_user_jensen_shannon_divergence) | Compare grouped visits-per-user distributions with Jensen-Shannon divergence. |
| [`visits_per_user_wasserstein_distance`](#fkmob.measures.evaluation.visits_per_user_wasserstein_distance) | Compare grouped visits-per-user distributions with Rust-backed Wasserstein distance. |
| [`od_matrix_common_part_of_commuters`](#fkmob.measures.evaluation.od_matrix_common_part_of_commuters) | Return CPC between two OD matrices, aligning labels when available. |
| [`profile_metric_wasserstein_distance`](#fkmob.measures.evaluation.profile_metric_wasserstein_distance) | Compare a scalar profile metric column with Rust-backed Wasserstein distance. |
| [`radius_of_gyration_wasserstein_distance`](#fkmob.measures.evaluation.radius_of_gyration_wasserstein_distance) | Compare radius-of-gyration distributions, optionally grouped by a column. |
| [`dwell_time_wasserstein_distance`](#fkmob.measures.evaluation.dwell_time_wasserstein_distance) | Compare dwell-time distributions in hours. |
| [`stvd_emd`](#fkmob.measures.evaluation.stvd_emd) | Compute the spatio-temporal Wasserstein distance between two distributions. |
| [`activity_distribution_jensen_shannon_divergence`](#fkmob.measures.evaluation.activity_distribution_jensen_shannon_divergence) | Compare categorical activity distributions with Jensen-Shannon divergence. |
| [`activity_transition_matrix_jensen_shannon_divergence`](#fkmob.measures.evaluation.activity_transition_matrix_jensen_shannon_divergence) | Compare activity transition matrices with Jensen-Shannon divergence. |
| [`motif_distribution_jensen_shannon_divergence`](#fkmob.measures.evaluation.motif_distribution_jensen_shannon_divergence) | Discover daily motifs for two visit datasets and compare motif distributions. |

::: fkmob.measures.evaluation
    options:
      show_source: false
