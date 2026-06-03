# Evaluation Measures

| API | Description |
| --- | --- |
| [`jensen_shannon_divergence`](#skmob2.measures.evaluation.jensen_shannon_divergence) | Return Jensen-Shannon divergence between two distributions. |
| [`matrix_jensen_shannon_divergence`](#skmob2.measures.evaluation.matrix_jensen_shannon_divergence) | Return Jensen-Shannon divergence between two category matrices. |
| [`time_bin_matrix_jensen_shannon_divergence`](#skmob2.measures.evaluation.time_bin_matrix_jensen_shannon_divergence) | Return mean per-column Jensen-Shannon divergence for time-bin matrices. |
| [`histogram_jensen_shannon_divergence`](#skmob2.measures.evaluation.histogram_jensen_shannon_divergence) | Bin two value arrays and return Jensen-Shannon divergence. |
| [`wasserstein_distance`](#skmob2.measures.evaluation.wasserstein_distance) | Return Rust-backed 1D Wasserstein distance between empirical samples. |
| [`column_distribution_jensen_shannon_divergence`](#skmob2.measures.evaluation.column_distribution_jensen_shannon_divergence) | Compare grouped numeric column distributions with Jensen-Shannon divergence. |
| [`column_distribution_wasserstein_distance`](#skmob2.measures.evaluation.column_distribution_wasserstein_distance) | Compare grouped numeric column distributions with Rust-backed Wasserstein distance. |
| [`visits_per_user_jensen_shannon_divergence`](#skmob2.measures.evaluation.visits_per_user_jensen_shannon_divergence) | Compare grouped visits-per-user distributions with Jensen-Shannon divergence. |
| [`visits_per_user_wasserstein_distance`](#skmob2.measures.evaluation.visits_per_user_wasserstein_distance) | Compare grouped visits-per-user distributions with Rust-backed Wasserstein distance. |
| [`od_matrix_common_part_of_commuters`](#skmob2.measures.evaluation.od_matrix_common_part_of_commuters) | Return CPC between two OD matrices, aligning labels when available. |
| [`profile_metric_wasserstein_distance`](#skmob2.measures.evaluation.profile_metric_wasserstein_distance) | Compare a scalar profile metric column with Rust-backed Wasserstein distance. |
| [`radius_of_gyration_wasserstein_distance`](#skmob2.measures.evaluation.radius_of_gyration_wasserstein_distance) | Compare radius-of-gyration distributions, optionally grouped by a column. |
| [`dwell_time_wasserstein_distance`](#skmob2.measures.evaluation.dwell_time_wasserstein_distance) | Compare dwell-time distributions in hours. |
| [`stvd_emd`](#skmob2.measures.evaluation.stvd_emd) | Compute the spatio-temporal Wasserstein distance between two distributions. |
| [`activity_distribution_jensen_shannon_divergence`](#skmob2.measures.evaluation.activity_distribution_jensen_shannon_divergence) | Compare categorical activity distributions with Jensen-Shannon divergence. |
| [`activity_transition_matrix_jensen_shannon_divergence`](#skmob2.measures.evaluation.activity_transition_matrix_jensen_shannon_divergence) | Compare activity transition matrices with Jensen-Shannon divergence. |
| [`motif_distribution_jensen_shannon_divergence`](#skmob2.measures.evaluation.motif_distribution_jensen_shannon_divergence) | Discover daily motifs for two visit datasets and compare motif distributions. |
| [`common_part_of_commuters`](#skmob2.measures.evaluation.common_part_of_commuters) | Return the common part of commuters (CPC) between two flow arrays. |
| [`common_part_of_links`](#skmob2.measures.evaluation.common_part_of_links) | Return the common part of links (CPL) between two flow arrays. |
| [`common_part_of_commuters_distance`](#skmob2.measures.evaluation.common_part_of_commuters_distance) | Return the common part of commuters by distance (CPCD). |
| [`r_squared`](#skmob2.measures.evaluation.r_squared) | Return the coefficient of determination R-squared. |
| [`mse`](#skmob2.measures.evaluation.mse) | Return the mean squared error between true and predicted values. |
| [`rmse`](#skmob2.measures.evaluation.rmse) | Return the root mean squared error between true and predicted values. |
| [`nrmse`](#skmob2.measures.evaluation.nrmse) | Return the normalized root mean squared error (RMSE / sum(true)). |
| [`max_error`](#skmob2.measures.evaluation.max_error) | Return the maximum signed error max(true_i - pred_i). |
| [`information_gain`](#skmob2.measures.evaluation.information_gain) | Return the information gain of true over predicted values. |
| [`kullback_leibler_divergence`](#skmob2.measures.evaluation.kullback_leibler_divergence) | Return the Kullback-Leibler divergence between true and predicted values. |
| [`pearson_correlation`](#skmob2.measures.evaluation.pearson_correlation) | Return the Pearson correlation coefficient and its two-tailed p-value. |
| [`spearman_correlation`](#skmob2.measures.evaluation.spearman_correlation) | Return the Spearman rank-order correlation coefficient and its p-value. |

::: skmob2.measures.evaluation
    options:
      show_source: false
