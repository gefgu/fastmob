---
icon: lucide/chart-line
---

# Evaluation Measures

| API | Description |
| --- | --- |
| [`common_part_of_commuters`](#fastmob.measures.evaluation.common_part_of_commuters) | Compare Trips or sparse FlowDataFrames with Rust-backed CPC. |
| [`common_part_of_links`](#fastmob.measures.evaluation.common_part_of_links) | Return the common part of links (CPL) between two flow arrays. |
| [`common_part_of_commuters_distance`](#fastmob.measures.evaluation.common_part_of_commuters_distance) | Return the common part of commuters by distance (CPCD). |
| [`compare_to`](#fastmob.measures.evaluation.compare_to) | Compare a value column between two dataframes, optionally grouped by an existing column. Also available as `BaseDataFrame.compare_to()`. |
| [`r_squared`](#fastmob.measures.evaluation.r_squared) | Return the coefficient of determination R-squared. |
| [`rmse`](#fastmob.measures.evaluation.rmse) | Return the root mean squared error between true and predicted values. |
| [`nrmse`](#fastmob.measures.evaluation.nrmse) | Return the normalized root mean squared error (RMSE / sum(true)). |
| [`information_gain`](#fastmob.measures.evaluation.information_gain) | Return the information gain of true over predicted values. |
| [`kullback_leibler_divergence`](#fastmob.measures.evaluation.kullback_leibler_divergence) | Return the Kullback-Leibler divergence between true and predicted values. |
| [`max_error`](#fastmob.measures.evaluation.max_error) | Return the maximum signed error max(true_i - pred_i). |
| [`mse`](#fastmob.measures.evaluation.mse) | Return the mean squared error between true and predicted values. |
| [`jensen_shannon_divergence`](#fastmob.measures.evaluation.jensen_shannon_divergence) | Return Jensen-Shannon divergence between two distributions. |
| [`wasserstein_distance`](#fastmob.measures.evaluation.wasserstein_distance) | Return Rust-backed 1D Wasserstein distance between empirical samples. |
| [`stvd_emd`](#fastmob.measures.evaluation.stvd_emd) | Compute the spatio-temporal Wasserstein distance between two distributions. |

::: fastmob.measures.evaluation
    options:
      show_source: false
