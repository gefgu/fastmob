---
icon: lucide/chart-line
---

# Evaluation Measures

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`ComparisonResult`](#fastmob.measures.evaluation.ComparisonResult) | Result of `compare_to`: an overall value and, if grouped, a per-group breakdown. |  |
| [`common_part_of_commuters`](#fastmob.measures.evaluation.common_part_of_commuters) | Compare two Trips or two FlowDataFrames through sparse Rust CPC. | [Common part of commuters and links](../../learn/fit-and-evaluate.md#common-part-of-commuters-and-links) |
| [`common_part_of_commuters_distance`](#fastmob.measures.evaluation.common_part_of_commuters_distance) | Compare `distance_km` distributions in two Trips using Rust CPCD. | [Common part of commuters and links](../../learn/fit-and-evaluate.md#common-part-of-commuters-and-links) |
| [`common_part_of_links`](#fastmob.measures.evaluation.common_part_of_links) | Compare active sparse OD links in two Trips or FlowDataFrames. | [Common part of commuters and links](../../learn/fit-and-evaluate.md#common-part-of-commuters-and-links) |
| [`compare_to`](#fastmob.measures.evaluation.compare_to) | Compare `value_col` between two dataframes, optionally grouped by `group_col`. | [Compare by group](../../learn/fit-and-evaluate.md#compare-by-group) |
| [`information_gain`](#fastmob.measures.evaluation.information_gain) | Return the information gain (KL divergence variant) of true over pred. | [Information and distribution metrics](../../learn/fit-and-evaluate.md#information-and-distribution-metrics) |
| [`jensen_shannon_divergence`](#fastmob.measures.evaluation.jensen_shannon_divergence) | Return Jensen-Shannon divergence between two distributions. | [Information and distribution metrics](../../learn/fit-and-evaluate.md#information-and-distribution-metrics) |
| [`kullback_leibler_divergence`](#fastmob.measures.evaluation.kullback_leibler_divergence) | Return the Kullback–Leibler divergence \(D_\mathrm{KL}(\mathrm{true}\,\\|\,\mathrm{pred})\). | [Information and distribution metrics](../../learn/fit-and-evaluate.md#information-and-distribution-metrics) |
| [`max_error`](#fastmob.measures.evaluation.max_error) | Return the maximum signed error max(true_i - pred_i). | [Error metrics on matched flows](../../learn/fit-and-evaluate.md#error-metrics-on-matched-flows) |
| [`mse`](#fastmob.measures.evaluation.mse) | Return the mean squared error between true and predicted values. | [Error metrics on matched flows](../../learn/fit-and-evaluate.md#error-metrics-on-matched-flows) |
| [`nrmse`](#fastmob.measures.evaluation.nrmse) | Return the normalized root mean squared error (RMSE / sum(true)). | [Error metrics on matched flows](../../learn/fit-and-evaluate.md#error-metrics-on-matched-flows) |
| [`r_squared`](#fastmob.measures.evaluation.r_squared) | Return the coefficient of determination R². | [Error metrics on matched flows](../../learn/fit-and-evaluate.md#error-metrics-on-matched-flows) |
| [`rmse`](#fastmob.measures.evaluation.rmse) | Return the root mean squared error between true and predicted values. | [Error metrics on matched flows](../../learn/fit-and-evaluate.md#error-metrics-on-matched-flows) |
| [`stvd_emd`](#fastmob.measures.evaluation.stvd_emd) | Compute the spatio-temporal Wasserstein distance between two distributions. | [Compare when things happen](../../learn/fit-and-evaluate.md#compare-when-things-happen-spatio-temporal-distance) |
| [`wasserstein_distance`](#fastmob.measures.evaluation.wasserstein_distance) | Return Rust-backed 1D Wasserstein distance between empirical samples. | [Information and distribution metrics](../../learn/fit-and-evaluate.md#information-and-distribution-metrics) |
<!-- api-table:end -->

::: fastmob.measures.evaluation
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Common part of commuters and links](../../learn/fit-and-evaluate.md#common-part-of-commuters-and-links) · [Compare by group](../../learn/fit-and-evaluate.md#compare-by-group) · [Information and distribution metrics](../../learn/fit-and-evaluate.md#information-and-distribution-metrics) · [Error metrics on matched flows](../../learn/fit-and-evaluate.md#error-metrics-on-matched-flows) · [Compare when things happen](../../learn/fit-and-evaluate.md#compare-when-things-happen-spatio-temporal-distance)

