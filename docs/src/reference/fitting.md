# Fitting

Fastkit-Mobility includes fitting utilities for common empirical mobility laws: distributions of
distances (maximum-likelihood fits with confidence intervals and a comparison of alternative forms),
visitation laws, and daily-location distributions. Use these functions after preparing the
corresponding trajectory-derived data. Gravity models are fitted with `Gravity.fit` (see
[Models](models.md)) and flows are scored with `compare_flow_models` (see
[Evaluation](measures/evaluation.md)).

For how the estimators work and what to report, see [How the fits work](../learn/fitting-methods.md).
For end-to-end examples, see [Compare to mobility laws](../learn/compare-to-mobility-laws.md) and the
[fit and evaluate guide](../learn/fit-and-evaluate.md).

## Catalogue

<!-- api-table:start -->
**Distributions of distances**

| API | Description | Example |
| --- | --- | --- |
| [`fit_distance_distribution`](#fastmob.measures.fitting.distance_distribution.fit_distance_distribution) | Fit a distribution to positive values by maximum likelihood, with confidence intervals. | [How sure is the fit? Range, uncertainty and alternatives](../learn/compare-to-mobility-laws.md#how-sure-is-the-fit-range-uncertainty-and-alternatives) |
| [`compare_distance_distributions`](#fastmob.measures.fitting.distance_distribution.compare_distance_distributions) | Fit several families to the same values and rank them by AIC. | [How sure is the fit? Range, uncertainty and alternatives](../learn/compare-to-mobility-laws.md#how-sure-is-the-fit-range-uncertainty-and-alternatives) |
| [`DistributionFit`](#fastmob.measures.fitting.distance_distribution.DistributionFit) | A fitted distribution of positive values, with its uncertainty and fit quality. | [How sure is the fit? Range, uncertainty and alternatives](../learn/compare-to-mobility-laws.md#how-sure-is-the-fit-range-uncertainty-and-alternatives) |
| [`DistributionFit.cdf`](#fastmob.measures.fitting.distance_distribution.DistributionFit.cdf) | Cumulative distribution function at `x` of the fitted distribution. |  |
| [`DistributionFit.pdf`](#fastmob.measures.fitting.distance_distribution.DistributionFit.pdf) | Probability density at `x` of the fitted distribution, normalised on `[xmin, xmax]`. |  |
| [`DistributionFit.summary`](#fastmob.measures.fitting.distance_distribution.DistributionFit.summary) | The estimates as a table. | [How sure is the fit? Range, uncertainty and alternatives](../learn/compare-to-mobility-laws.md#how-sure-is-the-fit-range-uncertainty-and-alternatives) |
| [`fit_values_to_truncated_powerlaw`](#fastmob.measures.fitting.fit_values_to_truncated_powerlaw) | Fit a truncated power law to positive values using a log histogram. | [Law 1](../learn/compare-to-mobility-laws.md#law-1-jump-lengths-follow-a-truncated-power-law) |
| [`log_truncated_powerlaw`](#fastmob.measures.fitting.log_truncated_powerlaw) | Evaluate `log(c * (x + r0)**(-beta) * exp(-x / kappa))`. | [Law 1](../learn/compare-to-mobility-laws.md#law-1-jump-lengths-follow-a-truncated-power-law) |

**Other laws**

| API | Description | Example |
| --- | --- | --- |
| [`fit_visitation_law`](#fastmob.measures.fitting.fit_visitation_law) | Fit the universal visitation law from staypoints. | [Law 2](../learn/compare-to-mobility-laws.md#law-2-the-universal-visitation-law) |
| [`VisitationLawFit`](#fastmob.measures.fitting.VisitationLawFit) | Fitted universal visitation law and the data used to estimate it. |  |
| [`fit_daily_location_lognormal`](#fastmob.measures.fitting.fit_daily_location_lognormal) | Fit a lognormal distribution to per-user daily distinct-location counts. | [Law 3](../learn/compare-to-mobility-laws.md#law-3-distinct-places-per-day-are-lognormal) |
<!-- api-table:end -->

## API

::: fastmob.measures.fitting.distance_distribution.fit_distance_distribution
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How sure is the fit? Range, uncertainty and alternatives](../learn/compare-to-mobility-laws.md#how-sure-is-the-fit-range-uncertainty-and-alternatives)

::: fastmob.measures.fitting.distance_distribution.compare_distance_distributions
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How sure is the fit? Range, uncertainty and alternatives](../learn/compare-to-mobility-laws.md#how-sure-is-the-fit-range-uncertainty-and-alternatives)

::: fastmob.measures.fitting.distance_distribution.DistributionFit
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How sure is the fit? Range, uncertainty and alternatives](../learn/compare-to-mobility-laws.md#how-sure-is-the-fit-range-uncertainty-and-alternatives)

::: fastmob.measures.fitting.VisitationLawFit
    options:
      show_source: false

::: fastmob.measures.fitting.fit_daily_location_lognormal
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Law 3](../learn/compare-to-mobility-laws.md#law-3-distinct-places-per-day-are-lognormal)

::: fastmob.measures.fitting.fit_values_to_truncated_powerlaw
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Law 1](../learn/compare-to-mobility-laws.md#law-1-jump-lengths-follow-a-truncated-power-law)

::: fastmob.measures.fitting.fit_visitation_law
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Law 2](../learn/compare-to-mobility-laws.md#law-2-the-universal-visitation-law)

::: fastmob.measures.fitting.log_truncated_powerlaw
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Law 1](../learn/compare-to-mobility-laws.md#law-1-jump-lengths-follow-a-truncated-power-law)

