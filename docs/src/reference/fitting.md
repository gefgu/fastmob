# Fitting

Fastkit-Mobility includes fitting utilities for common empirical mobility laws, including
truncated power laws, visitation laws, and daily-location distributions. Use
these functions after preparing the corresponding trajectory-derived data.

For an end-to-end example, see the [fit and evaluate guide](../learn/fit-and-evaluate.md).

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`VisitationLawFit`](#fastmob.measures.fitting.VisitationLawFit) | Fitted universal visitation law and the data used to estimate it. |  |
| [`fit_daily_location_lognormal`](#fastmob.measures.fitting.fit_daily_location_lognormal) | Fit a lognormal distribution to per-user daily distinct-location counts. | [Law 3](../learn/compare-to-mobility-laws.md#law-3-distinct-places-per-day-are-lognormal) |
| [`fit_values_to_truncated_powerlaw`](#fastmob.measures.fitting.fit_values_to_truncated_powerlaw) | Fit a truncated power law to positive values using a log histogram. | [Law 1](../learn/compare-to-mobility-laws.md#law-1-jump-lengths-follow-a-truncated-power-law) |
| [`fit_visitation_law`](#fastmob.measures.fitting.fit_visitation_law) | Fit the universal visitation law from staypoints. | [Law 2](../learn/compare-to-mobility-laws.md#law-2-the-universal-visitation-law) |
| [`log_truncated_powerlaw`](#fastmob.measures.fitting.log_truncated_powerlaw) | Evaluate `log(c * (x + r0)**(-beta) * exp(-x / kappa))`. | [Law 1](../learn/compare-to-mobility-laws.md#law-1-jump-lengths-follow-a-truncated-power-law) |
<!-- api-table:end -->

## API

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

