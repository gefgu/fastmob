# Privacy Risk

Rust-backed functions that assess re-identification risk from trajectory data.

Privacy location comparisons use H3 cells at resolution 12 by default, which
groups small GPS jitter into one representative location. Every function
accepts `h3_resolution` from 0 through 15. Forced-instance output uses the
H3 cell center.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`location_risk`](#fastmob.privacy.location_risk) | Assess re-identification risk from visited locations. | [Risk from places the attacker knows](../learn/privacy-risk.md#risk-from-places-the-attacker-knows) |
| [`location_sequence_risk`](#fastmob.privacy.location_sequence_risk) | Assess re-identification risk from ordered location sequences. | [Other attacker models](../learn/privacy-risk.md#other-attacker-models) |
| [`location_time_risk`](#fastmob.privacy.location_time_risk) | Assess re-identification risk from locations at a time precision. | [Other attacker models](../learn/privacy-risk.md#other-attacker-models) |
| [`unique_location_risk`](#fastmob.privacy.unique_location_risk) | Assess re-identification risk from each user's distinct locations. | [Other attacker models](../learn/privacy-risk.md#other-attacker-models) |
| [`location_frequency_risk`](#fastmob.privacy.location_frequency_risk) | Assess re-identification risk from locations and visit frequencies. | [Other attacker models](../learn/privacy-risk.md#other-attacker-models) |
| [`location_probability_risk`](#fastmob.privacy.location_probability_risk) | Assess re-identification risk from locations and visit probabilities. | [Other attacker models](../learn/privacy-risk.md#other-attacker-models) |
| [`location_proportion_risk`](#fastmob.privacy.location_proportion_risk) | Assess re-identification risk from locations and relative visit frequencies. | [Other attacker models](../learn/privacy-risk.md#other-attacker-models) |
| [`home_work_risk`](#fastmob.privacy.home_work_risk) | Assess re-identification risk from the two most frequent locations. | [Other attacker models](../learn/privacy-risk.md#other-attacker-models) |
<!-- api-table:end -->

::: fastmob.privacy.location_risk

::: fastmob.privacy.location_sequence_risk

::: fastmob.privacy.location_time_risk

::: fastmob.privacy.unique_location_risk

::: fastmob.privacy.location_frequency_risk

::: fastmob.privacy.location_probability_risk

::: fastmob.privacy.location_proportion_risk

::: fastmob.privacy.home_work_risk
