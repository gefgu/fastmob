# Privacy Risk

Rust-backed functions that assess re-identification risk from trajectory data.

Privacy location comparisons use H3 cells at resolution 12 by default, which
groups small GPS jitter into one representative location. Every function
accepts `h3_resolution` from 0 through 15. Forced-instance output uses the
H3 cell center.

::: fastmob.privacy.location_risk

::: fastmob.privacy.location_sequence_risk

::: fastmob.privacy.location_time_risk

::: fastmob.privacy.unique_location_risk

::: fastmob.privacy.location_frequency_risk

::: fastmob.privacy.location_probability_risk

::: fastmob.privacy.location_proportion_risk

::: fastmob.privacy.home_work_risk
