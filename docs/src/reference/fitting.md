# Fitting

| API | Description |
| --- | --- |
| [`log_truncated_powerlaw`](#fastmob.measures.fitting.mobility_laws.log_truncated_powerlaw) | Log of a truncated power-law (Gonzalez et al. 2008). |
| [`fit_values_to_truncated_powerlaw`](#fastmob.measures.fitting.mobility_laws.fit_values_to_truncated_powerlaw) | Fit a truncated power-law to a 1-D array of positive values. |
| [`compute_visitation_law_data`](#fastmob.measures.fitting.mobility_laws.compute_visitation_law_data) | Compute per-user, per-location inputs for the universal visitation law. |
| [`bin_visitation_law_data`](#fastmob.measures.fitting.mobility_laws.bin_visitation_law_data) | Aggregate visitation-law rows into binned `(rf, rho)` values. |
| [`fit_visitation_law`](#fastmob.measures.fitting.mobility_laws.fit_visitation_law) | Fit the universal visitation law to binned `(rf, rho)` values. |
| [`visitation_law_curve`](#fastmob.measures.fitting.mobility_laws.visitation_law_curve) | Build a smooth curve for `rho(r, f) = mu * (r*f)^(-eta)`. |
| [`daily_location_lognormal_fit`](#fastmob.measures.fitting.mobility_laws.daily_location_lognormal_fit) | Fit a lognormal distribution to daily distinct-location counts. |

::: fastmob.measures.fitting.mobility_laws.log_truncated_powerlaw
    options:
      show_source: false

---

::: fastmob.measures.fitting.mobility_laws.fit_values_to_truncated_powerlaw
    options:
      show_source: false

---

::: fastmob.measures.fitting.mobility_laws.compute_visitation_law_data
    options:
      show_source: false

---

::: fastmob.measures.fitting.mobility_laws.bin_visitation_law_data
    options:
      show_source: false

---

::: fastmob.measures.fitting.mobility_laws.fit_visitation_law
    options:
      show_source: false

---

::: fastmob.measures.fitting.mobility_laws.visitation_law_curve
    options:
      show_source: false

---

::: fastmob.measures.fitting.mobility_laws.daily_location_lognormal_fit
    options:
      show_source: false
