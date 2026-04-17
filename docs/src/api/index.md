# API Reference

skmob2 exposes 8 public functions organised across 5 measure modules.

All functions accept any Narwhals-compatible eager DataFrame (pandas, polars, …) and return a result in the same backend as the input.

| Function | Module | Description |
|---|---|---|
| `jump_lengths` | `skmob2.measures.jump_lengths` | Haversine distances between consecutive GPS fixes per user |
| `radius_of_gyration` | `skmob2.measures.radius_of_gyration` | Spread of a user's movements around their center of mass |
| `od_matrix` | `skmob2.measures.od` | Origin-destination trip counts |
| `od_metrics_per_area` | `skmob2.measures.od` | Per-area flow metrics derived from an OD matrix |
| `log_truncated_powerlaw` | `skmob2.measures.mobility_laws` | Log of the Gonzalez et al. 2008 truncated power-law |
| `fit_values_to_truncated_powerlaw` | `skmob2.measures.mobility_laws` | Fit a 1-D array to the truncated power-law model |
| `activity_transition_matrix` | `skmob2.measures.activity` | Activity-type transition probabilities across users |
| `intermittance_and_degree_of_return` | `skmob2.measures.individual` | Intermittancy and degree-of-return per user |

See [Measures](measures.md) for full parameter and return-type documentation.
