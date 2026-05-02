# API Reference

skmob2 exposes 56 public functions from the package root.

The public API is defined by `skmob2.__all__`. Most DataFrame functions accept any Narwhals-compatible eager DataFrame, such as pandas or polars, and return a result in the same backend as the input.

## Preprocessing

| Function | Module | Description |
|---|---|---|
| `filter` | `skmob2.preprocessing.filter` | Remove trajectory points using speed and loop filters. |
| `compress` | `skmob2.preprocessing.compress` | Compress consecutive trajectory points within a spatial radius. |
| `stay_locations` | `skmob2.preprocessing.stay_locations` | Detect stay locations from raw trajectory points. |
| `cluster` | `skmob2.preprocessing.cluster` | Cluster trajectory coordinates into location identifiers. |
| `cdr_to_visitation_df` | `skmob2.preprocessing.cdr` | Convert CDR-like records into visitation rows. |
| `cdr_to_trips_df` | `skmob2.preprocessing.cdr` | Convert visitation rows into trip rows. |

See [Preprocessing](preprocessing.md) for signatures and parameters.

## Spatial Measures

| Function | Module | Description |
|---|---|---|
| `jump_lengths` | `skmob2.measures.spatial.jump_lengths` | Haversine distances between consecutive fixes. |
| `radius_of_gyration` | `skmob2.measures.spatial.radius_of_gyration` | User movement spread around the center of mass. |
| `k_radius_of_gyration` | `skmob2.measures.spatial.k_radius_of_gyration` | Radius of gyration over the top-k most frequent locations. |
| `number_of_visits` | `skmob2.measures.spatial.number_of_visits` | Number of visits per user. |
| `number_of_locations` | `skmob2.measures.spatial.number_of_locations` | Number of distinct locations per user. |
| `maximum_distance` | `skmob2.measures.spatial.maximum_distance` | Maximum pairwise distance among a user's points. |
| `distance_straight_line` | `skmob2.measures.spatial.distance_straight_line` | Straight-line distance from first to last point. |
| `waiting_times` | `skmob2.measures.spatial.waiting_times` | Time gaps between consecutive points. |
| `home_location` | `skmob2.measures.spatial.home_location` | Inferred home location from nighttime observations. |
| `max_distance_from_home` | `skmob2.measures.spatial.max_distance_from_home` | Maximum distance from inferred home. |

See [Spatial Measures](spatial.md) for signatures and parameters.

## Flow Measures

| Function | Module | Description |
|---|---|---|
| `od_matrix` | `skmob2.measures.flows.od` | Origin-destination trip counts. |
| `od_metrics_per_area` | `skmob2.measures.flows.od` | Per-area metrics from an OD matrix. |
| `random_location_entropy` | `skmob2.measures.flows.random_location_entropy` | Random location entropy. |
| `uncorrelated_location_entropy` | `skmob2.measures.flows.uncorrelated_location_entropy` | Uncorrelated location entropy. |
| `visits_per_location` | `skmob2.measures.flows.visits_per_location` | Visit counts by location. |
| `homes_per_location` | `skmob2.measures.flows.homes_per_location` | Inferred home counts by location. |
| `visits_per_time_unit` | `skmob2.measures.flows.visits_per_time_unit` | Visit counts by time interval. |
| `mean_square_displacement` | `skmob2.measures.flows.mean_square_displacement` | Mean square displacement over a time offset. |

See [Flow Measures](flows.md) for signatures and parameters.

## Visit Measures

| Function | Module | Description |
|---|---|---|
| `activity_transition_matrix` | `skmob2.measures.visits.activity` | Activity transition probabilities. |
| `intermittance_and_degree_of_return` | `skmob2.measures.visits.mobility_profiling` | Intermittance and degree of return per user. |
| `exploration_profiling` | `skmob2.measures.visits.mobility_profiling` | Exploration profiling features per user. |
| `regularity` | `skmob2.measures.visits.regularity` | Regularity measure per user. |
| `fast_diversity` | `skmob2.measures.visits.fast_diversity` | Low-level suffix-array diversity for a sequence. |
| `diversity` | `skmob2.measures.visits.diversity` | Trajectory diversity per user. |
| `trajectory_entropy` | `skmob2.measures.visits.entropy` | Entropy of visit trajectories. |
| `trajectory_predictability` | `skmob2.measures.visits.entropy` | Predictability derived from trajectory entropy. |
| `random_entropy` | `skmob2.measures.visits.random_entropy` | Random entropy of visited locations. |
| `uncorrelated_entropy` | `skmob2.measures.visits.uncorrelated_entropy` | Uncorrelated entropy of visited locations. |
| `recency_rank` | `skmob2.measures.visits.recency_rank` | Location ranks by recency. |
| `frequency_rank` | `skmob2.measures.visits.frequency_rank` | Location ranks by visit frequency. |
| `individual_mobility_network` | `skmob2.measures.visits.individual_mobility_network` | Per-user mobility network edges. |
| `location_frequency` | `skmob2.measures.visits.location_frequency` | Location visit frequencies. |
| `real_entropy` | `skmob2.measures.visits.real_entropy` | Real entropy of trajectories. |
| `discover_daily_motifs_from_agents` | `skmob2.measures.visits.motifs` | Daily mobility motifs and motif distribution. |
| `mean_area_volume` | `skmob2.measures.visits.mean_area_volume` | Mean area volume over visits. |

See [Visit Measures](visits.md) for signatures and parameters.

## Privacy Attacks

| Class | Module | Description |
|---|---|---|
| `LocationAttack` | `skmob2.privacy` | Privacy risk from known visited locations. |
| `LocationSequenceAttack` | `skmob2.privacy` | Privacy risk from known ordered location sequences. |
| `LocationTimeAttack` | `skmob2.privacy` | Privacy risk from known locations at a selected time precision. |
| `UniqueLocationAttack` | `skmob2.privacy` | Privacy risk from known unique locations. |
| `LocationFrequencyAttack` | `skmob2.privacy` | Privacy risk from known locations and visit frequencies. |
| `LocationProbabilityAttack` | `skmob2.privacy` | Privacy risk from known locations and visit probabilities. |
| `LocationProportionAttack` | `skmob2.privacy` | Privacy risk from relative visit-frequency proportions. |
| `HomeWorkAttack` | `skmob2.privacy` | Privacy risk from the top-two frequent locations. |

See [Privacy Attacks](privacy.md) for signatures and parameters.

## Fitting

| Function | Module | Description |
|---|---|---|
| `log_truncated_powerlaw` | `skmob2.measures.fitting.mobility_laws` | Log of the truncated power-law model. |
| `fit_values_to_truncated_powerlaw` | `skmob2.measures.fitting.mobility_laws` | Fit positive values to the truncated power-law model. |

See [Fitting](fitting.md) for signatures and parameters.

## Evaluation

| Function | Module | Description |
|---|---|---|
| `common_part_of_commuters` | `skmob2.measures.evaluation` | Common part of commuters between two arrays. |
| `common_part_of_links` | `skmob2.measures.evaluation` | Common part of links between two arrays. |
| `common_part_of_commuters_distance` | `skmob2.measures.evaluation` | Distance derived from common part of commuters. |
| `r_squared` | `skmob2.measures.evaluation` | Coefficient of determination. |
| `mse` | `skmob2.measures.evaluation` | Mean squared error. |
| `rmse` | `skmob2.measures.evaluation` | Root mean squared error. |
| `nrmse` | `skmob2.measures.evaluation` | Normalized root mean squared error. |
| `information_gain` | `skmob2.measures.evaluation` | Information gain between true and predicted values. |
| `kullback_leibler_divergence` | `skmob2.measures.evaluation` | KL divergence between true and predicted distributions. |
| `pearson_correlation` | `skmob2.measures.evaluation` | Pearson correlation and p-value. |
| `spearman_correlation` | `skmob2.measures.evaluation` | Spearman correlation and p-value. |
| `max_error` | `skmob2.measures.evaluation` | Maximum absolute error. |

See [Evaluation](evaluation.md) for signatures and parameters.

## STVD-EMD

| Function | Module | Description |
|---|---|---|
| `stvd_emd` | `skmob2.measures.stvd_emd` | Spatiotemporal variation distance using sliced EMD. |

See [STVD-EMD](stvd-emd.md) for signatures and parameters.
