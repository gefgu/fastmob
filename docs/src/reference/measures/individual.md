# Individual Measures

| API | Description |
| --- | --- |
| [`distance_straight_line`](#skmob2.measures.individual.distance_straight_line) | Return the total trajectory length (km) for each user. |
| [`frequency_rank`](#skmob2.measures.individual.frequency_rank) | Return the frequency rank of each distinct location for every user. |
| [`home_location`](#skmob2.measures.individual.home_location) | Return the most-visited nighttime location for each user. |
| [`individual_mobility_network`](#skmob2.measures.individual.individual_mobility_network) | Return the individual mobility network as a directed edge-list DataFrame. |
| [`jump_lengths`](#skmob2.measures.individual.jump_lengths) | Compute jump lengths (km) for each user in the trajectory. |
| [`k_radius_of_gyration`](#skmob2.measures.individual.k_radius_of_gyration) | Compute the k-radius of gyration (km) for each user in the trajectory. |
| [`location_frequency`](#skmob2.measures.individual.location_frequency) | Return visit frequency for each distinct location per user. |
| [`max_distance_from_home`](#skmob2.measures.individual.max_distance_from_home) | Return the maximum Haversine distance (km) from each user's home location. |
| [`maximum_distance`](#skmob2.measures.individual.maximum_distance) | Return the maximum distance (km) covered in a single movement for each user. |
| [`number_of_locations`](#skmob2.measures.individual.number_of_locations) | Return the number of distinct locations visited by each user. |
| [`number_of_visits`](#skmob2.measures.individual.number_of_visits) | Return the total number of trajectory points (visits) for each user. |
| [`radius_of_gyration`](#skmob2.measures.individual.radius_of_gyration) | Compute the radius of gyration (km) for each user in the trajectory. |
| [`random_entropy`](#skmob2.measures.individual.random_entropy) | Return the random entropy of mobility for each user. |
| [`real_entropy`](#skmob2.measures.individual.real_entropy) | Return the real (true) entropy of mobility for each user. |
| [`recency_rank`](#skmob2.measures.individual.recency_rank) | Return the recency rank of each distinct location for every user. |
| [`uncorrelated_entropy`](#skmob2.measures.individual.uncorrelated_entropy) | Return the uncorrelated entropy of mobility for each user. |
| [`waiting_times`](#skmob2.measures.individual.waiting_times) | Return the waiting times (seconds) between consecutive GPS fixes for each user. |
| [`activity_transition_matrix`](#skmob2.measures.individual.activity_transition_matrix) | Compute the activity transition matrix for a visits DataFrame. |
| [`daily_activity_distribution`](#skmob2.measures.individual.daily_activity_distribution) | Compute a daily activity distribution matrix over fixed time bins. |
| [`visit_purpose_distribution`](#skmob2.measures.individual.visit_purpose_distribution) | Compute visit-purpose counts and percentages. |
| [`regularity`](#skmob2.measures.individual.regularity) | Compute regularity per user. |
| [`diversity`](#skmob2.measures.individual.diversity) | Compute trajectory diversity per user using suffix-array entropy. |
| [`trajectory_entropy`](#skmob2.measures.individual.trajectory_entropy) | Compute Kontoyiannis entropy of mobility trajectories per user. |
| [`trajectory_predictability`](#skmob2.measures.individual.trajectory_predictability) | Compute per-user maximum predictability via Fano's inequality. |
| [`discover_daily_motifs_from_agents`](#skmob2.measures.individual.discover_daily_motifs_from_agents) | Discover daily mobility motifs for all agents in a dataset. |
<!-- | [`intermittance_and_degree_of_return`](#skmob2.measures.individual.intermittance_and_degree_of_return) | Compute intermittancy and degree of return per user using vectorized operations. | -->
<!-- | [`exploration_profiling`](#skmob2.measures.individual.exploration_profiling) | Compute intermittency and degree of return, then cluster users into mobility profiles. | -->
<!-- | [`fast_diversity`](#skmob2.measures.individual.fast_diversity) | Compute the diversity of a sequence using suffix arrays. | -->
<!-- | [`mean_area_volume`](#skmob2.measures.individual.mean_area_volume) | Mean user volume per area and 10-minute time bin, averaged across days of the week. | -->

::: skmob2.measures.individual.distance_straight_line
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.frequency_rank
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.home_location
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.individual_mobility_network
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.jump_lengths
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.k_radius_of_gyration
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.location_frequency
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.max_distance_from_home
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.maximum_distance
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.number_of_locations
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.number_of_visits
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.radius_of_gyration
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.random_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.real_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.recency_rank
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.uncorrelated_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.waiting_times
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.activity_transition_matrix
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.daily_activity_distribution
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.visit_purpose_distribution
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.intermittance_and_degree_of_return
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.exploration_profiling
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.regularity
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.fast_diversity
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.diversity
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.trajectory_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.trajectory_predictability
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: skmob2.measures.individual.discover_daily_motifs_from_agents
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
<!-- ---

::: skmob2.measures.individual.mean_area_volume
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false -->