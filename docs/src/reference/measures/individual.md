# Individual Measures

| API | Description |
| --- | --- |
| [`distance_straight_line`](#fastmob.measures.individual.distance_straight_line.distance_straight_line) | Return the total trajectory length (km) for each user. |
| [`frequency_rank`](#fastmob.measures.individual.frequency_rank.frequency_rank) | Return the frequency rank of each distinct location for every user. |
| [`home_location`](#fastmob.measures.individual.home_location.home_location) | Return the most-visited nighttime location for each user. |
| [`individual_mobility_network`](#fastmob.measures.individual.individual_mobility_network.individual_mobility_network) | Return the individual mobility network as a directed edge-list DataFrame. |
| [`jump_lengths`](#fastmob.measures.individual.jump_lengths.jump_lengths) | Compute jump lengths (km) for each user in the trajectory. |
| [`k_radius_of_gyration`](#fastmob.measures.individual.k_radius_of_gyration.k_radius_of_gyration) | Compute the k-radius of gyration (km) for each user in the trajectory. |
| [`location_frequency`](#fastmob.measures.individual.location_frequency.location_frequency) | Return visit frequency for each distinct location per user. |
| [`max_distance_from_home`](#fastmob.measures.individual.max_distance_from_home.max_distance_from_home) | Return the maximum Haversine distance (km) from each user's home location. |
| [`maximum_distance`](#fastmob.measures.individual.maximum_distance.maximum_distance) | Return the maximum distance (km) covered in a single movement for each user. |
| [`number_of_locations`](#fastmob.measures.individual.number_of_locations.number_of_locations) | Return the number of distinct locations visited by each user. |
| [`number_of_visits`](#fastmob.measures.individual.number_of_visits.number_of_visits) | Return the total number of trajectory points (visits) for each user. |
| [`radius_of_gyration`](#fastmob.measures.individual.radius_of_gyration.radius_of_gyration) | Compute the radius of gyration (km) for each user in the trajectory. |
| [`random_entropy`](#fastmob.measures.individual.random_entropy.random_entropy) | Return the random entropy of mobility for each user. |
| [`real_entropy`](#fastmob.measures.individual.real_entropy.real_entropy) | Return the real (true) entropy of mobility for each user. |
| [`recency_rank`](#fastmob.measures.individual.recency_rank.recency_rank) | Return the recency rank of each distinct location for every user. |
| [`uncorrelated_entropy`](#fastmob.measures.individual.uncorrelated_entropy.uncorrelated_entropy) | Return the uncorrelated entropy of mobility for each user. |
| [`waiting_times`](#fastmob.measures.individual.waiting_times.waiting_times) | Return the waiting times (seconds) between consecutive GPS fixes for each user. |
| [`activity_transition_matrix`](#fastmob.measures.individual.activity_transition_matrix) | Compute the activity transition matrix for a visits DataFrame. |
| [`daily_activity_distribution`](#fastmob.measures.individual.daily_activity_distribution) | Compute a daily activity distribution matrix over fixed time bins. |
| [`visit_purpose_distribution`](#fastmob.measures.individual.visit_purpose_distribution) | Compute visit-purpose counts and percentages. |
| [`regularity`](#fastmob.measures.individual.regularity.regularity) | Compute regularity per user. |
| [`diversity`](#fastmob.measures.individual.diversity.diversity) | Compute trajectory diversity per user using suffix-array entropy. |
| [`trajectory_entropy`](#fastmob.measures.individual.trajectory_entropy) | Compute Kontoyiannis entropy of mobility trajectories per user. |
| [`trajectory_predictability`](#fastmob.measures.individual.trajectory_predictability) | Compute per-user maximum predictability via Fano's inequality. |
| [`discover_daily_motifs_from_agents`](#fastmob.measures.individual.discover_daily_motifs_from_agents) | Discover daily mobility motifs for all agents in a dataset. |
<!-- | [`intermittance_and_degree_of_return`](#fastmob.measures.individual.intermittance_and_degree_of_return) | Compute intermittancy and degree of return per user using vectorized operations. | -->
<!-- | [`exploration_profiling`](#fastmob.measures.individual.exploration_profiling) | Compute intermittency and degree of return, then cluster users into mobility profiles. | -->
<!-- | [`fast_diversity`](#fastmob.measures.individual.fast_diversity) | Compute the diversity of a sequence using suffix arrays. | -->
<!-- | [`mean_area_volume`](#fastmob.measures.individual.mean_area_volume) | Mean user volume per area and 10-minute time bin, averaged across days of the week. | -->

::: fastmob.measures.individual.distance_straight_line
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.frequency_rank
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.home_location
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.individual_mobility_network
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.jump_lengths
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.k_radius_of_gyration
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.location_frequency
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.max_distance_from_home
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.maximum_distance
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.number_of_locations
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.number_of_visits
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.radius_of_gyration
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.random_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.real_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.recency_rank
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.uncorrelated_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.waiting_times
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

<a id="fastmob.measures.individual.activity_transition_matrix"></a>

::: fastmob.measures.individual.activity_transition_matrix
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

<a id="fastmob.measures.individual.daily_activity_distribution"></a>

::: fastmob.measures.individual.daily_activity_distribution
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

<a id="fastmob.measures.individual.visit_purpose_distribution"></a>

::: fastmob.measures.individual.visit_purpose_distribution
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.intermittance_and_degree_of_return
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.exploration_profiling
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.regularity
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.fast_diversity
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

::: fastmob.measures.individual.diversity
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

<a id="fastmob.measures.individual.trajectory_entropy"></a>

::: fastmob.measures.individual.trajectory_entropy
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

<a id="fastmob.measures.individual.trajectory_predictability"></a>

::: fastmob.measures.individual.trajectory_predictability
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
---

<a id="fastmob.measures.individual.discover_daily_motifs_from_agents"></a>

::: fastmob.measures.individual.discover_daily_motifs_from_agents
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false
<!-- ---

::: fastmob.measures.individual.mean_area_volume
    options:
      show_source: false
      show_root_toc_entry: false
      show_root_heading: false -->