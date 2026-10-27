---
icon: lucide/user-round
---

# Individual measures

Individual measures describe each person's movement, place use, regularity,
and inferred behavior. Most accept a trajectory-like dataframe and return one
row, scalar, or variable-length result per user. Pass explicit column names
when auto-detection does not match your schema.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`activity_transition_matrix`](#fastmob.measures.individual.activity_transition_matrix) | Compute the activity transition matrix for a visits DataFrame. |  |
| [`compute_profiles`](#fastmob.measures.individual.compute_profiles) | Compute per-user mobility-profile metrics and assign profile labels. |  |
| [`daily_activity_distribution`](#fastmob.measures.individual.daily_activity_distribution) | Compute a daily activity distribution matrix over fixed time bins. |  |
| [`daily_motifs`](#fastmob.measures.individual.daily_motifs) | Compute one home-anchored mobility motif per user and day. |  |
| [`distance_straight_line`](#fastmob.measures.individual.distance_straight_line.distance_straight_line) | Return the total trajectory length (km) for each user. |  |
| [`diversity`](#fastmob.measures.individual.diversity.diversity) | Compute trajectory diversity per user using suffix-array entropy. |  |
| [`exploration_profiling`](#fastmob.measures.individual.exploration_profiling) | Compute intermittency and degree of return, then cluster users into mobility profiles. |  |
| [`frequency_rank`](#fastmob.measures.individual.frequency_rank.frequency_rank) | Return the frequency rank of each distinct location for every user. | [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks) |
| [`home_location`](#fastmob.measures.individual.home_location.home_location) | Return the most-visited nighttime location for each user. | [Home and work](../../learn/measure-individual-mobility.md#home-and-work) |
| [`individual_mobility_network`](#fastmob.measures.individual.individual_mobility_network.individual_mobility_network) | Return the individual mobility network as a directed edge-list DataFrame. | [A user's mobility network](../../learn/measure-individual-mobility.md#a-users-mobility-network) |
| [`intermittance_and_degree_of_return`](#fastmob.measures.individual.intermittance_and_degree_of_return) | Compute intermittancy and degree of return per user using vectorized operations. |  |
| [`jump_lengths`](#fastmob.measures.individual.jump_lengths.jump_lengths) | Compute jump lengths (km) for each user in the trajectory. | [Measure jump lengths](../../learn/analyze-a-simple-trajectory.md#measure-jump-lengths) |
| [`jump_lengths_road`](#fastmob.measures.individual.jump_lengths_road) | Road-network jump lengths (km): distance between consecutive stops for the same user, sorted by datetime |  |
| [`k_radius_of_gyration`](#fastmob.measures.individual.k_radius_of_gyration.k_radius_of_gyration) | Compute the k-radius of gyration (km) for each user in the trajectory. | [How far and how widely](../../learn/measure-individual-mobility.md#how-far-and-how-widely-radius-of-gyration-maximum-distance-jump-lengths) |
| [`location_frequency`](#fastmob.measures.individual.location_frequency.location_frequency) | Return visit frequency for each distinct location per user. | [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks) |
| [`max_distance_from_home`](#fastmob.measures.individual.max_distance_from_home.max_distance_from_home) | Return the maximum Haversine distance (km) from each user's home location. | [Home and work](../../learn/measure-individual-mobility.md#home-and-work) |
| [`maximum_distance`](#fastmob.measures.individual.maximum_distance.maximum_distance) | Return the maximum distance (km) covered in a single movement for each user. | [Find the longest single jump](../../learn/analyze-a-simple-trajectory.md#find-the-longest-single-jump) |
| [`motif_distribution`](#fastmob.measures.individual.motif_distribution) | Aggregate daily motif IDs into count and percentage columns. |  |
| [`number_of_locations`](#fastmob.measures.individual.number_of_locations.number_of_locations) | Return the number of distinct locations visited by each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |
| [`number_of_visits`](#fastmob.measures.individual.number_of_visits.number_of_visits) | Return the total number of trajectory points (visits) for each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |
| [`radius_of_gyration`](#fastmob.measures.individual.radius_of_gyration.radius_of_gyration) | Compute the radius of gyration (km) for each user in the trajectory. | [Measure the radius of gyration](../../learn/analyze-a-simple-trajectory.md#measure-the-radius-of-gyration) |
| [`radius_of_gyration_road`](#fastmob.measures.individual.radius_of_gyration_road) | Road-network radius of gyration (km) per user: RMS network distance from each of a user's stops to the arithmetic-mean centroid of their stops |  |
| [`random_entropy`](#fastmob.measures.individual.random_entropy.random_entropy) | Return the random entropy of mobility for each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |
| [`real_entropy`](#fastmob.measures.individual.real_entropy.real_entropy) | Return the real (true) entropy of mobility for each user. |  |
| [`recency_rank`](#fastmob.measures.individual.recency_rank.recency_rank) | Return the recency rank of each distinct location for every user. | [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks) |
| [`regularity`](#fastmob.measures.individual.regularity.regularity) | Compute regularity per user. |  |
| [`trajectory_entropy`](#fastmob.measures.individual.trajectory_entropy) | Compute Kontoyiannis entropy of mobility trajectories per user. |  |
| [`trajectory_predictability`](#fastmob.measures.individual.trajectory_predictability) | Compute per-user maximum predictability via Fano's inequality. |  |
| [`uncorrelated_entropy`](#fastmob.measures.individual.uncorrelated_entropy.uncorrelated_entropy) | Return the uncorrelated entropy of mobility for each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |
| [`visit_purpose_distribution`](#fastmob.measures.individual.visit_purpose_distribution) | Compute visit-purpose counts and percentages. |  |
| [`waiting_times`](#fastmob.measures.individual.waiting_times.waiting_times) | Return the waiting times (seconds) between consecutive GPS fixes for each user. | [Measure waiting times](../../learn/analyze-a-simple-trajectory.md#measure-waiting-times) |
| [`work_location`](#fastmob.measures.individual.work_location.work_location) | Return each user's most-visited weekday daytime location. | [Home and work](../../learn/measure-individual-mobility.md#home-and-work) |
<!-- api-table:end -->

## Which measure answers which question

| Question | Representative APIs |
| --- | --- |
| How far and how often does someone move? | `jump_lengths`, `radius_of_gyration`, `waiting_times` |
| Which places matter? | `home_location`, `work_location`, `location_frequency` |
| Is their movement regular or predictable? | `regularity`, entropy, predictability, motifs |
| Does route distance matter? | `jump_lengths_road`, `radius_of_gyration_road` |

For a task-focused introduction, see
[Measuring people and populations](../../learn/individual-and-collective-measures.md).

## API

::: fastmob.measures.individual.activity_transition_matrix
    options:
      show_source: false

::: fastmob.measures.individual.compute_profiles
    options:
      show_source: false

::: fastmob.measures.individual.daily_activity_distribution
    options:
      show_source: false

::: fastmob.measures.individual.daily_motifs
    options:
      show_source: false

::: fastmob.measures.individual.distance_straight_line
    options:
      show_source: false

::: fastmob.measures.individual.diversity
    options:
      show_source: false

::: fastmob.measures.individual.exploration_profiling
    options:
      show_source: false

::: fastmob.measures.individual.frequency_rank
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks)

::: fastmob.measures.individual.home_location
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Home and work](../../learn/measure-individual-mobility.md#home-and-work)

::: fastmob.measures.individual.individual_mobility_network
    options:
      show_source: false

<!-- example-link -->**Worked example:** [A user's mobility network](../../learn/measure-individual-mobility.md#a-users-mobility-network)

::: fastmob.measures.individual.intermittance_and_degree_of_return
    options:
      show_source: false

::: fastmob.measures.individual.jump_lengths
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Measure jump lengths](../../learn/analyze-a-simple-trajectory.md#measure-jump-lengths)

::: fastmob.measures.individual.jump_lengths_road
    options:
      show_source: false

::: fastmob.measures.individual.k_radius_of_gyration
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How far and how widely](../../learn/measure-individual-mobility.md#how-far-and-how-widely-radius-of-gyration-maximum-distance-jump-lengths)

::: fastmob.measures.individual.location_frequency
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks)

::: fastmob.measures.individual.max_distance_from_home
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Home and work](../../learn/measure-individual-mobility.md#home-and-work)

::: fastmob.measures.individual.maximum_distance
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Find the longest single jump](../../learn/analyze-a-simple-trajectory.md#find-the-longest-single-jump)

::: fastmob.measures.individual.motif_distribution
    options:
      show_source: false

::: fastmob.measures.individual.number_of_locations
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

::: fastmob.measures.individual.number_of_visits
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

::: fastmob.measures.individual.radius_of_gyration
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Measure the radius of gyration](../../learn/analyze-a-simple-trajectory.md#measure-the-radius-of-gyration)

::: fastmob.measures.individual.radius_of_gyration_road
    options:
      show_source: false

::: fastmob.measures.individual.random_entropy
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

::: fastmob.measures.individual.real_entropy
    options:
      show_source: false

::: fastmob.measures.individual.recency_rank
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks)

::: fastmob.measures.individual.regularity
    options:
      show_source: false

::: fastmob.measures.individual.trajectory_entropy
    options:
      show_source: false

::: fastmob.measures.individual.trajectory_predictability
    options:
      show_source: false

::: fastmob.measures.individual.uncorrelated_entropy
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

::: fastmob.measures.individual.visit_purpose_distribution
    options:
      show_source: false

::: fastmob.measures.individual.waiting_times
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Measure waiting times](../../learn/analyze-a-simple-trajectory.md#measure-waiting-times)

::: fastmob.measures.individual.work_location
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Home and work](../../learn/measure-individual-mobility.md#home-and-work)

