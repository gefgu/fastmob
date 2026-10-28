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
**Distance and movement**

| API | Description | Example |
| --- | --- | --- |
| [`jump_lengths`](#fastmob.measures.individual.jump_lengths.jump_lengths) | Compute jump lengths (km) for each user in the trajectory. | [Measure jump lengths](../../learn/analyze-a-simple-trajectory.md#measure-jump-lengths) |
| [`distance_straight_line`](#fastmob.measures.individual.distance_straight_line.distance_straight_line) | Return the total trajectory length (km) for each user. | [How far did they travel in total?](../../learn/visit-based-measures.md#how-far-did-they-travel-in-total) |
| [`maximum_distance`](#fastmob.measures.individual.maximum_distance.maximum_distance) | Return the maximum distance (km) covered in a single movement for each user. | [Find the longest single jump](../../learn/analyze-a-simple-trajectory.md#find-the-longest-single-jump) |
| [`max_distance_from_home`](#fastmob.measures.individual.max_distance_from_home.max_distance_from_home) | Return the maximum Haversine distance (km) from each user's home location. | [Home and work](../../learn/measure-individual-mobility.md#home-and-work) |
| [`radius_of_gyration`](#fastmob.measures.individual.radius_of_gyration.radius_of_gyration) | Compute the radius of gyration (km) for each user in the trajectory. | [Measure the radius of gyration](../../learn/analyze-a-simple-trajectory.md#measure-the-radius-of-gyration) |
| [`k_radius_of_gyration`](#fastmob.measures.individual.k_radius_of_gyration.k_radius_of_gyration) | Compute the k-radius of gyration (km) for each user in the trajectory. | [How far and how widely](../../learn/measure-individual-mobility.md#how-far-and-how-widely-radius-of-gyration-maximum-distance-jump-lengths) |
| [`waiting_times`](#fastmob.measures.individual.waiting_times.waiting_times) | Return the waiting times (seconds) between consecutive GPS fixes for each user. | [Measure waiting times](../../learn/analyze-a-simple-trajectory.md#measure-waiting-times) |

**Places**

| API | Description | Example |
| --- | --- | --- |
| [`home_location`](#fastmob.measures.individual.home_location.home_location) | Return the most-visited nighttime location for each user. | [Home and work](../../learn/measure-individual-mobility.md#home-and-work) |
| [`work_location`](#fastmob.measures.individual.work_location.work_location) | Return each user's most visited place on weekdays during working hours. | [Home and work](../../learn/measure-individual-mobility.md#home-and-work) |
| [`location_frequency`](#fastmob.measures.individual.location_frequency.location_frequency) | Return visit frequency for each distinct location per user. | [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks) |
| [`frequency_rank`](#fastmob.measures.individual.frequency_rank.frequency_rank) | Return the frequency rank of each distinct location for every user. | [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks) |
| [`recency_rank`](#fastmob.measures.individual.recency_rank.recency_rank) | Return the recency rank of each distinct location for every user. | [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks) |
| [`number_of_locations`](#fastmob.measures.individual.number_of_locations.number_of_locations) | Return the number of distinct locations visited by each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |
| [`number_of_visits`](#fastmob.measures.individual.number_of_visits.number_of_visits) | Return the total number of trajectory points (visits) for each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |

**Regularity and predictability**

| API | Description | Example |
| --- | --- | --- |
| [`real_entropy`](#fastmob.measures.individual.real_entropy.real_entropy) | Return the real (true) entropy of mobility for each user. | [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable) |
| [`random_entropy`](#fastmob.measures.individual.random_entropy.random_entropy) | Return the random entropy of mobility for each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |
| [`uncorrelated_entropy`](#fastmob.measures.individual.uncorrelated_entropy.uncorrelated_entropy) | Return the uncorrelated entropy of mobility for each user. | [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy) |
| [`trajectory_entropy`](#fastmob.measures.individual.trajectory_entropy) | Compute Kontoyiannis entropy of mobility trajectories per user. | [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable) |
| [`trajectory_predictability`](#fastmob.measures.individual.trajectory_predictability) | Compute per-user maximum predictability via Fano's inequality. | [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable) |
| [`regularity`](#fastmob.measures.individual.regularity.regularity) | Compute regularity per user. | [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable) |
| [`diversity`](#fastmob.measures.individual.diversity.diversity) | Compute trajectory diversity per user using suffix-array entropy. | [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable) |
| [`intermittance_and_degree_of_return`](#fastmob.measures.individual.intermittance_and_degree_of_return) | Compute intermittancy and degree of return per user. | [Return and exploration](../../learn/visit-based-measures.md#return-and-exploration) |
| [`exploration_profiling`](#fastmob.measures.individual.exploration_profiling) | Compute intermittency and degree of return, then cluster users into mobility profiles. | [Return and exploration](../../learn/visit-based-measures.md#return-and-exploration) |
| [`compute_profiles`](#fastmob.measures.individual.compute_profiles) | Compute per-user mobility-profile metrics and assign profile labels. | [One table of profiles](../../learn/visit-based-measures.md#one-table-of-profiles) |

**Motifs and mobility networks**

| API | Description | Example |
| --- | --- | --- |
| [`daily_motifs`](#fastmob.measures.individual.daily_motifs) | Compute one home-anchored mobility motif per user and day. | [The same from a plain visit table](../../learn/motifs.md#the-same-from-a-plain-visit-table) |
| [`motif_distribution`](#fastmob.measures.individual.motif_distribution) | Count the days of each daily mobility motif and give their percentage. | [How common is each motif?](../../learn/motifs.md#how-common-is-each-motif) |
| [`individual_mobility_network`](#fastmob.measures.individual.individual_mobility_network.individual_mobility_network) | Return the individual mobility network as a directed edge-list DataFrame. | [A user's mobility network](../../learn/measure-individual-mobility.md#a-users-mobility-network) |

**Activities and visit purposes**

| API | Description | Example |
| --- | --- | --- |
| [`activity_transition_matrix`](#fastmob.measures.individual.activity_transition_matrix) | Compute the activity transition matrix for a visits DataFrame. | [What are people doing?](../../learn/visit-based-measures.md#what-are-people-doing) |
| [`daily_activity_distribution`](#fastmob.measures.individual.daily_activity_distribution) | Return how activity purposes are spread over the hours of the day. | [What are people doing?](../../learn/visit-based-measures.md#what-are-people-doing) |
| [`visit_purpose_distribution`](#fastmob.measures.individual.visit_purpose_distribution) | Compute visit-purpose counts and percentages. | [What are people doing?](../../learn/visit-based-measures.md#what-are-people-doing) |

**Along the road network**

| API | Description | Example |
| --- | --- | --- |
| [`jump_lengths_road`](#fastmob.measures.individual.jump_lengths_road) | Jump lengths (km) measured along the road network. | [Road distance versus straight line](../../learn/road-network-analysis.md#road-distance-versus-straight-line) |
| [`radius_of_gyration_road`](#fastmob.measures.individual.radius_of_gyration_road) | Radius of gyration (km) per user, measured along the road network. | [Road distance versus straight line](../../learn/road-network-analysis.md#road-distance-versus-straight-line) |
<!-- api-table:end -->

## Which measure answers which question

| Question | Representative APIs |
| --- | --- |
| How far and how often does someone move? | [`jump_lengths`](#fastmob.measures.individual.jump_lengths), [`radius_of_gyration`](#fastmob.measures.individual.radius_of_gyration), [`waiting_times`](#fastmob.measures.individual.waiting_times) |
| Which places matter? | [`home_location`](#fastmob.measures.individual.home_location), [`work_location`](#fastmob.measures.individual.work_location), [`location_frequency`](#fastmob.measures.individual.location_frequency) |
| Is their movement regular or predictable? | [`real_entropy`](#fastmob.measures.individual.real_entropy), [`random_entropy`](#fastmob.measures.individual.random_entropy), [`daily_motifs`](#fastmob.measures.individual.daily_motifs), [`compute_profiles`](#fastmob.measures.individual.compute_profiles) |
| Does route distance matter? | [`jump_lengths_road`](#fastmob.measures.individual.jump_lengths_road), [`radius_of_gyration_road`](#fastmob.measures.individual.radius_of_gyration_road) |

For a task-focused introduction, see
[Measuring people and populations](../../learn/individual-and-collective-measures.md).

## API

<a id="fastmob.measures.individual.jump_lengths"></a>

::: fastmob.measures.individual.jump_lengths
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Measure jump lengths](../../learn/analyze-a-simple-trajectory.md#measure-jump-lengths)

<a id="fastmob.measures.individual.distance_straight_line"></a>

::: fastmob.measures.individual.distance_straight_line
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How far did they travel in total?](../../learn/visit-based-measures.md#how-far-did-they-travel-in-total)

<a id="fastmob.measures.individual.maximum_distance"></a>

::: fastmob.measures.individual.maximum_distance
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Find the longest single jump](../../learn/analyze-a-simple-trajectory.md#find-the-longest-single-jump)

<a id="fastmob.measures.individual.max_distance_from_home"></a>

::: fastmob.measures.individual.max_distance_from_home
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Home and work](../../learn/measure-individual-mobility.md#home-and-work)

<a id="fastmob.measures.individual.radius_of_gyration"></a>

::: fastmob.measures.individual.radius_of_gyration
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Measure the radius of gyration](../../learn/analyze-a-simple-trajectory.md#measure-the-radius-of-gyration)

<a id="fastmob.measures.individual.k_radius_of_gyration"></a>

::: fastmob.measures.individual.k_radius_of_gyration
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How far and how widely](../../learn/measure-individual-mobility.md#how-far-and-how-widely-radius-of-gyration-maximum-distance-jump-lengths)

<a id="fastmob.measures.individual.waiting_times"></a>

::: fastmob.measures.individual.waiting_times
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Measure waiting times](../../learn/analyze-a-simple-trajectory.md#measure-waiting-times)

<a id="fastmob.measures.individual.home_location"></a>

::: fastmob.measures.individual.home_location
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Home and work](../../learn/measure-individual-mobility.md#home-and-work)

<a id="fastmob.measures.individual.work_location"></a>

::: fastmob.measures.individual.work_location
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Home and work](../../learn/measure-individual-mobility.md#home-and-work)

<a id="fastmob.measures.individual.location_frequency"></a>

::: fastmob.measures.individual.location_frequency
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks)

<a id="fastmob.measures.individual.frequency_rank"></a>

::: fastmob.measures.individual.frequency_rank
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks)

<a id="fastmob.measures.individual.recency_rank"></a>

::: fastmob.measures.individual.recency_rank
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [Which places matter](../../learn/measure-individual-mobility.md#which-places-matter-frequency-and-recency-ranks)

<a id="fastmob.measures.individual.number_of_locations"></a>

::: fastmob.measures.individual.number_of_locations
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

<a id="fastmob.measures.individual.number_of_visits"></a>

::: fastmob.measures.individual.number_of_visits
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

<a id="fastmob.measures.individual.real_entropy"></a>

::: fastmob.measures.individual.real_entropy
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable)

<a id="fastmob.measures.individual.random_entropy"></a>

::: fastmob.measures.individual.random_entropy
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

<a id="fastmob.measures.individual.uncorrelated_entropy"></a>

::: fastmob.measures.individual.uncorrelated_entropy
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How many places](../../learn/measure-individual-mobility.md#how-many-places-locations-visits-and-entropy)

::: fastmob.measures.individual.trajectory_entropy
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable)

::: fastmob.measures.individual.trajectory_predictability
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable)

<a id="fastmob.measures.individual.regularity"></a>

::: fastmob.measures.individual.regularity
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable)

<a id="fastmob.measures.individual.diversity"></a>

::: fastmob.measures.individual.diversity
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [How regular, diverse and predictable?](../../learn/visit-based-measures.md#how-regular-diverse-and-predictable)

::: fastmob.measures.individual.intermittance_and_degree_of_return
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Return and exploration](../../learn/visit-based-measures.md#return-and-exploration)

::: fastmob.measures.individual.exploration_profiling
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Return and exploration](../../learn/visit-based-measures.md#return-and-exploration)

::: fastmob.measures.individual.compute_profiles
    options:
      show_source: false

<!-- example-link -->**Worked example:** [One table of profiles](../../learn/visit-based-measures.md#one-table-of-profiles)

::: fastmob.measures.individual.daily_motifs
    options:
      show_source: false

<!-- example-link -->**Worked example:** [The same from a plain visit table](../../learn/motifs.md#the-same-from-a-plain-visit-table)

::: fastmob.measures.individual.motif_distribution
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How common is each motif?](../../learn/motifs.md#how-common-is-each-motif)

<a id="fastmob.measures.individual.individual_mobility_network"></a>

::: fastmob.measures.individual.individual_mobility_network
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

<!-- example-link -->**Worked example:** [A user's mobility network](../../learn/measure-individual-mobility.md#a-users-mobility-network)

::: fastmob.measures.individual.activity_transition_matrix
    options:
      show_source: false

<!-- example-link -->**Worked example:** [What are people doing?](../../learn/visit-based-measures.md#what-are-people-doing)

::: fastmob.measures.individual.daily_activity_distribution
    options:
      show_source: false

<!-- example-link -->**Worked example:** [What are people doing?](../../learn/visit-based-measures.md#what-are-people-doing)

::: fastmob.measures.individual.visit_purpose_distribution
    options:
      show_source: false

<!-- example-link -->**Worked example:** [What are people doing?](../../learn/visit-based-measures.md#what-are-people-doing)

::: fastmob.measures.individual.jump_lengths_road
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Road distance versus straight line](../../learn/road-network-analysis.md#road-distance-versus-straight-line)

::: fastmob.measures.individual.radius_of_gyration_road
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Road distance versus straight line](../../learn/road-network-analysis.md#road-distance-versus-straight-line)

