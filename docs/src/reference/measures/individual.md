---
icon: lucide/user-round
---

# Individual measures

Individual measures describe each person's movement, place use, regularity,
and inferred behavior. Most accept a trajectory-like dataframe and return one
row, scalar, or variable-length result per user. Pass explicit column names
when auto-detection does not match your schema.

| Question | Representative APIs |
| --- | --- |
| How far and how often does someone move? | `jump_lengths`, `radius_of_gyration`, `waiting_times` |
| Which places matter? | `home_location`, `work_location`, `location_frequency` |
| Is their movement regular or predictable? | `regularity`, entropy, predictability, motifs |
| Does route distance matter? | `jump_lengths_road`, `radius_of_gyration_road` |

For a task-focused introduction, see
[Measure people and populations](../../learn/individual-and-collective-measures.md).

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

::: fastmob.measures.individual.home_location
    options:
      show_source: false

::: fastmob.measures.individual.individual_mobility_network
    options:
      show_source: false

::: fastmob.measures.individual.intermittance_and_degree_of_return
    options:
      show_source: false

::: fastmob.measures.individual.jump_lengths
    options:
      show_source: false

::: fastmob.measures.individual.jump_lengths_road
    options:
      show_source: false

::: fastmob.measures.individual.k_radius_of_gyration
    options:
      show_source: false

::: fastmob.measures.individual.location_frequency
    options:
      show_source: false

::: fastmob.measures.individual.max_distance_from_home
    options:
      show_source: false

::: fastmob.measures.individual.maximum_distance
    options:
      show_source: false

::: fastmob.measures.individual.motif_distribution
    options:
      show_source: false

::: fastmob.measures.individual.number_of_locations
    options:
      show_source: false

::: fastmob.measures.individual.number_of_visits
    options:
      show_source: false

::: fastmob.measures.individual.radius_of_gyration
    options:
      show_source: false

::: fastmob.measures.individual.radius_of_gyration_road
    options:
      show_source: false

::: fastmob.measures.individual.random_entropy
    options:
      show_source: false

::: fastmob.measures.individual.real_entropy
    options:
      show_source: false

::: fastmob.measures.individual.recency_rank
    options:
      show_source: false

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

::: fastmob.measures.individual.visit_purpose_distribution
    options:
      show_source: false

::: fastmob.measures.individual.waiting_times
    options:
      show_source: false

::: fastmob.measures.individual.work_location
    options:
      show_source: false
