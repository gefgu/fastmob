---
icon: lucide/users-round
---

# Collective Measures

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`homes_per_location`](#fastmob.measures.collective.homes_per_location.homes_per_location) | Return the number of users whose home is at each distinct location. | [Homes and visits per place](../../learn/population-measures.md#homes-and-visits-per-place) |
| [`mean_square_displacement`](#fastmob.measures.collective.mean_square_displacement.mean_square_displacement) | Return the mean square displacement (km²) across all users. | [How far do people spread over time?](../../learn/population-measures.md#how-far-do-people-spread-over-time) |
| [`random_location_entropy`](#fastmob.measures.collective.random_location_entropy.random_location_entropy) | Return the random entropy for each distinct location across all users. | [How many different people use a place?](../../learn/population-measures.md#how-many-different-people-use-a-place) |
| [`uncorrelated_location_entropy`](#fastmob.measures.collective.uncorrelated_location_entropy.uncorrelated_location_entropy) | Return the uncorrelated entropy for each distinct location across all users. | [How many different people use a place?](../../learn/population-measures.md#how-many-different-people-use-a-place) |
| [`visits_per_location`](#fastmob.measures.collective.visits_per_location.visits_per_location) | Return the total number of visits for each distinct location. | [Homes and visits per place](../../learn/population-measures.md#homes-and-visits-per-place) |
| [`visits_per_time_unit`](#fastmob.measures.collective.visits_per_time_unit.visits_per_time_unit) | Return the number of trajectory points per time unit across all users. | [Homes and visits per place](../../learn/population-measures.md#homes-and-visits-per-place) |
| [`od_matrix`](#fastmob.measures.collective.od_matrix) | Compute an Origin-Destination matrix from a trips DataFrame. | [From check-ins to flows](../../learn/population-measures.md#from-check-ins-to-flows) |
| [`od_metrics_per_area`](#fastmob.measures.collective.od_metrics_per_area) | Compute per-area mobility metrics from a long-format OD DataFrame. | [From check-ins to flows](../../learn/population-measures.md#from-check-ins-to-flows) |
| [`mean_area_volume`](#fastmob.measures.collective.mean_area_volume) | Mean user volume per area and 10-minute time bin, averaged across days of the week. | [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy) |
| [`build_stvd`](#fastmob.measures.collective.build_stvd) | Aggregate staypoints against a global Locations catalogue into an STVD frame. | [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy) |
| [`interest_network`](#fastmob.measures.collective.interest_network.interest_network) | Return pairs of global locations visited by the same people. | [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy) |
<!-- api-table:end -->

::: fastmob.measures.collective.homes_per_location
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Homes and visits per place](../../learn/population-measures.md#homes-and-visits-per-place)

---

::: fastmob.measures.collective.mean_square_displacement
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How far do people spread over time?](../../learn/population-measures.md#how-far-do-people-spread-over-time)

---

::: fastmob.measures.collective.random_location_entropy
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How many different people use a place?](../../learn/population-measures.md#how-many-different-people-use-a-place)

---

::: fastmob.measures.collective.uncorrelated_location_entropy
    options:
      show_source: false

<!-- example-link -->**Worked example:** [How many different people use a place?](../../learn/population-measures.md#how-many-different-people-use-a-place)

---

::: fastmob.measures.collective.visits_per_location
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Homes and visits per place](../../learn/population-measures.md#homes-and-visits-per-place)

---

::: fastmob.measures.collective.visits_per_time_unit
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Homes and visits per place](../../learn/population-measures.md#homes-and-visits-per-place)

---

::: fastmob.measures.collective.od_matrix
    options:
      show_source: false

<!-- example-link -->**Worked example:** [From check-ins to flows](../../learn/population-measures.md#from-check-ins-to-flows)

---

::: fastmob.measures.collective.od_metrics_per_area
    options:
      show_source: false

<!-- example-link -->**Worked example:** [From check-ins to flows](../../learn/population-measures.md#from-check-ins-to-flows)

---

::: fastmob.measures.collective.mean_area_volume
    options:
      show_source: false

<!-- example-link -->**Worked example:** [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy)

---

::: fastmob.measures.collective.build_stvd
    options:
      show_source: false

<!-- example-link -->**Worked example:** [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy)

---

::: fastmob.measures.collective.interest_network.interest_network
    options:
      show_source: false

<!-- example-link -->**Worked example:** [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy)

