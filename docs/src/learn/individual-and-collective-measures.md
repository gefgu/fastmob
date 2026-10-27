# Measuring people and populations

**Question.** Which measure answers my research question, and what does it mean?

Individual measures describe each person; collective measures describe where and when the whole population moves. To run the individual measures on real data, open [Measure individual mobility](measure-individual-mobility.md).

## Which measure for which question

| Question | Start with | Unit |
| --- | --- | --- |
| How widely does each person range? | `radius_of_gyration`, `maximum_distance` | km |
| How long are their trips and waits? | `jump_lengths`, `waiting_times` | km, seconds |
| How many places, how evenly used? | `number_of_locations`, `uncorrelated_entropy` | count, bits |
| Which places matter? | `frequency_rank`, `home_location`, `work_location` | places |
| When do visits occur? | `visits_per_time_unit` (collective) | visits per interval |
| Which origins and destinations connect? | `od_matrix` (collective) | trips |

## Choosing the input

Point-based measures such as the radius of gyration use raw trajectories. Visit-based measures (`diversity`, `regularity`, motif measures) need visit tables, such as staypoints. OD measures need trips or an origin-destination layout. Do not average a per-person result by hand when a collective function already defines the population.

## Assumptions

Distances are straight lines in kilometres, times are seconds, and hour-of-day measures treat hours in the data's own time zone, so convert to local time first.

## Related API

[Individual measures](../reference/measures/individual.md) ·
[Collective measures](../reference/measures/collective.md)
