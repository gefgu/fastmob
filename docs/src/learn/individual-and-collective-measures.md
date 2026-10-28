# Measuring people and populations

**Question.** Which measure answers my research question, and what does it mean?

Individual measures describe each person; collective measures describe where and when the whole population moves. To run the individual measures on real data, open [Measure individual mobility](measure-individual-mobility.md).

## Which measure for which question

| Question | Start with | Unit |
| --- | --- | --- |
| How widely does each person range? | [`radius_of_gyration`](../reference/measures/individual.md#fastmob.measures.individual.radius_of_gyration), [`maximum_distance`](../reference/measures/individual.md#fastmob.measures.individual.maximum_distance) | km |
| How long are their trips and waits? | [`jump_lengths`](../reference/measures/individual.md#fastmob.measures.individual.jump_lengths), [`waiting_times`](../reference/measures/individual.md#fastmob.measures.individual.waiting_times) | km, seconds |
| How many places, how evenly used? | [`number_of_locations`](../reference/measures/individual.md#fastmob.measures.individual.number_of_locations), [`uncorrelated_entropy`](../reference/measures/individual.md#fastmob.measures.individual.uncorrelated_entropy) | count, bits |
| Which places matter? | [`frequency_rank`](../reference/measures/individual.md#fastmob.measures.individual.frequency_rank), [`home_location`](../reference/measures/individual.md#fastmob.measures.individual.home_location), [`work_location`](../reference/measures/individual.md#fastmob.measures.individual.work_location) | places |
| When do visits occur? | [`visits_per_time_unit`](../reference/measures/collective.md#fastmob.measures.collective.visits_per_time_unit) (collective) | visits per interval |
| Which origins and destinations connect? | [`od_matrix`](../reference/measures/collective.md#fastmob.measures.collective.od_matrix) (collective) | trips |

## Choosing the input

Point-based measures such as the radius of gyration use raw trajectories. Visit-based measures (`diversity`, `regularity`, motif measures) need visit tables, such as staypoints. OD measures need trips or an origin-destination layout. Do not average a per-person result by hand when a collective function already defines the population.

## Assumptions

Distances are straight lines in kilometres, times are seconds, and hour-of-day measures treat hours in the data's own time zone, so convert to local time first.

## Related API

[Individual measures](../reference/measures/individual.md) ·
[Collective measures](../reference/measures/collective.md) ·
[Evaluation measures](../reference/measures/evaluation.md)

Worked examples: [Measure individual mobility](measure-individual-mobility.md), [Routine, predictability and mobility profiles](visit-based-measures.md) and [Population measures](population-measures.md), [Flows between places](flows-between-places.md) and [Occupancy and phone records](occupancy-and-phone-records.md).
