# Measure people and populations

Individual measures answer how each person moves; collective measures summarize
where and when the population moves.

```python
from fastmob.measures.individual import radius_of_gyration, home_location
from fastmob.measures.collective import visits_per_time_unit, od_matrix

per_person_range = radius_of_gyration(traj)
homes = home_location(traj)
visits_by_hour = visits_per_time_unit(visits, freq="1h")
flows = od_matrix(trips)
```

Use raw trajectories for point-based measures such as radius of gyration.
Use staypoint/visit data for visitation metrics, and trips or explicitly
origin-destination-shaped data for OD metrics. Do not aggregate a per-user
result manually when a collective API already defines the intended population
semantics.

| Question | Start with |
| --- | --- |
| How far does each user range? | `radius_of_gyration` |
| Which places does each user revisit? | `location_frequency`, `home_location` |
| When do visits occur? | `visits_per_time_unit` |
| Which origins and destinations are connected? | `od_matrix` |

See [individual measures](../reference/measures/individual.md) and
[collective measures](../reference/measures/collective.md) for complete
schemas and options.
