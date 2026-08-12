# Create staypoints and locations

Use staypoints to move from noisy position fixes to meaningful visits, then
cluster repeated visits into user-specific locations.

```python
from fastmob import Positionfixes

fixes = Positionfixes(traj)
staypoints = fixes.generate_staypoints(minutes_for_a_stop=20, spatial_radius_km=0.2)
locations, assigned_staypoints = staypoints.generate_user_locations(epsilon_km=0.2)
```

`assigned_staypoints` contains the location assignment for each retained stay.
The `locations` catalogue is user-scoped: the same numeric location ID is not
assumed to mean the same place for different users.

To infer a purpose for each recurring location:

```python
labeled_locations = locations.identify(assigned_staypoints)
```

The built-in frequency heuristic labels one likely home, one likely work
location, and all remaining locations as `other`. Inspect the inferred labels
before treating them as ground truth.

See [data structures](../reference/data-structures.md) for the
Positionfixes → Staypoints → Triplegs → Trips → Tours hierarchy, or open the
[Staypoints reference](../reference/data-structures/staypoints.md) for its
columns and API.

<!-- Visual placeholder: pings → staypoints → labeled user locations workflow. -->
