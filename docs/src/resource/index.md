# Resources

## Choose the right starting point

| Need | Resource |
| --- | --- |
| A minimal runnable analysis | [Start with a trajectory](../learn/analyze-a-simple-trajectory.md) |
| Column names or backend behavior | [Columns and backends](../learn/use-custom-columns-and-dataframe-backends.md) |
| Accepted inputs and return schemas | [API reference](../reference/index.md) |
| Performance evidence | [Benchmarks](../features/benchmarks.md) |
| Recent behavior changes | [Release notes](../release_notes/index.md) |

## Data and ecosystem

Fastmob is designed for trajectory, staypoint, trip, and origin-destination
data. The [data reference](../reference/data.md) documents bundled dataset
loaders. Its public API is informed by the
[scikit-mobility](https://scikit-mobility.github.io/scikit-mobility/) ecosystem;
Fastmob documents its own behavior and performance contracts independently.

## Glossary

- **Position fix** — one timestamped latitude/longitude observation.
- **Staypoint** — a period spent within a small spatial area.
- **Location** — a recurring place formed by clustering staypoints.
- **Trip leg** — movement between two stays.
- **OD flow** — an aggregate count or value from origin to destination.
- **STVD** — spatiotemporal visitation distribution.
