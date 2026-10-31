# Utilities

<!-- api-table:start -->
**Geographic distance**

| API | Description | Example |
| --- | --- | --- |
| [`haversine`](#fastmob.utils.haversine) | Compute elementwise great-circle distances between coordinate pairs. |  |
<!-- api-table:end -->

## Straight-line distance between coordinates

Use [`fastmob.utils.haversine`](#fastmob.utils.haversine) to calculate
straight-line surface distances between matching latitude/longitude pairs. The
coordinates are in degrees. Choose metres or kilometres with `unit`, and a
NumPy array, pandas Series, or Arrow array with `output`. Calculation uses the
native parallel batch kernel, so input rows are not processed by a Python loop.

For columns from a pandas DataFrame:

```python
from fastmob.utils import haversine

trips["distance_m"] = haversine(
    trips["latitude_source"], trips["longitude_source"],
    trips["latitude_target"], trips["longitude_target"],
    unit="m", output="numpy",
)
```

Results follow input row order, regardless of pandas index labels. Use
`unit="km"` for kilometres and `output="pandas"` or `output="arrow"` for
other containers. For routing along roads, see [Network](network.md).

::: fastmob.utils.haversine
    options:
      show_source: false
