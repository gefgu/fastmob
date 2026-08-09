---
icon: lucide/route
---

# TrajDataFrame

`TrajDataFrame` is fastmob's general-purpose wrapper for timestamped latitude
and longitude observations. It accepts eager Narwhals-compatible DataFrames,
detects conventional column names, and records the resolved user, time, and
coordinate metadata for later operations.

## Required data

Provide latitude, longitude, and datetime columns; a user-ID column is optional.
Pass explicit column names when automatic detection is not appropriate. The
wrapper preserves the native backend and can sort observations by user and time.

```python
import pandas as pd
from fastmob import TrajDataFrame

traj = TrajDataFrame(pd.DataFrame({
    "uid": [1, 1],
    "datetime": ["2024-01-01 08:00", "2024-01-01 08:10"],
    "lat": [47.37, 47.38],
    "lng": [8.54, 8.55],
}), sort=True)

jumps = traj.jump_lengths()
```

For typed hierarchy generation, use [`Positionfixes`](positionfixes.md), which
extends this class.

## API

::: fastmob.core.trajectory_dataframe.TrajDataFrame
    options:
      show_source: false
