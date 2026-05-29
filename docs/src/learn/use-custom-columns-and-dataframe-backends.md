# How to use custom columns and dataframe backends

This guide shows you how to run skmob2 measures when your trajectory dataframe uses nonstandard column names or a dataframe backend such as Polars.

<!-- For a beginner-friendly walkthrough with the default column names, start with [Analyze a simple trajectory](../tutorials/analyze-a-simple-trajectory.md). For complete function details, refer to the [API reference](../api/index.md). -->

## Before you start

Your dataframe must contain columns for:

- datetime
- latitude
- longitude
- user ID, unless the whole dataframe should be treated as one user's trajectory

The datetime column should already contain datetime-like values that your dataframe backend understands.

## Use explicit column names

Pass column name overrides to the measure you want to compute:

```python
import pandas as pd
from skmob2 import jump_lengths, radius_of_gyration

traj = pd.DataFrame({
    "person": ["alice", "alice", "alice"],
    "observed_at": pd.to_datetime([
        "2020-01-01 08:00:00",
        "2020-01-01 09:00:00",
        "2020-01-01 10:00:00",
    ]),
    "latitude_deg": [41.8902, 41.9028, 41.9109],
    "longitude_deg": [12.4922, 12.4964, 12.4818],
})

jumps = jump_lengths(
    traj,
    datetime_col="observed_at",
    lat_col="latitude_deg",
    lng_col="longitude_deg",
    uid_col="person",
)

rg = radius_of_gyration(
    traj,
    datetime_col="observed_at",
    lat_col="latitude_deg",
    lng_col="longitude_deg",
    uid_col="person",
)
```

Use explicit overrides whenever your data does not use skmob2's default column names.

## Use Polars input

Pass a Polars dataframe directly. skmob2 returns a Polars dataframe for dataframe results:

```python
import polars as pl
from skmob2 import jump_lengths

traj = pl.DataFrame({
    "person": ["alice", "alice", "alice"],
    "observed_at": [
        "2020-01-01 08:00:00",
        "2020-01-01 09:00:00",
        "2020-01-01 10:00:00",
    ],
    "latitude_deg": [41.8902, 41.9028, 41.9109],
    "longitude_deg": [12.4922, 12.4964, 12.4818],
}).with_columns(pl.col("observed_at").str.to_datetime())

result = jump_lengths(
    traj,
    datetime_col="observed_at",
    lat_col="latitude_deg",
    lng_col="longitude_deg",
    uid_col="person",
)

print(type(result))
```

The printed type should be a Polars dataframe type.

## Omit the user column for single-user data

If the dataframe contains one user's trajectory and has no user column, omit `uid_col`:

```python
jumps = jump_lengths(
    traj,
    datetime_col="observed_at",
    lat_col="latitude_deg",
    lng_col="longitude_deg",
)
```

skmob2 treats the whole dataframe as one trajectory.

## Verify the result

Check these points before using the result downstream:

- The returned dataframe backend matches the input backend for dataframe-returning measures.
- The result contains one row per user when `uid_col` is provided.
- Distances are expressed in kilometers.
- If a required column is missing, pass its name explicitly with the matching `*_col` argument.

