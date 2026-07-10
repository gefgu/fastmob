# Analyze a simple trajectory

In this tutorial, we will create a small trajectory dataset, compute two mobility measures, and inspect the results.

We will use pandas so the example is easy to read. fkmob can also work with other eager dataframe backends; when you are ready to adapt this example to your own column names or backend, see TODO.
<!-- [Use custom columns and dataframe backends](../how-to-guides/use-custom-columns-and-dataframe-backends.md). -->

## Before we start

Make sure fkmob and pandas are installed in your Python environment:

```bash
pip install fkmob pandas
```

Start a Python session or create a file named `simple_trajectory.py`.

## Step 1: Create a trajectory dataframe

First, create a small dataframe with two users, timestamps, and coordinates:

```python
import pandas as pd

df = pd.DataFrame({
    "uid": ["alice", "alice", "alice", "bob", "bob", "bob"],
    "datetime": pd.to_datetime([
        "2020-01-01 08:00:00",
        "2020-01-01 09:00:00",
        "2020-01-01 10:00:00",
        "2020-01-01 08:00:00",
        "2020-01-01 09:00:00",
        "2020-01-01 10:00:00",
    ]),
    "lat": [41.8902, 41.9028, 41.9109, 40.7128, 40.7306, 40.7580],
    "lng": [12.4922, 12.4964, 12.4818, -74.0060, -73.9352, -73.9855],
})

print(df)
```

The output should look something like:

```text
     uid            datetime      lat      lng
0  alice 2020-01-01 08:00:00  41.8902  12.4922
1  alice 2020-01-01 09:00:00  41.9028  12.4964
2  alice 2020-01-01 10:00:00  41.9109  12.4818
3    bob 2020-01-01 08:00:00  40.7128 -74.0060
4    bob 2020-01-01 09:00:00  40.7306 -73.9352
5    bob 2020-01-01 10:00:00  40.7580 -73.9855
```

Notice that each row is one point in a user's trajectory.

## Step 2: Compute jump lengths

Now compute the distance between each user's consecutive points:

```python
from fkmob import jump_lengths

jumps = jump_lengths(df)
print(jumps)
```

The output should look something like:

```text
     uid                                  jump_lengths
0  alice    [1.442..., 1.518...]
1    bob    [6.286..., 5.236...]
```

Notice that fkmob returns one row per user. The `jump_lengths` value is a list because each user has more than one movement between points.

## Step 3: Compute radius of gyration

Next, compute how widely each user moves around their center of mass:

```python
from fkmob import radius_of_gyration

rg = radius_of_gyration(df)
print(rg)
```

The output should look something like:

```text
     uid  radius_of_gyration
0  alice             0.9...
1    bob             3.8...
```

Notice that this result has one number per user. A larger radius of gyration means the user's points are more spread out.

## Step 4: Run the whole example

Let's put the pieces together:

```python
import pandas as pd
from fkmob import jump_lengths, radius_of_gyration

df = pd.DataFrame({
    "uid": ["alice", "alice", "alice", "bob", "bob", "bob"],
    "datetime": pd.to_datetime([
        "2020-01-01 08:00:00",
        "2020-01-01 09:00:00",
        "2020-01-01 10:00:00",
        "2020-01-01 08:00:00",
        "2020-01-01 09:00:00",
        "2020-01-01 10:00:00",
    ]),
    "lat": [41.8902, 41.9028, 41.9109, 40.7128, 40.7306, 40.7580],
    "lng": [12.4922, 12.4964, 12.4818, -74.0060, -73.9352, -73.9855],
})

print(jump_lengths(df))
print(radius_of_gyration(df))
```

You should see two small result dataframes, both grouped by user.

## What we have made

You have created a minimal trajectory dataset and used fkmob to compute two user-level mobility measures. The same pattern works for larger trajectory dataframes: build a dataframe with time, latitude, longitude, and user columns, then pass it to the measure you want to compute.

