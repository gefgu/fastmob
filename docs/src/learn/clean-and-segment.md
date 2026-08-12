# Clean and segment a trajectory

Use this recipe when raw positions contain implausible jumps or when a long
trace must be split into movement episodes. It preserves the dataframe backend
and makes no assumptions beyond time, latitude, and longitude columns.

## 1. Filter implausible movement

```python
from fastmob.preprocessing import filter

clean = filter(traj, method="smart_greedy", max_speed_kmh=160)
```

`filter` removes rows that imply unrealistic movement under the selected
strategy. Tune the speed threshold to the transport modes represented by your
data; do not use a car threshold for walking-only traces.

## 2. Split observation gaps

```python
from fastmob.preprocessing import segment

segmented = segment(clean, method="observation_gap", gap_s=30 * 60)
```

The returned frame adds a segment identifier. Each user receives independent
segments, so a gap for one person cannot split another person's trajectory.

## 3. Verify the result

```python
print(segmented.select(["uid", "datetime", "segment_id"]).head())
```

Check that segment boundaries correspond to genuine collection gaps rather
than routine sampling intervals. See the [preprocessing reference](../reference/preprocessing.md)
for speed, stop, temporal, direction-change, and value-change strategies.

<!-- Visual placeholder: before/after trajectory map with removed outlier and gap boundary. -->
