# skmob2

<p align="center">
  <img src="assets/logo.png" alt="skmob2 logo" width="280">
</p>

**skmob2** is a high-performance reimplementation of the [skmob](https://github.com/scikit-mobility/scikit-mobility) mobility-analysis library.

It exposes the same measure API but replaces the Python/pandas internals with a Rust extension (via PyO3) for compute-heavy kernels, and wraps the Python layer with [Narwhals](https://narwhals-dev.github.io/) so any eager dataframe (pandas, polars, …) is accepted as input.

## Key Features

- **Backend-agnostic:** pass a pandas, polars, or any other Narwhals-compatible DataFrame — skmob2 works without changes.
- **Rust-accelerated core:** Haversine distance calculations and other compute-heavy kernels run in compiled Rust via PyO3.
- **Drop-in API:** function signatures mirror the original skmob library so migration is straightforward.

## Installation

```bash
pip install skmob2
```

For development or building from source, see the [Getting Started](getting-started.md) guide.

## Quick Example

```python
import pandas as pd
from skmob2 import jump_lengths

df = pd.DataFrame({
    "uid": ["alice", "alice", "alice", "bob", "bob"],
    "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
    "lat": [41.9, 42.0, 42.1, 40.7, 40.8],
    "lng": [12.5, 12.5, 12.5, -74.0, -74.0],
})

result = jump_lengths(df)
print(result)
```

## Public API

| Function | Description |
|---|---|
| [`jump_lengths`][skmob2.measures.spatial.jump_lengths.jump_lengths] | Haversine distances between consecutive GPS fixes per user |
| [`radius_of_gyration`][skmob2.measures.spatial.radius_of_gyration.radius_of_gyration] | Spread of a user's movements around their center of mass |
| [`od_matrix`][skmob2.measures.flows.od.od_matrix] | Origin-destination trip counts |
| [`od_metrics_per_area`][skmob2.measures.flows.od.od_metrics_per_area] | Per-area flow metrics from an OD matrix |
| [`log_truncated_powerlaw`][skmob2.measures.fitting.mobility_laws.log_truncated_powerlaw] | Log of the truncated power-law (Gonzalez et al. 2008) |
| [`fit_values_to_truncated_powerlaw`][skmob2.measures.fitting.mobility_laws.fit_values_to_truncated_powerlaw] | Fit a 1-D array to the truncated power-law model |
| [`activity_transition_matrix`][skmob2.measures.visits.activity.activity_transition_matrix] | Activity-type transition probabilities |
| [`intermittance_and_degree_of_return`][skmob2.measures.visits.mobility_profiling.intermittance_and_degree_of_return] | Intermittancy and degree-of-return per user |
