# fastmob

High-performance mobility analysis with backend-agnostic dataframe support and Rust-accelerated compute kernels.

## Key Features

- **Backend-agnostic dataframes**: use pandas, Polars, or other eager Narwhals-compatible dataframes as inputs.
- **Rust-accelerated kernels**: run compute-heavy mobility operations through a compiled PyO3 extension.
- **Trajectory preprocessing**: filter noisy trajectories, compress movement records, detect stay locations, cluster stops, and transform CDR records.
- **Mobility measures**: compute jump lengths, radius of gyration, waiting times, location frequencies, entropy, predictability, origin-destination matrices, and related individual or collective metrics.
- **Modeling and generation**: build gravity, radiation, EPR, DITRAS, Markov diary, GeoSim, and spatial-temporal social trajectory models.
- **Privacy analysis**: evaluate mobility-data privacy risk with location, frequency, dataframe, and attack-oriented utilities.
- **Tessellation and IO helpers**: work with spatial tiles, trajectory dataframes, flow dataframes, datasets, and file-based mobility inputs.
- **Optional visualization**: build backend-neutral, portable ECharts charts through `fastmob.vis`.
- **Validation-focused development**: maintain correctness tests, benchmark outputs, profiling support, and compatibility checks for performance-sensitive workflows.

## Installation

```bash
pip install fastmob
```

For charts and HTML/SVG export:

```bash
pip install "fastmob[vis]"
```

For development from source:

```bash
uv sync --extra dev
env -u CONDA_PREFIX uv run maturin develop
```

## Quick Example

```python
import pandas as pd
from fastmob import jump_lengths

df = pd.DataFrame(
    {
        "uid": ["alice", "alice", "alice", "bob", "bob"],
        "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
        "lat": [41.9, 42.0, 42.1, 40.7, 40.8],
        "lng": [12.5, 12.5, 12.5, -74.0, -74.0],
    }
)

result = jump_lengths(df)
print(result)
```

## Development Checks

```bash
uv run pytest
uv run ruff check
```
