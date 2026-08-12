# fastmob

<p align="center">
  <img src="assets/logo_filled.svg" alt="fastmob logo" width="280">
</p>

**fastmob** is the fast mobility-analysis library for Python: a Rust-accelerated,
backend-agnostic toolkit for preparing trajectories, modelling movement, and
measuring individual and collective mobility.

It combines native Rust kernels with a Python API and
[Narwhals](https://narwhals-dev.github.io/narwhals/) dataframe support, so the
same workflow works with pandas, Polars, and other eager dataframe backends.
Fastmob includes trajectory processing, the Positionfixes → Staypoints →
Triplegs → Trips → Tours hierarchy, networks, privacy, fitting, and mobility
generation models. Compatibility with [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility)
is supported where it helps teams migrate, but it is not the library's scope or
identity.

# Key Features

- **Backend-agnostic**: pass a pandas, polars, or any other Narwhals-compatible DataFrame — fastmob works without changes.

- **Rust-accelerated core**: parallel, zero-copy kernels for compute-heavy mobility workloads.
    
- **Mobility-native toolkit**: work from raw position fixes through stays,
  locations, trips, tours, flows, networks, privacy, fitting, and generation.
    
- **Migration-friendly**: familiar scikit-mobility APIs and reproducible parity
  checks make adoption straightforward when compatibility matters.

- **Zero-copy by design**: fastmob processes data where it lives, avoiding
  unnecessary dataframe copies and memory pressure.

- **Lightweight**: Python wheels ship compact, release-ready binaries.


## Installation

```bash
pip install fastmob
```

## Validation and Performance

fastmob is tested as a mobility library and a performance project. The
correctness suite covers its native APIs across pandas and Polars, while optional
scikit-mobility comparisons verify compatibility where the projects overlap. The
benchmark suite uses standalone perf-counter scripts on representative mobility
workloads, including Brightkite-derived data slices.

For reproducible benchmark commands and methodology, see the
[benchmarks page](features/benchmarks.md).

## Quick Example

```python
import pandas as pd
from fastmob import jump_lengths

df = pd.DataFrame({
    "uid": ["alice", "alice", "alice", "bob", "bob"],
    "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
    "lat": [41.9, 42.0, 42.1, 40.7, 40.8],
    "lng": [12.5, 12.5, 12.5, -74.0, -74.0],
})

result = jump_lengths(df)
print(result)
```

# Getting Started

## Requirements

- Python >= 3.9
- Rust toolchain (for building from source)

## Install from PyPI

=== "pip"

    ```bash
    pip install fastmob
    ```

=== "uv"

    ```bash
    uv add fastmob
    ```

=== "generation fitting"

    ```bash
    pip install statsmodels
    ```

=== "visualization extra"

    ```bash
    pip install "fastmob[vis]"
    ```

## Install for Development (from source)

=== "Initial setup"

    ```bash
    git clone https://github.com/gefgu/fastmob.git
    cd fastmob
    bash scripts/setup_env.sh
    source .venv/bin/activate
    ```

=== "Rebuild Rust extension"

    ```bash
    maturin develop
    ```

=== "Docs"

    ```bash
    uv sync --group docs
    uv run --group docs zensical serve
    uv run --group docs zensical build
    ```

Development requires `uv` and a Rust toolchain. After setup the compiled Rust extension (`.so`) is placed directly in `fastmob/`, so the package is importable from the repo root without a separate pip install.

## Column Name Auto-Detection

fastmob auto-detects required columns by scanning a priority list:

| Semantic role | Accepted column names (in order) |
|---|---|
| datetime | `datetime`, `check-in_time`, `timestamp`, `time` |
| latitude | `lat`, `latitude` |
| longitude | `lng`, `lon`, `longitude` |
| user ID (optional) | `uid`, `user`, `user_id` |

When a user-ID column is absent, the entire dataframe is treated as a single individual.

You can also pass column names explicitly via keyword arguments (`datetime_col`, `lat_col`, `lng_col`, `uid_col`).

## Your First Measure: jump_lengths

```python
import pandas as pd
from fastmob import jump_lengths

# Build a minimal trajectory dataframe
df = pd.DataFrame({
    "uid": ["alice", "alice", "alice"],
    "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    "lat": [0.0, 1.0, 2.0],
    "lng": [0.0, 0.0, 0.0],
})

# Compute jump lengths (Haversine distances in km between consecutive points)
result = jump_lengths(df)
print(result)
# uid  jump_lengths
# alice  [111.19..., 111.19...]
```

## Using Polars

fastmob is backend-agnostic. Pass a polars DataFrame and you get a polars DataFrame back:

```python
import polars as pl
from fastmob import jump_lengths

df = pl.DataFrame({
    "uid": ["alice", "alice", "alice"],
    "datetime": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"],
    "lat": [0.0, 1.0, 2.0],
    "lng": [0.0, 0.0, 0.0],
}).with_columns(pl.col("datetime").str.to_datetime())

result = jump_lengths(df)  # returns a polars DataFrame
```

## Optional Extras

| Extra | Installs | Used by |
|---|---|---|
| `geo` | GeoPandas, Shapely, PyProj | geospatial data conversion and tessellation; H3 is native Rust |
| `vis` | `fastmob-vis` | `fastmob.vis` generic ECharts visualization |

Install an extra with:

```bash
pip install "fastmob[geo]"
# or with uv:
uv sync --extra geo
```

## Generation Models

```python
import pandas as pd
from fastmob.models import Gravity, SpatialEPR

tessellation = pd.DataFrame({
    "tile_id": [0, 1, 2],
    "lat": [45.0, 45.1, 45.2],
    "lng": [7.0, 7.1, 7.2],
    "relevance": [5, 6, 7],
    "tot_outflow": [10, 12, 14],
})

flows = Gravity().generate(tessellation, out_format="probabilities")

start = pd.Timestamp("2020-01-01 00:00:00")
end = pd.Timestamp("2020-01-01 06:00:00")
trajectories = SpatialEPR().generate(start, end, tessellation, n_agents=2, random_state=0)
```
