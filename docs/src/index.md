# Scikit-Mobility 2

<p align="center">
  <img src="assets/logo.png" alt="skmob2 logo" width="280">
</p>

**skmob2** is a high-performance reimplementation of the [skmob](https://github.com/scikit-mobility/scikit-mobility) mobility-analysis library.

It exposes the same measure API but replaces the Python/pandas internals with a Rust extension (via PyO3) for compute-heavy kernels, and wraps the Python layer with [Narwhals](https://narwhals-dev.github.io/narwhals/) so any eager dataframe (pandas, polars, …) is accepted as input.

# Key Features

- **Backend-agnostic**: pass a pandas, polars, or any other Narwhals-compatible DataFrame — skmob2 works without changes.

- **Rust-accelerated core**:  500x median speedup due to rust parallelized and zero-copy operations.
    
- **Drop-in API**: function signatures mirror the original skmob library so migration is straightforward.
    
- **Measured validation**: correctness tests, Python coverage, profiling, and standalone speed benchmarks make compatibility and performance claims reproducible.

- **Zero-Copy**: Scikit-Mobility 2 process the data where it lives. Instead of copying, it directly access your dataframe in the memory, saving memory and making the processing faster.

- **Lightweight**: Python Wheels ship with less than 30KB. 


## Installation

```bash
pip install skmob2
```

## Validation and Performance

skmob2 is tested as both a compatibility project and a performance project. The correctness suite covers the Python package, exercises pandas and Polars inputs, and includes optional comparisons with skmob where those dependencies are installed. The benchmark suite uses standalone perf-counter scripts on representative mobility workloads, including Brightkite-derived data slices.

<!-- For the reasoning behind this validation model and the commands used to reproduce it, see [Correctness, coverage, and benchmarking](explanations/correctness-coverage-and-benchmarking.md). -->

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

# Getting Started

## Requirements

- Python >= 3.8
- Rust toolchain (for building from source)

## Install from PyPI

=== "pip"

    ```bash
    pip install skmob2
    ```

=== "uv"

    ```bash
    uv add skmob2
    ```

=== "generation extra"

    ```bash
    pip install "skmob2[generation]"
    ```

## Install for Development (from source)

=== "Initial setup"

    ```bash
    git clone https://github.com/gefgu/skmob2.git
    cd skmob2
    bash scripts/setup_env.sh
    source .venv/bin/activate
    ```

=== "Rebuild Rust extension"

    ```bash
    maturin develop
    ```

=== "Docs"

    ```bash
    uv sync --extra docs
    uv run --extra docs zensical serve
    uv run --extra docs zensical build
    ```

Development requires `uv` and a Rust toolchain. After setup the compiled Rust extension (`.so`) is placed directly in `skmob2/`, so the package is importable from the repo root without a separate pip install.

## Column Name Auto-Detection

skmob2 auto-detects required columns by scanning a priority list:

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
from skmob2 import jump_lengths

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

skmob2 is backend-agnostic. Pass a polars DataFrame and you get a polars DataFrame back:

```python
import polars as pl
from skmob2 import jump_lengths

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
| `ai` | `scikit-learn` | `cluster` |
| `fitting` | `scipy` | `fit_values_to_truncated_powerlaw` |
| `diversity` | `pydivsufsort` | diversity measures (future) |
| `generation` | `scipy`, `powerlaw`, `statsmodels`, `python-igraph`, `tqdm` | `skmob2.models` generation APIs |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | documentation build |

Install an extra with:

```bash
pip install "skmob2[fitting]"
# or with uv:
uv sync --extra fitting
```

## Generation Models

```python
import pandas as pd
from skmob2.models import Gravity, SpatialEPR

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
