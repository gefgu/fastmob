# Getting Started

## Requirements

- Python >= 3.8
- Rust toolchain (for building from source)

## Install from PyPI

```bash
pip install skmob2
```

## Install for Development (from source)

Development requires `uv` and a Rust toolchain.

```bash
# Clone the repo
git clone https://github.com/gefgu/skmob2.git
cd skmob2

# First-time setup: creates .venv, builds the Rust extension, installs all dev deps
bash scripts/setup_env.sh
source .venv/bin/activate
```

After setup the compiled Rust extension (`.so`) is placed directly in `skmob2/`, so the package is importable from the repo root without a separate pip install.

To rebuild the Rust extension after editing `src/lib.rs`:

```bash
maturin develop
```

## Install the Docs Toolchain

```bash
uv sync --extra docs

# Serve locally with live reload
uv run zensical serve

# Build and validate
uv run zensical build
```

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
