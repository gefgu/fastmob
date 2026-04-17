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
bash tests/setup_env.sh
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
uv run mkdocs serve

# Build and validate (strict mode, zero warnings)
uv run mkdocs build --strict
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
| `fitting` | `scipy` | `fit_values_to_truncated_powerlaw` |
| `diversity` | `pydivsufsort` | diversity measures (future) |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | documentation build |

Install an extra with:

```bash
pip install "skmob2[fitting]"
# or with uv:
uv sync --extra fitting
```
