# fastmob

Fastmob is a high-performance Python library for mobility analysis. It accepts
eager dataframe backends through Narwhals (including pandas, Polars, and
PyArrow-compatible workflows) and uses Rust for compute-intensive trajectory
kernels.

## Install

```bash
pip install fastmob
```

Install optional capabilities only when needed:

```bash
pip install duckdb  # fetch road/rail networks from Overture
pip install statsmodels  # fit the Gravity Poisson-GLM
pip install "fastmob[vis]"  # visualization package
```

## Quickstart

```python
import pandas as pd
from fastmob import jump_lengths, radius_of_gyration

traj = pd.DataFrame(
    {
        "uid": ["alice", "alice", "alice"],
        "datetime": pd.date_range("2024-01-01", periods=3, freq="h"),
        "lat": [41.8902, 41.9028, 41.9109],
        "lng": [12.4922, 12.4964, 12.4818],
    }
)

print(jump_lengths(traj))
print(radius_of_gyration(traj))
```

Fastmob auto-detects common time, latitude, longitude, and user-ID column
names. Pass explicit `datetime_col`, `lat_col`, `lng_col`, and `uid_col`
arguments when your schema differs.

## Documentation

- [Getting started and task recipes](https://gefgu.github.io/fastmob/learn/)
- [API reference](https://gefgu.github.io/fastmob/reference/)
- [Network analysis and map matching](https://gefgu.github.io/fastmob/reference/network/)
- [Release notes](https://gefgu.github.io/fastmob/release_notes/)

## Development

```bash
git clone https://github.com/gefgu/fastmob.git
cd fastmob
bash scripts/setup_env.sh
source .venv/bin/activate
pytest
```

To preview documentation locally:

```bash
uv sync --group docs
uv run --group docs zensical serve
```

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for documentation conventions and
the repository workflows.

## Status and support

Fastmob is actively developed. Please include versions, dataframe backend, a
minimal reproducible input, expected behavior, and the full error when opening
an issue.
