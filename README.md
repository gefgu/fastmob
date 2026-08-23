<p align="center">
  <img src="docs/src/assets/logo_filled.svg" alt="Fastkit-Mobility logo" width="280">
</p>

# Fastkit-Mobility

[![PyPI version](https://img.shields.io/pypi/v/fastmob)](https://pypi.org/project/fastmob/)
[![PyPI downloads](https://img.shields.io/pypi/dm/fastmob)](https://pypi.org/project/fastmob/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastmob)](https://pypi.org/project/fastmob/)
[![License](https://img.shields.io/pypi/l/fastmob)](LICENSE)

Fastkit-Mobility (`fastmob` on PyPI) is a high-performance Python library for
mobility analysis. It combines native Rust kernels with a
[Narwhals](https://narwhals-dev.github.io/narwhals/)-based API, so the same
code works unchanged against pandas, Polars, and other eager dataframe
backends — no rewrites, no backend-specific branches.

Beyond individual/collective mobility measures, fastmob covers the full
pipeline: raw position fixes → staypoints → triplegs → trips → tours,
trajectory preprocessing (outlier filtering, simplification, segmentation,
smoothing), mobility models (next-location prediction, gravity/radiation,
synthetic diary generation), and network/social analysis (road/rail-matched
distances, co-presence contact networks, mobility-law fitting).

## Why fastmob

- **Backend-agnostic** — pass a pandas, Polars, or any other
  Narwhals-compatible DataFrame; fastmob works without changes.
- **Rust-accelerated core** — compute-heavy kernels run in parallel,
  zero-copy Rust instead of Python loops.
- **Full trajectory hierarchy** — positionfixes, staypoints, triplegs,
  trips, tours, and locations, following the trackintel model.
- **Migration-friendly** — API shapes mirror
  [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility)
  where compatibility matters, with reproducible parity tests against it.
- **Measured, not claimed** — correctness tests, coverage, and
  reproducible benchmarks back every performance and compatibility claim.

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
