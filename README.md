<p align="center">
  <img src="docs/src/assets/logo_filled.svg" alt="Fastkit-Mobility logo" width="280">
</p>

# Fastkit-Mobility

[![PyPI version](https://img.shields.io/pypi/v/fastmob)](https://pypi.org/project/fastmob/)
[![PyPI downloads](https://img.shields.io/pypi/dm/fastmob)](https://pypi.org/project/fastmob/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastmob)](https://pypi.org/project/fastmob/)
[![License](https://img.shields.io/pypi/l/fastmob)](LICENSE)

Fastkit-Mobility (`fastmob` on PyPI) turns raw mobility data into useful trips,
stays, and measures. Its compiled Rust engine and [Narwhals](https://narwhals-dev.github.io/narwhals/)-based API keep the same workflow running on pandas, Polars, and other eager dataframe backends.

Start with position fixes, derive staypoints and triplegs, then build trips and
tours. The hierarchy follows established mobility data models while keeping the
input as an ordinary dataframe.

## Why fastmob

- **Backend-agnostic** — pass a pandas, Polars, or any other
  Narwhals-compatible DataFrame; fastmob works without changes.
- **Rust-accelerated core** — compute-heavy kernels run in parallel,
  zero-copy Rust instead of Python loops.
- **A hierarchy for real movement data** — positionfixes, staypoints,
  triplegs, trips, tours, and locations.
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

## Turn raw GPS points into staypoints and trips

```python
import pandas as pd
from fastmob import Positionfixes

gps = pd.DataFrame(
    {
        "uid": ["alice"] * 9,
        "datetime": pd.date_range("2024-01-01 08:00", periods=9, freq="10min"),
        "lat": [41.8902] * 3 + [41.8950, 41.9000, 41.9050] + [41.9109] * 3,
        "lng": [12.4922] * 3 + [12.4950, 12.4980, 12.4800] + [12.4818] * 3,
    }
)

fixes = Positionfixes(gps)
staypoints = fixes.generate_staypoints(minutes_for_a_stop=20, spatial_radius_km=0.2)
triplegs = fixes.generate_triplegs(staypoints)
activity_stays = staypoints.create_activity_flag(time_threshold_min=20)
trips = triplegs.generate_trips(activity_stays)
print(trips.df[["started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id"]])
```

Fastmob auto-detects common time, latitude, longitude, and user-ID column
names. Pass explicit `datetime_col`, `lat_col`, `lng_col`, and `uid_col`
arguments when your schema differs.

Next: [measure people and populations](https://gefgu.github.io/fastmob/learn/individual-and-collective-measures/), [clean and segment trajectories](https://gefgu.github.io/fastmob/learn/clean-and-segment/), [route and map-match GPS traces](https://gefgu.github.io/fastmob/learn/network-analysis/), or explore [mobility models](https://gefgu.github.io/fastmob/reference/models/).

## Full GeoLife benchmark

The full [GeoLife GPS Trajectories](https://www.microsoft.com/en-my/download/details.aspx?id=52367) dataset contains **24,876,978 position fixes**. On an AMD Ryzen 9 9950X3D (16C/32T, 88 GB RAM), Fastmob detected staypoints in **0.59 s**, versus **99.84 s** for Trackintel: **170× faster**.

The comparison uses the same raw GPS corpus and staypoint thresholds (100 m radius, 20-minute dwell, 15-minute observation gap). The detectors have related but non-identical edge and duplicate handling, so the reproducible benchmark reports both timings and comparable detected-stay counts rather than claiming byte-for-byte identical output.

Run the full benchmark locally to record the exact speedup and hardware for your environment:

```bash
python benchmarks/geolife_staypoints.py \
  --data-dir "/path/to/Geolife Trajectories 1.3" \
  --output benchmarks/results/geolife_staypoints.json
```

See the [end-to-end GeoLife comparison notebook](https://gefgu.github.io/fastmob/learn/notebooks/stay-point-detection-geolife/) for the data preparation, equivalent Trackintel call, and visual output comparison.

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
