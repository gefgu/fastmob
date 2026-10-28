<p align="center">
  <img src="docs/src/assets/logo_filled.svg" alt="Fastkit-Mobility logo" width="280">
</p>

# Fastkit-Mobility

[![PyPI version](https://img.shields.io/pypi/v/fastmob)](https://pypi.org/project/fastmob/)
[![Total downloads](https://static.pepy.tech/badge/fastmob)](https://pepy.tech/projects/fastmob)
[![Python versions](https://img.shields.io/pypi/pyversions/fastmob)](https://pypi.org/project/fastmob/)
[![License](https://img.shields.io/pypi/l/fastmob)](LICENSE)

**Fastkit-Mobility** helps researchers turn location data into insights about how
people move. Clean GPS traces, find stops and trips, measure travel patterns,
and simulate mobility using familiar Python dataframes. Its Rust engine handles
compute-intensive work while you work with pandas or Polars.

[Get started](https://gefgu.github.io/fastmob/learn/) · [Explore the examples](https://gefgu.github.io/fastmob/learn/) · [API reference](https://gefgu.github.io/fastmob/reference/) · [Benchmarks](https://gefgu.github.io/fastmob/features/benchmarks/)


<p align="center">
  <img src="docs/src/assets/images/home/home-geolife-staypoints-triplegs.png" alt="GPS fixes, detected stops and trips for one day of a GeoLife participant in Beijing" width="650">
</p>

One day of GeoLife GPS data: grey points are GPS fixes, gold circles are detected
stops (staypoints), and coloured lines show movement between them (triplegs). Follow the [complete GeoLife example](https://gefgu.github.io/fastmob/learn/notebooks/stay-point-detection-geolife/).

## Try it

Install Fastkit-Mobility and pandas for this example (Python 3.9 or newer):

```bash
pip install fastmob pandas
```

Calculate each person's **radius of gyration**: a measure, in kilometres, of
how widely their observed locations spread around their centre of movement.

```python
import pandas as pd
from fastmob import radius_of_gyration

df = pd.DataFrame({
    "uid": ["alice"] * 3 + ["bob"] * 3,
    "datetime": pd.to_datetime(["2020-01-01 08:00", "2020-01-01 09:00", "2020-01-01 10:00"] * 2),
    "lat": [41.9, 42.0, 42.1, 40.7, 40.8, 41.0],
    "lng": [12.5, 12.5, 12.6, -74.0, -74.0, -73.9],
})

print(radius_of_gyration(df))
```

```text
     uid  radius_of_gyration
0  alice            9.878827
1    bob           14.423721
```

In this small example, Bob's observed locations are more spread out than
Alice's. Explore more [individual and collective measures](https://gefgu.github.io/fastmob/learn/individual-and-collective-measures/).

## What can you study?

- **Clean and segment traces:** filter noisy GPS data and separate movement from stops. [Learn how](https://gefgu.github.io/fastmob/learn/clean-and-segment/).
- **Find stops and recurring places:** identify where people spend time and group repeated visits to locations. [Learn how](https://gefgu.github.io/fastmob/learn/staypoints-and-locations/).
- **Measure movement:** study individual ranges and travel patterns, or flows across a population. [Explore measures](https://gefgu.github.io/fastmob/learn/individual-and-collective-measures/).
- **Work with transport networks:** route and match GPS traces to road and rail networks. [Explore network analysis](https://gefgu.github.io/fastmob/learn/network-analysis/).
- **Model mobility:** generate synthetic trajectories or flows and compare models with observations. [Explore models](https://gefgu.github.io/fastmob/reference/models/).


## Performance and compatibility

Fastkit-Mobility uses Rust for compute-intensive operations and supports pandas,
Polars and other eager dataframes through Narwhals. Many measure names and
signatures follow [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility).

<p align="center">
  <img src="docs/src/assets/images/home/home-benchmark-speedup.png" alt="Execution times for five mobility measures: Fastkit-Mobility with pandas compared with scikit-mobility on approximately four million Brightkite check-ins" width="800">
</p>

Average execution time for five measures on approximately 4 million time-sorted
Brightkite check-ins. Fastkit-Mobility uses pandas; the comparison is with
scikit-mobility. Times are averages of three runs on an Intel Core i7-10700 with
Python 3.12, and the chart uses a logarithmic scale. The scikit-mobility input
contained 3,999,891 rows and the Fastkit-Mobility input 4,000,000. Results depend
on the operation, data and environment. See the [raw benchmark record](https://github.com/gefgu/fastmob_benchmarks/blob/main/benchmarks/measures/results/py312_intel_core_i7_10700_16c_brightkite4M/measures_benchmarks.json)
and [more comparisons](https://gefgu.github.io/fastmob/features/benchmarks/).

## More resources

- [Guides and tutorials](https://gefgu.github.io/fastmob/learn/) · [API reference](https://gefgu.github.io/fastmob/reference/) · [Release notes](https://gefgu.github.io/fastmob/release_notes/)
- [Contributing and development setup](docs/CONTRIBUTING.md)
- [Report an issue](https://github.com/gefgu/fastmob/issues): include package versions, dataframe backend, a minimal example, expected behavior and the full error.