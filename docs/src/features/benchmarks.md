# Benchmark Results

These plots compare Fastkit-Mobility with every library that offers the same function. Bars show average time (or peak extra memory) on a log axis, and the label on a Fastkit-Mobility bar is its advantage over the best other library. A bar reading *timeout* or *error* means the library did not finish; those operations are kept, not hidden.

The numbers are environment-specific, so read them as benchmark evidence for these runs rather than as a universal performance guarantee.

## Summary

| Suite | Compared with | Data | Hardware | Headline result |
|---|---|---|---|---|
| [Individual and collective measures](#measures) | scikit-mobility | 4M Brightkite check-ins | Core i7-10700 | individual measures about 190x to 2,700x faster; two scikit-mobility measures time out; slower than scikit-mobility on `visits_per_location` with pandas (1.28 s against 1.18 s); `visits_per_time_unit` only 2.6x faster |
| [Preprocessing](#preprocessing) | scikit-mobility, MovingPandas, PTRAIL, MoveTK | 4M GeoLife points | Xeon Silver 4316 | see figure; MovingPandas times out on Douglas-Peucker simplification |
| [Preprocessing, full GeoLife](#preprocessing-on-full-geolife) | scikit-mobility, MovingPandas, PTRAIL | full GeoLife, 24.9M points | Ryzen 9 9950X3D | 40x to 2,800x faster where a competitor finishes; MovingPandas times out on seven operations |
| [MoveTK C++ suites](#movetk-c-outliers-and-simplification) | MoveTK | 4M GeoLife points | i7-10700, Ryzen 9 9950X3D, Xeon Silver 4316 | Chan-Chin and Imai-Iri cost 16 to 31 s in MoveTK, with no Fastkit-Mobility counterpart |
| [Staypoints and locations](#staypoints-and-locations) | trackintel | full GeoLife, 24.9M points | Core i7-10700 | staypoints 0.86 s against 316 s |
| [Next-location model](#next-location-model) | HuMobi | YJMob100K, 111.5M rows | Ryzen 9 9950X3D | fit 5-6x and predict 39-46x faster on YJMob100K; on Brightkite the pandas fit is only 1.7x faster |
| [Single function, every library](#one-function-every-library) | eight other libraries | 4M Brightkite check-ins | Core i7-10700 | `jump_lengths`; MoveTK ties at small sizes |
| [Generative models](#generative-models) | scikit-mobility | 1,000 to 10,000 locations, up to 10,000 agents | Core i7-10700 | EPR variants 70x to 180x faster; Gravity 1.5x to 1.7x and Radiation 2.5x to 3x faster; GeoSim, STS-EPR and Markov Diary 1.5x to 6x faster; Gravity uses more memory at 10,000 locations |
| [Across CPUs](#across-cpus) | trackintel (staypoints) | Brightkite, GeoLife, YJMob100K | i7-10700, Ryzen 9 9950X3D | same ordering on both CPUs; absolute times differ |
| [Road network](#road-network) | TransBigData, stplanr with OSRM | GeoLife Beijing | Core i7-10700 | much faster distances, routes and CH preparation; slower fetch-and-extract build |
| [Privacy risk](#privacy) | scikit-mobility | toy trajectories repeated 30x | Core i7-10700 | see figure |
| [Distributed baseline](#distributed-baseline-sparkmobility) | SparkMobility | full GeoLife, 24.9M points | Xeon Silver 4316 | see figure |
| [Scaling without a competitor](#scaling-fastkit-mobility-only) | none | YJMob100K up to 111.5M rows | Core i7-10700 | motifs, mobility laws, social recast |

## Setup

| Item | Value |
|---|---|
| Hardware | Intel Core i7-10700 (16 threads) unless stated; Intel Xeon Silver 4316 (40 threads) for preprocessing and SparkMobility; AMD Ryzen 9 9950X3D for the next-location suite. A single figure never mixes machines. Most suites were also run on the Ryzen and the Xeon, and those results are in the raw repository. |
| Python | 3.12 for Fastkit-Mobility (3.13 on the Xeon preprocessing run). Each competitor runs in its own environment: scikit-mobility, for instance, ran on 3.10 in the privacy suite and 3.11 elsewhere. |
| Timing | Average wall-clock time of repeated runs, with data loading excluded. The `jump_lengths` competitors are timed by `hyperfine` with the interpreter start-up and import baseline subtracted. |
| Memory | Peak resident memory above a load-only baseline. A bar labelled `<0.1 MB` means the operation stayed below what the method can resolve, not that it used no memory. |
| Input order | The measures figures use time-sorted input; preprocessing and SparkMobility use raw (unsorted) input. |
| Timeouts | 900 s for measures (3,600 s for `uncorrelated_location_entropy`). Preprocessing used 900 s per case (10,800 s for the full-GeoLife run); MovingPandas was rerun with 32,000 s and its Douglas-Peucker simplification still exceeded it. No SparkMobility case timed out. |
| Datasets | Brightkite (about 4M check-ins; scikit-mobility received 3,999,891 rows, Fastkit-Mobility 4,000,000), GeoLife (24.9M points in full) and YJMob100K (111.5M rows in full). |

Memory ratios are shown only when Fastkit-Mobility uses at least 5 MB, because smaller values sit inside the measurement noise. Raw results, scripts and per-library environments are in the [`fastmob_benchmarks`](https://github.com/gefgu/fastmob_benchmarks) repository.

## One function, every library

`jump_lengths` is the function the most libraries share. Competitors are timed by `hyperfine` in a separate process, with the start-up cost subtracted, while Fastkit-Mobility is timed in-process, so treat this plot as a ranking rather than an exact ratio. Differences below about 0.02 s are within measurement noise: MoveTK and Fastkit-Mobility are effectively tied at 1k and 10k rows, and the gap opens with the data size. The scikit-mobility bar here (42.8 s) differs from the measures figure (79.2 s) because the two suites time the work differently.

![jump_lengths speed across nine libraries](../assets/benchmarks/jump-lengths-libraries.png)

## Measures

### Individual

![Individual measures, execution time](../assets/benchmarks/measures-individual-speed.png)

![Individual measures, peak extra memory](../assets/benchmarks/measures-individual-memory.png)

### Collective

![Collective measures, execution time](../assets/benchmarks/measures-collective-speed.png)

![Collective measures, peak extra memory](../assets/benchmarks/measures-collective-memory.png)

## Preprocessing

MoveTK appears for operations it implements. Its Douglas-Peucker time (1.97 s) is its total minus its own file-parsing baseline and is not measured on the same boundary as the Fastkit-Mobility bars, so it is not a like-for-like win. MovingPandas timed out on that operation. Fastkit-Mobility times come from the 2026-08 table record (`preprocessing_benchmarks.json`); later per-case checkpoints from an unreleased development build are not used.

![Preprocessing, execution time](../assets/benchmarks/preprocessing-speed.png)

![Preprocessing, peak extra memory](../assets/benchmarks/preprocessing-memory.png)

## Preprocessing on full GeoLife

The same operations (apart from `cluster`) on the whole GeoLife dataset (24.9M points, six times the 4M slice above) on the Ryzen 9 9950X3D, with raw input order. Fastkit-Mobility finishes every plotted operation in about one to two seconds. `cluster` is not plotted because scikit-mobility errored on it at this size (Fastkit-Mobility took 15.7 s with pandas and 1.8 s with Polars). MovingPandas exceeded the 10,800 s budget on seven operations (Douglas-Peucker, top-down time ratio and max-distance simplification, the angle-change, stop and value-change segmenters, and the Kalman smoother), shown as *timeout*. Memory is not plotted: MovingPandas, PTRAIL and the pandas backend report near-zero deltas while Polars and scikit-mobility report gigabytes, which reflects how each is measured rather than a real ratio, so the figure would mislead.

![Preprocessing on full GeoLife, execution time](../assets/benchmarks/preprocessing-full-speed.png)

## MoveTK (C++): outliers and simplification

MoveTK is a C++ library, so these suites run its compiled example programs and measure them with GNU `time`, averaged over three runs on a 4M-row GeoLife slice (simplification uses coordinates projected to local planar kilometres, with a 0.05 km tolerance). Each MoveTK time includes parsing the CSV, so a parse-only baseline is listed; the preprocessing figure above uses the net value (total minus baseline) for MoveTK. `smart_greedy` is excluded because MoveTK did not finish it at 100k rows within five minutes. Chan-Chin and Imai-Iri have no Fastkit-Mobility counterpart and are close to each other because MoveTK's Imai-Iri uses Chan-Chin to create edges.

| Seconds (total, including parsing) | Core i7-10700 | Ryzen 9 9950X3D | Xeon Silver 4316 |
|---|---:|---:|---:|
| Outliers: parse-only baseline | 2.08 | 1.03 | 3.23 |
| Outliers: greedy | 4.43 | 2.30 | 8.11 |
| Outliers: Zheng | 4.67 | 2.80 | 7.05 |
| Simplification: parse-only baseline | 2.17 | 1.10 | 3.33 |
| Simplification: Douglas-Peucker | 2.95 | 1.90 | 5.29 |
| Simplification: Chan-Chin | 16.09 | 18.01 | 30.93 |
| Simplification: Imai-Iri | 16.34 | 18.07 | 29.40 |

For comparison, reading the outliers CSV takes 1.33 s (pandas) and 0.22 s (Polars) on the i7, 0.80 s and 0.20 s on the Ryzen, and 2.41 s and 0.36 s on the Xeon; the simplification CSV takes 1.39 s and 0.21 s, 0.83 s and 0.20 s, and 2.02 s and 0.27 s. On the Xeon, Fastkit-Mobility's algorithm-only times on the preprocessing input are 1.42 s (pandas) and 1.12 s (Polars) for greedy outliers, 1.36 s and 1.12 s for Zheng, and 3.25 s and 2.74 s for Douglas-Peucker. These are measured on a wider file and after loading, so they are not additive with the table; treat them as a scale reference, not an exact ratio. MoveTK's single-threaded programs run faster on the CPUs with the higher per-core clock, which is why the 16-thread i7 beats the 40-thread Xeon.

## Staypoints and locations

Compared with trackintel on the full GeoLife dataset (single runs). Only stages that produce comparable results are shown. Fastkit-Mobility's tripleg generation does not yet split on tracking gaps (388 triplegs against trackintel's 4,146), so triplegs, trips and tours are left out of the comparison.

![Staypoints and locations against trackintel, execution time](../assets/benchmarks/hierarchy-trackintel-speed.png)

## Next-location model

`NextLocationPredictor` compared with HuMobi's Markov chain. These runs were recorded on the Ryzen host and the absolute times may carry some disk contention. Fastkit-Mobility predicts the most frequent next location, while HuMobi draws from the transition frequencies, so the predicted locations are not expected to match one by one.

![Next-location fit and predict on YJMob100K](../assets/benchmarks/next-location-yjmob-speed.png)

![Next-location fit and predict on Brightkite](../assets/benchmarks/next-location-brightkite-speed.png)

## Generative models

Model generation compared with scikit-mobility. These speeds come from the i7-10700 runs in the benchmark repository, which merge Fastkit-Mobility results recorded under Python 3.11 and 3.12 (scikit-mobility ran on Python 3.10); the same values appear in its Ryzen table, so they are not an independent Ryzen measurement. DensityEPR and SpatialEPR at 10,000 agents are not plotted because neither library was measured there on the i7. Memory is not plotted: several scikit-mobility memory cells were never captured. Where both were measured, Gravity at 10,000 locations uses more memory in Fastkit-Mobility (24,474 MB) than in scikit-mobility (22,543 MB), while Radiation uses less (12,224 MB against 16,284 MB).

![Generative models, execution time](../assets/benchmarks/models-speed.png)

## Across CPUs

The suites were repeated on other machines, and no figure mixes them. The tables put the same Fastkit-Mobility workloads side by side as average seconds. The Ryzen 9 9950X3D is faster than the i7-10700 in every row, and on staypoints Fastkit-Mobility is ahead of trackintel on both.

**Individual measures, 4M Brightkite check-ins (time-sorted)**

| Measure | Backend | Core i7-10700 | Ryzen 9 9950X3D |
|---|---|---:|---:|
| `radius_of_gyration` | pandas | 0.100 | 0.058 |
| `radius_of_gyration` | Polars | 0.088 | 0.056 |
| `jump_lengths` | pandas | 0.187 | 0.063 |
| `jump_lengths` | Polars | 0.135 | 0.054 |
| `number_of_locations` | pandas | 0.142 | 0.041 |
| `number_of_locations` | Polars | 0.084 | 0.040 |
| `home_location` | pandas | 0.225 | 0.140 |
| `home_location` | Polars | 0.104 | 0.056 |

**Staypoint generation, full GeoLife (24.9M points; trackintel is a single run)**

| Library | Core i7-10700 | Ryzen 9 9950X3D |
|---|---:|---:|
| Fastkit-Mobility (pandas) | 1.13 | 0.59 |
| Fastkit-Mobility (Polars) | 0.86 | 0.47 |
| trackintel | 316 | 99.8 |

**Daily motifs, full YJMob100K (111.5M rows)**

| Backend | Core i7-10700 | Ryzen 9 9950X3D |
|---|---:|---:|
| pandas | 6.08 | 3.95 |
| Polars | 1.21 | 0.46 |

The Xeon Silver 4316 results for the motif suite are not shown: they come from a different stack (Python 3.13), the Polars run timed out on the full dataset (1,800 s), and run-to-run variance was high.

## Road network

Compared with TransBigData and with stplanr using a local OSRM server, on GeoLife Beijing (Core i7-10700). This is mixed evidence: building a road network by fetching and extracting it is slower in Fastkit-Mobility, but the two are not like for like: Fastkit-Mobility fetches live data from Overture on S3, while stplanr's `osrm-extract` reads a local file. The comparable preparation step (CH preparation) favours Fastkit-Mobility. Values below are seconds from single runs; cells marked n/a mean the library has no matching operation.

| Operation | Scale | Fastkit-Mobility | TransBigData | stplanr (OSRM) |
|---|---|---:|---:|---:|
| Build road network (fetch/extract) | 1 km² | 91.3 | n/a | 4.07 |
| Build road network (fetch/extract) | 25 km² | 136.2 | n/a | 0.36 |
| Build road network (fetch/extract) | 225 km² | 125.8 | n/a | 0.77 |
| Network distances | 10,000 OD pairs | 0.005 | 84.4 | timeout |
| Route geometry | 1,000 queries | 0.03 | 8.44 | 1.36 |
| Route geometry | 10,000 queries | 0.01 | 83.4 | timeout |
| OD desire lines | 1,000 OD pairs | 0.004 | n/a | 2.28 |
| OD desire lines | 10,000 OD pairs | 0.01 | n/a | timeout |
| Build road network (CH preparation) | 1 km² | 0.01 | n/a | 0.36 |
| Build road network (CH preparation) | 225 km² | 0.24 | n/a | 5.02 |

The stplanr timeouts at 10,000 queries come from a reproducible hang of the local OSRM server, not from stplanr itself. Larger TransBigData cases were extrapolated rather than measured and are left out.

## Privacy

Privacy-risk metrics on toy trajectories repeated 30 times, with scikit-mobility given a prebuilt trajectory dataframe.

![Privacy risk, execution time](../assets/benchmarks/privacy-speed.png)

## Distributed baseline: SparkMobility

SparkMobility runs in local mode on one host and includes Parquet input and output. Its downstream operations ran on 283,168 rows against 67,456 for Fastkit-Mobility, and the two systems implement different algorithms for some operations. Read these plots as a scale reference, not a like-for-like contest.

![Execution time against SparkMobility](../assets/benchmarks/sparkmobility-speed.png)

![Peak extra memory against SparkMobility](../assets/benchmarks/sparkmobility-memory.png)

## Scaling (Fastkit-Mobility only)

These functions have no comparable implementation in the other libraries, so the tables show how Fastkit-Mobility scales on YJMob100K. Each cell is seconds (and peak extra megabytes, where recorded); the Core i7-10700 was used.

**Daily motifs** (seconds / MB)

| Rows | pandas | Polars |
|---:|---:|---:|
| 100k | 0.0093 / 1.3 | 0.0028 / 0.5 |
| 1M | 0.0604 / 12.9 | 0.0120 / 5.3 |
| 10M | 0.546 / 130.6 | 0.101 / 54.3 |
| 100M | 5.48 / 1,326.7 | 1.13 / 563.6 |
| full (111.5M) | 6.08 / 1,484.2 | 1.21 / 633.2 |

**Mobility laws, full dataset** (average seconds; the truncated power law uses the 87.4M rows that qualify)

| Fit | pandas | Polars |
|---|---:|---:|
| daily location lognormal | 5.96 | 6.09 |
| truncated power law | 6.88 | 6.74 |
| visitation law | 58.0 | 44.9 |

**Social recast** (seconds): 0.029 s at 100k rows, 0.40 s at 1M and 72.6 s at 10M with pandas (73.2 s with Polars). The 100M tier was not run because memory use grows with the encounter count.

## Reproduce the figures

The figures are drawn from the raw results of the sibling `fastmob_benchmarks` repository:

```bash
uv run --no-project --with matplotlib python scripts/make_benchmark_figures.py
```

Each figure is one entry of `figures()` in that script, so a new comparison needs one new entry.
