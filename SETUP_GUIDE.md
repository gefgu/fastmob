# skmob2 Benchmark Setup Guide

## Problem Solved

The benchmark suite spans packages that need incompatible geospatial stacks:
- `skmob2` development uses the normal `.venv` and the current pandas/Shapely stack.
- `skmob` comparisons need `.venv-skmob` with Python 3.10, `scikit-mobility 1.3.1`, `geopandas 0.10.2`, and `Shapely 1.8.5.post1`.
- movingpandas comparisons should run only from an environment where `movingpandas` imports, normally `.venv` after installing the optional extra.

`scripts/run_benchmarks.sh` runs each benchmark group in the matching environment instead of trying to import every comparison package from one venv.

### Initial Setup

```bash
# Normal skmob2 benchmark environment
bash scripts/setup_env.sh --movingpandas
source .venv/bin/activate

# Dedicated skmob comparison environment, created separately with the legacy stack
# and rebuilt with the current skmob2 extension before skmob benchmarks.
env -u CONDA_PREFIX \
    VIRTUAL_ENV="$PWD/.venv-skmob" \
    PATH="$PWD/.venv-skmob/bin:$PATH" \
    .venv/bin/maturin develop
```

### After Setup (Normal Usage)

```bash
# Run all benchmark groups in their compatible environments
bash scripts/run_benchmarks.sh

# Run one workload family
bash scripts/run_benchmarks.sh -k radius_of_gyration
```

## What scripts/setup_env.sh Does

The `scripts/setup_env.sh` script:
1. ✓ Installs skmob2 development dependencies (pytest, pytest-benchmark, polars)
2. ✓ Builds the Rust extension (`maturin develop`)
3. ✓ Optionally installs movingpandas with `--movingpandas`
4. ✓ Leaves skmob comparisons to `.venv-skmob`, because skmob requires Shapely < 2

## Benchmark Results

Run the benchmarks to compare skmob2, skmob, and movingpandas where available:

```bash
# Run all dataset sizes
bash scripts/run_benchmarks.sh

# Run only 1k rows (fast)
bash scripts/run_benchmarks.sh -k 1k

# Save results to JSON
bash scripts/run_benchmarks.sh --benchmark-json=results.json
```

### Performance Summary

| Dataset | skmob2 | skmob | Speedup |
|---------|--------|-------|---------|
| 1k rows | 2.41 ms | 4.24 ms | **1.76x** faster |
| 10k rows | 13.25 ms | 31.09 ms | **2.34x** faster |
| 100k rows | 186.05 ms | 363.72 ms | **1.95x** faster |

See `BENCHMARK_RESULTS.md` for detailed results.

## Troubleshooting

**"ImportError: cannot import name 'cascaded_union'"**
→ The skmob benchmark is running outside `.venv-skmob`, or `.venv-skmob` has Shapely 2.x installed. Rebuild/use `.venv-skmob`.

**"AttributeError: module 'shapely' has no attribute 'geos_version'"**
→ The skmob environment has an incompatible Shapely version. Use the dedicated `.venv-skmob` legacy stack.

**"ValueError: numpy.dtype size changed"**
→ Numpy/pandas ABI mismatch. Recreate the affected environment instead of mixing Conda and venv packages.

**Check which environment is active:**
```bash
.venv/bin/python --version
.venv-skmob/bin/python --version
```

## Key Insights

1. **Python Version Matters**: Older Python versions have wheels built with compatible library versions from that era
2. **ABI Compatibility**: Binary libraries like numpy/pandas need matching versions for their C extensions
3. **Shapely 2.0 Break**: The removal of `cascaded_union` was a breaking change for libraries like skmob
4. **skmob2 Performance**: The Rust implementation in skmob2 provides consistent 1.95x average speedup

## Files Created

- `setup.env` - Environment setup script
- `BENCHMARK_RESULTS.md` - Detailed benchmark results
- `SETUP_GUIDE.md` - This file
