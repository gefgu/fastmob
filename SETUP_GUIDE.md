# skmob2 Benchmark Setup Guide

## Problem Solved

The original environment had **Python version incompatibility** issues:
- Python 3.12 had wheels built with newer libraries (Shapely 2.0+)
- scikit-mobility still uses `shapely.ops.cascaded_union` which was **removed in Shapely 2.0**
- This caused import failures when trying to run the original skmob for benchmarking

## Solution

Recreate the virtual environment with **Python 3.10** (compatible with the library versions from 2020-2021):

### Initial Setup (First Time)

```bash
# 1. Remove the old venv with Python 3.12
rm -rf .venv

# 2. Create a fresh venv with Python 3.10
python3.10 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Run the setup script
bash setup.env
```

### After Setup (Normal Usage)

```bash
# Just activate and run benchmarks
source .venv/bin/activate
bash tests/run_benchmarks.sh
```

## What setup.env Does

The `setup.env` script:
1. ✓ Installs skmob2 development dependencies (pytest, pytest-benchmark, polars)
2. ✓ Builds the Rust extension (`maturin develop`)
3. ✓ Installs Shapely 1.8.5 (compatible with skmob's cascaded_union)
4. ✓ Installs scikit-mobility 1.3.1
5. ✓ Ensures numpy/pandas ABI compatibility
6. ✓ Verifies both skmob and skmob2 can be imported

## Benchmark Results

Run the benchmarks to compare skmob2 vs original skmob:

```bash
# Run all dataset sizes (1k, 10k, 100k rows)
bash tests/run_benchmarks.sh

# Run only 1k rows (fast)
bash tests/run_benchmarks.sh -k 1k

# Save results to JSON
bash tests/run_benchmarks.sh --benchmark-json=results.json
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
→ You're using Python 3.12+. Use Python 3.10 instead.

**"AttributeError: module 'shapely' has no attribute 'geos_version'"**
→ Shapely 2.0+ is installed. Run: `pip install 'shapely<2.0'`

**"ValueError: numpy.dtype size changed"**
→ Numpy/pandas ABI mismatch. Run: `pip install 'numpy<2.0' 'pandas>=1.5.3,<2.0'`

**Check your Python version:**
```bash
python --version  # Should be 3.10.x
which python      # Should be in .venv/bin/python
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
