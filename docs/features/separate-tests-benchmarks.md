# Separating Correctness Tests from Benchmarks

## Goal

Reorganize the test suite so that:

- Correctness tests live in `tests/correctness/` and run in CI without external dependencies.
- Benchmarks live in `tests/benchmarks/` and use **pytest-benchmark** for proper statistical output.
- The old `benchmarks/` top-level folder is removed; its Brightkite fixture moves into `tests/benchmarks/conftest.py`.

---

## Final directory structure

```
tests/
  correctness/
    conftest.py          # synthetic fixtures, optional skmob comparison fixture
    test_individual.py   # correctness tests for skmob2/measures/individual.py
  benchmarks/
    conftest.py          # Brightkite loader + parametrized-size fixtures
    bench_individual.py  # pytest-benchmark tests for individual measures
benchmarks/              # DELETE — contents migrated to tests/
```

No new top-level package is introduced; both sub-directories are plain pytest directories (no `__init__.py` required unless needed for imports).

---

## Correctness tests (`tests/correctness/`)

### `tests/correctness/conftest.py`

Provides two kinds of fixtures:

1. **Synthetic fixtures** (always available, no network, no optional deps):
   - A small `pd.DataFrame` with 3 users and 5 GPS points each, with hand-computed known-good values stored alongside as constants.
   - Example fixture name: `synthetic_tdf` — returns a plain `pd.DataFrame` with columns `uid`, `datetime`, `lat`, `lng`.
   - Constants like `EXPECTED_JUMP_LENGTHS` map `uid -> list[float]` to the pre-computed ground truth.

2. **Optional skmob comparison fixture** (skipped when `skmob` is absent):
   - A session-scoped `brightkite_skmob` fixture that calls `pytest.importorskip("skmob")` at the top of the fixture body, then loads the Brightkite dataset as a `skmob.TrajDataFrame`.
   - The Brightkite data file path is `tests/benchmarks/data/loc-brightkite_totalCheckins.txt.gz` — shared with the benchmark fixtures so the file is only downloaded once regardless of which suite runs first.

### `tests/correctness/test_individual.py`

Two groups of tests:

**Group 1 — hardcoded synthetic (always runs):**

```python
def test_jump_lengths_known_values(synthetic_tdf):
    from skmob2.measures.individual import jump_lengths
    result = jump_lengths(synthetic_tdf, show_progress=False, merge=False)
    for uid, expected in EXPECTED_JUMP_LENGTHS.items():
        actual = result[result["uid"] == uid]["jump_lengths"].values[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
```

Add one test per measure in `skmob2/measures/individual.py`. Each test:
- Uses only `synthetic_tdf` (or a variant of it).
- Has no import of `skmob`, `polars`, or any optional dep.
- Asserts exact numeric output against pre-computed constants.

**Group 2 — skmob comparison (skipped when skmob absent):**

```python
@pytest.mark.skmob
def test_jump_lengths_matches_skmob(brightkite_skmob):
    from skmob2.measures.individual import jump_lengths as skmob2_jl
    from skmob.measures.individual import jump_lengths as skmob_jl
    ...
    assert np.allclose(skmob2_result, skmob_result, rtol=1e-5, atol=1e-5)
```

The `@pytest.mark.skmob` marker is registered in `tests/correctness/conftest.py` via `pytest_configure`. Tests in this group are automatically skipped (via `pytest.importorskip` inside the `brightkite_skmob` fixture) when `skmob` is not installed — no explicit `skipif` needed at the call site.

---

## Benchmarks (`tests/benchmarks/`)

### `tests/benchmarks/conftest.py`

Responsibilities:
1. **Brightkite download + cache** — identical logic to the current `benchmarks/conftest.py`, but the data directory is `tests/benchmarks/data/`. Session-scoped.
2. **Parametrized size fixtures** — expose a `dataset_size` fixture parametrized over `[1_000, 10_000, 100_000]` rows. A derived `brightkite_slice` fixture combines `brightkite_data` (full 100 k rows loaded once) with `dataset_size` to return a sliced copy.

```python
@pytest.fixture(scope="session")
def brightkite_raw():
    """Download/cache Brightkite; return full 100k-row DataFrame (plain pandas)."""
    ...  # same download logic as current benchmarks/conftest.py
    return df  # plain pd.DataFrame, not TrajDataFrame

@pytest.fixture(scope="session")
def brightkite_tdf(brightkite_raw):
    """Wrap full dataset as skmob.TrajDataFrame (skipped if skmob absent)."""
    skmob = pytest.importorskip("skmob")
    return skmob.TrajDataFrame(brightkite_raw, ...)

@pytest.fixture(params=[1_000, 10_000, 100_000], ids=["1k", "10k", "100k"])
def dataset_size(request):
    return request.param

@pytest.fixture
def bench_slice_pandas(brightkite_raw, dataset_size):
    return brightkite_raw.head(dataset_size).copy()

@pytest.fixture
def bench_slice_skmob(brightkite_tdf, dataset_size):
    skmob = pytest.importorskip("skmob")
    return skmob.TrajDataFrame(brightkite_tdf.head(dataset_size).copy(), ...)
```

### `tests/benchmarks/bench_individual.py`

One benchmark function per (measure, library) combination, parametrized over dataset size via the `bench_slice_*` fixtures. **No correctness assertions** — benchmarks only measure throughput.

```python
import pytest
from skmob2.measures.individual import jump_lengths as skmob2_jl

def test_jump_lengths_skmob2_pandas(benchmark, bench_slice_pandas):
    benchmark(skmob2_jl, bench_slice_pandas, show_progress=False, merge=False)

def test_jump_lengths_skmob(benchmark, bench_slice_skmob):
    skmob_jl = pytest.importorskip("skmob.measures.individual").jump_lengths
    benchmark(skmob_jl, bench_slice_skmob, show_progress=False, merge=False)
```

Key points:
- `benchmark(func, arg, **kwargs)` — pytest-benchmark handles repetition, warm-up, min/max/mean/stddev, and outlier detection automatically.
- skmob2 and skmob appear as separate rows in the benchmark table, making side-by-side comparison natural.
- `--benchmark-json=results.json` or `--benchmark-save=<name>` can be passed on the command line for persistence; no code changes needed.
- A `bench_individual_polars` variant can be added later by introducing a `bench_slice_polars` fixture in conftest.

---

## `pyproject.toml` changes

Add an optional-dependencies group for development:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-benchmark",
    "skmob",
    "polars",
    "tqdm",
]
```

Install with:

```bash
pip install -e ".[dev]"
```

`skmob` is listed here as a soft dependency: tests that need it call `pytest.importorskip("skmob")` and are skipped gracefully when it is absent. It is **not** added to `[project.dependencies]`.

---

## Running the suites

```bash
# Correctness only (fast, no network, no optional deps required)
pytest tests/correctness/

# Correctness including skmob comparison (requires skmob installed)
pytest tests/correctness/ -m skmob

# All correctness except skmob comparison
pytest tests/correctness/ -m "not skmob"

# Benchmarks — prints table to stdout
pytest tests/benchmarks/ -v

# Benchmarks — save JSON report
pytest tests/benchmarks/ --benchmark-json=results.json

# Benchmarks — save named snapshot for later comparison
pytest tests/benchmarks/ --benchmark-save=baseline

# Compare two snapshots
pytest-benchmark compare baseline 0001
```

---

## Migration steps (ordered)

### Step 1 — Create `tests/correctness/conftest.py`

- Define `synthetic_tdf` fixture: 3 users, 5 points each, fixed lat/lng values chosen so haversine distances are easy to hand-compute.
- Define `EXPECTED_JUMP_LENGTHS` dict at module level.
- Register the `skmob` marker: `config.addinivalue_line("markers", "skmob: requires skmob package")`.
- Define `brightkite_skmob` session fixture with `pytest.importorskip("skmob")` guard.

### Step 2 — Create `tests/correctness/test_individual.py`

- Port the correctness assertions from `benchmarks/test_jump_lengths.py` (`_normalize_result` logic) into the new synthetic fixture tests.
- Add `@pytest.mark.skmob` tests that call the `brightkite_skmob` fixture and compare against skmob output.
- Remove all timing/memory code — correctness tests assert values only.

### Step 3 — Create `tests/benchmarks/conftest.py`

- Move the `brightkite_data` fixture from `benchmarks/conftest.py` verbatim, adjusting the data path to `tests/benchmarks/data/`.
- Add `brightkite_raw` (plain DataFrame), `brightkite_tdf` (TrajDataFrame, guarded with importorskip), `dataset_size` parametrized fixture, `bench_slice_pandas`, and `bench_slice_skmob`.

### Step 4 — Create `tests/benchmarks/bench_individual.py`

- Write `test_jump_lengths_skmob2_pandas` and `test_jump_lengths_skmob` using the `benchmark` fixture.
- Remove the `_benchmark_call`, `_normalize_result`, and timing-print logic — pytest-benchmark replaces all of that.

### Step 5 — Update `pyproject.toml`

- Add `[project.optional-dependencies]` with the `dev` group as shown above.

### Step 6 — Delete `benchmarks/`

- Move the cached data file if it exists:
  ```bash
  mkdir -p tests/benchmarks/data
  mv benchmarks/data/loc-brightkite_totalCheckins.txt.gz tests/benchmarks/data/ 2>/dev/null || true
  rm -rf benchmarks/
  ```
- The data directory path in the new `conftest.py` (`tests/benchmarks/data/`) will find the moved file and skip the download.

### Step 7 — Update `CLAUDE.md`

Replace the "Tests / benchmarks" section with:

```markdown
## Tests

```bash
# Correctness (fast, no network)
pytest tests/correctness/

# Correctness + skmob comparison (requires skmob)
pytest tests/correctness/ -m skmob
```

## Benchmarks

```bash
# Run with printed table
pytest tests/benchmarks/ -v

# Save results to JSON
pytest tests/benchmarks/ --benchmark-json=results.json
```

The `brightkite_data` session fixture downloads the Brightkite check-in dataset (~100 k rows) on first run and caches it to `tests/benchmarks/data/`. Subsequent runs reuse the cached file.
```

---

## Notes on future measures

When adding a new measure (e.g., `radius_of_gyration`):

1. Add the Rust kernel and Python wrapper as described in CLAUDE.md.
2. In `tests/correctness/conftest.py`: add a new synthetic fixture or reuse `synthetic_tdf`, add the expected-values constant.
3. In `tests/correctness/test_individual.py`: add `test_<measure>_known_values` and `test_<measure>_matches_skmob`.
4. In `tests/benchmarks/bench_individual.py`: add `test_<measure>_skmob2_pandas` and `test_<measure>_skmob`.

No changes to `pyproject.toml` or conftest files needed for each new measure.
