---
name: skmob2 refactoring history
description: Completed refactors, deferred items, and known code smells with file+line references
type: project
---

## Completed (2026-04-17)

### R1: Centralize column candidate lists and add shared trajectory helpers to `_common.py`
- **Smell:** `jump_lengths.py` and `radius_of_gyration.py` each had identical ~20-line column-detection + validation + sort-column-assembly blocks. `activity.py` and `individual.py` each defined their own local candidate list variables.
- **Change:** Moved all candidate list constants to `_common.py`. Added `_detect_trajectory_columns()` and `_prepare_trajectory()` helpers.
- **Callers updated:** `jump_lengths.py`, `radius_of_gyration.py` now call `_prepare_trajectory`. `activity.py`, `individual.py` import shared candidate lists.
- **New test file:** `tests/correctness/test_common.py` (16 tests covering `_pick_existing_column`, `_detect_trajectory_columns`, `_prepare_trajectory`).

### R2: Fix pandas direct import in `od.py`
- **Smell:** `od.py` had `import pandas as pd` at module level, violating the Narwhals-only convention. Also had a fragile `isinstance(counts.to_native(), pd.DataFrame)` check.
- **Change:** Moved `import pandas as pd` inside each function body (both functions explicitly produce pandas output; this is intentional). Removed the isinstance check — `pd.DataFrame(counts.to_native())` always works.
- **Also:** Improved error messages to include candidate lists.

### R3: Add explicit column override params to `jump_lengths` and `radius_of_gyration`
- **Change:** Both public functions now accept `datetime_col`, `lat_col`, `lng_col`, `uid_col` as keyword-only arguments, passed through to `_prepare_trajectory`. Improves composability and explicitness for users.

### R4: Motif hashing redesign — canonical adjacency-matrix IDs (2026-04-17)
- **Smell:** `format_motif_id_v2` produced `"{n_nodes}_{n_edges}_{order}"` IDs that depended on insertion order into the dynamic library — two structurally identical graphs in different libraries got different IDs. `_is_isomorphic` used brute-force permutation search with no fast rejection.
- **Change:** Added `_degree_sequence` (fast prefilter), `_canonical_adjacency_form` (max-permutation big-endian binary string), and `_motif_id` (returns `"m{n_nodes}:{bits}"`). Updated `_is_isomorphic` to use canonical form comparison. Updated `classify_or_add_motif_v2` to call `_motif_id(user_graph)` directly. Removed `format_motif_id_v2` entirely. Updated `measures/__init__.py` and `skmob2/__init__.py`.
- **Tests updated:** `test_format_motif_id_v2_basic` → replaced with 3 new `_motif_id` tests. `test_classify_motif_1_returns_known_id` → now asserts `"m1:0"`. `test_classify_motif_2_returns_known_id` → now asserts `"m2:0110"`.
- **Key canonical IDs:** single node (motif 1) = `"m1:0"`, simple return (motif 2) = `"m2:0110"`.

### R5: Clean up redundant column candidates in `_common.py` (2026-04-17)
- **Smell:** `TIMESTAMP_CANDIDATES` contained `"local_timestamp"` (project-specific, non-standard). `USER_ID_CANDIDATES` contained `"ID"` (too generic, case-sensitive collision risk).
- **Change:** Removed `"local_timestamp"` from `TIMESTAMP_CANDIDATES`. Removed `"ID"` from `USER_ID_CANDIDATES`.
- **Net effect:** No test failures; only the 5 pre-existing pyarrow failures remain.

### R6: Reorganize measures into domain subfolders (2026-04-17)
- **Smell:** All 7 source files lived flat in `skmob2/measures/` with no grouping — hard to navigate, unclear which files relate to which domain.
- **Change:** Created four domain subfolders under `skmob2/measures/`:
  - `spatial/` — `jump_lengths.py`, `radius_of_gyration.py`
  - `visits/` — `activity.py`, `intermittance.py` (renamed from `individual.py`), `motifs.py`
  - `flows/` — `od.py`
  - `fitting/` — `mobility_laws.py`
- Each subfolder has an `__init__.py` re-exporting its public symbols.
- `skmob2/measures/__init__.py` and `skmob2/__init__.py` updated to import from subfolders.
- Relative imports in each file updated: `from ._common import` → `from .._common import`.
- Test files mirrored into `tests/correctness/spatial/`, `visits/`, `flows/`, `fitting/` with import paths updated accordingly.
- Old `individual.py` → `visits/intermittance.py` (content unchanged; all 6 functions remain in one file).
- Public API surface: unchanged. `from skmob2 import jump_lengths` etc. still works.
- Test result: 131 passed, 5 pre-existing pyarrow failures (unchanged from pre-R6 baseline).

## Migration artifact cleanup (2026-04-17)

### R7: Fix broken skmob2 import paths in mobility_analysis adapters
- **Smell:** All 6 adapter files (`motifs.py`, `individual.py`, `spatial.py`, `activity.py`, `cdr_measures.py`, `mobility_laws.py`) still used pre-R6 flat import paths (`skmob2.measures.od`, `skmob2.measures.individual`, etc.). These were broken by the R6 subfolder reorganization. Also `motifs.py` imported `format_motif_id_v2` from skmob2 which was removed in R4 (the adapter has its own local copy).
- **Change:** Updated all 6 adapter files to use new subfolder paths (`skmob2.measures.visits.*`, `skmob2.measures.flows.od`, `skmob2.measures.spatial.*`, `skmob2.measures.fitting.mobility_laws`). Removed 6 unused skmob2 aliases from `motifs.py` (only `_skmob2_discover_daily_motifs_from_agents` was actually called; the rest were shadowed by local implementations). Removed debug print block from `spatial.py` (lines 469-473 had a `# --- DEBUG: ---` block that printed to stdout).
- **Tests:** skmob2 baseline unchanged — 131 passed, 5 pre-existing pyarrow failures.
- **Note:** The adapter's own `format_motif_id_v2` at `motifs.py:141` is intentional — it's the local `N_E_id` formatter used by `classify_or_add_motif_v2` in that file. Not a migration artifact.

## Completed (2026-04-20) — Post-batch audit fixes

### R8: Fix Narwhals violation — `entropy.py` module-level `import pandas`
- Moved `import pandas as pd` from module level to inside each of `trajectory_entropy` and `trajectory_predictability` with `# noqa: PLC0415` comment.
- Return type annotations updated from `pd.DataFrame` to `Any` (both functions still always return pandas).
- All 21 entropy tests still pass.

### R9: Remove dead variable `n_rows` in `random_location_entropy.py`
- Line 69: `n_rows = len(locs)` was assigned and never used. Removed.

### R10: Extract `_shannon_entropy` to `_common.py`, remove duplication
- `_shannon(counts)` was defined identically in `uncorrelated_entropy.py` and `uncorrelated_location_entropy.py`.
- Moved to `_common.py` as `_shannon_entropy`. Added `import math` to `_common.py`.
- Both files now import `_shannon_entropy` from `_common`. Local definitions removed.
- Added 7 unit tests for `_shannon_entropy` in `test_common.py`.

### R11: `homes_per_location.py` — use `_pick_existing_column` from `_common`
- Removed local `lat_candidates`/`lng_candidates` lists and `next(...)` iteration.
- Now imports `_pick_existing_column, LAT_CANDIDATES, LNG_CANDIDATES` from `_common`.
- Error message updated to include candidate lists.

### R12: O(L×N) per-location filter loop in `uncorrelated_location_entropy.py`
- Replaced per-location `visit_counts.filter(...)` loop (runs a full dataframe scan per location) with a single-pass Python dict accumulation over sorted rows.
- Complexity reduced from O(L×N) to O(N log N). Behavior unchanged.
- Removed unused `import math`.

### R13: Wrong `@pytest.mark.skmob` on scipy-dependent tests in `test_evaluation.py`
- 8 tests for `kullback_leibler_divergence`, `pearson_correlation`, `spearman_correlation` were marked `@pytest.mark.skmob` but require scipy, not skmob.
- Added module-level `requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, ...)` and applied it to those 8 tests.
- Effect: tests now run in the default `pytest -m "not skmob"` suite when scipy is present. Net +8 passing tests.

### R14: `_kontoyiannis_entropy` — O(n²) → O(n) space
- Previous implementation allocated an n×n DP table (800 MB for 10k-point user).
- Rewritten with two rolling 1D arrays (`prev_row`, `curr_row`) plus a `col_max` accumulator.
- Memory reduced from O(n²) to O(n). All entropy/real_entropy tests pass unchanged.

## Completed (2026-04-20) — motifs.py pandas + Rust refactor

### R15: Remove module-level `import pandas` from `motifs.py`; use `_common.py` candidate lists; add Rust canonical-adjacency kernel
- **Smells:**
  1. `import pandas as pd` at module level (line 15) — violates Narwhals-only rule.
  2. Local candidate list variables `_UID_CANDIDATES`, `_LOC_CANDIDATES`, etc. defined inside `discover_daily_motifs_from_agents` — duplicated logic from `_common.py`.
  3. `_canonical_adjacency_form` was pure-Python n! × n² inner loop — compute-heavy for large dynamic libraries.
  4. Fragile `isinstance(native, pd.DataFrame)` using module-level import.
  5. `from itertools import permutations` unused after Rust delegation.
- **Changes:**
  1. Removed module-level `import pandas as pd`. Added `import pandas as pd  # noqa: PLC0415` inside each of the 4 functions that need it: `_compute_primary_home_node_id`, `build_motif_graph`, `_build_daily_motif_records`, `discover_daily_motifs_from_agents`.
  2. Replaced all 6 local candidate list variables with imports from `_common.py`.
  3. Added `canonical_adjacency_form` Rust kernel in `src/lib.rs`.
- **Tests:** 275 passed, 5 pre-existing pyarrow failures (baseline unchanged). All 29 motifs tests pass.
- **Note:** Row-iterative pandas logic was still inside function bodies — see R16.

### R16: Fully eliminate `import pandas` from `motifs.py` — replace with Narwhals (2026-04-20)
- **Smells:** 4 remaining `import pandas as pd` inside function bodies (`_compute_primary_home_node_id`, `build_motif_graph`, `_build_daily_motif_records`, `discover_daily_motifs_from_agents`). Per project constraint, `import pandas` is forbidden inside `skmob2/` — even inside function bodies.
- **Changes:**
  1. `_compute_primary_home_node_id`: now takes `nw.DataFrame`. Replaced pandas filter/copy/dt.hour/groupby/idxmax/value_counts with `nw.filter`, `nw.with_columns`, `nw.col.dt.hour()`, `nw.group_by().agg()`, `.sort(descending=True).row(0)`.
  2. `build_motif_graph`: eliminated `pd.DataFrame` prefix-row construction and `pd.concat`. The prefix node is now prepended directly as a Python string to the extracted `sequence` list — no intermediate DataFrame needed.
  3. `_build_daily_motif_records`: takes `nw.DataFrame`. Replaced `.sort_values()`/`pd.to_datetime()`/`.dt.date`/`.astype(str)`/`.iloc[0]`/`.iloc[-1]` with Narwhals equivalents. Used `nw.col().dt.truncate("1d")` instead of `.dt.date()` — the latter raises `NotImplementedError` on default pandas backend (returns object-dtype Series, not supported by Narwhals).
  4. `discover_daily_motifs_from_agents`: replaced `isinstance(native, pd.DataFrame)`+`.to_pandas()` with Narwhals-only `nw.from_native()`. Replaced `pd.DataFrame(all_records)` result assembly with `nw.from_dict(daily_data, backend=backend)`. Replaced `work_df.groupby()` with explicit user_id loop over `work_df.filter(nw.col(user_id_col) == uid)`. Used `backend=nw_df.implementation` (not deprecated `native_namespace=`) for `nw.from_dict`.
  5. Replaced `value_counts().sort_index().rename_axis().reset_index(name="count")` chain with `group_by("motif_id").agg(nw.len().alias("count")).sort("motif_id")`.
  6. Output now returns the same backend as the input (`.to_native()` on both result DataFrames).
- **Key Narwhals gotcha:** `.dt.date()` is not usable on default pandas backend — use `.dt.truncate("1d")` instead to group by calendar day.
- **Tests:** 275 passed, 5 pre-existing pyarrow failures (unchanged). All 29 motifs tests pass.

### R17: Move motif graph/canonicalization logic to Rust; trim Python wrapper to extraction + assembly (2026-04-20)
- **Smell:** `discover_daily_motifs_from_agents` iterated per-user in Python, doing N·D Narwhals `.filter` round-trips plus Python-level graph construction per day. Only the final permutation step was in Rust.
- **Changes:**
  1. Full rewrite of `src/motifs.rs` — adds `compute_daily_motifs` kernel that runs per-user processing in parallel via Rayon. Internally: `compute_primary_home_node_id` (night HOME by duration, fallback to most frequent HOME), `compute_motif_from_daily_visits` (sequence build + consecutive dedup + edge collection + canonical form), `process_single_user` (look-back/look-ahead + per-day loop). Extracted `canonical_adjacency_form_internal` as a pure Rust fn (called inside the kernel without PyO3 overhead). Kept `canonical_adjacency_form` as thin `#[pyfunction]` wrapper.
  2. `src/lib.rs` — added `m.add_function(wrap_pyfunction!(motifs::compute_daily_motifs, m)?)`.
  3. `skmob2/measures/visits/motifs.py` — stripped to: column detection → sort → flat list extraction → user_ranges computation → `_core.compute_daily_motifs(...)` call → result assembly with `nw.from_dict`. All Python helpers (`_DiGraph`, `_degree_sequence`, `_canonical_adjacency_form`, `_is_isomorphic`, `get_motif_library`, `_motif_id`, `_compute_primary_home_node_id`, `build_motif_graph`, `_build_daily_motif_records`) deleted. `discover_daily_motifs_from_agents` now returns a 2-tuple `(daily_df, dist_df)` (was returning only `daily_df` — fix needed for tests).
  4. Removed `get_motif_library` from all four `__init__.py` files (`visits/`, `measures/`, root `skmob2/`, `skmob2/measures/__init__.py`).
  5. Test file trimmed to 5 `test_discover_motifs_*` tests (removed tests for deleted helpers and the `dynamic_library_file` test).
- **date_id encoding:** `nw.col("start_timestamp").dt.truncate("1d").cast(nw.Int64) // (86400 * 1_000_000)` gives days-since-epoch as Int32. Reverse: `date_id * 86400 * 1_000_000 → cast(Datetime("us"))`.
- **Key Narwhals gotcha:** `.dt.date()` raises `NotImplementedError` on default pandas backend. Use `.dt.truncate("1d")` instead.
- **Tests:** 5 motif tests pass, full suite = 256 passed, 5 pre-existing pyarrow failures, 1 skipped, 17 deselected.

## Completed (2026-04-21) — preprocessing module cleanup

### R18: Replace `statistics.median` with `numpy.median` in `compress.py`
- `import statistics` was only used for `statistics.median`. Replaced with `numpy.median` (numpy is already a transitive dep via scikit-learn/pandas and is used in `cluster.py`).
- `float()` wrapper added to ensure the return type is always a plain Python float, not numpy scalar.

### R19: Fix shadowed builtin variable `l` in `cluster.py`
- `Counter(l for l in ...)`, `sorted(..., key=lambda l: ...)`, and `[remap[l] if l >= 0 ...]` all used `l` as a loop variable, shadowing the Python built-in.
- Renamed to `lbl` throughout the DBSCAN label remapping block.

### R20: Fix type annotation placement for `ranges` in all four preprocessing files
- `filter.py`, `compress.py`, `stay_locations.py`, `cluster.py` all had the type annotation `list[tuple[int, int]]` placed only on the `else`-branch assignment, making the `if`-branch unannotated.
- Moved annotations to a single declaration before the if/else in all four files. Applied same pattern to `uid_values: list` and `uid_series_list: list` in `stay_locations.py` and `cluster.py`.

### R21: Fix misleading docstring for `stop_radius_factor` in `stay_locations.py`
- Old: "Multiplier for spatial_radius_km (unused when spatial_radius_km is not None)."
- New: "Accepted for skmob API compatibility; not used by this implementation."
- The old wording implied there was a code path that used it; there isn't.

## Test baseline (2026-04-21)
- 286 passed, 7 failed (pre-existing Polars/pyarrow), 1 skipped, 21 deselected (skmob marker).
- Previous session baseline was 256 (R18–R21 added 30 new passing tests from new preprocessing module).

## Deferred / Known smells remaining

### D1: Split `visits/intermittance.py` by function — COMPLETED (2026-04-19)
- Split into: `intermittance.py` (just `intermittance_and_degree_of_return` + `_compute_single_idr`), `regularity.py`, `fast_diversity.py`, `diversity.py`, `entropy.py` (`trajectory_entropy` + `trajectory_predictability` + 3 private helpers: `_kontoyiannis_entropy`, `_fano_equation_term`, `_solve_max_predictability_with_fano`).
- Private entropy helpers kept in `entropy.py` (Option C — tightly coupled to entropy logic).
- Test files renamed: `test_individual_*.py` → `test_*.py`. New file `test_fast_diversity.py` for the suffix-array primitive.
- Important gotcha: module name `fast_diversity` collides with the re-exported function in `visits/__init__.py`. The monkeypatch test must use `sys.modules["skmob2.measures.visits.fast_diversity"]` instead of `import skmob2.measures.visits.fast_diversity as mod` to get the module object.
- Three `__init__.py` files updated: `visits/__init__.py`, `measures/__init__.py`, `skmob2/__init__.py`.
- Result: 131 passed, 5 pre-existing pyarrow failures (baseline unchanged).

### D2: `activity.py` and `intermittance.py` drop Narwhals mid-function
- Both convert to pandas for row-iterative loops (`iterrows`, `groupby`). This is correct given algorithmic requirements, but worth documenting explicitly.
- Not a bug — flagged for awareness.
- `motifs.py` was in the same category but has been fully converted to Narwhals in R16 — serves as a reference for how to port row-iterative pandas logic to Narwhals.

### D3: `_ROW_ORDER_COL` sentinel in `_common.py` — COMPLETED (2026-04-17)
- Added `.drop(_ROW_ORDER_COL)` as the final step in `_prepare_trajectory`, so the sentinel column is used for sort tiebreaking but never appears in the returned dataframe.
- Updated docstring step numbering accordingly (step 2 "attach row-order index" removed; steps 3–6 → 2–5).
- No callers referenced `_ROW_ORDER_COL` externally — grep confirmed it was only used inside `_prepare_trajectory`.
- Result: 131 passed, 5 pre-existing pyarrow failures (baseline unchanged).

### D4: Orchestration layer
- No `individual_mobility_summary()` type function exists that composes jump_lengths + radius_of_gyration in one call. This would be a useful orchestration primitive for users. Deferred until user confirms interest.
