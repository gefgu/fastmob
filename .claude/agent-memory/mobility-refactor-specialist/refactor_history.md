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

### D3: `_ROW_ORDER_COL` sentinel in `_common.py` — COMPLETED (2026-04-17)
- Added `.drop(_ROW_ORDER_COL)` as the final step in `_prepare_trajectory`, so the sentinel column is used for sort tiebreaking but never appears in the returned dataframe.
- Updated docstring step numbering accordingly (step 2 "attach row-order index" removed; steps 3–6 → 2–5).
- No callers referenced `_ROW_ORDER_COL` externally — grep confirmed it was only used inside `_prepare_trajectory`.
- Result: 131 passed, 5 pre-existing pyarrow failures (baseline unchanged).

### D4: Orchestration layer
- No `individual_mobility_summary()` type function exists that composes jump_lengths + radius_of_gyration in one call. This would be a useful orchestration primitive for users. Deferred until user confirms interest.
