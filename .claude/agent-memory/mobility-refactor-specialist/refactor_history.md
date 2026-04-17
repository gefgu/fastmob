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

## Deferred / Known smells remaining

### D1: `individual.py` filename (medium priority)
- `individual.py` contains only `intermittance_and_degree_of_return`. The name `individual.py` is a legacy monolithic-file name. Per project rules, it should be `intermittance.py`.
- **Blocker:** Renaming requires updating `measures/__init__.py`, `skmob2/__init__.py`, and the test file `test_individual_intermittance.py`. Low risk but touches public API path.
- **Deferred:** Not done yet — user should confirm rename is acceptable before proceeding.

### D2: `activity.py` and `individual.py` drop Narwhals mid-function
- Both convert to pandas for row-iterative loops (`iterrows`, `groupby`). This is correct given algorithmic requirements, but worth documenting explicitly.
- Not a bug — flagged for awareness.

### D3: `_ROW_ORDER_COL` sentinel in `_common.py`
- Used in `_prepare_trajectory` but leaked to callers via the returned df. Callers don't need to see it. Could be dropped from the result df in `_prepare_trajectory` as a cleanup.
- Low priority — no correctness impact.

### D4: Orchestration layer
- No `individual_mobility_summary()` type function exists that composes jump_lengths + radius_of_gyration in one call. This would be a useful orchestration primitive for users. Deferred until user confirms interest.
