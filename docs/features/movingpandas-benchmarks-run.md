# movingpandas Benchmarks — Run Tracking

Execution tracking file. Do not modify the original plan file.

## Steps

- [x] Step 1 — Add `dev-movingpandas` extras group in `pyproject.toml`
- [x] Step 2 — Add `--movingpandas` flag to `tests/setup_env.sh`
- [x] Step 3 — Add fixtures and `movingpandas` marker in `tests/benchmarks/conftest.py`
- [ ] Step 4 — Add `test_jump_lengths_movingpandas` to `tests/benchmarks/bench_jump_lengths.py`
- [ ] Step 5 — Add `test_radius_of_gyration_movingpandas` (with `_rog_for_tc` helper) to `tests/benchmarks/bench_radius_of_gyration.py`
