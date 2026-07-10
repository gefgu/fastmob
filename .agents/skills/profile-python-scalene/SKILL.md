---
name: profile-python-scalene
description: Profile Python functions in the fkmob repository with Scalene, reduce Scalene JSON output, and explain CPU and memory hotspots. Use when Codex needs to run or analyze Scalene profiles for fkmob callables, benchmark-like workloads, pandas/polars trajectory inputs, or existing Scalene .json files.
---

# Profile Python Scalene

## Workflow

Run from the repository root (`/home/gustavo/fkmob`) so imports, data paths, and `uv` environment resolution match the project.

Prefer explicit output paths outside the repo unless the user asks to persist artifacts:

```bash
uv run scalene run --memory --profile-only fkmob -o /tmp/scalene-profile.json .agents/skills/profile-python-scalene/scripts/profile_function.py --- fkmob.module:function --input-kind brightkite-pandas --rows 100000
```

Then inspect the profile with either Scalene's terminal view or the reducer:

```bash
uv run scalene view --cli -r /tmp/scalene-profile.json
uv run python .agents/skills/profile-python-scalene/scripts/reduce_scalene_json.py /tmp/scalene-profile.json --top 20
```

Use `--profile-only fkmob` for normal investigations. Add a comma-separated broader scope only when the question is about dependency overhead, for example `--profile-only fkmob,narwhals,pandas,polars`.

## fkmob Environment Fallback

In this repository, Codex may run in a sandbox where `/snap/bin/uv` fails with snap-confine permissions and the local `.venv` has no `pip`. If the wrapper script fails before profiling while trying to rebuild with `maturin develop --uv`, first check whether the compiled extension is already usable:

```bash
.venv/bin/python -c 'import fkmob._core as c; print(c.__file__)'
```

If that succeeds, profile directly with `.venv/bin/scalene` instead of forcing a rebuild:

```bash
.venv/bin/scalene run --profile-only fkmob --memory --off \
  -o /tmp/filter_4M_scalene.json \
  tests/profiling/brightkite_workloads.py --- \
  --workload filter --rows 4000000 --backend pandas \
  --implementation fkmob --scalene-function-profile

.venv/bin/python .agents/skills/profile-python-scalene/scripts/reduce_scalene_json.py \
  /tmp/filter_4M_scalene.json --top 25
```

If the extension is not importable, rebuild with the approved local path:

```bash
env -u CONDA_PREFIX uv run maturin develop --release
```

The repo Cargo config must keep `-C` as a separate flag before each rustc option, for example `["-C", "force-frame-pointers=yes", "-C", "symbol-mangling-version=v0"]`.

## Profiling Callables

Use `scripts/profile_function.py` to profile an importable callable. Targets use `module:function` syntax:

```bash
uv run scalene run --memory --profile-only fkmob -o /tmp/radius.json .agents/skills/profile-python-scalene/scripts/profile_function.py --- fkmob.measures.spatial.radius_of_gyration:radius_of_gyration --input-kind brightkite-pandas --rows 1000 --repeat 5
```

Input kinds:

- `brightkite-pandas`: default; loads `tests/shared/data/loc-brightkite_totalCheckins.txt.gz`, normalizes timestamps, and passes a pandas slice.
- `brightkite-polars`: loads the same Brightkite file as a polars DataFrame.
- `geolife-pandas`: uses `tests.shared.geolife.load_geolife_pandas`; this may download GeoLife if missing, so use it only when the user explicitly asks.
- `none`: calls the target without a positional input, useful for smoke tests or functions whose inputs are supplied through kwargs.

Pass keyword arguments as JSON:

```bash
uv run scalene run --memory --profile-only fkmob -o /tmp/filter.json .agents/skills/profile-python-scalene/scripts/profile_function.py --- fkmob.preprocessing.filter:filter --kwargs-json '{"max_speed_kmh": 500}' --rows 10000
```

Use `--repeat` to increase sample counts when Scalene reports little activity. Keep `--rows` small first, then scale up after the call succeeds.

## Reading Scalene JSON

Scalene JSON is detailed and often too large for direct context. Prefer the reducer first:

```bash
uv run python .agents/skills/profile-python-scalene/scripts/reduce_scalene_json.py /tmp/scalene-profile.json --top 20
```

The reduced JSON includes:

- Run metadata: program, elapsed seconds, memory enabled, max footprint, and peak-memory location.
- Top files by total CPU share.
- Top functions and lines by Python CPU, native C CPU, system CPU, total CPU, peak memory, allocation MB, allocation count, and growth MB.
- Leak entries when Scalene reports them.

When interpreting results, call out whether time is Python, native C, or system time. High native C time in pandas/polars/narwhals paths may indicate vectorized work rather than Python-loop overhead. High allocation count or growth on a line is often more actionable than peak memory alone.

## Guardrails

- Do not modify benchmark result JSON files while profiling.
- Do not run full-size profiles first; start with `--rows 1000` or `--rows 10000`.
- Do not use `geolife-pandas` unless the user asks for GeoLife or approves dataset download.
- Use `/tmp` for generated `.json` and `.html` profile artifacts by default.
- If `uv run scalene ...` fails because of sandbox or snap confinement, rerun the same command with the required approval rather than changing the profiling approach.
