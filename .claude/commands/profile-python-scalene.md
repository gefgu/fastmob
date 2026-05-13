# Profile Python Scalene

Profile Python functions in the skmob2 repository with Scalene, reduce Scalene JSON output, and explain CPU and memory hotspots.

## Workflow

Run from the repository root (`/home/gustavo/skmob2`) so imports, data paths, and `uv` environment resolution match the project.

Prefer explicit output paths outside the repo unless the user asks to persist artifacts:

```bash
uv run scalene run --memory --profile-only skmob2 -o /tmp/scalene-profile.json .agents/skills/profile-python-scalene/scripts/profile_function.py --- skmob2.module:function --input-kind brightkite-pandas --rows 100000
```

Then inspect the profile with either Scalene's terminal view or the reducer:

```bash
uv run scalene view --cli -r /tmp/scalene-profile.json
uv run python .agents/skills/profile-python-scalene/scripts/reduce_scalene_json.py /tmp/scalene-profile.json --top 20
```

Use `--profile-only skmob2` for normal investigations. Add a comma-separated broader scope only when the question is about dependency overhead.

## Profiling Callables

Use `scripts/profile_function.py` to profile an importable callable. Targets use `module:function` syntax:

```bash
uv run scalene run --memory --profile-only skmob2 -o /tmp/radius.json .agents/skills/profile-python-scalene/scripts/profile_function.py --- skmob2.measures.spatial.radius_of_gyration:radius_of_gyration --input-kind brightkite-pandas --rows 1000 --repeat 5
```

Input kinds:

- `brightkite-pandas`: loads `tests/shared/data/loc-brightkite_totalCheckins.txt.gz`, normalizes timestamps, and passes a pandas slice.
- `brightkite-polars`: loads the same Brightkite file as a polars DataFrame.
- `geolife-pandas`: uses `tests.shared.geolife.load_geolife_pandas`; may download GeoLife if missing — use only when the user explicitly asks.
- `none`: calls the target without a positional input.

Pass keyword arguments as JSON:

```bash
uv run scalene run --memory --profile-only skmob2 -o /tmp/filter.json .agents/skills/profile-python-scalene/scripts/profile_function.py --- skmob2.preprocessing.filter:filter --kwargs-json '{"max_speed_kmh": 500}' --rows 10000
```

Use `--repeat` to increase sample counts when Scalene reports little activity. Keep `--rows` small first, then scale up after the call succeeds.

## Reading Scalene JSON

Scalene JSON is often too large for direct context. Prefer the reducer first:

```bash
uv run python .agents/skills/profile-python-scalene/scripts/reduce_scalene_json.py /tmp/scalene-profile.json --top 20
```

The reduced JSON includes: run metadata, top files by CPU share, top functions and lines by Python/native/system CPU, peak memory, allocation MB, and leak entries.

When interpreting results: high native C time in pandas/polars/narwhals paths may indicate vectorized work rather than Python-loop overhead. High allocation count or growth on a line is often more actionable than peak memory alone.

## Guardrails

- Do not modify benchmark result JSON files while profiling.
- Do not run full-size profiles first; start with `--rows 1000` or `--rows 10000`.
- Do not use `geolife-pandas` unless the user asks or approves dataset download.
- Use `/tmp` for generated `.json` and `.html` profile artifacts by default.
- If `uv run scalene ...` fails due to sandbox or snap confinement, rerun the same command with the required approval rather than changing the profiling approach.

$ARGUMENTS
