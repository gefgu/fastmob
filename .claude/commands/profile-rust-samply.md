# Profile Rust Samply

Profile fkmob Rust-backed Python workloads with samply, reduce Firefox Profiler JSON profiles, and diagnose native/PyO3 performance bottlenecks.

## Workflow

Run from the repository root (`/home/gustavo/fkmob`) so `uv`, `maturin`, tests, data paths, and `fkmob._core` imports match the project.

Prefer the existing Brightkite runner for supported workloads:

```bash
bash tests/run_samply_profiles.sh --list
bash tests/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration --scope function
```

Use `--scope function` by default — it prepares data before Samply attaches so profiles emphasize the target function. Use `--scope full` only when startup, data loading, or import overhead is the suspected bottleneck.

Before running Samply, check `/proc/sys/kernel/perf_event_paranoid`. If greater than `1`, ask the user to lower it with `sudo sysctl kernel.perf_event_paranoid=1` before profiling. Do not attempt to run the sudo command yourself unless the user explicitly asks.

## Reducing Profiles

Samply writes Firefox Profiler JSON, usually compressed as `.json.gz`. Do not load large raw profiles directly into context. Reduce them first:

```bash
uv run python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py \
  .profiles/samply/fkmob/radius_of_gyration.json.gz --top 20
```

Useful options:

```bash
uv run python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py /tmp/profile.json.gz --filter fkmob:: --top 30
uv run python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py /tmp/profile.json.gz --thread python -o /tmp/reduced.json
```

Focus on:

- `top_leaf_frames`: where samples stop; good for tight loops or expensive calls.
- `top_inclusive_frames`: functions anywhere in hot stacks; good for callers and shared helpers.
- `top_stacks`: repeated call paths.
- `rust_focused`: filtered entries tied to `fkmob`, `src/`, Rust crate paths, or PyO3.
- Python conversion frames such as `PyFloat_FromDouble`, `PyList_New`, `pyo3::conversion::IntoPyObject::owned_sequence_into_pyobject`. When these dominate after the Rust kernel is optimized, treat the bottleneck as actionable by adding NumPy or Arrow return paths.

When reporting findings, state the sample count, whether the profile is function or full scope, and the workload size/backend. Connect hot Rust frames to files under `src/` before proposing edits.

## Generic Callable Fallback

For targets not in `tests.profiling.brightkite_workloads`, use the generic importable-callable wrapper:

```bash
samply record --save-only --rate 1000 -o /tmp/fkmob-profile.json.gz -- \
  uv run python .agents/skills/profile-rust-samply/scripts/profile_importable_samply.py \
  fkmob.module:function --input-kind brightkite-pandas --rows 10000 --repeat 3
```

Targets use `module:function` syntax. Input kinds: `brightkite-pandas`, `brightkite-polars`, `none`. Pass keyword arguments with `--kwargs-json '{"key": "value"}'`.

## Guardrails

- Keep large generated profiles in `/tmp` unless the user asks to persist them.
- Do not run full 4M-row profiles first.
- Do not silently change profiler scope to make a command pass; report what was profiled.
- Always check `/proc/sys/kernel/perf_event_paranoid` before Samply recording. If greater than `1`, ask the user to run `sudo sysctl kernel.perf_event_paranoid=1` and wait for confirmation. Do not try to bypass this setting.
- Do not edit Rust code solely from one noisy profile. Confirm the hot frame, inspect the corresponding Rust implementation, make a focused change, then rerun the smallest profile or benchmark that can validate the fix.
- When optimizing PyO3 conversion overhead, prefer adding sibling array-returning functions and updating Python wrappers to prefer them with a fallback to the existing `Vec<T>` functions. Do not break public Python APIs or remove normal `Vec<T>` paths unless requested.

$ARGUMENTS
