---
name: profile-rust-samply
description: Profile skmob2 Rust-backed Python workloads with samply, reduce Firefox Profiler JSON profiles, and diagnose native/PyO3 performance bottlenecks. Use when Codex needs to run Samply against skmob2 functions, analyze .json or .json.gz Firefox Profiler output, understand hot Rust frames in src/*.rs, or report and fix Rust-core performance issues.
---

# Profile Rust Samply

## Workflow

Run from the repository root (`/home/gustavo/skmob2`) so `uv`, `maturin`, tests, data paths, and `skmob2._core` imports match the project.

Prefer the existing Brightkite runner for supported workloads:

```bash
bash scripts/run_samply_profiles.sh --list
bash scripts/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration --scope function
```

Use `--scope function` by default. It prepares data before Samply attaches, so profiles emphasize the target function rather than imports and dataset loading. Use `--scope full` only when startup, data loading, Python wrapper setup, or import overhead is the suspected bottleneck.

The runner builds the extension in release mode with symbols by setting `CARGO_PROFILE_RELEASE_DEBUG=1` and running `maturin develop --release` with the repo fallback to `maturin develop --release --uv`. Start with `--rows 10000`; scale up only after the workload succeeds and the profile has enough samples.

Before running Samply, check `/proc/sys/kernel/perf_event_paranoid`. If it is greater than `1`,
ask the user to lower it with `sudo sysctl kernel.perf_event_paranoid=1` before profiling. Do not
attempt to run the sudo command yourself unless the user explicitly asks; Samply profiles should
only be launched after the user confirms the setting has been updated or the current value is
already `1` or lower.

## skmob2 Environment Fallback

In this repository, Codex may run in a sandbox where `/snap/bin/uv` fails with snap-confine permissions and the local `.venv` has no `pip`. If `scripts/run_samply_profiles.sh` fails before recording while trying to rebuild with `maturin develop --uv`, first check whether the compiled extension is already usable:

```bash
.venv/bin/python -c 'import skmob2._core as c; print(c.__file__)'
```

If that succeeds, keep the function-scope prepared-child behavior by running the Python Samply runner directly:

```bash
.venv/bin/python scripts/profile_brightkite_samply.py \
  --samply-bin /home/gustavo/.cargo/bin/samply \
  --rows 4000000 --workload filter --implementation skmob2 \
  --backend pandas --scope function --rate 1000 \
  --output-dir /tmp/samply_filter_4M

.venv/bin/python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py \
  /tmp/samply_filter_4M/skmob2/filter.json.gz --top 20 \
  -o /tmp/filter_4M_samply_reduced.json
```

If the extension is not importable, rebuild with the approved local path:

```bash
env -u CONDA_PREFIX uv run maturin develop --release
```

The repo Cargo config must keep `-C` as a separate flag before each rustc option, for example `["-C", "force-frame-pointers=yes", "-C", "symbol-mangling-version=v0"]`.

## Reducing Profiles

Samply writes Firefox Profiler JSON, usually compressed as `.json.gz`. Do not load large raw profiles directly into context. Reduce them first:

```bash
uv run python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py \
  .profiles/samply/skmob2/radius_of_gyration.json.gz --top 20
```

Useful options:

```bash
uv run python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py /tmp/profile.json.gz --filter skmob2:: --top 30
uv run python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py /tmp/profile.json.gz --thread python -o /tmp/reduced.json
```

Focus on:

- `top_leaf_frames`: where samples stop; good for tight loops or expensive calls.
- `top_inclusive_frames`: functions present anywhere in hot stacks; good for callers and shared helpers.
- `top_stacks`: repeated call paths.
- `rust_focused`: filtered entries likely tied to `skmob2`, `src/`, Rust crate paths such as `skmob2::`, PyO3, or the native extension.
- Python conversion/materialization frames such as `PyFloat_FromDouble`,
  `PyLong_FromUnsignedLongLong`, `PyObject_Malloc`, `PyList_New`,
  `pyo3::conversion::IntoPyObject::owned_sequence_into_pyobject`, and
  `pyo3::types::tuple::<impl ...>::into_pyobject`. When these dominate after
  the Rust kernel is optimized, treat the bottleneck as actionable by adding
  NumPy or Arrow return paths for flat numeric outputs. Keep the existing
  `Vec<T>`/tuple-of-`Vec<T>` pyfunctions as compatibility fallbacks unless the
  user explicitly asks to remove them.

When reporting findings, state the sample count, whether the profile is function or full scope, and the workload size/backend. Connect hot Rust frames to likely files under `src/` before proposing edits.

## Generic Callable Fallback

Use the repo runner first. For targets that are not in `tests.profiling.brightkite_workloads`, use the generic importable-callable wrapper:

```bash
samply record --save-only --rate 1000 -o /tmp/skmob2-profile.json.gz -- \
  uv run python .agents/skills/profile-rust-samply/scripts/profile_importable_samply.py \
  skmob2.module:function --input-kind brightkite-pandas --rows 10000 --repeat 3
```

Targets use `module:function` syntax. Input kinds are `brightkite-pandas`, `brightkite-polars`, and `none`. Pass keyword arguments with `--kwargs-json '{"key": "value"}'`.

This fallback profiles the whole process, including imports and input construction. Treat it as exploratory unless the target cannot be represented by the Brightkite runner.

## Guardrails

- Keep large generated profiles in `/tmp` unless the user asks to persist them; `.profiles/samply` is acceptable for repo profiling runs.
- Do not run full 4M-row profiles first.
- Do not silently change profiler scope to make a command pass; report what was profiled.
- Always check `/proc/sys/kernel/perf_event_paranoid` before Samply recording. If it is greater
  than `1`, ask the user to run `sudo sysctl kernel.perf_event_paranoid=1` and wait for confirmation
  before profiling. If Samply still fails due to Linux perf permissions, read and report the current
  value and the same manual command. Do not try to bypass this setting.
- Do not edit Rust code solely from one noisy profile. Confirm the hot frame, inspect the corresponding Rust implementation, make a focused change, then rerun the smallest profile or benchmark that can validate the fix.
- When optimizing PyO3 conversion overhead, prefer adding sibling array-returning
  functions (for example NumPy arrays for pandas-facing wrappers or Arrow arrays
  for Polars/list outputs) and update Python wrappers to prefer them with a
  fallback to the existing normal `Vec<T>` functions. Do not break public Python
  APIs or remove the normal `Vec<T>` paths unless requested.
