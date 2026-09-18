# Contributing to Fastmob

Thanks for helping improve Fastmob. Documentation fixes, bug reports, tests,
benchmark evidence, and code are all useful contributions. Small, focused pull
requests are the fastest to review.

## Start here

Before starting a large feature or behavioural change, open an issue describing
the problem, intended users, a small example, and any compatibility concerns.
For a bug, include Fastmob/Python versions, dataframe backend, a minimal input,
expected result, actual result, and the complete error. A failing regression
test is especially helpful.

For a straightforward typo, test fix, or tightly scoped bug fix, open a pull
request directly. Please avoid bundling unrelated refactors, formatting churn,
or generated benchmark results with the functional change.

## Set up a development checkout

Fastmob has Python APIs backed by compiled Rust kernels. Install the normal
contributor environment, then build the extension in editable mode:

```bash
git clone https://github.com/gefgu/fastmob.git
cd fastmob
bash scripts/setup_env.sh
source .venv/bin/activate
maturin develop
```

`scripts/setup_env.sh` creates `.venv` and installs the `dev` dependency group.
Use Python 3.12 explicitly when needed:

```bash
bash scripts/setup_env.sh --python 3.12
```

The main environment deliberately excludes third-party comparison libraries
with incompatible dependency stacks. Create those only when needed, for
example `bash scripts/setup_benchmark_env.sh skmob --python 3.10`. Supported
environment names and pins live in `benchmarks/environments/`.

## Make a focused change

- **Python API or dataframe behaviour:** keep public functions backend-agnostic
  through Narwhals and add pandas coverage; add Polars coverage when the change
  touches dispatch, schema, null, ordering, or return-type behaviour.
- **Rust kernel:** update the binding layer in `fastmob-py/` as well as the core
  implementation, preserve Arrow-compatible inputs, and rebuild with
  `maturin develop` before testing.
- **Public behaviour:** add a regression test in the closest existing module.
  Test the user-visible contract, including columns, units, ordering, errors,
  and edge cases—not only a private helper.
- **Compatibility work:** use the isolated comparison environment and state the
  intended tolerance. Related mobility implementations can differ at edges, so
  do not claim byte-for-byte parity unless the contract guarantees it.
- **Performance work:** add a reproducible benchmark or extend an existing one;
  record dataset, command, hardware, versions, timing method, and result
  semantics. Do not present one machine's result as a universal guarantee.

Keep commits readable and explain *why* a non-obvious decision was made. If you
used an AI tool, review the full diff, understand the change, and remain able to
answer review questions about it.

## Verify before opening a pull request

Run the smallest relevant checks while iterating, then the applicable final
checks below:

```bash
# A focused test while developing
pytest tests/correctness/preprocessing/test_stay_locations.py -q

# Normal correctness suite; excludes optional scikit-mobility comparisons
bash scripts/run_correctness.sh

# Python and Rust lint checks
bash scripts/run_lint.sh
```

Run optional comparison tests only in their matching environment. For example,
the correctness runner accepts `-m skmob`; GeoLife-backed checks can use
`--geolife-mode=slice` before a full-dataset run.

For documentation changes, run:

```bash
uv run --group docs python scripts/build_notebook_docs.py
uv run --group docs zensical build --strict
```

If a relevant check cannot run, say so in the pull request with the command,
environment, and reason. Do not edit generated notebook Markdown or its
`*_files/` assets; they are ignored build output.

## Documentation contributions

Choose the smallest page type that fits the reader's need:

- **Recipe:** a concrete task, prerequisites, minimal deterministic example,
  interpretation, and a link to the authoritative API reference.
- **Reference page:** a compact overview plus generated public docstrings; do
  not reproduce every parameter in prose.
- **Release note:** user-visible `Added`, `Changed`, `Fixed`, `Deprecated`, or
  `Removed` entries, with migration guidance for public API changes.

State the user outcome first. Document units, input/output schemas, backend and
optional-extra requirements, and meaningful ordering or fallback behaviour.
Use relative Markdown links and useful image alt text. Keep a diagram's source
beside its rendered asset, and mark an intentional future visual with
`<!-- Visual placeholder: ... -->` rather than leaving a TODO in published
content.

Notebooks under `docs/src/` are source files. After changing one, regenerate
the Markdown with the command above; the conversion adds a Binder link and
keeps saved outputs available in the rendered site.

## Pull request checklist

In the PR description, include:

- The problem and the user-facing outcome.
- Any API, schema, performance, compatibility, or migration impact.
- Tests and checks run, plus anything intentionally not run.
- Documentation and release-note updates, or why they are unnecessary.
- Reproduction details for bugs and benchmark claims.

Keep the PR to one purpose, respond to review feedback with context, and update
tests and documentation when the reviewed behaviour changes. Contributors and
maintainers use the same review process: a change is ready when its behaviour,
evidence, and maintenance cost are clear.
