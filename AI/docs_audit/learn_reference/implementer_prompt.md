# Implementation prompt

Improve Fastkit-Mobility's Learn and Reference documentation using the brief,
critic rubric, and bounded loop in this directory. Carry the campaign through
implementation and verification; do not stop at a proposal. Read repository
instructions first, use graphify for codebase discovery, and preserve unrelated
working-tree changes and previous audit records.

## Scope and priorities

Work on `docs/src/learn/`, `docs/src/reference/`, their navigation, notebook
conversion and launch support, and narrowly necessary shared styles or tooling.
Change API docstrings only to correct or clarify documented behavior, without
changing runtime behavior. Home, Resources, About, and release notes are outside
this campaign except for regressions caused by shared changes.

Prioritize correctness and execution, then reader comprehension and wayfinding,
then scannability and visual polish. Never trade a working example or accurate
interpretation for a higher visual score. Use real outputs rather than decorative
illustrations. Keep prose branding Fastkit-Mobility; use `fastmob` for imports
and installation.

## Learn architecture

Audit existing content before deciding chapter names. Start with this provisional
sequence and merge or adjust groups to fit implemented capabilities:

1. Getting started and understanding mobility data.
2. Preparing trajectories: cleaning, segmentation, stops, and locations.
3. Measuring mobility: individuals, populations, and relevant comparisons.
4. Spatial and network analysis: routes, matching, interactions, and motifs.
5. Modelling and evaluating mobility: fitting, generation, and evaluation.

Provide a compact theme outline, one explicit first tutorial, and task-oriented
links. Within each chapter use: research question → concept → method choice →
worked example → interpretation and limitations → related APIs and references.
Define unfamiliar terms on first use. Explain required columns, coordinate and
distance units, time ordering, CRS/timezone assumptions, and missing values where
relevant. Link technical detail rather than forcing every reader through it.

Use consistent chapter and page labels across sidebar, headings, and overview.
Avoid arbitrary icons and unnecessary nesting. Put essential meaning in text;
use concise outcome-oriented thumbnails where they help select an example.
Use calm surfaces, readable plots, accessible contrast in light/dark themes,
and a compact mobile list. Do not equate lower contrast with readability.

Each tutorial should open with its question, expected result, prerequisites,
data source, and launch links. Show an authentic representative output early.
Finish with interpretation, limitations, next steps within Learn, and API links.
Distinguish conceptual reading pages from executable tutorials visibly.

## Reference architecture

Build a complete public API inventory from exports, documented APIs, and public
submodules. Reconcile discrepancies rather than trusting the current nav or
`__all__` alone. Include public optional-feature APIs even if extras are needed.

The overview should put a concise module catalogue first. Module pages start
with compact function/class tables: linked symbol, one-line purpose, and example
link where available. Every public symbol must have reachable details. Details
include signature, parameters, return schema and units, important assumptions,
errors/limitations, related APIs, and worked-example links as applicable. Use
generated documentation where reliable; inspect the rendered result. Detailed
entries can remain on a module page if links land directly on the right symbol.
Avoid expanding every detail in the initial catalogue or duplicating tutorials.
Link conceptual explanations back to Learn. Preserve existing URLs or provide
an explicit redirect/migration map for moved pages and anchors.

## Notebook contract

- Every executable Learn example has a canonical `.ipynb` source; generated
  Markdown and output folders are build artifacts. Avoid duplicate hand-kept
  walkthroughs. Short illustrative API snippets may remain, but do not count
  toward example coverage unless backed by a verified runnable example.
- Provide working Colab and Binder links to the canonical repository notebook.
  Verify repository, ref, notebook path, and dependency setup. Review the current
  converter, which injects Binder links, and add Colab support as needed.
- Run all cells in order from a fresh kernel, empty data cache, and directory
  outside the checkout. Install declared published dependencies and extras;
  require no editable local install, compiler, mounted Drive, credentials,
  uploads, repository-relative input files, or manual cell edits. If a required
  API is unreleased, report the mismatch and explicitly pin a reproducible
  installable revision rather than silently relying on workspace code.
- Fetch real data over public internet URLs in the notebook. Prefer stable,
  versioned sources, cite provenance/license, disclose download size and sampling,
  use a writable temporary/cache directory, and fail clearly on download errors.
  Network outages must not silently substitute fabricated data. Clearly labelled
  synthetic micro-examples are allowed and need no external download.
- Declare all dependencies, seeds, and relevant environment versions. Avoid GPU,
  paid services, local credentials, and implicit state. Keep saved useful outputs
  and explain results without flooding the page with installation/download logs.
- The final executed cell displays measured dependency-setup time, download time,
  load/preparation time, analysis/plot time, and total elapsed notebook runtime.
  Start the total timer before setup, mark phases without overlaps, and explain
  that total includes execution overhead. Use wall-clock timing, not CPU timing.
  Report cache status, data size/rows, environment, and measurement date. State
  download = 0 when synthetic; distinguish cached runs from cold downloads.
  Do not label analysis-only timing as notebook runtime or invent estimates.
- Local isolated execution is evidence of portability, not proof of Colab
  execution. Attempt actual Colab verification when access exists; otherwise mark
  `colab: portability-checked; hosted execution unverified`. Binder needs only
  static portability/link/environment review; never claim it ran if it did not.

## At least 80% meaningful function example coverage

Create `coverage.csv` in the campaign run directory with columns:
`canonical_symbol,module,aliases,kind,public_evidence,example_path,cell_id,
use_case,output_explanation,execution_log,status,exclusion_reason`.
Use separate rows for additional examples if useful; compute totals over unique
canonical symbols, never rows. Resolve re-export aliases to one symbol. Count
public functions and public class methods in the function denominator; track
class construction separately. Exclude only private implementation details,
third-party re-exports, and duplicate aliases with written evidence. Do not omit
hard-to-demonstrate or optional-feature functions to improve the percentage.
Freeze the reconciled inventory before optimisation; log later changes.

A covered function must be called in an executed notebook code cell with complete
inputs and a stated use case. Its result or observable effect must be inspected
and explained. Imports, mentions, unused helper definitions, generated API
listings, screenshots of code, and dead branches do not count. A workflow can
cover multiple functions when each use is meaningful and traceable. Link from
each covered function's reference entry to its example page and section.

Acceptance is `covered_unique_functions / eligible_unique_functions >= 0.80`
(equivalently at least `ceil(0.80 * eligible)`), plus 100% catalogue/detail
coverage. Report global and per-module counts, the uncovered list, and reasons.
Flag modules with zero examples even when the global threshold passes; avoid
building bloated notebooks solely to collect calls. Prefer coherent research
workflows plus focused examples for remaining use cases.

## Evidence and completion

Follow `LOOP.md`. Save inventories, execution logs, screenshots, critic reports,
changes, and unresolved issues under a new campaign directory. Run notebook
conversion before documentation tests and builds. Run `graphify update .` after
code changes. Do not commit, publish, or deploy unless separately requested.
Final reporting must distinguish implemented, executed, portability-reviewed,
and unverified work, with exact API example coverage and remaining blockers.
