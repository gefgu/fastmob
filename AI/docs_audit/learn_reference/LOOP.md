# Bounded implementation and critique loop

## Stage 0: Baseline and scope

Create a new `AI/docs_audit/runs/learn-reference/<campaign-id>/` directory.
Inventory all Learn pages/notebooks and public APIs; reconcile exports, aliases,
methods and optional modules. Save `inventory.md`, `coverage.csv`, current
navigation, and a baseline journey review before editing. Record the source
snapshot, pre-existing changes, dependency versions, and evidence limitations.
Do not reuse old PASS verdicts: their scope excluded content depth.

Propose the theme navigation and map every existing page to its destination.
Keep URL continuity and log redirects. Prioritize coverage gaps by research use
case; do not choose only functions that are easiest to demonstrate.

## Stage 1: Representative pilot

Implement and review these together as one reader journey:

- Learn overview and theme navigation.
- The existing first-trajectory tutorial, converted to a canonical runnable
  notebook where needed.
- Cleaning/stop-detection concepts and a real-data worked notebook.
- Reference overview plus corresponding Preprocessing and Individual Measures
  catalogue/detail sections.

This default pilot follows the existing pages; it is not a restriction on later
coverage. Use at most three review snapshots: baseline, substantive revision,
targeted correction. Review content, execution and visual evidence together.
Pilot acceptance requires local gates for pilot assets and the quality threshold;
global coverage remains pending. Resolve the template and navigation before
replicating them across the library.

## Stage 2: Expand by theme

Apply the accepted patterns to all Learn and Reference content. Group examples
into coherent workflows and add focused notebooks where necessary. Update the
coverage ledger after each batch. Explicitly evaluate underrepresented modules.
Perform a full campaign audit once all notebooks and API entries are ready.
Allow at most three full campaign review snapshots, including that first audit.

## Each revision cycle

1. Save the candidate snapshot and exact scope under `r<N>/`. Convert notebook
   sources, execute changed/new notebooks in isolated fresh environments with
   clean data caches, and save per-notebook logs and timing outputs. On final
   acceptance, all notebooks need execution evidence for the final snapshot;
   unchanged notebook evidence may be reused only when source, dependencies,
   shared helpers and data versions match. Re-run anything affected by changes.
2. Run in order:

   ```bash
   uv run --group docs python scripts/build_notebook_docs.py
   uv run --group docs --with pytest pytest tests/docs -q
   uv run --group docs zensical build --strict
   ```

   Execute notebooks separately before conversion when outputs changed. The
   converter/build does not execute them. Inspect generated pages, launch links,
   symbol anchors, search results and representative notebook outputs in-browser.
3. Capture current desktop light/dark and 390px mobile views, fold and full page,
   for hubs and representative detail pages. Include long API pages and actual
   rendered notebook pages; rotate additional modules through review. Existing
   `scripts/docs_round.sh` can assist capture, but omits notebook conversion;
   run conversion first and use unique round IDs so old evidence is preserved.
4. Give `critic_prompt.md` and the evidence bundle to a fresh independent reviewer
   session. A separate critic agent/session is appropriate when available; do
   not review the implementer's intentions as evidence. Save `critic.md`.
5. Fix P0, then P1. Save `changes.md` with issue IDs, changed paths, verification,
   and declined requests with reasons. Update coverage without changing its
   denominator silently. Address P2 only after gates and core journeys work.
6. Recheck affected assets and capture fresh evidence. Keep issue IDs stable.
   A regression is diagnosed and corrected with a focused patch; restore only
   campaign-owned edits if needed. Never reset the working tree or revert
   unrelated changes automatically because an average score fell.

## Stop rules

Stop successfully only when all campaign gates pass, quality mean >= 8.0 with
every dimension >= 7, and a final regression sweep finds no P0/P1 issue.

Stop iteration and report incomplete work when the review budget is exhausted,
or mean gain is < 0.3 across two successive revision intervals with no reduction
in P0/P1 issues and no improvement in hard gates. A plateau is not success.
Do not add unbounded rounds to chase a critic number. Report exact remaining
work and evidence needed to restart. Functional gates take priority over scores.

Allow one final cross-theme regression review after expansion (at most seven
formal reviews total: three pilot, three campaign, one regression). This checks
Learn↔Reference↔notebook journeys, complete inventory/coverage, and shared-style
effects on Home and one other page. It is not a new redesign round.

## Final artifacts

Save `SUMMARY.md` containing final navigation, covered/eligible counts and
percentage, per-module coverage, uncovered symbols, notebook execution matrix,
Colab/Binder portability versus hosted status, timing/data evidence, score
trajectory, fixed/open issues, commands/results, and links to evidence. Include
any blocked API/version mismatch or missing hosted verification candidly.
No automatic commit or publishing step is part of this loop.
