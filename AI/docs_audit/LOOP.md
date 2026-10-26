# Docs improvement loop

Goal: raise each top-level docs page to >= 8.0/10 per the Opus critic
(`critic_prompt.md`), with no dimension below 6, using bounded rounds. The whole
process is logged in `AI/docs_audit/runs/` so it stays auditable.

## Pages in scope (top-level, in this order)
1. Home (`index.md`) — most important; sets the visual language for the rest
2. Learn index (`learn/index.md`)
3. Reference index (`reference/index.md`)
4. Features overview (`features/index.md`)
5. Resources (`resource/index.md`)
6. About (`about/index.md`)
7. Release Notes (`release_notes/index.md`)

Site-wide items (nav tree, theme/contrast, icons) are handled in **Round 0**
and by "CROSS-PAGE NOTES", not per page.

## Round budget
- **Round 0 (once, global):** critic reviews nav tree + Home + one Learn page
  and one Reference page for site-wide problems. Fix nav/theme/icon issues first,
  because per-page scores are meaningless on a broken frame.
- **Per page: minimum 2, maximum 4 critic rounds** (round 1 = baseline, no edits
  before it).
- **Stop early** when verdict is PASS (>= 8.0, no dimension < 6).
- **Stop on plateau:** if Overall improves by < 0.5 across two consecutive
  rounds, stop and report instead of grinding.
- **Stop on regression:** if Overall drops, revert that round's edits and
  continue from the previous best.
- **Global cap:** 7 pages x 4 rounds + Round 0 = at most ~30 critic calls.
- **Final pass (1 round):** re-score all pages together after everything is
  done, since later nav/theme changes can break earlier pages. Only fix
  regressions.

Why these numbers: round 1 gives a baseline; most gains come in rounds 2-3 from
the P0/P1 list; round 4 is polish with diminishing returns. A cap prevents
chasing a number past the point the critic is just inventing nits.

## Procedure per round
1. **Capture** (Playwright or similar; build with `bash scripts/serve_docs.sh`):
   desktop light, desktop dark, mobile 390px, full-page, above-the-fold crop.
   Save to `AI/docs_audit/runs/<page>/r<N>/`.
2. **Critic:** spawn a *fresh* Opus agent with the system prompt in
   `critic_prompt.md`, the screenshots, the page Markdown, the nav tree, and
   the previous round's output. Save its answer as `r<N>/critic.md`.
3. **Triage (implementer = Claude):** apply all P0, then P1 that don't conflict;
   skip P2 until round >= 3. Write `r<N>/changes.md` listing what was changed
   and what was deliberately declined and why.
4. **Guardrails while editing:**
   - Overall structure and presentation only; don't rewrite technical content
     or change API claims.
   - Don't put numbers (speedups, benchmarks) in docs that you haven't taken
     from existing benchmark results.
   - Prefer removing/merging over adding. No new top-level nav items without
     removing others.
   - Generated notebook `.md` files are not edited directly.
   - Run `uv run --group docs zensical build --strict` and
     `uv run --group docs --with pytest pytest tests/docs -q` after each round.
5. **Re-capture and loop** until a stop condition fires.
6. **Commit** once per page when it stops (focused commit, only docs files).

## Images
When a critic requests an image:
- **Prefer notebook-real-data**: add/extend a notebook under `docs/src/` (or a
  script under `scripts/`) that generates the figure from Brightkite/GeoLife
  with fastmob calls; save the PNG at the requested location and keep the
  notebook so the figure is reproducible.
- **Diagrams** that are conceptual (pipeline, hierarchy): SVG or Mermaid first.
- **AI-generated** only for things real data can't show. Do **not** generate
  these; append an entry to `AI/docs_audit/image_requests.md` with:
  `filename | location | where it is used | prompt | size/aspect | style notes`
  so it can be handed to Codex. Insert the `<img>` reference with alt text and
  a placeholder-safe fallback so the build doesn't break before the file exists.

## Run log
Each page ends with `runs/<page>/SUMMARY.md`: score trajectory per round,
issues fixed, issues declined (with reason), open P2s, image requests. A
top-level `runs/INDEX.md` lists final scores for all pages.

## Judge-drift controls
- Fresh critic each round; it only sees the previous scores/issues, not
  the implementer's reasoning.
- Scores are anchored in the rubric (8 = "I'd show this as a good example").
- If a page reaches 8 on round 1-2 with obviously unaddressed problems, rerun
  the critic once more with "assume the score is 2 points too generous" before
  accepting.
