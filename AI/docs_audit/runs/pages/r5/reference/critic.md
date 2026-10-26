PAGE: docs/src/reference/index.md   ROUND: 5
SCORES: first_impression=7 engagement=7 navigation=8 scannability=7 visual=6 ia_role=7 conciseness=6 trust=7
OVERALL: 6.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile full page, "Modules" table: adding the third column made the table wider than 390px. "Key functions" is cut at the right edge: the header reads "Key" and the cells show only "Tra", "fil", "int", "mat"... So the new column is effectively invisible on phones. Description cells wrap to 4-5 lines, which makes each row very tall (the table runs about 1,500px on mobile). This is a regression from r4, where both tables fit.
   FIX: In docs/src/reference/index.md, drop the Description column and keep `API | Key functions`. Put the one-line description into the API cell as a second line (e.g. `[Preprocessing](preprocessing.md)<br><small>Filtering, staypoints, H3</small>`). Or hide the Description column under 600px with a CSS rule scoped to this table, for example via an `attr_list` class on a wrapper div (`.ref-modules td:nth-child(2){display:none}`) in the site extra CSS.
2. [visual] [P1] Mobile fold, code block: the line `"2020-01-01 08:00", periods=3, freq="h"` is still clipped at the right edge (`freq="h`). It is 47 characters with its indent. The r4 change log says lines were kept under 40, but this one was not.
   FIX: Break it again: `"2020-01-01 08:00",` on one line, then `periods=3, freq="h",` on the next, both at 8-space indent. Every line is then <= 36 chars.
3. [engagement] [P2] Desktop full page, Modules table, "Key functions" column: the names render as plain inline code, not links. The reader still has to open the module page and search to reach `stay_locations` or `match_trajectory`. The column only half does its job.
   FIX: Link each name to its generated anchor, e.g. `[`stay_locations`](preprocessing.md#fastmob.preprocessing.stay_locations)`, using the same anchor scheme as the summary tables on the leaf pages.
4. [visual] [P1] Mobile full page, footer: "Next" is still clipped to "N" at the right edge, and "Data structures overview" is not visible.
   FIX: In the site extra CSS, under 600px set `.md-footer__inner{flex-direction:column}` and give `.md-footer__link` `width:100%`. Or hide the Previous link on mobile so Next gets the full row.
5. [conciseness] [P2] Code block: 11 of 13 lines still build a toy dataframe. On mobile, the "Start from your input" heading lands at the very bottom of the fold.
   FIX: Use a bundled loader if `fastmob.data.load_dataset` has a small sample (3 lines total). Otherwise accept it.
6. [navigation] [P2] Footer "Previous: Compare mobility laws notebook" crosses into Learn. Known, deferred, not scored.
7. [visual] [P2] Deferred items only: header logo white box, black header in light mode, no icons on reference leaves. Not scored.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile code line clipped): not fixed -- the line was re-split, but the `"2020-01-01 08:00", periods=3, freq="h"` line is still clipped at 390px.
2 (mobile TOC button overlap / footer Next clipped): partial -- the floating TOC button is gone from the mobile fold and full page, and the first task-table row reads in full. The footer "Next" is still clipped to "N".
3 (no Key APIs column): partial -- the column was added and looks good on desktop, but the names are not linked, and on mobile the column is cut off (regression, see issue 1).
4 (toy dataframe too long): not fixed -- the line count grew slightly because of the date_range split.
5 (cross-section Previous link): not fixed (deferred).
6 (deferred visuals): not fixed (deferred).

WHAT WORKS (max 3, so it is not broken in the next round):
- Desktop light and dark: the input/output pair and the one-line backend sentence under the lede show the main promise in 3 seconds. Both themes have clean contrast, and link colors read well in dark mode.
- On desktop, the Key functions column gives a scannable "where is X" map. Keep it, but make it work on mobile.
- The mobile TOC toggle fix: "Start from your input" now wraps cleanly inside 390px with nothing overlapping it.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show how the reference modules connect along the pipeline (raw points -> Data structures/Preprocessing -> Trajectory/Network/Integration -> Measures/Fitting/Models -> Evaluation/Privacy/Visualization). It could replace the "Start from your input" table.
  preferred_source: diagram
  suggested_filename: reference-module-map.png
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- The mobile floating TOC button is now hidden site-wide. Confirmed fixed here.
- The footer Next link is still clipped at 390px. It needs a stacked layout site-wide.
- Three-column tables overflow at 390px. Any page with a code-heavy third column needs a mobile rule that hides or stacks a column.
- Code snippets: enforce line length <= 40 chars (including indent) for mobile.
