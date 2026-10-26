PAGE: docs/src/reference/index.md   ROUND: 4
SCORES: first_impression=7 engagement=6 navigation=8 scannability=7 visual=6 ia_role=7 conciseness=6 trust=7
OVERALL: 6.8/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile fold, code block: the `"datetime": pd.date_range("2020-01-01 08:00", periods=3, freq="h"),` line is still cut at the right edge (`"2020-01-01 08:0`). It is the longest line in the first thing a phone reader sees. Desktop light/dark now fit.
   FIX: In docs/src/reference/index.md, split the call over lines, e.g. `"datetime": pd.date_range(` / `        "2020-01-01 08:00",` / `        periods=3, freq="h"),`. Or drop `uid` and use `periods=3, freq="h"` with a shorter start string. Keep every line <= 40 chars.
2. [visual] [P1] Mobile fold and full page, "Start from your input" table, first row: the floating round TOC button (≡) still sits on top of the "Data structures, then" cell. That text is cut to "the". The r3 change log says the toggle was hidden, but it still renders in the r4 mobile screenshots. In the mobile footer, "Next" is still clipped to "N" at the right edge.
   FIX: Check the site CSS rule that is meant to hide the mobile TOC toggle. It does not match on this page, so the selector or the stylesheet load order is likely wrong. In the footer, let `.md-footer__link` wrap or stack Previous/Next vertically under 600px so "Next: Data structures overview" shows in full.
3. [engagement] [P1] Modules table (12 rows): it still only repeats the sidebar labels with prose descriptions. A reader looking for "where is `stay_locations`" or "`map_match`" cannot find it from the hub without opening each page. The output block was added (fixed), but the Key APIs column was not.
   FIX: Add a third column "Key APIs" to the Modules table with 2-3 linked flagship callables per row (Preprocessing: `filter`, `stay_locations`, `compress`; Measures: `radius_of_gyration`, `jump_lengths`; Network: `map_match`; Data: the loader names). On mobile, drop the Description column instead, so the table stays two columns wide.
4. [conciseness] [P2] Code block: 7 of its 11 lines build a toy dataframe. On mobile, this pushes "Start from your input" past the fold. This is quickstart material on a reference hub.
   FIX: If `fastmob.data` has a small bundled loader, use it (`df = fastmob.data.load_...()` plus the call, 3 lines). Otherwise keep the dataframe and accept this as polish.
5. [navigation] [P2] Footer: "Previous: Compare mobility laws notebook" jumps back into Learn. Known, deferred.
   FIX: None this round. If touched later, hide the previous link on section-index pages through a template override.
6. [visual] [P2] Known deferred items only: header logo white box, black header in light mode, no icons on reference leaves. Not scored.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (engagement: no output / no Key APIs column): partial -- an output block with the real value (9.878827) and a backend sentence were added, but there is no Key APIs column.
2 (code line clipped on desktop + mobile): partial -- fixed on desktop light/dark, still clipped on mobile.
3 (mobile tables overflow / footer Next clipped): partial -- both tables now wrap inside 390px with no mid-word cuts, but the footer "Next" is still clipped to "N".
4 (no version): fixed -- the footer reads "fastmob 0.3.2 · BSD-3-Clause licensed".
5 (toy dataframe too long): partial -- 13 lines down to 11 with date_range; still dominates the mobile fold.
6 (cross-section Previous link): not fixed (deferred).
7 (deferred visuals; mobile TOC button overlap): not fixed -- the TOC button still overlaps the first task-table row on mobile.

WHAT WORKS (max 3, so it is not broken in the next round):
- The input/output pair right under the lede, followed by "The result is a pandas DataFrame because df is pandas; pass Polars and you get Polars," shows the backend-preserving promise in about 3 seconds on desktop.
- Task-first structure ("Start from your input", then "Modules"), and the sidebar "Reference overview" active state matches the Modules table 1:1 in the same order.
- The footer version string (0.3.2) and the short "Column and backend conventions" admonition close the page credibly.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show how the reference modules connect along the pipeline (raw points -> Data structures/Preprocessing -> Trajectory/Network/Integration -> Measures/Fitting/Models -> Evaluation/Privacy/Visualization). It could replace the "Start from your input" table.
  preferred_source: diagram
  suggested_filename: reference-module-map.png
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- The mobile floating TOC toggle is still visible and overlaps content, even though the r3 change log says it is hidden. Verify the CSS site-wide.
- Footer Next link is clipped at 390px site-wide. It needs a wrap or stacked layout.
- Code blocks on mobile do not wrap. Keep snippet lines <= 40 chars.
- Cross-section prev/next links at section boundaries remain (deferred).
