PAGE: docs/src/reference/index.md   ROUND: 6
SCORES: first_impression=7 engagement=7 navigation=8 scannability=7 visual=7 ia_role=7 conciseness=6 trust=7
OVERALL: 7.0/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile full page, "Modules" table, "Key functions" column: the table now fits at 390px, but the column is only about 80px wide, so inline-code names break in the middle of identifiers: "TrajDataFram|e", "Positionfixe|s", "stay_location|s", "match_traject|ory", "build_road_gr|aph", "join_with_poi|s", "fit_values_to|_truncated_po|werlaw", "location_ris|k". A broken identifier cannot be read or copied, so the column is still close to useless on phones. Each row is also 3-4 lines tall.
   FIX: In docs/src/reference/index.md, show one key function per module, not two (drop `Positionfixes`, `segment`, `smooth`, `build_road_graph`, `join_with_events`, `jump_lengths`, `load_geolife_trajectories`, `SpatialEPR`). Then add a mobile rule scoped to this table in the site extra CSS: wrap the table in `<div class="ref-modules" markdown>` and under 600px set `.ref-modules td:nth-child(2) code{word-break:normal;overflow-wrap:normal;white-space:nowrap}` with `.ref-modules td:nth-child(1){width:55%}`. Another option is to stack the cells under 600px (`.ref-modules tr{display:block}`, `td{display:block}`) so the function list sits under the description at full width.
2. [conciseness] [P2] Fold, code block: 11 of 15 lines still build a toy dataframe (the date_range split added 3 lines). On mobile the "Start from your input" heading is pushed to the very bottom of the fold.
   FIX: If `fastmob.data.load_dataset` has a small bundled sample, replace the dataframe with `df = fastmob.data.load_dataset("<sample>").head(...)` (3-4 lines in total). If not, collapse `date_range` back onto one line using a shorter call such as `pd.date_range("2020-01-01", periods=3, freq="h")` (35 chars at an 8-space indent, so it fits on mobile).
3. [engagement] [P2] Desktop full page, Key functions column: the names are still plain inline code, not links. A reader who spots `stay_locations` must open Preprocessing and search for it.
   FIX: Link each name to its generated anchor, e.g. `` [`stay_locations`](preprocessing.md#fastmob.preprocessing.stay_locations) ``.
4. [navigation] [P2] Footer "Previous: Compare mobility laws notebook" crosses into Learn. Known and deferred; not scored.
5. [visual] [P2] Deferred: white box behind the header logo, black header in light mode, no icons on reference leaf pages. Not scored.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile Modules table overflow, 3 columns): partial. The table is now 2 columns and fits inside 390px, but the narrow Key functions column breaks identifiers mid-word (new issue 1).
2 (mobile code line clipped): fixed. Every line fits at 390px in mobile-fold; `freq="h",` is fully visible.
3 (Key functions not linked): not fixed.
4 (mobile footer Next clipped): fixed. The pager stacks: Previous on one row, then "Next: Data structures overview" right-aligned and fully visible.
5 (toy dataframe too long): not fixed. It is now 15 lines.
6 (cross-section Previous link): not fixed (deferred).
7 (deferred visuals): not fixed (deferred).

WHAT WORKS (max 3, so it is not broken in the next round):
- Desktop light and dark: a clean single-column hub. The input/output pair and the "pass Polars and you get Polars" sentence show the backend promise in seconds, and both themes have good contrast, including yellow links on dark.
- The two-column Modules layout on desktop: a linked module name with a muted one-line description underneath, plus the key functions. It is a scannable "where is X" map. Keep it.
- The mobile footer pager stack and the code-line wrapping are fixed. Do not regress them.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show how the reference modules connect along the pipeline (raw points -> Data structures/Preprocessing -> Trajectory/Network/Integration -> Measures/Fitting/Models -> Evaluation/Privacy/Visualization). It could replace the "Start from your input" table.
  preferred_source: diagram
  suggested_filename: reference-module-map.png
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- Footer pager stacking on phones is confirmed fixed here.
- Any table that has an inline-code column needs a mobile rule that prevents mid-identifier breaks: nowrap on code, plus stacked cells or a wider code column. This probably affects the summary tables on the reference leaf pages too.
- Code snippets: keep lines at 40 characters or fewer, including indent. This page now complies.
