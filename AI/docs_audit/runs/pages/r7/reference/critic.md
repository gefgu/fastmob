PAGE: docs/src/reference/index.md   ROUND: 7
SCORES: first_impression=7 engagement=7 navigation=8 scannability=8 visual=7 ia_role=7 conciseness=6 trust=7
OVERALL: 7.1/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] REGRESSION. Mobile fold, code block, `date_range` line: the line ends at `periods=3, freq=` and `"h"` is cut off at the right edge of the block. The new phone gutter on code blocks (about 16px each side, added for Features this round) makes the block narrower, so a line that fit in r6 overflows now. The first code a phone reader sees on the reference hub is clipped.
   FIX: In docs/src/reference/index.md, split the call over two arguments lines: `"2020-01-01", periods=3,` then `freq="h",` on its own line. A better option is to drop `freq` entirely, as in `pd.date_range("2020-01-01", periods=3)`: it defaults to daily, and the radius of gyration does not depend on time. Keep every line at 34 characters or fewer, indent included, because the gutter now costs about 6 characters.
2. [conciseness] [P2] Fold, code block: building the toy dataframe still takes 8 of the 13 lines, so the API call that matters is the last line. On mobile, "Start from your input" sits at the bottom of the fold.
   FIX: Use a bundled sample if `fastmob.data.load_dataset` provides one (`df = fastmob.data.load_dataset("<sample>")`). If not, put `datetime` on one line (`pd.date_range("2020-01-01", periods=3)`) and drop the `uid` column so `df` is 5 lines.
3. [engagement] [P2] Modules table, key-function chips (`stay_locations`, `match_trajectory`, ...): the chips are still plain inline code. A reader who spots a function must open the module page and search for it.
   FIX: Link each chip to its generated anchor, e.g. `` [`stay_locations`](preprocessing.md#fastmob.preprocessing.stay_locations) ``.
4. [scannability] [P2] Modules table header: there is a single column headed "Module", so the header row adds nothing but a bordered empty band (desktop full page, about 40px). Names also differ from the sidebar: the sidebar says "Data Structures" and "Input/Output", while the table says "Data structures" and "Input/output".
   FIX: Remove the header row by turning the table into a definition-style list or a `<div class="ref-modules">` card list with the same content. Make the casing match zensical.toml nav titles (sentence case in both places).
5. [navigation] [P2] Footer "Previous: Compare mobility laws notebook" crosses into Learn. Known and deferred; not scored.
6. [visual] [P2] Deferred: header logo image clipped ("Fastkit Mobilit"), black header in light mode, no icons on reference leaf pages. Not scored.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile Key functions breaking mid-identifier): fixed. Each module is now one stacked cell: link, muted description, and chips. At 390px every identifier is whole (`fit_values_to_truncated_powerlaw`, `load_geolife_trajectories`, `match_trajectory`), and each row is 3-4 short lines.
2 (toy dataframe too long): partial. It went from 15 to 13 lines, but the data is still 8 of them.
3 (Key functions not linked): not fixed.
4 (cross-section Previous link): not fixed (deferred).
5 (deferred visuals): not fixed (deferred).
r6 "what works" check, mobile code-line fit: REGRESSED. `freq="h"` is clipped again (new issue 1).

WHAT WORKS (max 3, so it is not broken in the next round):
- The stacked Modules cell (linked name, muted one-line description, key-function chips) works at both desktop and 390px with no broken identifiers. It is a clean "where is X" map, and that earns the scannability 8. Keep this structure.
- The input/output pair plus "pass Polars and you get Polars" sells the backend promise above the fold. Both themes have good contrast, including yellow links on dark.
- The "Start from your input" task-to-module table is short and gives a first-time reader a next step. The mobile footer pager stays stacked and fully visible.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show how the reference modules connect along the pipeline (raw points -> Data structures/Preprocessing -> Trajectory/Network/Integration -> Measures/Fitting/Models -> Evaluation/Privacy/Visualization). It could sit above or replace "Start from your input".
  preferred_source: diagram
  suggested_filename: reference-module-map.png
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- The new phone gutter on code blocks narrows mobile code width by about 32px. Re-check every page whose snippets were tuned to the old 40-character limit. The new safe limit is about 34 characters, indent included. Reference index is already broken by this.
- Sidebar nav titles ("Data Structures", "Input/Output") use title case, while page and table text use sentence case. Pick one in zensical.toml.
