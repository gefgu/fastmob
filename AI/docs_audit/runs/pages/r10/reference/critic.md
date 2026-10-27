PAGE: docs/src/reference/index.md   ROUND: 10
SCORES: first_impression=7 engagement=8 navigation=8 scannability=8 visual=7 ia_role=7 conciseness=6 trust=7
OVERALL: 7.3/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Module-map figure (new this round), mobile fold and mobile full page: the diagram is scaled from about 690px to about 350px wide, so its box labels ("Data structures", "Integration", "Visualization") render at about 4-5px and cannot be read. On desktop the labels sit at the top edge of each box, not centred vertically ("Trajectory", "Measures", "Evaluation" hug the top border), so the figure looks unfinished.
   FIX: Regenerate docs/src/assets/images/reference/reference-module-map.png (or better, an SVG) with vertically centred labels. For phones, either add a vertical variant (Raw points at the top, stages stacked downwards, 3 boxes per row) and swap it in with `<picture><source media="(max-width: 600px)" ...>`, or wrap the figure so it scrolls horizontally at a readable size. Also make each box a link to its module page if it becomes an SVG.
2. [conciseness] [P1] The page now has three routes to the same modules in a row: the module-map figure, "Start from your input" (5 rows), and "Modules" (12 rows). On mobile the reader scrolls about 2.5 screens past figure and task table before reaching the module list, which is what a reference hub is for. The toy dataframe is also still 8 of the 12 code lines.
   FIX: Keep the figure and the Modules list, and merge "Start from your input" into the figure caption as one sentence ("Raw points? start at Data structures and Preprocessing; have a trajectory? go to Trajectory and Measures"), or move the task table below Modules. In the code block, use `fastmob.data.load_dataset(...)` if a small bundled sample exists; if not, drop the `uid` column (single-user frames are allowed) to save a line.
3. [scannability] [P2] Modules table, desktop and mobile: the header row "Module" over a single column is still an empty bordered band of about 40px. Sidebar casing still differs from the table ("Data Structures", "Input/Output", "Contact Network" in the sidebar against "Data structures", "Input/output" in the table). Not fixed since r7.
   FIX: Replace the one-column table with a `<div class="ref-modules">` list using the same stacked cell (link, muted description, chips), or hide the thead with CSS for this table. In zensical.toml, rename "Data Structures" to "Data structures", "Input/Output" to "Input/output", and "Contact Network" to "Contact network".
4. [visual] [P2] Header, desktop light and dark: the new outlined logo already reads "Fastkit- Mobility" (stacked), and the site title "Fastkit-Mobility" appears right next to it. The name is shown twice within 100px. The logo itself is no longer clipped, and the white header with the gold rule (light) and black header (dark) look clean and calm.
   FIX: Site-wide: either use the "F" mark only (icon without wordmark) as the logo, or hide the `.md-header__title` text on desktop when the wordmark logo is shown. Keep the mobile header as it is (title only, no logo), which reads well.
5. [trust] [P2] Key-function chips are now links (fixed from r7), but they point at the module page, not the function. Clicking `stay_locations` drops the reader at the top of Preprocessing.
   FIX: Use the generated anchors, e.g. `[`stay_locations`](preprocessing.md#fastmob.preprocessing.stay_locations)`, and `fastmob.vis` -> `visualization.md`. Run `zensical build --strict` to catch wrong anchors.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile `freq="h"` clipped): fixed. `freq` is gone and every line fits at 390px with no clipping, including `"2020-01-01", periods=3`.
2 (toy dataframe too long): partial. 12 lines now, but the data is still 8 of them.
3 (Key functions not linked): partial. The chips are links, but only to the module page, not to anchors.
4 (Modules header row / casing mismatch): not fixed.
5 (cross-section Previous link): fixed. The footer shows only "Next: Data structures overview", so the pager stays inside Reference.
6 (deferred visuals: clipped logo, black light-mode header): fixed. The logo is whole, and the light header is white with a gold rule. A new issue is the duplicated name (issue 4).
r7 image request (module map): done, but it is not readable on mobile (issue 1).

WHAT WORKS (max 3, so it is not broken in the next round):
- The input/output pair plus "pass Polars and you get Polars" sells the backend promise above the fold. Code lines now fit at 390px. Keep every line at 34 characters or fewer.
- The stacked Modules cell (linked name, muted description, linked chips) reads well on desktop and on phone with no broken identifiers. This, with the pipeline figure on desktop, earns scannability 8 and engagement 8.
- The new header (white with a gold rule in light mode, black in dark mode) and the section-bound pager ("Next: Data structures overview" only) are calm and give clear wayfinding. Contrast is good in both themes.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: A phone-legible version of the module map, showing the same pipeline stacked vertically so labels stay at 12px or more at 390px width.
  preferred_source: diagram
  suggested_filename: reference-module-map-vertical.svg
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- Header logo wordmark plus site title shows "Fastkit-Mobility" twice on every desktop page. Pick one.
- Diagram PNGs that are wide (5 or more columns) become illegible on phones site-wide. Audit other pages for the same problem, and prefer SVG with a vertical mobile variant.
- Sidebar title case ("Data Structures", "Input/Output", "Contact Network") against sentence case everywhere else is still open in zensical.toml.
