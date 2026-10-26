PAGE: docs/src/reference/index.md   ROUND: 2
SCORES: first_impression=7 engagement=5 navigation=7 scannability=7 visual=7 ia_role=7 conciseness=7 trust=6
OVERALL: 6.6/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P1] Below the lede, desktop and mobile: the only "show" is a single call, `fastmob.radius_of_gyration(df)`, with no `df` defined and no output, followed by a 12-row link table. A cold reader still cannot see what an API unit looks like (signature, parameters, returned frame) or which functions live in each module. The module table repeats the sidebar word for word, so it adds little.
   FIX: In docs/src/reference/index.md, add a third column "Key APIs" to the `API | Description` table listing 2-3 flagship callables per module as links to their anchors (e.g. Preprocessing: `filter`, `stay_locations`, `compress`; Measures: `radius_of_gyration`, `jump_lengths`). Grow the code block to 4-5 lines: build a tiny dataframe (or `fastmob.data.load_...`) and show the printed result (2-3 rows of `uid | radius_of_gyration`) as a second block or a comment.
2. [navigation] [P1] Left sidebar, desktop light/dark: the list starts at "Data Structures". The "Reference overview" entry declared first in zensical.toml is not rendered, so nothing in the sidebar is highlighted on this page and there is no "you are here" marker or way back to the hub from deeper reference pages except the tab.
   FIX: Ensure the overview entry renders: either keep `{ "Reference overview" = "reference/index.md" }` and check that the tab index is not being hidden (Zensical `navigation.indexes`/tabs behaviour), or rename it so it appears as the first sidebar item with active styling. Verify on this page that one sidebar item is highlighted.
3. [navigation] [P2] Footer prev/next, desktop: "Previous: Compare mobility laws notebook" jumps back into Learn, a cross-section neighbour that makes no sense from a reference hub. Mobile shows only "Next".
   FIX: Accept as a nav-order artifact or suppress the cross-section Previous link on section index pages (Zensical footer config / page front matter `hide: [footer]` is too blunt; prefer leaving Next only consistently on both layouts).
4. [visual] [P2] Mobile (390px), code block under the lede: the comment is clipped at "# one row per" with a horizontal scroller, so the one explanatory sentence on mobile is cut off. A floating TOC button overlaps the "Integration" description row.
   FIX: Move the comment to its own line above the call (`# one row per user, returned in df's backend`), so no line exceeds ~38 chars. TOC button overlap is site-wide (see cross-page).
5. [ia_role] [P2] "Start from your input" table, second half of page: it re-links the same modules as the table above, so the page is two link tables back to back. Useful, but it should come first because it is the task-based entry and the module table duplicates the sidebar.
   FIX: Move "Start from your input" directly under the code block and put the module table (with the Key APIs column from issue 1) below it under an H2 "Modules" so the TOC shows both sections.
6. [trust] [P2] Footer: license, PyPI and GitHub links are now present (good) but there is still no version anywhere on the page; a reference hub is where readers check "which version is this documenting".
   FIX: Add the version to the footer `copyright` string in zensical.toml (or a "Documents fastmob vX.Y.Z" line under the H1 lede).
7. [visual] [P2] Deferred known items only: no per-page sidebar icons, dark-mode header logo white box. Not penalised further.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (sidebar truncated after Fitting / Data Structures expanded / no overview item): partial -- all 12 modules now render and groups are collapsed; the "Reference overview" item is still not visible/highlighted.
2 (Modules list vs nav mismatch, Visualization, Contact Network): fixed -- API | Description table matches the nav order 1:1, Visualization added to nav, contact-network moved under measures.
3 (no code/glimpse of API): partial -- one-line code example added, but no output and no flagship function names.
4 (generic 3-line intro, "Fastmob" casing): fixed -- one-sentence lede, lowercase fastmob.
5 (narrow table, link colour differs between themes): fixed for width (tables span content column); link accent is olive in light / yellow in dark, consistent brand family -- acceptable.
6 (footer lacks version/license; ambiguous "Next: Overview"; Learn Previous): partial -- "Next: Data structures overview" is clear, license/PyPI/GitHub added; still no version; Previous still points into Learn.
7 (mobile TOC button overlap; missing icons): not fixed (icons deferred; TOC button still overlaps table rows on mobile).

WHAT WORKS (max 3, so it is not broken in the next round):
- Above the fold now answers "what is in here" in one line, with a code block and the full module table starting on the first screen (desktop).
- The module table and the sidebar agree exactly, in the same order, all 12 modules; tables span the column width and read cleanly in light and dark.
- "Start from your input" task table and the short "Column and backend conventions" note remain compact and useful.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Map of how reference modules connect along the pipeline (raw points -> Data structures/Preprocessing -> Trajectory/Network/Integration -> Measures/Fitting/Models -> Evaluation/Privacy/Visualization), placed beside or replacing the "Start from your input" table.
  preferred_source: diagram
  suggested_filename: reference-module-map.png
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- Section-index sidebar entries ("Reference overview", likely also "Learn overview") do not render in the sidebar; check all sections.
- Mobile floating TOC button overlaps content rows on table-heavy pages; consider hiding it on short pages or giving it an opaque backdrop/offset.
- Footer has license and links now but no version string site-wide.
- Cross-section prev/next links (Learn notebook -> Reference hub) appear on every section boundary.
