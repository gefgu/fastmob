PAGE: docs/src/reference/index.md   ROUND: 3
SCORES: first_impression=7 engagement=5 navigation=8 scannability=7 visual=6 ia_role=7 conciseness=6 trust=6
OVERALL: 6.5/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P1] Code block under the lede, all screenshots: it is now 13 lines, but it still ends at `fastmob.radius_of_gyration(df)` and never shows the result. The reader sees a dataframe being built and then nothing comes back. There is still no glimpse of which callables live in each module: the Modules table has the same 12 rows as the sidebar, with prose descriptions only.
   FIX: In docs/src/reference/index.md, add an output block right after the call (a `text` fence with the 2-line printed frame: `uid  radius_of_gyration` / `alice  <value>`; get the real value by running it). Add a third column "Key APIs" to the Modules table with 2-3 linked flagship callables per row (e.g. Preprocessing: `filter`, `stay_locations`, `compress`; Measures: `radius_of_gyration`, `jump_lengths`; Network: `map_match`, ...).
2. [visual] [P1] REGRESSION. In the code block on desktop light/dark, the `"datetime": pd.to_datetime([...])` line runs past the block edge (cut off at `"2020-01-01 10`). On mobile (390px) the datetime line and the comment are both cut off (`"2020-01-01 08`, `returned as a pandas DataF`). The first thing a reader sees on every layout is a clipped line.
   FIX: Shorten the dataframe. Use `"datetime": pd.date_range("2020-01-01 08:00", periods=3, freq="h"),` and shorten the comment to `# one row per user, same backend as df` (<= 38 chars), or split it over two lines. No line should go over ~40 characters, so the block fits on mobile.
3. [visual] [P1] Mobile full page, both tables: "Start from your input" and "Modules" run past the right edge of the viewport ("preprocessi", "individual measure", Descriptions cut mid-word). In the footer, "Next" is also cut to "N". The page reads as broken on a phone. The first column wraps to 1 word per line ("Raw GPS / or / check-in / points") while the second column overflows.
   FIX: For the task table, shorten the right-column cells to a single link each (e.g. "Raw points" -> [Data structures](...)) and the left cells to 2-3 words, so the table fits in 358px. If table overflow is site-wide, add CSS in the docs stylesheet: `.md-typeset table:not([class]) { display: table; width: 100%; table-layout: fixed; } td { overflow-wrap: anywhere; }` for screens under 600px. Check that the footer Next link is no longer clipped.
4. [trust] [P2] Footer and lede: still no version anywhere ("BSD-3-Clause licensed · PyPI · GitHub"). The reference hub is the page where readers check which release is documented.
   FIX: Add `vX.Y.Z` to the `copyright` string in zensical.toml, or add a single "Documents fastmob vX.Y.Z" line under the H1.
5. [conciseness] [P2] Code block: 10 of its 13 lines build a toy dataframe, which pushes the task table below the fold on mobile. That is quickstart content, not a reference hub.
   FIX: Once issue 2 is fixed, keep the dataframe to ~5 short lines (date_range, scalar uid). Or replace it with a bundled-dataset loader (`df = fastmob.data.load_...(...)`) if one exists, so the block is 3 lines plus output.
6. [navigation] [P2] Footer: "Previous: Compare mobility laws notebook" still jumps back across sections into Learn (desktop and mobile).
   FIX: Accept as a nav-order artifact, or hide the previous link on section-index pages through a template override. Low priority.
7. [visual] [P2] Deferred known items only (header logo white box in dark mode, black header in light mode, no sidebar icons on reference leaves). Not scored. The mobile floating TOC button still sits on top of the first table row.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (engagement: no output, no flagship APIs / Key APIs column): partial -- df is now defined, but there is still no output and no Key APIs column.
2 (Reference overview not rendered/highlighted in sidebar): fixed -- "Reference overview" is the first sidebar item with active highlight in light and dark.
3 (cross-section Previous link into Learn): not fixed.
4 (mobile comment clipped; TOC button overlap): regressed -- the comment is still clipped on mobile, and the longer datetime line now clips on desktop too; the TOC button overlap remains.
5 (task table should come first, module table under an H2 "Modules"): fixed -- order swapped, both H2s appear in the "On this page" TOC.
6 (no version on page/footer): not fixed.
7 (deferred icons / dark logo): not fixed (deferred).

WHAT WORKS (max 3, so it is not broken in the next round):
- Sidebar now marks "Reference overview" as active and matches the Modules table 1:1 in the same order, so readers can tell where they are.
- Task-first structure: "Start from your input" comes before "Modules", and the right TOC lists both. On desktop the page is a clean two-step hub.
- The compact "Column and backend conventions" note at the end, with a link to the Learn page, makes a good closing pointer.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show how the reference modules connect along the pipeline (raw points -> Data structures/Preprocessing -> Trajectory/Network/Integration -> Measures/Fitting/Models -> Evaluation/Privacy/Visualization). It could replace the "Start from your input" table, which breaks on mobile.
  preferred_source: diagram
  suggested_filename: reference-module-map.png
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- Tables overflow the viewport at 390px on table-heavy pages, and the footer Next link is clipped. A site-wide mobile table CSS rule is needed.
- Code blocks on mobile have no wrap. Keep doc examples to <= 40 characters per line or enable wrapping for short snippets.
- There is no version string in the footer anywhere on the site.
- Cross-section prev/next links still appear at section boundaries.
- The mobile floating TOC button overlaps content.
