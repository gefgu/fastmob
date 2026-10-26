PAGE: reference/index.md   ROUND: 1
SCORES: first_impression=6 engagement=4 navigation=5 scannability=6 visual=6 ia_role=6 conciseness=6 trust=5
OVERALL: 5.5/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [navigation] [P0] Left sidebar, desktop light and dark full shots: the Reference tree stops at "Fitting". Models, Input/Output, Privacy and Data (all in the zensical.toml nav) are not rendered, and the Reference "Overview" entry for this very page is not shown either, so the sidebar has no "you are here" item. The sidebar also opens with the "Data Structures" group fully expanded (9 children), which pushes every other module below it and makes Data Structures look like the whole reference.
   FIX: Check why the sidebar is truncated (sticky sidebar height / max-height in docs/src/stylesheets/extra.css, or the nav block in zensical.toml) and confirm all 13 Reference entries render on the index page. Make sure the "Overview" item for reference/index.md is visible and highlighted. Collapse nested groups (Data Structures, Measures) by default on the index (drop `navigation.expand` or equivalent) so the top level reads as a short, flat list of modules.
2. [ia_role] [P1] "Modules" bullet list vs sidebar: the list and the nav disagree. "Fitting" is in the sidebar but missing from the list; "visualization" is in the list but has no nav entry; "Contact Network" sits in nav under Measures but lives in reference/social/. Last bullet lumps four modules into one run-on line ("Input/output, privacy, data, and visualization"). A reader cannot trust either list to be complete.
   FIX: Make one ordered source of truth. Replace the "Modules" bullets in docs/src/reference/index.md with an `API | Description` table (one row per module, same order as the zensical.toml Reference nav, Fitting included, one row each for I/O, Privacy, Data, Visualization). Then either add Visualization to the nav or drop it from the page, and move reference/social/contact-network.md under reference/measures/ (or move the nav entry).
3. [engagement] [P1] Whole page: nothing is shown, only prose, a table of links and a link list. For the reference hub of a library whose pitch is a scikit-mobility-style API, there is no glimpse of what an API unit looks like (signature, one call).
   FIX: Under the H1, replace the 3-line intro paragraph with a one-sentence lede plus a 4-6 line code block showing the canonical shape (`import fastmob as fm`; `fm.TrajDataFrame(df)`; `fm.radius_of_gyration(tdf)` returning a dataframe in the input backend). Alternatively, turn the "Modules" table into a grid of cards (icon + module name + 1 line + 2-3 flagship function names as links), the scikit-learn API index pattern.
4. [first_impression] [P1] Intro paragraph above the fold ("The reference is the authoritative contract ... behavior that matters at production scale") is generic and takes three lines without saying what is in the reference. "Fastmob" is capitalized here but "fastmob" everywhere else (header, wordmark).
   FIX: Cut to one line, e.g. "Every public function and class in fastmob, grouped by module. Inputs can be any pandas or polars dataframe." Use lowercase "fastmob" consistently.
5. [visual] [P2] "Start from your input" table is narrow (about 460px in a 690px column on desktop), leaving dead space on the right and making it look unfinished next to the full-width admonition below. Light-mode links are olive underlined, dark-mode links are blue: the accent color changes between themes.
   FIX: Let the table span the content width (CSS `table { width: 100% }` for `.md-typeset` tables in extra.css) or give it a fixed two-column layout. Pick one brand-derived link color family for both themes (cross-page, see below).
6. [trust] [P2] Footer and page bottom: only "Made with Zensical"; no version, license, or "last updated". The mobile footer shows only a "Next: Overview" link (no Previous), and "Next: Overview" is ambiguous (which Overview?). Desktop "Previous: Compare mobility laws notebook" jumps back into Learn, which is a confusing cross-section neighbour for a reference hub.
   FIX: Retitle the Data Structures overview nav entry to "Data structures overview" (zensical.toml) so prev/next labels are self-explanatory. Add version + license to the footer via `copyright` in zensical.toml (cross-page).
7. [navigation] [P2] Mobile, after the input table: a floating TOC button overlaps the space just above the "Modules" heading, and no section icons appear for top-level reference leaves (Preprocessing, Trajectory, Network, Integration, Fitting) while Data Structures / Measures children all have icons.
   FIX: Add `icon: lucide/...` front matter to every top-level reference page (preprocessing.md, trajectory.md, network.md, integration.md, fitting.md, models.md, io.md, privacy.md, data.md) so icon use is uniform.

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Start from your input" (You have -> Start here) table is a genuinely useful task-based entry point; keep it near the top.
- The "Column and backend conventions" note is short, relevant to every API and links to the Learn page that explains it; keep it as the closing block.
- Page is short, single-column, and readable on mobile (390px) with no horizontal scroll; tables wrap cleanly.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show how the reference modules relate along the data pipeline (raw points -> preprocessing -> trajectory / data structures hierarchy -> measures / network -> evaluation/models), so the "Start from your input" table has a visual map.
  preferred_source: diagram
  suggested_filename: reference-module-map.png
  suggested_location: docs/src/assets/images/reference/

CROSS-PAGE NOTES:
- Sidebar truncation after "Fitting" was reported in round 0 and is still present; likely a site-wide sidebar height/scroll issue affecting every Reference page.
- Header is solid black in light theme, and link color is olive in light vs blue in dark; unify accent tokens in extra.css.
- 7 top-level tabs (Home, Features, Learn, Reference, Resources, About, Release Notes) remain; round-0 recommendation to collapse to ~5 still stands.
- Footer has no version, license, or project links on any page.
