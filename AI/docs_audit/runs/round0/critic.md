```
PAGE: site (round 0)   ROUND: 0
SCORES: first_impression=4 engagement=4 navigation=4 scannability=5 visual=5 ia_role=4 conciseness=4 trust=5
OVERALL: 4.4/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [first_impression] [P0] Home, above the fold (desktop and mobile): a plain "fastmob" H1, then a logo that reads "Fastkit Mobilit" (a different, truncated-looking brand name), then two dense paragraphs of prose. There is no figure, no CTA button, no code, and no proof of speed. On mobile the whole first screen is H1 + logo + prose, and "Key Features" only starts at the bottom edge. A 60-second visitor cannot see what the library produces or why it is faster.
   FIX: Replace the top of docs/src/index.md with a hero block: a one-line tagline, two buttons ("Get started" -> learn/analyze-a-simple-trajectory, "API reference" -> reference/index.md), a 3-line code snippet (pip install + jump_lengths call), and one real figure (the benchmark speedup chart from docs/src/assets/benchmarks/ or a trajectory map). Drop the duplicate "fastmob" H1 above the logo, use the wordmark that says "fastmob" (or explain Fastkit Mobility once), and cut the second paragraph to one sentence (the scikit-mobility scope disclaimer belongs on About).
2. [navigation] [P0] Learn section, site-wide: Learn pages render with NO left sidebar (learn index and learn/analyze-a-simple-trajectory: empty left column). On a child page the "Learn" tab is not even highlighted (analyze page header: every tab is grey), so the reader loses their place and cannot move to the next tutorial. Cause: zensical.toml uses `{ "Learn" = "learn/index.md", children = [...] }`, which the project's own CLAUDE.md says Zensical does not support; "Release Notes" uses the same broken form.
   FIX: In zensical.toml, rewrite Learn and Release Notes in the native nested-list form used by Reference/Measures: `{ "Learn" = [ { "Overview" = "learn/index.md" }, ... ] }` (navigation.indexes makes the first entry the section index). Validate with `uv run --group docs zensical build --strict` and re-screenshot the analyze page to confirm the sidebar and active tab.
3. [ia_role] [P1] Top-level IA is 7 tabs with mixed roles (Home, Features, Learn, Reference, Resources, About, Release Notes). Home also acts as an install guide, dev-setup guide, column-name reference and model showcase: the full Home shot contains "Installation" AND "Install from PyPI" (pip shown twice), "Quick Example" AND "Your First Measure: jump_lengths" (the same call twice), a dev-from-source tab block, a column auto-detection table, a Polars section, and an unexplained "Generation Models" block that ends the page with no next step.
   FIX: Collapse the tabs to 5: Home, Get started (install + Learn), User guide/Examples, API reference, About (fold Features/Benchmarks, Resources and Release Notes under About or the footer). On Home, keep one install line and one example. Move "Install for Development" to a CONTRIBUTING/About page, the column table to learn/use-custom-columns..., and "Generation Models" to Learn. End Home with a 3-card "Next steps" grid (Learn / Reference / Benchmarks).
4. [visual] [P1] Theme: the light theme is not light. The header bar is solid black in light mode, and Learn index cards are near-black slabs (#1b1b1b) with saturated yellow text on a white page (learn desktop-light: 8 black cards stacked down 1,300px), which reads as harsh and inconsistent with the cream code blocks and pale-blue admonitions on the same page. In dark mode the logo renders as a white rectangle (home desktop-dark, logo area) and the header logo is a white box too. Light-mode link color is olive (#6b6b00-ish) and dark-mode links switch to blue, so the accent is not consistent across themes.
   FIX: In the theme CSS (docs/src stylesheets / DESIGN.md tokens), make light-mode cards use the surface token (white/very light grey with a 1px border, as the dark theme already does) and keep yellow only for the small eyebrow/arrow. Use a transparent-background SVG logo for dark mode (`logo` variant or CSS `filter`/separate asset). Pick one link color per theme derived from the brand accent (a dark gold in light, the same yellow family in dark), not olive vs blue.
5. [conciseness] [P1] Reference > Measures > Individual is one 55,000px page. Each function repeats the same ~12-line Brightkite `read_csv` loader in its Example (jump_lengths, waiting_times, ...). Example outputs overflow and are clipped on the right edge (the `location id` column is cut mid-hash), and the right-hand TOC lists most functions twice (distance_straight_line > distance_straight_line, diversity > diversity, ...) because a module heading and a function heading are both emitted (visible at the page end: "fastmob.measures.individual.work_location" followed by a "work_location(...)" heading).
   FIX: Split docs/src/reference/measures/individual.md into one short page per function group (or use `show_root_heading`/`show_submodules` options so only the function heading is emitted, which fixes the TOC duplication). Replace the repeated Brightkite loader with a 4-line synthetic DataFrame or a single `fastmob.data.load_brightkite_sample()` call, and trim printed outputs to columns that fit (`df[["uid","datetime","lat","lng"]]`).
6. [scannability] [P1] API units are hard to scan. The signature renders as a highlighted pill per wrapped line (individual fold: three separate grey bars for activity_transition_matrix), the Type column wraps "str or / None" onto two code chips, raw Sphinx markup leaks (jump_lengths_road: ":func:`~fastmob.measures...jump_lengths`"), and "See Also" is an unlinked run-on paragraph. The intro "Question / Representative APIs" table uses code spans that are not links, and the page lacks the `API | Description` summary table the project conventions require.
   FIX: Render signatures as a single code block (mkdocstrings `separate_signature: true`, `line_length` ~ 80). Convert docstrings' `:func:` roles to Markdown cross-refs. Make "See Also" a linked list. Add the `API | Description` summary table right after the H1, with each name linked to its anchor, and make the Question-table entries links.
7. [trust] [P2] Version signals disagree: the header badge says fastmob-v0.3.2, but the nav's Release Notes stop at v0.1.3. Home "Requirements" lists "Rust toolchain" as if users need it to `pip install` wheels. The footer is only "Made with Zensical" with no license, version, or links.
   FIX: Add release notes for 0.2.x-0.3.2 (or one "Changelog" page linking to GitHub releases) and drop per-version nav children. Reword Requirements to "Python >= 3.9 (Rust only needed when building from source)". Add a footer with license, current version, GitHub, and PyPI links (`extra.social` / copyright in zensical.toml).

PREVIOUS ISSUES STATUS (rounds >= 2): n/a (round 0)

WHAT WORKS (max 3, so it is not broken in the next round):
- Reference index "Start from your input" table (You have -> Start here) is a good task-based entry into the reference; keep it and its placement.
- Icons on Reference sidebar children (Data Structures, Measures) are consistent, line-style, and meaningful. Extend the same style to the remaining leaves rather than removing it.
- The analyze-a-simple-trajectory tutorial has a good shape: numbered steps in the right TOC, code followed by expected output and a one-line "Notice that..." interpretation for each step. Mobile code blocks go full-bleed without horizontal page scroll.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Home hero proof of speed: fastmob vs scikit-mobility wall time at 1k-4M rows for 3-4 measures, so the "fast" claim is shown, not told.
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: Brightkite (cached tests/shared/data/loc-brightkite_totalCheckins.txt.gz); use existing benchmark JSON from scripts/run_benchmarks.sh --benchmark-json for jump_lengths, radius_of_gyration, filter, stay_locations; log-log line or grouped bar of median time vs rows, fastmob pandas/polars vs skmob, annotate the speedup at 4M.
- purpose: Show what the first tutorial computes: alice's and bob's 3 points on a map with jump-length labels on each segment and a circle of radius_of_gyration around each center of mass.
  preferred_source: notebook-real-data
  suggested_filename: simple-trajectory-jumps-rg.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: the 6-row DataFrame from learn/analyze-a-simple-trajectory.md; fastmob.jump_lengths and fastmob.radius_of_gyration; plot two small-multiple panels (Rome, New York), points + connecting segments labeled in km, dashed circle of radius rg around the centroid.
- purpose: Explain the Positionfixes -> Staypoints -> Triplegs -> Trips -> Tours hierarchy that Home names in prose, in one picture.
  preferred_source: diagram
  suggested_filename: mobility-hierarchy.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: (optional real-data variant) one GeoLife user-day: generate_staypoints/generate_triplegs/generate_trips, plot raw fixes grey, staypoints as circles, triplegs colored per trip.

CROSS-PAGE NOTES:
Home
- Two H1s ("fastmob" and "# Key Features" in index.md); demote Key Features and every later section to H2.
- "Key Features" is a bullet list of bold claims with no numbers or links. Turn it into a 3x2 card grid with an icon each, linking to the relevant Learn/Reference page and to features/benchmarks.md for the speed claim.
- The page ends on a cropped "Generation Models" code block (trajectory line runs off the box) with no closing CTA.
Learn index
- Page H1 is "How-to guides" while the tab says "Learn". The nav lists 9 entries, but the page shows 8 cards: "Compare mobility laws notebook" has no card.
- The same tutorial has three names: nav "Start with a trajectory", card "Inspect a minimal trajectory", page H1 "Analyze a simple trajectory". Use one title everywhere.
- Cards have no thumbnails. Per the scikit-learn benchmark, give each card one small figure (the gallery pattern), and label the two notebooks consistently ("Notebook" badge) instead of "SCENARIO 0n" eyebrows that carry no information.
- The bottom admonition "Use the right level of detail" repeats the intro sentence; remove one.
Learn: analyze-a-simple-trajectory
- No figure anywhere on the page (full shot is code/output only); see image request 2.
- No previous/next footer links and no "Next: Columns and backends" at the end of "What we have made"; enable `navigation.footer` in zensical.toml theme features.
- Step 4 repeats the entire DataFrame from Step 1 verbatim (~20 lines); show only the added lines or collapse it into a `??? "Full script"` block.
- Light-mode code blocks use a cream/beige background that clashes with the white page and pale-blue admonitions; align with the neutral surface token.
Reference index
- The left sidebar is cut off after "Fitting" in both the fold and full shots (Models, Input/Output, Privacy, Data not visible without scrolling the sidebar itself), and the Reference "Overview" item is not shown. Check the sidebar height/sticky behavior.
- Top-level leaves (Preprocessing, Trajectory, Network, Integration, Fitting, ...) have no icons while Data Structures/Measures children do; either add `icon:` front matter to every reference page or to none.
- The "Modules" bullet list duplicates the sidebar; replace it with the `API | Description`-style two-column table or remove it. "visualization" is linked here but has no nav entry.
- "Contact Network" sits under Measures but its file lives in reference/social/; move the file or the nav entry so URL and nav agree.
Reference > Measures > Individual
- Mobile: code spans in the Question table break mid-identifier ("radius_of_gyration_laz / y"). Add `overflow-wrap: anywhere` only at underscores (`word-break: normal` + `<wbr>`) or let that column scroll.
- References and See Also admonitions are expanded by default and repeat for each function; collapse them by default (`??? info`).
- Example outputs wider than the content column are clipped instead of scrolling horizontally.
```
