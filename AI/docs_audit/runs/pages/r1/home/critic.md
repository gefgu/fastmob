```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 1
SCORES: first_impression=7 engagement=7 navigation=7 scannability=8 visual=6 ia_role=7 conciseness=7 trust=7
OVERALL: 7.0/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile (mobile-full, "Fast on real data" and "From raw GPS fixes to tours"): both figures are scaled down to the 343px column, so their labels are ~5-6px tall and unreadable ("0.10s (315x faster)", the stage names, and the generate_*() caption are illegible). These two figures are the page's main proof; on a phone they are decoration.
   FIX: Ship a mobile variant of each and swap with <picture><source media="(max-width: 600px)" srcset="..."></picture> in index.md: benchmark as a 5-row chart with larger fonts (or a plain 3-column table "measure | skmob | fastmob | speedup" rendered under 600px), and the hierarchy diagram laid out vertically (top-to-bottom). Alternatively wrap each figure in a horizontally scrollable container with min-width: 640px.
2. [engagement] [P1] "Fast on real data" chart (desktop light/dark): the fastmob bars (0.09-0.15s on a 0-200s axis) render as a 1px yellow tick, so the "faster" series is invisible and the reader only sees grey bars plus yellow text. In light mode the yellow labels (#d4b800-ish on white) are low-contrast. The chart says "fastmob is small" rather than showing a comparison.
   FIX: Regenerate docs/src/assets/images/home/home-benchmark-speedup.png as a log-scale x-axis (0.01s-1000s) dot or bar chart so both series are visible, or plot "speedup (x)" bars directly (315x ... 1329x) with the seconds as labels. Use a darker gold (e.g. #8a6d00) for fastmob labels in the light variant, or produce separate light/dark PNGs with the `#only-light` / `#only-dark` image suffixes.
3. [visual] [P1] Desktop layout (all desktop shots): the content column starts at x~370 of 1440 with an empty ~360px left gutter (no sidebar is shown on Home, toc hidden), so the page looks off-centre and the hero is narrower than it needs to be. In dark mode the hierarchy diagram is five saturated #ffff66 slabs on near-black, the brightest thing on the page, louder than the primary CTA.
   FIX: Add `hide: [navigation, toc]` to the index.md front matter (or a home-only CSS rule in docs/src/stylesheets/extra.css to centre .md-content with max-width ~960px). For mobility-hierarchy.svg, use CSS-variable-driven fills (or a dark variant via #only-dark) with an outlined/low-saturation node style in dark mode and yellow only on the border.
4. [ia_role] [P1] Hierarchy diagram ("From raw GPS fixes to tours"): "Locations" is drawn as the last step after "Tours" (Tours -> Locations arrow), while the alt text and the actual API say Staypoints cluster into Locations (and Tours return to a Location). A cold reader learns the wrong pipeline from the one figure meant to explain it.
   FIX: Redraw docs/src/assets/images/home/mobility-hierarchy.svg with Locations branching off Staypoints (Staypoints -> Locations, above or below), and a dashed "returns to" link from Tours to Locations if needed. Update the caption to include generate_locations().
5. [conciseness] [P2] The page has two link grids with the same destinations: hero buttons (Get started / API reference / Benchmarks) and the closing "Next steps" cards (Learn / Reference / Benchmarks). "Why fastmob" re-states the hero lead (pandas/Polars, Rust kernels) and the benchmark section ("Native speed").
   FIX: Cut "Why fastmob" to two bullets that add new information (Familiar API / skmob compatibility, Tested), or fold them into the hero lead as one line. Change the Next-steps cards to destinations the hero does not already offer (e.g. "Columns and backends", "Stay-point notebook", "Release notes / what's new"), or drop the hero's third button.
6. [first_impression] [P2] Hero code block (fold, all viewports): the snippet calls jump_lengths and radius_of_gyration, but the single output block shows only radius_of_gyration, with identical 9.07904 for both users; it is also visually detached from the tabs (it sits outside them, and on Polars the output does not match the Polars code). The fold ends with no figure: the chart starts below 900px on desktop and after ~1000px on mobile.
   FIX: In index.md, show only the call whose output is printed (drop jump_lengths from the hero, or print its output), give alice and bob different coordinates so the two values differ, and move the output inside each tab. Consider a two-column hero on desktop (text + buttons left, code right) so the benchmark heading appears within the first screen.
7. [trust] [P2] Page end (desktop/mobile full): footer is still only "Made with Zensical" with no license, version, GitHub or PyPI link; the footer pager says "Next: Overview" with no section name (it is Features > Overview), which reads like a dead end on a landing page.
   FIX: Set `copyright` and `extra.social` (GitHub, PyPI) in zensical.toml and show the current version. Either hide the pager on Home (`hide: [footer]` is too broad; use a home-only CSS rule on .md-footer__link) or rename the Features overview nav entry so the pager reads "Features".

PREVIOUS ISSUES STATUS (rounds >= 2):
- R0#1 (Home hero, no figure/CTA/code, logo name): fixed for hero/CTA/code/figure; partial for branding (header logo still reads "Fastkit-Mobility" next to the "fastmob" wordmark).
- R0#3 (Home mixes install/dev-setup/column table/Generation Models, duplicate install and example, no closing CTA): fixed. One install line, one example, dev setup linked out, Generation Models removed, Next-steps grid added.
- R0#4 (theme): partial. Light-mode cards are now light surfaces with a border (fixed); header bar is still solid black in light mode; links still olive in light vs blue in dark.
- R0#7 (trust/version): partial. Requirements wording fixed (no Rust needed for wheels); Release Notes now list v0.3.2/v0.2.0; footer still has no license/version/links.
- R0 cross-page Home notes: two H1s fixed; Key Features turned into a card grid (fixed, though cards have no icons); cropped Generation Models ending fixed.
- R0 image requests: home-benchmark-speedup.png delivered (defects in issue 2); mobility-hierarchy delivered (defects in issues 1, 3, 4).

WHAT WORKS (max 3, so it is not broken in the next round):
- Hero: an H1 that states the claim, a one-sentence plain-language lead, a strong yellow primary CTA, then pip install and pandas/Polars tabs. A stranger can tell what this is within 5 seconds.
- Section rhythm: short H2 sections (Fast on real data / hierarchy / What you can do / Why / Next steps), each a sentence plus one visual or grid. Very scannable on desktop and mobile.
- "What you can do" 3x2 card grid linking to the Learn pages; cards look right in both light and dark and stack cleanly on mobile.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show what fastmob produces on real data, not only how fast: one user's day of raw GPS fixes with detected staypoints and triplegs, as the hero/right-column visual or the top of "What you can do".
  preferred_source: notebook-real-data
  suggested_filename: home-geolife-staypoints-triplegs.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: GeoLife, one user-day; Positionfixes(df).generate_staypoints() and .generate_triplegs(); plot raw fixes as small grey dots, triplegs as colored lines per tripleg, staypoints as larger yellow-outlined circles, light basemap or none; export 1200x700 plus a 600x600 mobile crop, transparent or theme-matched background.
- purpose: Mobile-readable benchmark proof (see issue 1/2): speedup per measure at 4M rows.
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-mobile.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: same benchmark JSON as home-benchmark-speedup.png; horizontal bars of speedup (x) per measure, log x-axis, labels >= 14px at 600px width, light and dark variants.

CROSS-PAGE NOTES:
- Header logo asset (assets/logo_filled.svg) still renders the "Fastkit-Mobility" mark beside the "fastmob" site name in both themes; use a fastmob mark.
- Light theme header bar is still solid black; link color is olive in light and blue in dark. Unify to one accent family per theme in extra.css.
- Top nav still has 7 tabs (Home, Features, Learn, Reference, Resources, About, Release Notes); Release Notes lists 6 per-version children. Consider collapsing as suggested in round 0 issue 3.
- Footer pager shows bare "Overview" for section indexes site-wide; consider naming section overview pages after their section.
```
