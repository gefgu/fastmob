```
PAGE: docs/src/resource/index.md   ROUND: 10
SCORES: first_impression=7 engagement=8 navigation=8 scannability=8 visual=7 ia_role=8 conciseness=8 trust=7
OVERALL: 7.6/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [trust] [P1] The numbers contradict each other inside the Datasets section. The new figure caption says "Brightkite: 4.7M check-ins worldwide" (desktop-light-fold, left panel). The bullet directly below says "about 4M location check-ins". A researcher reading two different sizes six lines apart starts doubting the benchmark numbers too.
   FIX: Pick one figure. SNAP lists 4,491,143 check-ins, so either regenerate the figure caption to say "4.5M" or change the bullet in docs/src/resource/index.md line 29 to match the number the figure was built from. Use the same number on the Benchmarks page.
2. [visual] [P1] On mobile (mobile-full, Class/Meaning table) rows are still 3-4 lines tall. The Class column holds one short identifier but takes about 45% of the width. The table is still the longest block on the phone page, about 1.2 screens. This was flagged in rounds 6 and 7 and is not fixed.
   FIX: Wrap the table in `<div class="glossary-table" markdown>` and add a rule to docs/src/stylesheets/extra.css: `@media (max-width:600px){ .glossary-table td:first-child, .glossary-table th:first-child { width:1%; white-space:nowrap; padding-right:.6em } }`.
3. [visual] [P2] On mobile (mobile-fold bottom / mobile-full, Datasets) the new two-panel dataset figure shrinks to about 95px tall. Its in-image captions ("Brightkite: 4.7M check-ins worldwide", "GeoLife: GPS traces around Beijing") are about 5px and unreadable, and the Beijing panel turns into a grey smudge.
   FIX: Serve a stacked variant on phones through `<picture><source media="(max-width:600px)" srcset="...resources-datasets-vertical.png">`, the same pattern the hierarchy diagram already uses. Or move the two captions out of the PNG into HTML text under the figure.
4. [visual] [P2] On desktop (desktop-light-full, "Terms used in these docs") the hierarchy diagram is still about 565px wide in an about 690px column. Its sublabels ("raw GPS / check-ins", "tours return to a Location") are still about 8px. This is not fixed since round 7.
   FIX: In extra.css, set `.figure--hierarchy img { width:100%; max-width:none; height:auto }` so the SVG fills the column.
5. [scannability] [P2] On desktop (desktop-light-fold, Datasets) there is about 80px of empty space between the bottom of the dataset figure and the first bullet. That is more than the gap above the H2. The figure looks detached from the list it illustrates.
   FIX: Reduce the bottom margin of `.figure` on this page (or globally) to about 0.8em. Optionally put a one-line caption under the image so it reads as part of the list.
6. [engagement] [P2] The PyPI and Source code cards use the identical `</>` icon (desktop-light-fold, first two cards). The icon then signals nothing, and the row looks copy-pasted.
   FIX: In docs/src/resource/index.md, give PyPI a package/box icon (add `icon-package.svg` under docs/src/assets/images/icons/) and keep `icon-code.svg` for Source code.
7. [trust] [P2] The license line under Datasets is still generic ("Check each dataset's own page...").
   FIX: This is optional. Add a short clause per bullet only where the source states its terms. Do not add a section.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (ia_role, "fastmob" in intro prose): fixed. The intro now reads "...background reading for Fastkit-Mobility."
2 (visual, tall mobile glossary rows): not fixed.
3 (visual, undersized desktop hierarchy diagram): not fixed.
4 (visual, teal tip admonition body): fixed. The "Citing Fastkit-Mobility" tip now has a pale-yellow body with a yellow title bar in light mode and an olive one in dark mode, in line with the theme.
5 (trust, generic license line): not fixed. This was optional and remains P2.
6 (engagement, deferred dataset thumbnails / card icons): fixed. The Datasets section now has a real-data Brightkite world map and GeoLife Beijing figure, and the cards carry icons. This raises engagement to 8.
Cross-page (About entry named "About fastmob"): fixed. The sidebar and pager now read "About Fastkit-Mobility".

WHAT WORKS (max 3, so it is not broken in the next round):
- The real-data Datasets figure: a Brightkite world scatter in the accent gold next to a GeoLife Beijing road web. In one glance it shows what each dataset is, in both themes. This is what earns engagement 8.
- The three equal-height link cards with `pip install fastmob` in the PyPI card make the first screen actionable on desktop and mobile.
- The footer pager now stays inside the About section (Previous: About Fastkit-Mobility / Next: Development setup), and the sidebar highlights Resources. Wayfinding is clean.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a phone-readable version of the datasets figure, with the two panels stacked so each map is legible at 390px
  preferred_source: notebook-real-data
  suggested_filename: resources-datasets-vertical.png
  suggested_location: docs/src/assets/images/resources/
  notebook_recipe: Brightkite (cached tests/shared/data/loc-brightkite_totalCheckins.txt.gz) lat/lng scatter in the accent gold above GeoLife traces from load_geolife_trajectories (grey lines, Beijing bbox). 1 column x 2 rows, about 800x1000px, captions at least 28px in the image or omitted in favour of HTML captions. The caption count must match the bullet text.

CROSS-PAGE NOTES:
- New header, light mode: the white bar with the gold bottom rule is clean and the outlined logo is no longer clipped. But the logo wordmark ("Fastkit- / Mobility", stacked) sits directly beside the site-title text "Fastkit-Mobility", so the name appears twice within 60px. Either hide the title text when the logo carries the wordmark, or use the mark-only "F" logo next to the text title.
- New header, dark mode: it is pure black on a pure-black page with no visible rule. The header/body boundary disappears, and only the About tab underline marks it. Keep a 1px rule in dark mode too (gold at low opacity, or `--md-default-fg-color--lightest`).
- Mobile header: it shows only the text title, no logo mark. That is acceptable, but it is inconsistent with desktop.
- Footer pager kept within the section: this works on desktop. On mobile, Previous and Next stack with about 60px of dead space between them (mobile-full bottom). Tighten the gap.
- The Brightkite size (4M / 4.5M / 4.7M) should be one number site-wide (Home, Benchmarks, Resources).
```
