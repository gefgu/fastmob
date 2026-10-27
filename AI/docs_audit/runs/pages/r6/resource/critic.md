```
PAGE: docs/src/resource/index.md   ROUND: 6
SCORES: first_impression=6 engagement=6 navigation=8 scannability=8 visual=7 ia_role=7 conciseness=8 trust=7
OVERALL: 7.1/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [first_impression] [P1] Above the fold (desktop-light-fold, desktop-dark-fold) the three cards look the same as in round 5. The PyPI card is still five lines ("Install with / pip install fastmob; / prebuilt wheels for / Linux, macOS and / Windows.") next to three-line siblings. The install command sits mid-sentence on its own wrapped line instead of opening the card. This is the round-5 fix, and it was not applied.
   FIX: In docs/src/resource/index.md, change the PyPI card body to `<p><code>pip install fastmob</code><br>Wheels for Linux, macOS, Windows.</p>`. Keep the other two bodies at two short lines ("Rust kernels, Python API, tests." / "Bugs and questions on GitHub.") so all three cards match in height.
2. [trust] [P1] The Papers section (desktop-light-full, under "Background reading") is new and useful, but none of the four citations is a link. A researcher still has to search for each one. The license line ("Check each dataset's own page...") is generic and sits after Papers, away from the Datasets list it refers to.
   FIX: Link each paper title to its DOI (JSS 10.18637/jss.v103.i04; CEUS trackintel DOI; KDD 2011 DOI; WWW 2009 DOI). Move the license sentence to sit directly under the Datasets list, and name the terms where they are known (e.g. "Brightkite: SNAP, free for research use").
3. [visual] [P2] On desktop (desktop-light-full, "Terms used in these docs") the horizontal diagram is still about 470px wide in an about 565px column, and its sublabels are tiny. The "movement between stops" sublabel runs into the Triplegs box border. On mobile, the new vertical diagram (mobile-full) fixes the crop, but the same sublabel is clipped ("movement between stop") at the box edge.
   FIX: In mobility-hierarchy.svg and mobility-hierarchy-vertical.svg, widen the Triplegs node or shorten the sublabel to "between stops". In extra.css, let the desktop figure fill the column (`.figure img { width:100%; max-width:none }`).
4. [visual] [P2] On mobile (mobile-full, Class/Meaning table) the Class column now holds only the identifier, but it still takes about half the width. Every Meaning cell wraps to 3-4 lines, and the table runs about 1.3 screens.
   FIX: Give the first column a narrow width on small screens (e.g. `@media (max-width:600px){ .md-typeset table td:first-child{ width:1%; white-space:nowrap } }`, scoped to this page's table), or turn the table into a definition list below 600px.
5. [engagement] [P2] The Datasets section is still text-only (the round-4 and round-5 image request, still open). It is the only section without a visual cue.
   FIX: Add the datasets thumbnail strip (see IMAGE REQUESTS) under "## Datasets".
6. [visual] [P2] The "Citing fastmob" tip (bottom of desktop-light-full) is still the only teal element on a yellow/olive theme.
   FIX: Site-wide, map the `tip` admonition colour to the theme accent in extra.css.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (trust, dataset licenses / papers): partial. A Papers list with four correct citations and a license pointer were added. There are no links or DOIs, and the license line is generic and placed after Papers.
2 (first_impression, uneven cards / pip line): not fixed. The PyPI card is unchanged at five lines.
3 (visual, diagram scale and mobile crop): partial. The mobile vertical variant via <picture> works with no scroll hint, and Tours and the dashed label are visible. The desktop figure is still small, and the Triplegs sublabel overflows its box in both variants.
4 (visual, tall mobile glossary rows): partial. The prose prefix moved into Meaning, but the column split still makes rows 3-4 lines tall.
5 (navigation, mobile footer "Develop..." ellipsis): fixed. The pager stacks and shows "Development setup" in full.
6 (engagement, datasets thumbnail): not fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- The mobile vertical hierarchy diagram served via <picture> is readable at 390px with no crop or scroll hint. Keep this pattern.
- Clear wayfinding: the About sidebar highlights Resources, the right TOC (Datasets / Background reading > Papers / Terms) matches the page, and the footer pager now stacks cleanly on phones.
- The Papers list gives the page real reference value: the two library papers plus the source papers for Brightkite and GeoLife, each tied to its dataset.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show readers what each dataset looks like before they pick a loader (Brightkite: global check-in scatter; GeoLife: dense Beijing GPS traces).
  preferred_source: notebook-real-data
  suggested_filename: datasets-thumbnails.png
  suggested_location: docs/src/assets/images/resource/
  notebook_recipe: Load a 10k-row sample of tests/shared/data/loc-brightkite_totalCheckins.txt.gz and one GeoLife user via load_geolife_trajectories. Draw two side-by-side panels in matching style: a world lat/lng scatter (Brightkite) and one user's trajectory over Beijing (GeoLife). Use a transparent background and keep each panel about 320px tall.

CROSS-PAGE NOTES:
- The tip admonition is teal on a yellow/olive theme. Map it to the accent site-wide.
- SVG sublabels in hierarchy diagrams overflow node boxes (Triplegs). Check the same SVG on Home.
- Deferred (P2 only): header logo white box, black light-mode header, footer Previous crossing sections, no card icons.
```
