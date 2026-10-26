```
PAGE: docs/src/resource/index.md   ROUND: 5
SCORES: first_impression=6 engagement=6 navigation=8 scannability=8 visual=7 ia_role=7 conciseness=8 trust=6
OVERALL: 7.0/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [trust] [P1] Datasets and Background reading (desktop-light-full, mid page) have not changed since round 3. No dataset has a license or terms, so a researcher cannot tell whether Brightkite or GeoLife can be reused or cited. Background reading lists only library homepages and no papers. This is the main thing keeping the page from being a reference people come back to.
   FIX: In docs/src/resource/index.md, end each dataset bullet with its terms and a citation pointer, e.g. "· SNAP, cite Cho et al. KDD 2011" for Brightkite and "· Microsoft Research, cite Zheng et al. 2010" for GeoLife. Under Background reading, add a "Papers" sub-list with 2-3 entries: the scikit-mobility paper (Pappalardo et al., J. Stat. Softw. 2022), the trackintel paper (Martin et al., CEUS 2023), and Brightkite's source paper.
2. [first_impression] [P2] Above the fold (desktop-light-fold, desktop-dark-fold), the three link cards are still flat text boxes. The PyPI card puts `pip install fastmob` on its own line, which pushes the card to five lines while its siblings have three. The cards look uneven, and the most useful action (copying the install command) does not stand out.
   FIX: Shorten the PyPI card body to "<code>pip install fastmob</code><br>Wheels for Linux, macOS, Windows" so the command starts the card and fits on one line. Trim the other two bodies to about the same length so all three cards are the same height. Card icons stay deferred.
3. [visual] [P2] On desktop the hierarchy diagram (desktop-light-full, "Terms used in these docs") is about 550px wide in a 670px column. Its sublabels ("where you stopped", "trips back to a place", "tours return to a Location") are about 9px. On mobile (mobile-full) it is still cropped at Trips, so Tours and the dashed label sit behind a "scroll →" hint.
   FIX: Let the SVG fill 100% of the content width on desktop (in extra.css, set `width:100%; max-width:none` on `.figure-scroll img`). Below 600px, serve a top-to-bottom variant through `<picture><source media="(max-width:600px)" srcset="...-vertical.svg">`, as Features already does for its chart.
4. [visual] [P2] On mobile (mobile-full, Terms table) the identifiers now stay whole, but every row wraps over 3-4 lines because the "Trajectory:" prose prefix and the class name share a narrow column. The table is about 1.5 screens tall on a phone.
   FIX: Drop the prose prefix and show only the linked class name in the Term column (e.g. "[`Staypoints`]"). Move the plain-English name into the Meaning cell ("Staypoint: a period spent..."). The first column then becomes one short line.
5. [navigation] [P2] On mobile (mobile-full, footer) the Next label is cut off as "Develop...".
   FIX: This is site-wide. Let the footer prev/next titles wrap to two lines below 600px instead of ellipsizing (`white-space: normal` on `.md-footer__title`).
6. [engagement] [P2] Datasets is still text-only, so readers cannot see what each dataset looks like. This is the round-4 image request, still open.
   FIX: Add the datasets thumbnail strip (see IMAGE REQUESTS) directly under the "## Datasets" heading.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (visual, floating mobile TOC button overlapping content): fixed. mobile-fold and mobile-full no longer show the 38px button, and the Datasets bullets are unobstructed.
2 (visual, mobile glossary identifiers breaking mid-word): fixed. TrajDataFrame, Positionfixes and FlowDataFrame render intact at 390px. Rows are still tall (new issue 4).
3 (first_impression, flat cards / pip wrap): partial. The code span no longer splits, but it still drops to its own line and makes the PyPI card taller. Icons are deferred.
4 (trust, dataset licenses / papers): not fixed.
5 (visual, diagram scale and mobile crop): not fixed.
6 (ia_role, glossary size): unchanged, which is acceptable as stated last round.

WHAT WORKS (max 3, so it is not broken in the next round):
- Mobile is now clean. There is no floating overlay, the cards stack at full width, the identifiers stay whole and the list spacing is tight. Keep the `.md-sidebar--secondary` hide rule.
- The Datasets list: each bullet links its source and names the loader or page that uses it.
- Wayfinding: the About sidebar highlights Resources, the right TOC lists the three sections, and the footer prev/next and version line make the page feel maintained.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show readers what each dataset looks like before they pick a loader (Brightkite: global check-in scatter; GeoLife: dense Beijing GPS traces).
  preferred_source: notebook-real-data
  suggested_filename: datasets-thumbnails.png
  suggested_location: docs/src/assets/images/resource/
  notebook_recipe: Load a 10k-row sample of tests/shared/data/loc-brightkite_totalCheckins.txt.gz and one GeoLife user via load_geolife_trajectories. Draw two side-by-side panels in matching style: a world lat/lng scatter for Brightkite and one user's trajectory line over Beijing for GeoLife. Use a transparent background so it works in both themes, and keep each panel about 320px tall.

CROSS-PAGE NOTES:
- The mobile footer next/prev titles are ellipsized ("Develop..."). Allow wrapping site-wide.
- The tip admonition's teal (Citing fastmob) is still the only teal on a yellow/olive theme. Consider mapping `tip` to the theme accent.
- Diagrams in a `.figure-scroll` wrapper are cropped on phones. A vertical SVG variant served through `<picture>` (as on Features) would remove the scroll hint everywhere.
- Deferred (P2 only): header logo white box, black light-mode header, footer Previous crossing sections, no card icons.
```
