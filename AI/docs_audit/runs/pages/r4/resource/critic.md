```
PAGE: docs/src/resource/index.md   ROUND: 4
SCORES: first_impression=6 engagement=6 navigation=8 scannability=8 visual=6 ia_role=7 conciseness=8 trust=6
OVERALL: 6.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] The floating mobile TOC button is still there. The change log says it was hidden, but mobile-fold (bottom right) and mobile-full show it sitting on the first Datasets bullet and covering "check-ins". Either the CSS rule did not ship or its selector misses this element.
   FIX: In extra.css, hide the floating TOC button below 600px and target the element that actually renders it. Inspect the 390px build, since the round-3 rule clearly does not match. Then re-screenshot to confirm it is gone.
2. [visual] [P1] Mobile glossary (mobile-full, Terms table): merging the columns created a new defect. The inline-code class names now break mid-identifier, as in "TrajDataFram|e", "Positionfixe|s" and "FlowDataFram|e". Each term cell also wraps over 2-3 lines.
   FIX: Give `.md-typeset td code` `white-space: nowrap` (or `word-break: keep-all`) for this table. Alternatively, drop the "Trajectory:" / "Position fix:" prose prefix and show only the linked class name, which is short enough to fit. A third option: below 600px, render the term on its own line above the meaning (a definition list instead of a table).
3. [first_impression] [P2] Above the fold (desktop-light-fold, desktop-dark-fold), the three link cards are still flat, icon-less text boxes. The PyPI card still breaks `pip install fastmob` across two lines, so it is taller than its siblings.
   FIX: Add a lucide icon per card (package / github / life-buoy) with the same markup Features uses. Shorten the PyPI body to "`pip install fastmob` · wheels for Linux, macOS, Windows" and add `white-space: nowrap` on that code span so all three cards are the same height.
4. [trust] [P2] The Datasets list still has no license or terms. A researcher cannot tell whether Brightkite or GeoLife can be reused. Background reading is still library links only, with no papers.
   FIX: Append the size and license to each dataset bullet (e.g. "· SNAP terms", "· Microsoft Research license"). Add 2 papers under Background reading: the scikit-mobility JSS paper and the trackintel paper.
5. [visual] [P2] On desktop the hierarchy diagram renders at about 550px wide inside a 670px column. Its box sublabels ("where you stopped", "tours return to a Location") are about 9px and hard to read. On mobile it is cropped (Trips and Locations cut off) behind a "scroll →" hint.
   FIX: Let the SVG scale to 100% of the content width on desktop. On mobile, accept the scroll, or offer a vertical (top-to-bottom) variant of the SVG below 600px via `<picture>`/media query.
6. [ia_role] [P2] The glossary plus diagram is still the largest block on a meta/resources page. It is coherent now, but it duplicates the Home hierarchy diagram and the Data structures reference.
   FIX: This is acceptable as is. If the page is trimmed further, collapse the Terms table into a `??? info "Terms used in these docs"` details block so the diagram stays visible and the table expands on demand.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (engagement, no icons/figure): partial. The hierarchy diagram now illustrates the glossary. The cards still have no icons.
2 (visual, mobile tables / TOC overlap): partial. The Datasets table became a list, so there is no more hidden column, and the glossary is two columns. However, the identifiers now break mid-word on mobile, and the floating TOC button still overlaps content (it was claimed hidden).
3 (visual, GeoLife identifier break / pip wrap): partial. `load_geolife_trajectories` now sits intact in the list. `pip install fastmob` still wraps inside the PyPI card on desktop.
4 (trust, license/papers/citation): partial. The citation is now a "Citing fastmob" tip admonition. There are still no dataset licenses and no papers.
5 (ia_role, glossary order / placement): fixed for order. Location now follows Staypoint, and the diagram sits above the table. The glossary is still on this page (now acceptable).
6 (first_impression, flat cards): not fixed (icons were deferred).

WHAT WORKS (max 3, so it is not broken in the next round):
- The Datasets section as a three-bullet list. Each bullet links its source and names the loader or page that uses it, and it reads cleanly on desktop and mobile.
- The diagram sits directly above the term table, and the table follows hierarchy order. The glossary finally has a picture, and its order matches the picture.
- Wayfinding: the About sidebar has the section index first with Resources highlighted, the right TOC lists the three sections, prev/next goes to About / Development setup, and the footer shows the version and license.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show readers what each bundled dataset looks like before they pick a loader (Brightkite: global check-in scatter; GeoLife: dense Beijing GPS traces).
  preferred_source: notebook-real-data
  suggested_filename: datasets-thumbnails.png
  suggested_location: docs/src/assets/images/resource/
  notebook_recipe: Load a 10k-row Brightkite sample (tests/shared/data/loc-brightkite_totalCheckins.txt.gz) and one GeoLife user via load_geolife_trajectories. Draw two side-by-side panels in matching style: a world lat/lng scatter for Brightkite and one user's trajectory line over Beijing for GeoLife. Use a transparent background so it works in both themes.

CROSS-PAGE NOTES:
- The floating mobile TOC button is still rendered at 390px despite the round-3 change log. Verify the selector site-wide.
- Inline code inside table cells breaks mid-identifier on mobile. Add a site-wide `td code { white-space: nowrap }` (or `overflow-wrap: normal`) and let the table scroll instead.
- The tip admonition's teal (Citing fastmob) is the only teal accent on a yellow/olive-themed page. Consider restyling `tip` to the theme accent.
- Deferred (P2 only): header logo white box, black light-mode header, footer Previous crossing sections.
```
