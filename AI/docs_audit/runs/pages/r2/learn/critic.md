PAGE: learn/index.md   ROUND: 2
SCORES: first_impression=7 engagement=7 navigation=7 scannability=7 visual=6 ia_role=8 conciseness=7 trust=6
OVERALL: 6.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] The guide grid is ragged on desktop. The Route card has no thumbnail, so it is about 130px tall (desktop-light-full, right column y~957-1089) next to the 330px Measures card. "Fit and evaluate" is then left alone in row 4, with an empty right column from y~1090 to y~1630 (about 770px real). Cards are not stretched to row height, so the grid reads as broken, not as a gallery. It looks the same in dark mode.
   FIX: In docs/src/stylesheets/extra.css, make `.how-to-grid` use `align-items: stretch` with equal-height cards. Switch to 3 columns at >=1200px, which is possible now that the TOC is hidden: 7 guides in 3 columns gives rows of 3/3/1, or put "Start here" full-width and the other 6 in 3/3. Until the road-network figure exists, give the Route card a placeholder thumbnail (a neutral `.how-to-card__thumb--placeholder` block with a route icon) so all guide cards have the same structure.
2. [engagement] [P1] The "Start with a trajectory" thumbnail is the first image the reader sees, and the weakest one. It shows two identical vertical lines inside identical dashed circles, labelled "alice"/"bob", both "9.1 km", and the labels are about 6px and unreadable (desktop-light-fold, x~400-797, y~411-634). It does not look like a trajectory and does not show any difference between the two users.
   FIX: Regenerate learn-thumb-trajectory.png. Use one 2D path with 6-8 points that move in both lat and lng, numbered by time order, plus the rg circle around the centroid. Remove the in-image text or make it at least 14px at card size. If the guide's data really is collinear, change the thumbnail's sample data to points that are not collinear.
3. [trust] [P1] The metadata contradicts the thumbnail. The "Clean and segment trajectories" card says "Synthetic data", but its thumbnail alt text (and figure) is "the longest movement segments of one GeoLife user". Readers who notice will stop trusting the other tags. The metadata row also has only a data source: no time estimate, and nothing that says which guides run on real data out of the box.
   FIX: In docs/src/learn/index.md, either change that meta to "GeoLife" (if the guide uses it) or regenerate the thumbnail from the guide's synthetic data. Add a reading time to every `.how-to-card__meta` (e.g. "5 min · Synthetic data"), and show "Binder" on notebook cards as a small badge/pill instead of plain text.
4. [engagement] [P2] Both Notebook-example cards are text only (desktop-light-full, y~1708-1868). After six figure cards, the section that should be the most visual (end-to-end runs on real data) looks like an afterthought.
   FIX: Give each notebook card a thumbnail taken from its own captured output. Use the stay-point map for the GeoLife notebook and the power-law fit plot for the mobility-laws notebook. Do not reuse the guide thumbnails: export the notebooks' own figures into docs/src/assets/images/learn/.
5. [scannability] [P2] Every guide card has the eyebrow "GUIDE" directly under the "Guides" H2, and every notebook card has "NOTEBOOK" under "Notebook examples". The eyebrow repeats the section heading 9 times and adds nothing. Only "START HERE" carries information.
   FIX: Remove the `how-to-card__eyebrow` span from all cards except the start card in docs/src/learn/index.md. Optionally replace it with a topic tag (Preprocessing / Measures / Models).
6. [visual] [P2] Thumbnail text (axis ticks on the Measures histogram and the Fit plot, "alice/bob" labels) is far too small to read at card width, on both desktop and mobile (mobile-full: tick labels are illegible). The figures are hard to read in both themes.
   FIX: Re-export the thumbnails at card size (about 400x225 CSS px) with tick and label fonts of at least 12px, or remove ticks and keep one short axis label. Thumbnails only need to show the shape, not data values.
7. [navigation] [P2] The footer still shows "Previous: Motifs" (from Features), so the prev/next chain crosses into another section. The change log says this was addressed, but the screenshot still shows it. Known-deferred items (header logo text/box, black header) are not counted against this page.
   FIX: Either accept it, or hide the footer "Previous" on section index pages (front matter / template override) so the hub only offers "Next: Start with a trajectory".

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (text-only grid, add thumbnails): partial: 5 of 7 guide cards have real-data figures and 1 has a backend diagram; the Route card and both notebook cards still have none; the trajectory thumbnail is weak.
2 (hub doesn't match section / missing mobility-laws card): fixed: Guides and Notebook examples sections, one card per sidebar entry.
3 (three names per guide, useless SCENARIO eyebrow): fixed for names (card H2 = nav label, whole card is the link); the eyebrow is now "GUIDE"/"NOTEBOOK", which is still redundant (issue 5).
4 (no starting point, jargon intro): fixed: intro says "start with the first guide", "Start here" marker plus accent border on card 1, and the "parameter contract" phrase is gone.
5 (duplicated tip): fixed: removed.
6 (uneven card spacing, olive accent, unused TOC column): partial: internal gaps and link wrap are gone, the light accent is now dark gold, and the TOC is hidden; but the grid is still 2 columns and is now ragged at row level (issue 1).
7 (no metadata/effort signals): partial: a data-source row was added, but there are no time estimates, Binder is plain text, and one tag contradicts its figure (issue 3).

WHAT WORKS (max 3, so it is not broken in the next round):
- Every sidebar entry has exactly one card with the same title, grouped into Guides vs Notebook examples. This one-to-one match between hub and sidebar is what earns ia_role=8.
- The staypoint (GeoLife) and fit (Brightkite power law) thumbnails show real output, are recognisable at a glance, and look correct in both light and dark themes.
- The "Start here" card (accent border plus eyebrow) and the one-paragraph intro give a cold reader a clear entry point. Mobile stacking is clean, with no horizontal scroll.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Replace the trajectory thumbnail so it shows a real 2D path in time order and its radius of gyration.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-trajectory.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Use one GeoLife or Brightkite user with 8-15 points over a day. Run fastmob.radius_of_gyration on them. Plot the points connected in time order (first point marked), the centroid, and a dashed rg circle. 16:9, no ticks, no in-image text smaller than 14px at 400px width.
- purpose: Card thumbnail for the stay-point detection notebook.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-staypoints.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Export the main map figure already captured in learn/notebooks/stay-point-detection-geolife (positionfixes in grey, fastmob staypoints in gold, trackintel staypoints as hollow markers). 16:9 crop.
- purpose: Card thumbnail for the compare-mobility-laws notebook.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-laws.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Export the notebook's Brightkite jump-length distribution plot with the fitted power law overlaid, cropped to 16:9 with large tick labels. Make it visually different from learn-thumb-fit.png, e.g. show multiple candidate laws.

CROSS-PAGE NOTES:
- The prev/next footer still crosses section boundaries (Learn index -> "Previous: Motifs"). Apply the same decision to every section index.
- learn/comparision-to-mobility-laws.md: the filename typo still appears in the URL. Rename it and add a redirect.
- Thumbnail style guide: set one export recipe (size, font minimum, background transparent or matching the card surface, accent gold) so Features/Home figures and Learn thumbnails look like one family.
- Deferred and only noted: header logo text "Fastkit Mobilit" / white box in dark mode, black header in light mode, and the Route card waiting on a road-network figure.
