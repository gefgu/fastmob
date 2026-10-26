PAGE: learn/index.md   ROUND: 3
SCORES: first_impression=7 engagement=7 navigation=7 scannability=7 visual=7 ia_role=8 conciseness=8 trust=6
OVERALL: 7.1/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] The 3-column grid fixed equal row heights, but it now leaves orphans. "Fit and evaluate mobility results" sits alone in row 3 with two empty columns (desktop-light-full, y~1140-1497, about 600px of blank space to its right). Notebook examples has 2 cards in a 3-column grid, so its right third is empty (y~1588-1826). The page still reads as unfinished at the bottom. Dark mode looks the same.
   FIX: In docs/src/learn/index.md and docs/src/stylesheets/extra.css, make the "Start here" card a full-width horizontal hero card (`.how-to-card--start { grid-column: 1 / -1; }` with the thumbnail on the left and text on the right at >=1200px). The other 6 guides then fill exactly 2 rows of 3. Give the Notebook examples grid a modifier (`.how-to-grid--two`) that uses 2 equal columns across the full content width.
2. [engagement] [P1] The Route and map-match card is the only guide with no thumbnail and no meta line (desktop-light-full, right column y~742-1121). In a figure gallery, it looks broken or unfinished. It also sits in the middle of the grid, where it stands out the most. (The real figure is deferred, but the empty card is not.)
   FIX: Until the road-network figure exists, add a neutral placeholder thumb (`.how-to-card__thumb--placeholder`: surface-tinted 16:9 block with a centred lucide route icon) and a meta line with the guide's data source. This keeps the card structure the same as the others.
3. [trust] [P1] The metadata row only gives a data source. There is no time estimate. "Binder" on the notebook cards is plain grey text, not a launch affordance (y~1786). The Start card's alt text still says "Two short trajectories with jump lengths...", but the new figure shows one GeoLife outing. Also, the figure's "radius of gyration: 4.5 km" label overlaps the path.
   FIX: Add a reading time to every `.how-to-card__meta` ("5 min · GeoLife"). Render Binder as a small pill or badge. Update the alt text in index.md to describe the single path and its rg circle. Re-export learn-thumb-trajectory.png with the label moved off the path, or removed.
4. [engagement] [P2] Both Notebook-example cards are text only (y~1588-1826). This is the end-to-end, real-data section, but it is the least visual one on the page. (Deferred; noted only.)
   FIX: Export each notebook's own main figure as its card thumbnail (see IMAGE REQUESTS).
5. [visual] [P2] Thumbnail text is still unreadable at card size. Look at the histogram tick labels on "Measure people and populations", the ticks and legend on the Fit plot, and the "fastmob Rust kernel" box label in the backends diagram (desktop-light-full, y~880-890, y~1175-1290; mobile-full: illegible). This is less important now that the shapes read clearly.
   FIX: Re-export learn-thumb-measures.png and learn-thumb-fit.png without tick labels and legend, keeping at most one axis label at >=12px at 400px width. Raise the font of the centre node in learn-thumb-backends.svg to match the outer boxes.
6. [navigation] [P2] The footer still offers "Previous: Motifs" (y~1883), which crosses into Features. This is a hub page, so the only useful pager link is "Next: Start with a trajectory".
   FIX: Hide the previous link on section index pages (template override or front-matter flag), or accept it site-wide and drop this item.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (ragged grid, Route card short, Fit alone): partial: cards are now equal height in a 3-column grid, but Fit is still orphaned in row 3, and the Route card still has no thumbnail.
2 (weak trajectory thumbnail): mostly fixed: it is now a real 2D GeoLife path in time order with an rg circle. The label overlaps the path, and the alt text is stale.
3 (meta contradicts thumbnail; no time; Binder plain): partial: "Figure: GeoLife" is now honest, but there are no time estimates and Binder is still plain text.
4 (text-only notebook cards): not fixed (deferred).
5 (redundant GUIDE/NOTEBOOK eyebrows): fixed: only "Start here" remains.
6 (tiny thumbnail text): not fixed.
7 (Previous: Motifs pager): not fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- The hub matches the sidebar one to one: each sidebar entry has a card with the same title, and cards are grouped into Guides vs Notebook examples. This earns ia_role=8.
- The page is lean (conciseness=8): a three-sentence intro, then cards. There is no filler, and the eyebrow noise is gone.
- The real-data thumbnails (segments, staypoints, histogram, power-law fit) and the new trajectory path are recognisable at a glance in both themes. The accent-bordered "Start here" card is a clear entry point. Mobile stacks cleanly with no horizontal scroll.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Card thumbnail for the stay-point detection notebook.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-staypoints.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Export the main map already captured in learn/notebooks/stay-point-detection-geolife: positionfixes in grey, fastmob staypoints in gold, trackintel staypoints as hollow markers. Crop to 16:9 with no ticks.
- purpose: Card thumbnail for the compare-mobility-laws notebook, visually distinct from learn-thumb-fit.png.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-laws.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Plot the Brightkite jump-length distribution on log-log axes with 2-3 candidate laws overlaid in distinct colours. Use 16:9, no tick labels, and at most a short legend at >=12px.
- purpose: Thumbnail for Route and map-match, showing a noisy GPS trace snapped to a road network.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-route.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Take a small OSM road graph from the network-analysis guide. Draw the raw trace as grey points and the map-matched route as a gold line over light-grey roads. Use 16:9 with no text.

CROSS-PAGE NOTES:
- Section index pages: decide once whether the "Previous" pager link should cross into another section, and apply the decision everywhere.
- Thumbnail export recipe (size, no tick labels, minimum font size, card-surface background) should be shared by Learn, Features, and Home figures.
- learn/comparision-to-mobility-laws.md: the filename typo is still in the URL.
- Deferred, noted only: header logo white box in dark mode, black header in light mode, missing Route/notebook thumbnails.
