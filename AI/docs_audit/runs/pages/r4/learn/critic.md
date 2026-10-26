PAGE: learn/index.md   ROUND: 4
SCORES: first_impression=8 engagement=7 navigation=7 scannability=8 visual=7 ia_role=8 conciseness=8 trust=7
OVERALL: 7.5/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] The Route and map-match card is now the only hole in an otherwise regular 3x2 grid. It sits in the middle of row 2 (desktop-light-full, x~690-985, y~1097-1475) with about 250px of empty space above its title, while its neighbours both have figures. (The figure itself is deferred, but the card still looks broken.)
   FIX: Until learn-thumb-route.png exists, add a `.how-to-card__thumb--placeholder` block. Make it the same 16:9 size and surface tint as the other thumbs, with a centred lucide `route` icon, so the title lines up with "Measure people..." and "Fit and evaluate...".
2. [trust] [P1] The meta lines contradict the figures and use three different formats. The Start hero shows a real GeoLife outing (its alt text says so), but its meta says "Synthetic data" (y~573). "Clean and segment" says "Figure: GeoLife", while the other cards say just "GeoLife" or "Brightkite". There are still no reading-time estimates.
   FIX: In docs/src/learn/index.md, use one format: "<n> min · <data>". For the Start card, either say "Synthetic data · figure: GeoLife", or regenerate the thumbnail from the guide's synthetic trajectory. Drop the "Figure:" prefix everywhere else.
3. [visual] [P2] Thumbnail text is still unreadable at card size. Examples: the histogram tick labels on "Measure people" (y~1248), the axis ticks and legend on the Fit plot (y~1135-1255), and "fastmob Rust kernel" in the backends diagram (y~763). On mobile-full they are just smudges.
   FIX: Re-export learn-thumb-measures.png and learn-thumb-fit.png with no tick labels or legend (keep at most one axis label at >=12px rendered). In learn-thumb-backends.svg, set the centre node's font size to match the pandas/Polars boxes.
4. [engagement] [P2] Both Notebook-example cards are text only (y~1575-1775). This is the real-data, end-to-end section, yet it is the least visual one on the page. (Deferred; noted only.)
   FIX: Use each notebook's main figure as its thumbnail (see IMAGE REQUESTS from round 3; still valid).
5. [trust] [P2] The new "Binder" pill looks like a button, but it sits inside the card's anchor. Clicking it opens the rendered page, not Binder, so the affordance misleads.
   FIX: Either restyle it as a non-interactive tag (no border; e.g. "Runs on Binder"), or move the card link to the title and make the badge its own `<a>` to the Binder URL.
6. [navigation] [P2] The footer still shows "Previous: Motifs" (y~1855), which crosses into Features. (Deferred site-wide.)
   FIX: Hide the previous link on section index pages via a template override, or accept it site-wide.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (orphaned Fit card, half-empty notebook row): fixed. The Start card is a full-width hero, the six guides fill exactly 2 rows of 3, and the notebooks are 2-up.
2 (Route card with no thumbnail or meta): partial. The meta line "Road and rail networks" was added, but there is no thumbnail or placeholder, so the hole is still in the middle of the grid.
3 (no time estimate, plain Binder, stale alt text, label on the path): partial. Alt text is updated and the rg label is moved above the path. Binder is now a pill (see issue 5). There are still no time estimates, and the Start meta now contradicts its own figure.
4 (text-only notebook cards): not fixed (deferred).
5 (tiny thumbnail text): not fixed.
6 (Previous: Motifs pager): not fixed (deferred).

WHAT WORKS (max 3, so it is not broken in the next round):
- The full-width "Start here" hero, with a real figure on the left and the eyebrow, title and blurb on the right, is above the fold in both themes. It gives a cold reader an obvious first click. This earns first_impression=8.
- The grid is now regular (hero, then 3x2, then 2-up), and the cards are equal height. The eye can find any guide in under 3 seconds, and the cards match the sidebar one to one with the same titles. This earns scannability=8 and ia_role=8.
- The page is lean: a three-sentence intro, then cards, with no filler (conciseness=8). Mobile stacks cleanly in a single column with no horizontal scroll, and dark mode has consistent card borders and good contrast.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Thumbnail for Route and map-match: a noisy GPS trace snapped to a road network.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-route.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Take a small OSM road graph from the network-analysis guide. Draw roads in light grey, the raw trace as grey points, and the map-matched route as a gold line. Use 16:9 with no text.
- purpose: Card thumbnail for the stay-point detection notebook.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-staypoints.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Reuse the notebook's main GeoLife map: positionfixes in grey, fastmob staypoints in gold, trackintel staypoints as hollow markers. Use 16:9 with no ticks.
- purpose: Card thumbnail for the compare-mobility-laws notebook, visually distinct from learn-thumb-fit.png.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-laws.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Plot the Brightkite jump-length distribution on log-log axes with 2-3 candidate laws in distinct colours. Use 16:9, no tick labels, and a legend of at most 3 short entries at >=12px.

CROSS-PAGE NOTES:
- Share one thumbnail export recipe across Learn, Features and Home: 16:9, no tick labels, minimum 12px rendered text, and a transparent or card-surface background. In dark mode the Learn thumbs have no panel, while in light mode they sit on a grey panel; pick one treatment.
- Section index pages: decide once whether the "Previous" pager may cross into another section.
- learn/comparision-to-mobility-laws.md: the filename typo is still in the URL.
- Deferred, noted only: header logo white box, black header in light mode, missing Route and notebook thumbnails.
