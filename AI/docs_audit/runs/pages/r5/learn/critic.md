PAGE: learn/index.md   ROUND: 5
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] Text inside the thumbnails is still unreadable at card size.
   - Fit plot: the tick labels and legend (desktop-light-full, y~1130-1255).
   - Measure histogram: the x ticks and axis label (y~1245-1258).
   - Backends diagram: the "fastmob Rust kernel" centre node (y~763).
   - Route thumb: the new caption "raw GPS (red) snapped to the road (gold)" (y~1139). The SVG sets it at 20px in a 640px viewBox, which renders at about 7px. In dark mode its #777 fill is close to the card surface.
   On mobile-full all of these are smudges.
   FIX:
   - Re-export learn-thumb-fit.png and learn-thumb-measures.png with no tick labels and no legend.
   - In learn-thumb-backends.svg, make the centre node's font the same size as the pandas/Polars boxes.
   - In learn-thumb-route.svg, delete the `<text class="t">` caption. The card title and alt text already say what it shows.
2. [engagement] [P2] The Route thumbnail does not show "snapping". The grey `.road` path is drawn exactly under the gold `.route` path, so the road is invisible. The only visible grey is a short dead-end stub (`M200 330 L200 200`) that reads as a stray bar (y~1215-1250). Nothing links each red point to the line.
   FIX: In learn-thumb-route.svg:
   - Add 1-2 more grey road branches that the route does not take, so the road network is visible.
   - Draw a thin dashed grey connector from each red point to its nearest spot on the gold line.
   - Shorten or remove the stub.
3. [visual] [P2] Thumbnail backgrounds differ by theme. In light mode every thumb sits on a grey panel. In dark mode the panels disappear, the figures float on the card surface, and the hero figure loses its frame (desktop-dark-full, y~395-620). Each version is fine alone, but they look like two design systems.
   FIX: In the shared card CSS, give `.how-to-card__thumb` a background token (`--thumb-surface`) with a light and a dark value. Use a slightly raised surface in dark mode so the panel still shows.
4. [trust] [P2] The cards still have no time or effort estimate. A "Short, runnable guides" hub is where a reader decides how much time to spend.
   FIX: Optionally add "~N min" to each `how-to-card__meta` line, e.g. "Data: GeoLife · ~10 min". Keep the current "Data:" format.
5. [engagement] [P2] The two Notebook-example cards are still text-only (y~1600-1800). Deferred, noted only.
   FIX: Use each notebook's main figure as its thumbnail (see IMAGE REQUESTS).
6. [navigation] [P2] The footer still shows "Previous: Motifs", which crosses into Features. Deferred site-wide, noted only.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (Route card hole in the grid): fixed. The card now has a 16:9 thumbnail, so the 3x2 grid is uniform and the titles line up.
2 (meta lines contradict the figures, mixed formats, no time estimates): mostly fixed. All cards now use "Data: ...". The Start card honestly says "synthetic (figure: GeoLife)". Time estimates are still missing (issue 4, now P2).
3 (thumbnail text unreadable): not fixed. The new Route caption adds another tiny label (issue 1).
4 (text-only notebook cards): not fixed (deferred).
5 (Binder pill looked like a button): fixed. It is now plain meta text: "· Binder notebook".
6 (Previous: Motifs pager): not fixed (deferred).
No regressions. The mobile fold is clean: the floating TOC button is gone, and the Start hero card fills the first screen without overlap.

WHAT WORKS (max 3, so it is not broken in the next round):
- The full-width "Start here" hero with a real GeoLife figure sits above the fold on desktop and on mobile. A cold reader gets one obvious first click, and the "Next: Start with a trajectory" pager reinforces it. This earns first_impression=8 and navigation=8.
- A regular grid: hero, then 3x2 guides, each with a figure, then 2-up notebooks. Card titles match the sidebar one to one, and every card has a uniform "Data: ..." meta line. This earns scannability=8, ia_role=8 and trust=8.
- The page is lean: a three-sentence intro and then cards, with no filler (conciseness=8). The single-column stack on mobile has no horizontal scroll.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Card thumbnail for the stay-point detection notebook.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-staypoints.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: GeoLife, using the notebook's main map. Show positionfixes in grey, fastmob staypoints in gold, and trackintel staypoints as hollow markers. 16:9, no ticks, no text.
- purpose: Card thumbnail for the compare-mobility-laws notebook, visually distinct from learn-thumb-fit.png.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-laws.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Brightkite jump-length distribution on log-log axes, with 2-3 candidate laws in distinct colours. 16:9, no tick labels or legend.

CROSS-PAGE NOTES:
- Use one thumbnail export recipe across Learn, Features and Home: 16:9, no tick labels or legends, no in-image captions under 12px rendered, and one themed panel token for both modes.
- Section index pages: decide once whether the "Previous" pager may cross into another section (deferred).
- The filename typo `learn/comparision-to-mobility-laws.md` is still in the URL.
- Deferred, noted only: header logo white box, black header in light mode, missing notebook thumbnails.
