PAGE: learn/index.md   ROUND: 6
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] The thumbnail panels are still missing in dark mode. The change log says "thumbnails share one surface token", but on desktop-dark-full there is no visible panel behind any thumb. The hero figure (y~395-620) and all six grid thumbs (y~700-830, y~1127-1260) float on the card surface. On desktop-light-full every thumb sits on a grey #f3-ish panel. The token either resolves to the card colour in dark mode or is not applied to `.how-to-card__thumb`.
   FIX: In the shared card CSS, set the dark value of `--thumb-surface` a visible step above the card. For example, if the card is about #1c1c1c, use about #262626. Confirm that `.how-to-card__thumb { background: var(--thumb-surface) }` is not overridden by a transparent PNG/SVG rule in dark mode.
2. [visual] [P2] Some in-image text is still tiny.
   - Measure histogram: the x-axis label "radius of gyration per user" (y~1254).
   - Fit plot: "jump length" (y~1254).
   - Backends SVG: the centre node's "fastmob Rust kernel" (y~765). It is still much smaller than the pandas/Polars labels.
   - Hero figure: the title "radius of gyration: 4.5 km" (y~405). It is mid-grey, and on the dark card it has low contrast.
   On mobile-full all of these are illegible.
   FIX:
   - Re-export learn-thumb-measures.png and learn-thumb-fit.png without axis labels.
   - In learn-thumb-backends.svg, set the centre node's font-size to the size the side boxes use.
   - Either drop the hero's in-image title (the alt text and card copy cover it) or render it in a theme-neutral darker tone at 14px or more.
3. [trust] [P2] There are still no time or effort estimates on the cards (carried over, optional).
   FIX: Append "· ~N min" to each `how-to-card__meta` line.
4. [engagement] [P2] The two notebook cards are text-only (y~1600-1800). Deferred, noted only.
5. [navigation] [P2] The footer "Previous: Motifs" crosses into Features. Deferred, noted only.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (thumbnail text unreadable): partial.
   - Fixed: the tick labels and legends are gone, and the Route caption is removed.
   - Not fixed: the axis labels on the measures and fit thumbs, and the small Backends centre node.
2 (Route thumb does not show snapping): fixed. Extra grey road branches are visible, and each red point has a short connector to the gold route. The stray stub is gone.
3 (thumb background differs by theme): not fixed. The change was claimed, but dark-mode panels are still invisible (issue 1).
4 (no time estimates): not fixed (optional).
5 (text-only notebook cards): not fixed (deferred).
6 (Previous: Motifs pager): not fixed (deferred). On mobile the pager now stacks cleanly.
No regressions:
- Desktop fold: the hero is still above the fold.
- Mobile fold: the image-on-top Start card fits the first screen without overlap.
- Mobile: single column, no horizontal scroll.

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Start here" hero with a real GeoLife figure is above the fold on desktop and mobile. It is the one obvious first click, and the "Next: Start with a trajectory" pager backs it up (first_impression=8, navigation=8).
- A uniform hero, 3x2 guide grid and 2-up notebook layout. The card titles match the sidebar one to one, and every card has a consistent "Data: ..." meta line (scannability=8, ia_role=8, trust=8).
- The redrawn Route thumbnail now explains map matching at a glance. Each guide thumb shows real output from its own guide (engagement=8). The intro is three sentences with no filler (conciseness=8).

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Card thumbnail for the stay-point detection notebook.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-staypoints.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: GeoLife. Positionfixes in grey, fastmob staypoints in gold, trackintel staypoints as hollow markers. 16:9, no text.
- purpose: Card thumbnail for the compare-mobility-laws notebook, distinct from learn-thumb-fit.png.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-laws.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Brightkite jump-length distribution on log-log axes, with 2-3 candidate laws in distinct colours. 16:9, no labels.

CROSS-PAGE NOTES:
- Check the dark value of the `--thumb-surface` token site-wide (Home and Features thumbs likely have the same invisible panel).
- One thumbnail recipe for all pages: 16:9, no axis labels, ticks or legends, and no in-image text under about 12px rendered.
- The filename typo `learn/comparision-to-mobility-laws.md` is still in the URL.
- Deferred, noted only: the header logo white box, the black light-mode header, the Previous pager crossing sections, and the missing notebook thumbnails.
