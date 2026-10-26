```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 3
SCORES: first_impression=8 engagement=7 navigation=8 scannability=7 visual=6 ia_role=7 conciseness=8 trust=8
OVERALL: 7.4/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile (390px), benchmark chart (mobile-full, y~640-960): the chart is the desktop PNG shrunk into a horizontal-scroll box. Its text is about 7px tall, the legend is cut to "scikit-mobil", every grey scikit-mobility bar and its seconds value (31.4s ... 182.8s) sits off-screen to the right, and "(1208x faste" is clipped. The x-axis caption is cut to "Execution time on 4M Brightkite". No fade or "scroll ->" hint shows that more is there. On a phone the page's main proof shows yellow bars with nothing to compare against.
   FIX: Export a second, portrait chart (about 600x700, larger font, legend stacked above, value labels inside or under the bars) and serve it with `<picture><source media="(max-width: 600px)" srcset="assets/images/home/home-benchmark-speedup-mobile.png"><img src="...home-benchmark-speedup.png" alt="..."></picture>` in index.md. Do not rely on figure-scroll for the hero chart. A simpler fallback is a mobile chart that plots only the speedup factor (315x ... 1329x) as five bars.
2. [visual] [P1] Mobile, hierarchy diagram (mobile-full, y~1560-1700): it gets the same scroll-box treatment. Box text is about 6px. "Trips", "Tours" and "Locations" are clipped at the right edge, and "Tours" and the "tours return to a Location" label are entirely off-screen. With no scroll cue, a reader sees half a diagram.
   FIX: Add a vertical variant of mobility-hierarchy.svg (Positionfixes at the top, then Staypoints/Triplegs, then Trips, then Tours, with Locations beside Staypoints) and switch it in at <=600px with `<picture>`, as in issue 1. If figure-scroll stays for other figures, add a right-edge fade gradient on `.figure-scroll` in docs/src/stylesheets/extra.css under 600px.
3. [scannability] [P1] Desktop, chart size (desktop-light-fold plus the top of the full page): the chart renders about 840px tall at full content width (1190px), so it fills the whole fold and continues below it. The pip line and the code example now start around y~1080, a full screen below the CTAs. The chart's text (legend 22px, labels up to 24px bold) is bigger than the page's own H2s, so the chart shouts over the headings.
   FIX: Cap the hero chart with `max-width: 820px; margin: 0 auto` on a dedicated class (e.g. `.hero-chart img`), or re-export it at about 16:6 with chart text at about 13-14px at display size. That way the bottom of the chart and the `pip install` line both land in a 900px fold.
4. [ia_role] [P2] Hero to code transition (desktop full, y~1080): `pip install fastmob` and the pandas/Polars tabs follow the chart with no heading or lead-in. After a large figure they read as a separate, unlabelled section. The output block is still a second grey card with its own copy button, which makes it look like a snippet to run (R2#4 not fixed). On mobile the pip block still runs edge to edge while the code tabs keep a 16px inset.
   FIX: Put an H2 such as "Try it" (or a short `<p class="figure-caption">Install and run</p>`) above the pip block. Add a `.hero-output` style (no copy button, tighter top margin, a small "Output" label, slightly different background). Give the pip block the same 16px horizontal inset on mobile.
5. [visual] [P2] Hierarchy diagram, desktop (desktop-light-full, y~830-1100): the Staypoints -> Locations curve crosses the Triplegs -> Trips arrow. The note "tours return to a Location" floats at the far right, detached from the dashed arrow it describes. The diagram is left-aligned with about 200px of empty space on its right.
   FIX: In mobility-hierarchy.svg, move Locations to the top of column 3 (above Trips) so its arrow from Staypoints curves upward without a crossing. Route the dashed Locations -> Tours arrow along the top and place its label on the dashed line. Centre the figure (`margin: 0 auto` on the img).
6. [visual] [P2] Chart, light mode (desktop-light-fold): category labels, the axis and the scikit-mobility values are mid-grey (#8a8a8a-ish) on white, which looks washed out next to the near-black page text. The yellow "(315x faster)" labels on white are also low contrast.
   FIX: When re-exporting, use the page's body-text colour for labels in the light variant and a darker gold (the light-mode link colour) for the fastmob labels. Alternatively ship light and dark PNGs and swap them with `#only-light` / `#only-dark` or `<picture media="(prefers-color-scheme: dark)">`.
7. [visual] [P2] Deferred brand items, still visible in every screenshot: the header logo reads "Fastkit-Mobilit" beside "fastmob", there is a white box behind the logo in dark mode, and the header is black in light mode. Noted only; not scored.
   FIX: Re-export the logo as a transparent fastmob mark when the source asset is available.

PREVIOUS ISSUES STATUS (rounds >= 2):
- R2#1 (figures printed as raw Markdown): fixed. Both figures render as real images in light, dark and mobile.
- R2#2 (round-1 figure issues unverifiable): partial. Desktop: fastmob bars are visible on the log axis, the log scale is labelled, the dark-mode diagram uses dark fills with yellow outlines and is no longer the brightest element, and Locations is placed correctly. Mobile legibility is not fixed (issues 1-2).
- R2#3 (fold ends on a bare heading, no figure): partial. Desktop fold now shows H1, lead, CTAs and most of the chart, so the fix landed. Mobile: the lead is still 5 lines at a large size, the "Benchmarks" button still wraps to a second row, and the chart in the mobile fold is clipped. The chart is now oversized (issue 3).
- R2#4 (output looks like a second snippet; pip block misaligned on mobile): not fixed (carried in issue 4).
- R2#5 (deferred brand items): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- The first screen on desktop: the claim H1 ("hundreds of times faster"), a two-sentence lead, a yellow primary CTA, then the real-data proof right away. The proof is a log-scale chart that names the dataset (4M Brightkite check-ins), the comparison (fastmob pandas vs scikit-mobility), per-measure seconds and speedups, and links "All measures and methodology". This is what earns first_impression and trust their 8s.
- Dark mode: the chart and the redrawn hierarchy diagram (dark fills, yellow outlines, muted arrows) sit comfortably on the dark page with no glaring white boxes.
- The lower half: "What you can do" 3x2 cards, a two-bullet "Why fastmob", the Python and wheels note, a "Next steps" grid, a named pager ("Next: Features overview") and a footer with license, PyPI and GitHub. Everything is concise and stacks cleanly on mobile.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: A phone-readable version of the speed proof: five measures, each with the speedup factor large and the two times small, readable at 390px without scrolling.
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-mobile.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: Same Brightkite 4M benchmark numbers as home-benchmark-speedup.png; portrait 600x700; one horizontal bar per measure showing the speedup factor (linear axis 0-1400x), with the label "31.4s -> 0.10s" under each measure name; font at least 16px at export size; transparent background, or light and dark variants.
- purpose: A vertical hierarchy diagram for mobile so all six levels are visible without horizontal scrolling.
  preferred_source: diagram
  suggested_filename: mobility-hierarchy-vertical.svg
  suggested_location: docs/src/assets/images/home/
- purpose: (Carried from rounds 1-2, still not delivered.) Show what fastmob produces on real data, not only how fast: one GeoLife user-day of raw fixes with detected staypoints and triplegs, at the top of "What you can do".
  preferred_source: notebook-real-data
  suggested_filename: home-geolife-staypoints-triplegs.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: GeoLife, one user-day; Positionfixes(df).generate_staypoints() and .generate_triplegs(); raw fixes as small grey dots, triplegs as coloured lines, staypoints as larger yellow-outlined circles; light or no basemap; 1200x700 plus a 600x600 mobile crop.

CROSS-PAGE NOTES:
- `.figure-scroll` is now used site-wide (Features charts as well). Without a visible scroll cue it hides content on phones everywhere. Add a right-edge fade or "scroll ->" hint under 600px in extra.css, and prefer `<picture>` mobile variants for key figures.
- matplotlib-exported charts use their own mid-grey text palette, which does not match the theme's text tokens in light mode. Standardise one chart style (label colour, font size at display width) for every docs figure.
- Header logo and the black light-mode header remain deferred.
```
