```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 4
SCORES: first_impression=8 engagement=7 navigation=8 scannability=8 visual=7 ia_role=8 conciseness=8 trust=8
OVERALL: 7.8/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P1] Whole page (desktop-light-full and mobile-full): the only visuals are a timing chart and a box diagram. Nothing shows what fastmob produces on real data, such as stops, trips or a map. A researcher deciding whether to try the library sees proof of speed but no picture of the result. The "What you can do" 3x2 grid (desktop-full y~830-1180) is six text-only cards. This request has been carried since round 1 and is the main thing holding engagement at 7.
   FIX: Add one real-data figure, home-geolife-staypoints-triplegs.png (see IMAGE REQUESTS), directly under the "What you can do" H2 and above the card grid, using a `<picture>` mobile crop. Optionally add a 16:9 thumbnail at the top of each card (the Learn notebook thumbnails can be reused) so the grid stops being a wall of six text boxes.
2. [visual] [P2] Desktop hierarchy diagram (desktop-light-full y~1510-1710; also dark): the round-3 layout defects remain. The Staypoints -> Locations curve still crosses the Triplegs -> Trips arrow. The label "tours return to a Location" still floats about 120px to the right of the dashed arrow it describes. Centring is now fixed.
   FIX: In docs/src/assets/images/home/mobility-hierarchy.svg, put Locations below Staypoints in column 2 (under Triplegs) or route its curve under the Triplegs box so nothing crosses. Place the label on the dashed Locations -> Tours path, e.g. directly under Tours, left-aligned to the arrowhead.
3. [visual] [P2] Mobile vertical diagram (mobile-full y~1700-2150): the Locations -> Tours dashed relation and its label have been dropped. Locations hangs alone at the bottom with a long curve from Staypoints that bypasses Trips and Tours, which makes it read like a dead end. The desktop and mobile diagrams now tell different stories.
   FIX: In mobility-hierarchy-vertical.svg, place Locations to the left of Tours on the same row (Staypoints -> Locations straight down the left column) and add the short dashed Locations -> Tours arrow with a "returned to" label.
4. [ia_role] [P2] "Try it" block (desktop-full y~1290-1360; mobile y~1450-1520): the output table is still a second grey code card with its own copy button, so it reads as another snippet to run (carried from R2#4). On mobile the `pip install fastmob` block still runs edge to edge (x=0-375) while the tabbed code below keeps a 16px inset.
   FIX: Wrap the output in a `.hero-output` style with no copy button (`data-md-no-copy` or a `text` block with `.no-copy`), a small "Output" label, a lighter or bordered background and a tighter top margin. Give `.md-typeset > .highlight` 16px horizontal margins under 600px in docs/src/stylesheets/extra.css so the pip block matches the tabs.
5. [visual] [P2] Desktop chart in light mode (desktop-light-fold y~460-860): category labels, the axis, the legend and the scikit-mobility values are still mid-grey (#8a8a8a-ish) on white. The gold "(315x faster)" labels are low contrast on white (R3#6 not fixed). Separately, the mobile chart's title "speedup vs scikit-mobility" touches the right edge of the 390px viewport (mobile-full y~660).
   FIX: Re-export home-benchmark-speedup.png with labels in the page's body-text colour and the fastmob labels in the darker light-mode link gold. Alternatively ship light and dark variants via `#only-light` / `#only-dark`. In the mobile PNG, centre the title or wrap it onto two lines with a 16px right margin.
6. [first_impression] [P2] Mobile fold (mobile-fold.png): the lead paragraph is still 5 lines at roughly 22px. The "Benchmarks" button still wraps alone onto a second row. Only the first 2.5 bars of the chart make it into the fold.
   FIX: Under 600px in extra.css, set `.hero-lead` to about 18px. Either make the three `.hero-button`s `flex: 1 1 auto` with smaller padding so they fit in one row, or drop "Benchmarks" on mobile, since the caption link directly below already goes there.
7. [visual] [P2] Deferred brand items, visible in every screenshot: the header logo reads "Fastkit-Mobilit" beside "fastmob", the logo sits on a white box in dark mode, the header is black in light mode, and the footer "Previous" link crosses sections. Noted only; not scored.
   FIX: Swap in a transparent fastmob mark when the source asset exists.

PREVIOUS ISSUES STATUS (rounds >= 2):
- R3#1 (mobile benchmark chart clipped in a scroll box): fixed. A portrait speedup-only chart (1329x ... 315x, large gold labels) is served at <=600px, with no horizontal scroll. One minor problem: the title touches the right edge (issue 5).
- R3#2 (mobile hierarchy diagram clipped): fixed. The vertical SVG shows all six levels at a readable size. The Locations -> Tours relation was lost along the way (issue 3).
- R3#3 (chart oversized on desktop, shouting over the H2s): fixed. The chart is capped at about 820px and about 420px tall, and its text no longer outweighs the headings. The pip line still lands just below a 900px fold, which is acceptable.
- R3#4 (no lead-in before the code; output looks like a snippet; pip block inset on mobile): partial. The "Try it" H2 is added. The output styling and the mobile pip inset are not fixed (issue 4).
- R3#5 (diagram crossing, floating label, left-aligned): partial. The diagram is centred. The crossing and the floating label remain (issue 2).
- R3#6 (grey chart labels in light mode): not fixed (issue 5).
- R3#7 (deferred brand items): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- Desktop first screen: the claim H1, a two-sentence lead, a yellow primary CTA, a caption that names the dataset and comparison, and a right-sized log-scale chart with per-measure seconds and speedups, all within 900px. This is landing-page best practice and earns first_impression and trust their 8s.
- Page order (proof -> Try it -> concept diagram -> capabilities -> why -> next steps -> named pager -> footer with version 0.3.2, license, PyPI and GitHub) is predictable and has no filler. Each H2 is followed by one compact element, so the eye finds any section in seconds (scannability, conciseness).
- The `<picture>` mobile variants: the phone gets a purpose-built speedup chart and a vertical diagram instead of shrunken desktop images. Keep this pattern for every key figure.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show what fastmob produces, not only how fast: one GeoLife user-day of raw fixes with detected staypoints and triplegs, placed under "What you can do".
  preferred_source: notebook-real-data
  suggested_filename: home-geolife-staypoints-triplegs.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: GeoLife, one user-day; Positionfixes(df).generate_staypoints() and .generate_triplegs(); raw fixes as small grey dots, triplegs as coloured lines, staypoints as larger yellow-outlined circles; light or no basemap; 1200x600 plus a 600x600 mobile crop for <picture>; transparent background or light and dark variants.

CROSS-PAGE NOTES:
- Chart styling: matplotlib exports still use their own mid-grey text. Standardise one docs chart style (body-text-coloured labels, about 13-14px at display width, light and dark variants) across Home, Features and Benchmarks.
- Code blocks that sit directly in the page (not inside tabs) run edge to edge on phones while tabbed blocks keep a 16px inset. Fix this once in extra.css.
- Output blocks site-wide carry copy buttons. Add a shared "output" block style with no copy button and use it on every page.
- Header logo, black light-mode header and footer "Previous" crossing sections remain deferred.
```
