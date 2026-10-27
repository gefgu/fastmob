```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 6
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] The Triplegs subtitle "movement between stops" is now wider than its box in every variant. It touches or crosses the left and right borders: desktop-light-full y~1725 (box x~496-636), desktop-dark-full y~1725 where the yellow text overlaps the yellow border, and mobile-full y~1780 (box x~197-350). It is the only visibly broken element left on the page.
   FIX: In mobility-hierarchy.svg and mobility-hierarchy-vertical.svg, change the Triplegs subtitle to "moving segments" (15 chars, the same width as "where you stopped"), or widen the Triplegs and Staypoints boxes by about 24px. Re-check both themes.
2. [visual] [P2] On mobile there is still about 130px of empty space between the "dashed: tours return to a Location" note and the "What you can do" H2 (mobile-full y~2080-2230). This was carried from R5#5 and was not addressed.
   FIX: Crop the viewBox of mobility-hierarchy-vertical.svg to its content plus about 8px of bottom padding. If the gap is CSS rather than the SVG, reduce `.figure` margin-bottom under 600px in docs/src/stylesheets/extra.css.
3. [visual] [P2] On mobile, the `pip install fastmob` block still runs edge to edge at x=0-375 (mobile-full y~1040-1080). The tabbed code card and the output card below it are inset 16px. This was carried from R4#4 and R5#3; the r5 change log fix ("code blocks keep right padding") did not cover it.
   FIX: In extra.css, under `@media (max-width: 600px)`, add `.md-typeset > .highlight { margin-inline: 0; border-radius: .4rem; }`, or whatever cancels the negative margin Material applies to top-level `.highlight` on small screens.
4. [visual] [P2] Light-mode benchmark chart labels are still mid-grey on white: category names, tick labels, legend, and the scikit-mobility seconds (desktop-light-fold y~460-860). They are readable but noticeably weaker than the body text. This was carried from R5#1c.
   FIX: Re-export home-benchmark-speedup.png and the -mobile variant with label colour #333 (or ship #only-light and #only-dark variants with body-text colours).
5. [visual] [P2] GeoLife figure on mobile (mobile-full y~2290-2565). The legend ("position fixes / triplegs / staypoints") renders at about 9px and the fix dots are barely visible. Desktop dark is now acceptable: the grey dotted fixes show. This was carried from R5#4.
   FIX: Add a 600x600 mobile crop with legend font of at least 14px at display size and serve it through `<picture>`, as the speedup chart and diagram already do.
6. [ia_role] [P2] The "Try it" output card (desktop-light-full y~1285-1365) no longer shows a copy icon, which is good. It is still styled identically to the code card above, so it reads as a second snippet.
   FIX: Give output blocks a 1px border with a transparent background, or a small "Output" label, in extra.css.
7. [visual] [P2] Deferred brand items, noted only: the "Fastkit-Mobilit" logo text, the white logo box in dark mode, and the black light-mode header. The footer pager now shows only "Next: Features overview", which is fine.

PREVIOUS ISSUES STATUS (rounds >= 2):
- R5#1a (copy button on the output block): fixed. No clipboard icon on the output card in light, dark or mobile.
- R5#1b (mobile hero buttons wrapping): fixed. "Get started" and "API reference" sit on one row; Benchmarks is hidden and the caption link still reaches it.
- R5#1c (grey light-mode chart labels): not fixed (issue 4).
- R5#2 (mobile chart title clipped to "scikit-mobilit"): fixed. "Speedup vs scikit-mobility" is fully visible with the subtitle below.
- R5#3 (mobile pip block edge to edge): not fixed (issue 3).
- R5#4 (GeoLife figure: faint fixes, tiny mobile legend, no context): partial. The fixes are darker and visible in dark mode. The mobile legend is still tiny and there is still no scale or basemap (issue 5).
- R5#5 (mobile dead space; Triplegs subtitle reading as truncated): partial. The subtitle text was completed, but it now overflows its box (issue 1). The dead space remains (issue 2).
- R5#6 (mobile fold lead too long): partial. The lead is still 5 lines, but with one button row the full five-bar speedup chart now fits in the first ~940px. Above the fold (mobile-fold) shows 4 of 5 bars. That is acceptable.
- R5#7 (deferred brand items): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- Desktop first screen: the claim H1, a two-sentence lead, a yellow primary CTA, a caption that names the dataset, and a log-scale chart with seconds and speedups, all above 860px. Mobile now mirrors this with a clean two-button row and a portrait speedup chart. This earns first_impression and trust their 8s.
- The proof-then-vocabulary-then-capability sequence. The benchmark chart is followed by a runnable pandas/Polars snippet with real output, then the hierarchy diagram, then the GeoLife real-data figure, then six task cards linked to Learn pages. This earns engagement, ia_role and scannability their 8s.
- `<picture>` mobile variants for both the chart and the diagram. Extend the same pattern to the GeoLife figure rather than shrinking the desktop image.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: The GeoLife user-day must be legible on phones (legend and fixes).
  preferred_source: notebook-real-data
  suggested_filename: home-geolife-staypoints-triplegs-mobile.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: GeoLife, same busiest user-day. Use Positionfixes(df).generate_staypoints() and .generate_triplegs(). Use a 600x600 canvas with a legend font of at least 14px at display size, fixes at s=4 and alpha 0.6, and an optional 1 km scale bar.

CROSS-PAGE NOTES:
- Top-level (non-tabbed) code blocks still run edge to edge on phones while tabbed blocks are inset. Fix this once in extra.css.
- One chart style should apply to Home, Features and Benchmarks: body-text-coloured labels in light mode and `<picture>` mobile variants.
- SVG diagram text should be checked against box widths in all themes. A copy change in r5 introduced an overflow here.
- The header logo and the black light-mode header remain deferred.
```
