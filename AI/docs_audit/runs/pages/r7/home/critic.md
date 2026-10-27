```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 7
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=7 ia_role=8 conciseness=8 trust=8
OVERALL: 7.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Regression in the mobile speedup chart (home-benchmark-speedup-mobile.png). The x axis now starts at 1x, so the tick labels collide and read "1x 10x100x1000x". "100x" and "1000x" overlap at mobile-full y~1010-1020, x~240-310. In R6 the axis showed a clean "100x  1000x". This sits right after the first screen, at the point where a phone reader is judging the speed claim. It probably comes from the r6 "mobile chart axis now starts at 1x" change, which was meant for Features.
   FIX: Regenerate home-benchmark-speedup-mobile.png with ticks only at 1x, 10x, 100x and 1000x, fontsize 9-10, and at least 70px between labels at display width. Alternatively, go back to the R6 axis range (100x-2000x), since every Home bar is above 300x. If the Home and Features charts share a script, give the Home variant its own xlim.
2. [visual] [P2] Mobile code-block gutters are now inconsistent the other way. The `pip install fastmob` block is inset about 32px on each side (x~32-343, mobile-full y~1030-1070). The tabbed pandas card and the output card below it are inset 16px (x~16-359). The block no longer bleeds edge to edge, but it now looks narrower than everything around it.
   FIX: In docs/src/stylesheets/extra.css, under `@media (max-width: 600px)`, set the top-level `.md-typeset > .highlight` (or `div.language-bash`) to `margin-inline: 0` so its edges line up with the tabbed block. Do not stack the new phone gutter on top of the content padding.
3. [visual] [P2] GeoLife figure on mobile (mobile-full y~2280-2560). The legend ("position fixes / triplegs / staypoints") still renders at about 8-9px, and the grey fix dots are barely visible. The figure uses only about 70% of the column width. This was carried from R5#4 and R6#5.
   FIX: Add home-geolife-staypoints-triplegs-mobile.png (a 600x600 crop with a legend font of at least 14px at display size) and serve it with `<picture><source media="(max-width: 600px)">`, the same pattern the chart and the diagram use.
4. [ia_role] [P2] The "Try it" output card (desktop-light-full y~1285-1365, mobile-full y~1350-1430) is still styled exactly like the code card above it, so it reads as a second snippet. This was carried from R6#6.
   FIX: In extra.css, give output blocks (the `text` fence that follows a code block) a transparent background with a 1px border, or add a small "Output" label.
5. [visual] [P2] The light-mode benchmark chart labels are darker than in R6 (#666 versus #777, measured on "radius of gyration"). Category names, tick labels and the scikit-mobility seconds are still clearly lighter than the #222 body text (desktop-light-fold y~460-860). This was carried from R6#4 and is partial.
   FIX: Re-export home-benchmark-speedup.png and the -mobile variant with label colour #333 or darker. Alternatively, ship #only-light and #only-dark variants that use body-text colours.
6. [trust] [P2] Naming: the alt text of the hero chart still says "fastmob execution time ... fastmob is 315x to 1329x faster" (index.md line 24). Screen readers and image search read this text as prose, and prose should say Fastkit-Mobility.
   FIX: Change the alt text to "Bar chart comparing scikit-mobility and Fastkit-Mobility execution time on 4M check-ins for five measures; Fastkit-Mobility is 315x to 1329x faster."
7. [visual] [P2] Deferred brand items, noted only and not scored: the logo is clipped at its right edge ("Fastkit-Mobility" reads "Mobiliy") and sits on a white box in dark mode, the header is black in light mode, and the cards have no icons. The footer pager shows only "Next: Features overview", which is fine.

PREVIOUS ISSUES STATUS (rounds >= 2):
- R6#1 (Triplegs subtitle overflowing its box): fixed. "moving segments" now fits inside the box on desktop light (y~1725), desktop dark and mobile.
- R6#2 (about 130px of mobile dead space under the vertical diagram): fixed. The gap from the "dashed: tours return to a Location" note to the "What you can do" H2 is now about 70px, which is normal section spacing.
- R6#3 (mobile pip block edge to edge): partial. It is now inset, but by 32px instead of the 16px the other blocks use (issue 2).
- R6#4 (grey light-mode chart labels): partial. Labels went from #777 to #666 but are still lighter than body text (issue 5).
- R6#5 (GeoLife legend tiny on mobile): not fixed (issue 3).
- R6#6 (output card looks like a code card): not fixed (issue 4).
- R6#7 (deferred brand items): deferred, unchanged.
- NEW regression: the mobile chart axis labels now overlap (issue 1).

WHAT WORKS (max 3, so it is not broken in the next round):
- The desktop first screen does the whole landing job above 860px: the claim H1, a two-sentence lead that names Fastkit-Mobility, a yellow primary CTA, a caption that names the dataset with a methodology link, and a log-scale chart with seconds and speedups. It works in both themes.
- The page runs proof, then try it, then vocabulary, then real data, then tasks, then next steps. The benchmark is followed by the runnable pandas/Polars snippet with output, then the hierarchy diagram (now clean in all themes), the GeoLife figure, six task cards, and three next-step cards. Keep this order.
- `<picture>` mobile variants for the chart and the diagram. The vertical diagram now crops tightly. Reuse this pattern for the GeoLife figure.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: The GeoLife user-day must be legible on phones, including the legend and the fixes.
  preferred_source: notebook-real-data
  suggested_filename: home-geolife-staypoints-triplegs-mobile.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: GeoLife, same busiest user-day. Use Positionfixes(df).generate_staypoints() and .generate_triplegs(). Use a 600x600 canvas with a legend font of at least 14px at display size, fixes at s=4 and alpha 0.6, tight bounds so the trace fills the frame, and an optional 1 km scale bar.

CROSS-PAGE NOTES:
- The r6 Features chart-axis change appears to have leaked into the Home mobile chart. Chart scripts shared between pages need a separate axis range for each figure, and every regenerated chart should be checked at 390px.
- The phone gutter for top-level code blocks is now added on top of the content padding. Fix this once in extra.css so top-level and tabbed blocks share one 16px edge on every page.
- One chart label colour (body text) should apply to Home, Features and Benchmarks in light mode.
- The header logo clipping and the white box in dark mode, plus the black light-mode header, remain deferred.
```
