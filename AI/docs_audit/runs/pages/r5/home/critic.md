```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 5
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=7 ia_role=8 conciseness=8 trust=8
OVERALL: 7.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Three round-4 claims in changes.md do not show in the r5 screenshots. (a) The "Try it" output block (desktop-light-full y~1285-1365, dark same; mobile-full y~1440-1520) still shows a copy icon at its top right and is styled exactly like the code card above it. (b) The mobile hero is not on one row: mobile-fold y~448-498 has "Benchmarks" wrapped onto its own full-width row. (c) The light-mode chart labels (desktop-light-fold y~460-860) are still mid-grey on white: category names, axis ticks, legend and the scikit-mobility values. Either the CSS and PNG changes did not reach the built site or the selectors miss.
   FIX: Rebuild the site and re-capture before changing anything else. If (a) persists, check that the output fences carry the attribute the copy-button suppression keys on. For pymdownx, wrap the fence in `<div class="result" markdown>` and add `.result .md-clipboard{display:none}` plus a lighter background or a 1px border with an "Output" label in docs/src/stylesheets/extra.css. For (b), under 600px set `.hero-actions{flex-wrap:nowrap}` and `.hero-button{flex:1 1 0;padding-inline:.6em;font-size:.85rem}`, or hide the Benchmarks button on mobile because the caption link below already goes there. For (c), re-export home-benchmark-speedup.png with labels in body-text colour, or ship #only-light and #only-dark variants.
2. [visual] [P1] This is a regression. The mobile chart title now clips at the viewport edge and reads "speedup vs scikit-mobilit" (mobile-fold y~642-674). The final "y" is cut off, which looks broken on the first screen a phone user sees.
   FIX: In home-benchmark-speedup-mobile.png, wrap the title onto two lines ("speedup vs" / "scikit-mobility, 4M check-ins") or centre it inside a 16px safe margin, then re-export.
3. [visual] [P2] On mobile, the `pip install fastmob` block still runs edge to edge at x=0-375 (mobile-full y~1115-1150). The tabbed code card and the output card below it keep the 16px inset. This is carried from R4#4.
   FIX: In extra.css, under `@media (max-width: 600px)`, set `.md-typeset > .highlight { margin-inline: 0; }` or give the top-level highlight the same border-radius and inset as the tabbed blocks. Check that `.md-content__inner` padding is not being cancelled by a negative margin on `.highlight`.
4. [visual] [P2] GeoLife figure (desktop-light-full y~1030-1490; mobile-full y~2340-2640). The figure itself is a big gain. Three defects remain. The grey position-fix dots are almost invisible, especially in dark mode (desktop-dark-full y~1060-1090), so the "raw fixes in, structure out" story loses its input layer. On mobile the legend text is about 9px and unreadable. The figure has no geographic context: there is no basemap or scale, so it reads as abstract coloured lines.
   FIX: Re-render with the fixes as larger dots (s~4, alpha~0.6) in a mid-grey that works on both themes. Add a faint basemap (CartoDB Positron, plus Dark Matter for an #only-dark variant) or at least a 1 km scale bar. Make the legend font about 2x larger in the 600x600 mobile crop and serve that crop with `<picture>`.
5. [visual] [P2] On mobile there is about 130px of dead space between the diagram's "dashed: tours return to a Location" note and the "What you can do" H2 (mobile-full y~2340-2470). The vertical SVG probably has bottom padding in its viewBox. Separately, the Triplegs subtitle "movement between" reads as a truncated phrase in both diagrams.
   FIX: Crop the viewBox of mobility-hierarchy-vertical.svg to its content. Change the Triplegs subtitle to "movement between stops" (desktop box width permitting) or to "moving segments".
6. [first_impression] [P2] Mobile fold (mobile-fold.png): the lead paragraph is still 5 lines at about 21px. Together with the wrapped button row, only 3 bars of the chart fit in the first screen. This is carried from R4#6.
   FIX: Under 600px in extra.css, set `.hero-lead { font-size: 1.05rem; line-height: 1.5; }`. With issue 1b fixed, the full speedup chart should reach the fold.
7. [visual] [P2] Deferred brand items, visible in every screenshot and noted only: the header logo text "Fastkit-Mobilit", the white logo box in dark mode, the black header in light mode, and the footer "Previous" link crossing sections.
   FIX: Swap in a transparent fastmob mark when the source asset exists.

PREVIOUS ISSUES STATUS (rounds >= 2):
- R4#1 (no real-data figure; "What you can do" was text-only): fixed. The GeoLife staypoints and triplegs figure now sits under the H2 with a caption that names the dataset and city. Polish items are in issue 4.
- R4#2 (desktop diagram crossing and floating label): fixed. Locations now sits above Staypoints, nothing crosses, and the label rests on the dashed Locations -> Tours path.
- R4#3 (mobile diagram dropped the Locations -> Tours relation): fixed. Locations and Tours share the bottom row, joined by a dashed arrow and an explanatory note.
- R4#4 (output looks like a snippet; mobile pip block runs edge to edge): not fixed. The copy button is still visible on the output and the pip block still runs edge to edge (issues 1a, 3).
- R4#5 (grey light-mode chart labels; mobile chart title touching the edge): not fixed, and the title part regressed. The labels are still grey, and the title is now clipped to "scikit-mobilit" (issues 1c, 2).
- R4#6 (mobile fold: long lead, wrapped button row): not fixed (issues 1b, 6).
- R4#7 (deferred brand items): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- Desktop first screen: the claim H1, a two-sentence lead, a yellow primary CTA, a caption that names the dataset and comparison, and a log-scale chart with per-measure seconds and speedups, all inside 900px. This earns first_impression and trust their 8s.
- The page now proves both speed (benchmark chart) and output (GeoLife staypoints and triplegs on real data). The hierarchy diagram explains the vocabulary before the capability cards use it. That pairing is what lifts engagement to 8.
- The `<picture>` mobile variants: a portrait speedup chart and a vertical hierarchy diagram that now shows all six levels and the Locations -> Tours relation. Keep this pattern for the GeoLife figure as well.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: The same GeoLife user-day, legible on phones and in dark mode, so readers can see the raw fixes turn into triplegs and stops.
  preferred_source: notebook-real-data
  suggested_filename: home-geolife-staypoints-triplegs-mobile.png (plus -dark variants)
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: GeoLife, same busiest user-day. Use Positionfixes(df).generate_staypoints() and .generate_triplegs(). Draw fixes at s=4 and alpha 0.6 in grey (#9a9a9a). Use 600x600 for mobile with a legend font of at least 14px at display size. Add a faint CartoDB basemap (Positron for light, DarkMatter for dark) or a 1 km scale bar. Serve the variants with `<picture>`, #only-light and #only-dark.

CROSS-PAGE NOTES:
- Verify the build pipeline: three changes reported as applied in r4/changes.md (no copy button on output blocks, one-row hero on phones, darker chart labels) are not visible in the r5 captures. Possible causes are a stale build, a CSS selector that misses, or the PNG not being regenerated. Confirm this before attributing a lack of progress on other pages.
- One docs chart style (body-text-coloured labels, safe margins on mobile exports, light and dark variants) should apply across Home, Features and Benchmarks.
- Top-level code blocks run edge to edge on phones while tabbed blocks keep the inset. Fix this once in extra.css.
- The header logo, black light-mode header and footer "Previous" link crossing sections remain deferred.
```
