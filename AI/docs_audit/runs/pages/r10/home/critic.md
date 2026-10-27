```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 10
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=7 ia_role=8 conciseness=8 trust=8
OVERALL: 7.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] In dark mode the hero benchmark chart labels are hard to read. desktop-dark-fold y~460-860: the category names ("radius of gyration" ... "max distance from home"), the scikit-mobility seconds ("63.4s", "338.3s"), the tick labels, the axis title and the legend all render #666 on the #0b0b0b page, about 3.5:1. Light mode uses the same #666 on white (5.7:1). It looks like one transparent PNG serves both themes, so the grey suits neither: too dim in dark and lighter than body text in light (R7#5 is still open). This is the page's main proof, and it is the first thing a reader looks at.
   FIX: Export two variants of home-benchmark-speedup.png and of -mobile.png. The light variant gets labels #333; the dark variant gets labels #d0d0d0 and scikit-mobility bars about #9a9a9a. Serve them with `#only-light` / `#only-dark` (or `<source media="(prefers-color-scheme: dark)">` inside the existing `<picture>`) in docs/src/index.md lines 21-26.
2. [visual] [P2] Regression in the mobile code-block gutter. In mobile-full y~1040-1080 the `pip install fastmob` block now bleeds edge to edge (grey background from x=0 to the viewport edge). The tabbed pandas card and the output card directly below are inset 16px (x~16-359). In R7 the inset was 32px. It has now overshot the other way.
   FIX: In docs/src/stylesheets/extra.css, under `@media (max-width: 600px)`, give the top-level `.md-typeset > .highlight` (or `div.language-bash`) `margin-inline: 0` with no negative margins, and the same border-radius as the tabbed block, so all three blocks share the 16px edge.
3. [ia_role] [P2] The "Try it" output card still looks exactly like a second code snippet. Desktop-light-full y~1285-1365 shows the same #f5f5f5 fill and radius as the code card above it, and the same is true in dark and on mobile. This is carried from R6#6 and R7#4.
   FIX: In extra.css, style a `text` fence that follows a code block inside a tab with a transparent background and a 1px border (var(--md-default-fg-color--lightest)), or add a small "Output" label above it with `::before`.
4. [visual] [P2] The new header shows the name twice. desktop-light-fold x~115-350 has the outlined logo with the stacked wordmark "Fastkit- / Mobility" and then, 10px to the right, the site title "Fastkit-Mobility" in bold. The two lockups compete, and the stacked mini wordmark (about 11px) is the less legible copy.
   FIX: Use the icon-only "F" mark as the header logo (logo-mark.svg) and keep the text title. Alternatively, keep the full lockup and hide `.md-header__title` on desktop. Mobile already shows only the title, so leave it as it is.
5. [visual] [P2] The search field is a solid black pill in an otherwise white header (desktop-light-fold x~880-1057, y~8-40). It is now the heaviest element in the header and draws the eye ahead of the yellow "Get started" CTA.
   FIX: In extra.css, restyle `.md-search__form` in light mode to a light-grey fill (#f2f2f2) with dark text and a 1px #e0e0e0 border. Keep the dark pill only in the black dark-mode header.
6. [trust] [P2] The alt text is still partly wrong. index.md line 24 now begins correctly but ends "fastmob is 423x to 2658x faster". This is carried from R7#6 and is partial.
   FIX: Replace "fastmob" with "Fastkit-Mobility" in that alt string.
7. [visual] [P2] The card icons repeat and carry little meaning. The same map icon is used on "Clean and segment", "Staypoints and locations" and "Networks and map matching", and the same `</>` icon is used on "Fit and evaluate", "Generate synthetic mobility" and the "Stay-point notebook" card (desktop-light-full y~1360-1600 and y~1810). Identical icons make the grid look templated and do not help the eye find a card.
   FIX: Give each card a distinct glyph (for example filter/scissors for clean, pin for staypoints, gauge for measure, route/graph for networks, chart-line for fit, sparkles/dice for generate, notebook for the notebook card) under docs/src/assets/images/icons/, all in the same outline style.

PREVIOUS ISSUES STATUS (rounds >= 2):
- R7#1 (mobile chart tick labels overlapping): fixed. The axis now reads "10x  1000x" with clear spacing (mobile-full y~1060).
- R7#2 (mobile pip block gutter): regressed. It went from a 32px inset to edge to edge (issue 2).
- R7#3 (GeoLife legend tiny on mobile): fixed. The mobile variant's legend reads at about 14px and is stacked vertically (mobile-full y~1670-1730).
- R7#4 (output card looks like a code card): not fixed (issue 3).
- R7#5 (grey light-mode chart labels): not fixed. They are still #666, and dark mode now shows that the same grey is too dim there (issue 1).
- R7#6 (alt text naming): partial (issue 6).
- R7#7 (deferred brand items): mostly fixed. The logo is no longer clipped, the white box in dark mode is gone, the light header is now white with a gold rule that separates it cleanly from the content, and the dark header blends into the black page. New, smaller issues: the name is shown twice (issue 4) and the search pill is heavy (issue 5). Home has no previous/next pager, which is correct for the landing page because the Next-steps cards do that job.

WHAT WORKS (max 3, so it is not broken in the next round):
- The first screen still does the whole landing job in both themes: the claim H1, a lead sentence that names Fastkit-Mobility and the pandas/Polars promise, a yellow primary CTA, a captioned benchmark with a methodology link, and per-measure seconds and speedups. The new white header with its gold rule makes this first screen calmer than R7's black bar.
- The page runs proof, then try it, then vocabulary diagram, then real GeoLife figure, then six task cards, then why, then next steps. Each step answers the reader's next question. Keep this order.
- The `<picture>` mobile variants now cover all three figures (chart, vertical hierarchy diagram, GeoLife). All of them are legible at 390px.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: The hero benchmark must be equally legible in light and dark themes.
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-dark.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: The same Brightkite 4M benchmark results and the same script as home-benchmark-speedup.png. Use a transparent background, text and ticks #d0d0d0, scikit-mobility bars #9a9a9a, and keep the gold Fastkit-Mobility bars. Also produce home-benchmark-speedup-mobile-dark.png from the mobile variant script.

CROSS-PAGE NOTES:
- Chart PNGs that are shared across themes use #666 labels everywhere (Home, and likely Features and Benchmarks). Adopt one rule: a light and a dark variant per chart, with body-text label colours.
- The top-level code-block gutter on phones is still unstable from round to round. Fix it once in extra.css so top-level and tabbed blocks share one 16px edge on every page, and add a 390px check for it.
- The header changes (white with a gold rule in light, black in dark) work well site-wide. The remaining header polish is the duplicated logo wordmark plus title, and the black search pill in light mode.
```
