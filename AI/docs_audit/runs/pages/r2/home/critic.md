```
PAGE: docs/src/index.md (Home, role: landing)   ROUND: 2
SCORES: first_impression=7 engagement=4 navigation=8 scannability=7 visual=4 ia_role=7 conciseness=8 trust=6
OVERALL: 6.4/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual/engagement/trust] [P0] REGRESSION. "Fast on real data" and "From raw GPS fixes to tours" (desktop light, desktop dark, mobile full): both figures are gone and the raw Markdown is printed as body text: "![Bar chart comparing scikit-mobility and fastmob ...](assets/images/home/home-benchmark-speedup.png)" and "![Diagram: Positionfixes produce ...](assets/images/home/mobility-hierarchy.svg)". Cause: the new `<div class="figure-scroll" markdown>` wrappers rely on the `md_in_html` extension, which is not enabled in zensical.toml (only admonition, rst_math, arithmatex, details, superfences, tabbed, highlight, inlinehilite, snippets are listed). The built site/index.html still contains the literal `<div class="figure-scroll" markdown>` and two `![` strings. The page's only proof of speed and its only explanatory diagram are now broken markup; a visitor reads this as an unmaintained site.
   FIX: Either add `[project.markdown_extensions.md_in_html]` (and `attr_list`) to zensical.toml, or, without a new extension, replace each wrapper in index.md with plain HTML: `<div class="figure-scroll"><img src="assets/images/home/home-benchmark-speedup.png" alt="..."></div>` (same for mobility-hierarchy.svg). Then add a docs test (tests/docs) that fails if any built page under site/ contains the text `![` outside <code>/<pre>, or a `markdown>` attribute, so this cannot ship again.
2. [engagement] [P1] Round-1 issues 1-4 (mobile legibility, invisible fastmob bars and light-mode label contrast, dark-mode diagram brightness, Locations placement) cannot be verified because neither figure renders. The change log says the chart moved to a log scale and the diagram was redrawn; none of that reaches the reader.
   FIX: After fixing issue 1, re-screenshot and check at 390px that the chart labels are readable inside the horizontal-scroll container (the container should show a visible scroll affordance, e.g. a fade on the right edge or a "scroll ->" hint under 600px), and that the dark-mode diagram is not the brightest element on the page.
3. [first_impression] [P1] Fold (desktop 1440x900): the screen is hero + install + a 13-line code block + output, and ends on the bare "Fast on real data" heading, with no figure in view. Mobile fold (390px): the lead paragraph runs 7 lines at a large size and the three buttons wrap to two rows, so the reader sees only the first 3 lines of code. Unchanged since round 1.
   FIX: In docs/src/stylesheets/extra.css reduce `.hero-lead` to ~1.1rem with max-width ~60ch, and under 600px make `.hero-button` secondary buttons smaller or drop "Benchmarks" from the hero (the section below is the benchmark). On desktop, a two-column hero (lead + buttons + pip on the left, tabbed code on the right) would pull the chart heading and part of the chart into the first screen.
4. [visual] [P2] Hero code (all viewports): the output block sits as a second, separate grey card under the code inside the tab, with its own copy button; reads as a second snippet to run. On mobile the pip block bleeds edge to edge while the tab code block keeps a 16px inset, so the two blocks are misaligned.
   FIX: Style the output as an attached result (e.g. a `.hero-output` class with reduced top margin, no copy button, slightly different background, a small "Output" label), and give the pip block the same horizontal inset as the code block on mobile.
5. [visual] [P2] Deferred brand items still visible in every screenshot: header logo reads "Fastkit-Mobility" beside the "fastmob" site name, white box behind the logo in dark mode, black header in light mode. Noted only; not scored beyond P2.
   FIX: Re-export assets/logo_filled.svg as a fastmob mark with a transparent background when the source asset is available.

PREVIOUS ISSUES STATUS (rounds >= 2):
- R1#1 (mobile figure legibility): not verifiable; figures do not render (regressed, see issue 1).
- R1#2 (invisible fastmob bars, light-mode label contrast): not verifiable; chart does not render (regressed).
- R1#3 (off-centre desktop layout / loud dark diagram): partial. Layout fixed: `hide: [navigation, toc]` applied, content now centred at x~119-1307. Dark diagram not visible (regressed).
- R1#4 (Locations drawn after Tours): not verifiable; diagram does not render. Alt text is correct.
- R1#5 (duplicate link grids, "Why" restates hero): fixed. "Why fastmob" is two bullets of new information; Next steps now point at Columns and backends, Stay-point notebook, Resources, none of which the hero offers.
- R1#6 (hero snippet/output mismatch): fixed. Single call, distinct values (9.878827 / 14.423721), output inside each tab with a Polars-format output. Fold still ends without a figure (carried as issue 3).
- R1#7 (footer/pager): fixed. Footer shows "BSD-3-Clause licensed · PyPI · GitHub"; pager reads "Next: Features overview".
- R1 cross-page notes: nav reduced to 5 tabs (fixed); link colour unified per theme, dark gold in light / yellow in dark (fixed); logo and black header deferred.

WHAT WORKS (max 3, so it is not broken in the next round):
- Hero: claim-as-H1, one-sentence lead, yellow primary CTA, pip install, pandas/Polars tabs whose code and printed output now match. A stranger knows what this is in 5 seconds.
- Navigation and close: 5-tab header, centred content with sidebar/TOC hidden, a "Next steps" grid with destinations the hero does not repeat, a named pager ("Features overview") and a footer with license, PyPI and GitHub.
- "What you can do" 3x2 card grid: consistent in light and dark, stacks cleanly on mobile, every card links to a Learn page.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show what fastmob produces on real data, not only how fast: one user-day of raw GPS fixes with detected staypoints and triplegs, as the visual at the top of "What you can do" or beside the hero code on desktop. (Carried from round 1; not delivered.)
  preferred_source: notebook-real-data
  suggested_filename: home-geolife-staypoints-triplegs.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: GeoLife, one user-day; Positionfixes(df).generate_staypoints() and .generate_triplegs(); raw fixes as small grey dots, triplegs as coloured lines, staypoints as larger yellow-outlined circles; no basemap or a light one; export 1200x700 plus a 600x600 mobile crop, transparent background.

CROSS-PAGE NOTES:
- Any other page that adopted `<div ... markdown>` wrappers (the changes log says "figures scroll sideways on mobile" site-wide) will have the same raw-Markdown failure until md_in_html is enabled; in the current docs/src only index.md uses it, but check generated/notebook pages after the fix.
- Add a site-wide docs test that scans the built site/ for unrendered Markdown image syntax (`![`) and literal `markdown` attributes on HTML elements.
- Header logo ("Fastkit-Mobility", white box in dark) and black light-mode header remain deferred.
```
