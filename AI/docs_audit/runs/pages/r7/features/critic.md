PAGE: docs/src/features/index.md   ROUND: 7
SCORES: first_impression=8 engagement=7 navigation=8 scannability=7 visual=7 ia_role=7 conciseness=7 trust=7
OVERALL: 7.3/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [conciseness] [P1] The page says the same three things twice in a row. The stat strip ("426x median speedup / pandas and Polars / scikit-mobility-style API") is followed directly by the Familiar API, Fast and Any dataframe cards, which repeat it (desktop-light-full y≈290-580). The "Fast" card's 426x then appears a third time in the chart section. On mobile the three pills take three rows (about 140px) before the first card (mobile-full, displayed y≈180-270). Not fixed since r6.
   FIX: In docs/src/features/index.md, delete the "Fast" card. The strip and the chart already carry the number, so the first grid drops to 5 cards. Also trim the Familiar API card `<p>` to "Port existing code with an import change." On phones, reduce `.stat-strip span` padding and font-size so the three pills fit in 1-2 rows.
2. [scannability] [P1] On mobile, six identical trait cards stack for about 950px of original height before any code or chart appears (mobile-full, displayed y≈295-945). A phone reader scrolls through roughly 2.5 screens of same-looking boxes. Not fixed since r3.
   FIX: Under 480px, make the first `.feature-grid` (only that one) a 2-column grid of compact cards: h3 only, `<p>` hidden, smaller padding. Doing issue 1 as well leaves 5 or 4 short tiles of about 250px in total.
3. [visual] [P2] The new mobile speedup chart now starts its axis at 1x, which is fixed, but its tick labels collide: "1x 10x100x1000x" runs together under the bars (mobile-full crop, chart bottom, about y≈2010 original). The category labels are still mid-gray small text.
   FIX: Re-export docs/src/assets/images/home/home-benchmark-speedup-mobile.png with the ticks thinned to 1x / 100x / 1000x, or drop the tick axis entirely because each bar already has its value. Make the category labels ≥ #555 at ≥13px display size.
4. [visual] [P2] The mobile code block no longer runs edge to edge, which is fixed. It is now inset about 32px on the left and about 47px on the right, so it no longer lines up with the headings and cards at the 16px gutter. The last token, "as m", touches the right edge of the block (mobile-full crop, "Switching is one import").
   FIX: In the phone media query, set the `.highlight`/`pre` margin-inline to 0 so the block shares the 16px content gutter, and keep its inner right padding. Shorten the example to `import skmob.measures.individual as m` → `import fastmob as m` if it still overflows.
5. [ia_role] [P2] "Motifs" (sidebar entry and third Dig deeper card) is an analysis how-to filed among the "why Fastkit-Mobility" pages. Not fixed.
   FIX: Move motifs.md under Learn in the zensical.toml nav and remove its card here, leaving Benchmarks and Optimizations as the Dig deeper cards.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile code block edge to edge): fixed. The block now has side margins. Its alignment is off, which is filed as new issue 4.
2 (stat strip duplicates cards, 3-row pills on mobile): not fixed.
3 (long mobile card stack): not fixed.
4 (mobile chart axis starts at 100x, light labels): partial. The axis now starts at 1x and the bar lengths are honest. The tick labels now overlap, and the category labels are still light gray.
5 (Motifs IA fit): not fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- First impression earns 8. The H1, a one-line lede that names Fastkit-Mobility correctly, and the stat strip tell a stranger what this is and why it matters within the first 330px, in both light and dark.
- Navigation earns 8 because the reader always knows where they are and where to go next: the highlighted "Features overview" sidebar entry, the active Features tab, the 3-item TOC, the Dig deeper cards, and a Previous/Next pager that stacks cleanly on mobile.
- The desktop chart, in both themes, shows scikit-mobility vs Fastkit-Mobility (pandas) timings with "(Nx faster)" labels and a log-scale caption. The methodology sentence links to Benchmarks. This is credible real-data proof, and the mobile chart is now an honest log scale starting at 1x.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a mobile speedup chart with tick labels that do not collide and readable category labels
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-mobile.png (re-export in place)
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: Same 4M Brightkite pandas numbers (5 runs). Horizontal log-scale bars from 1x with the value as text. Ticks at 1x/100x/1000x only, or no tick axis. Labels in #555 or darker at ≥13px display size, transparent background.
- purpose: show "any dataframe in, same backend out" (pandas/Polars -> Narwhals -> Arrow buffers -> Rust kernel -> same backend)
  preferred_source: diagram
  suggested_filename: features-backend-flow.png
  suggested_location: docs/src/assets/images/features/

CROSS-PAGE NOTES: The new phone code-block margin looks asymmetric (about 32px left, about 47px right) and misaligned with the 16px content gutter. Check it on Home, Learn and Reference, where the same CSS applies. The mobile speedup chart is shared with Home, so the tick collision appears there too. Features child pages still have no sidebar icons, while Measures pages do. Known deferrals (P2, not scored): the clipped header logo image, the black header in light mode, the footer "Previous" crossing sections (Previous=Home here), and no icons on cards.
