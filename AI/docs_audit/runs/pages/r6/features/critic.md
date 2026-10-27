PAGE: docs/src/features/index.md   ROUND: 6
SCORES: first_impression=8 engagement=7 navigation=8 scannability=7 visual=7 ia_role=7 conciseness=7 trust=7
OVERALL: 7.3/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] The "Switching is one import" code block on mobile still runs edge to edge (mobile-full, y≈1010-1070 displayed). It has no left or right gutter, while every card above and below sits inside 16px margins. The r5 change log says code blocks keep their right padding on phones. That may be true of the padding inside the block, but the block itself still spans the full viewport.
   FIX: In the site CSS, under `@media (max-width: 600px)`, set `.md-typeset .highlight, .md-typeset pre { margin-inline: 0; border-radius: <card radius>; }`. Make sure the content wrapper's 16px padding is not cancelled by Zensical's negative `margin: 0 -.8rem` rule on `pre` at phone widths, which is the likely cause.
2. [conciseness] [P2] The new stat strip ("426x median speedup / pandas and Polars / scikit-mobility-style API") repeats the Fast, Any dataframe and Familiar API cards almost word for word, directly underneath them (desktop-light-fold, y≈290-580). On mobile the three pills wrap into three rows, about 140px, before the first card (mobile-fold).
   FIX: Keep the strip and cut the card overlap. Remove the "Fast" card, since the strip and the chart already carry 426x, so the grid has 5 or 4 cards. Or shorten the card `<p>` text so the cards add only new information, for example Familiar API -> "Port existing code with an import change". On mobile, use `flex-wrap` with smaller pill padding so the strip fits in 2 rows.
3. [scannability] [P2] The mobile trait cards still stack into about 950px of identical boxes before the code or the chart appears (mobile-full, y≈295-950 displayed). Not fixed since r3.
   FIX: Under 480px, show `.feature-grid` (the first grid only) as a 2-column compact grid with the card `<p>` hidden, or cut to 4 cards (see issue 2).
4. [visual] [P2] The mobile speedup chart (mobile-full, y≈1135-1385 displayed) now shows its full title, "Speedup vs scikit-mobility / 4M Brightkite check-ins", which is fixed. The axis still starts at 100x, though, so 315x looks about a quarter the length of 1329x. Category labels and ticks are still light gray small text.
   FIX: Re-export home-benchmark-speedup-mobile.png with the x axis starting at 1x (log scale), and with category and tick labels in a darker gray (≥ #555 on light) at ≥13px display size.
5. [ia_role] [P2] "Motifs" (sidebar entry and third Dig deeper card) is still an analysis how-to filed among the "why fastmob" pages. Not fixed.
   FIX: Move motifs.md under Learn in the zensical.toml nav and drop its card here, leaving Benchmarks and Optimizations as the Dig deeper cards.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile chart title clipped, small gray labels, axis at 100x): partial. The title is no longer clipped. The labels are still light gray, and the axis still starts at 100x.
2 (text-only above the fold): fixed (adequately). The stat strip gives a concrete number and backend claim in the first screen on desktop and mobile. There is still no code or figure above the fold, but the 5-second test now passes.
3 (long mobile card stack): not fixed.
4 (mobile code block has no side gutter): not fixed. It is visibly still edge to edge in mobile-full, even though the change log says otherwise.
5 (Motifs IA fit): not fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- First impression earns 8. The H1, a one-line lede and the stat strip pills (426x / pandas and Polars / skmob-style API) tell a stranger what fastmob is and why it matters within the first 330px, in both light and dark.
- Wayfinding earns navigation=8: the highlighted "Features overview" sidebar entry, the active Features tab, the 3-item "On this page" TOC, the Dig deeper card row, and the Previous Home / Next Benchmarks pager, which now stacks cleanly on mobile.
- The desktop landscape chart, in light and dark, shows skmob and fastmob times with "(Nx faster)" labels, a log-scale axis caption, and a methodology sentence linking to Benchmarks. This is credible real-data proof. Keep it.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a portrait speedup chart whose bar lengths are honest at 390px and whose labels are readable
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-mobile.png (re-export in place)
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: Same 4M Brightkite pandas numbers (5 runs). Horizontal log-scale bars starting at 1x, with the value as text. Keep the current title and subtitle. Labels in darker gray at ≥13px display size, transparent background.
- purpose: show "any dataframe in, same backend out" (pandas/Polars -> Narwhals -> Arrow buffers -> Rust kernel -> same backend)
  preferred_source: diagram
  suggested_filename: features-backend-flow.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: n/a

CROSS-PAGE NOTES: The edge-to-edge mobile code block appears to be a site-wide CSS issue. Zensical's negative horizontal margin on `pre` at small widths is the likely cause, and the r5 "right padding" change did not address it. Check it on Home, Learn and Reference. The mobile speedup chart's axis starting at 100x also affects Home. Features child pages still have no sidebar icons, while Measures pages do. Known deferrals (P2, not scored): the header logo with the white box, the black header in light mode, the footer "Previous" crossing sections (Previous=Home here), and no icons on cards.
