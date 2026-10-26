PAGE: docs/src/features/index.md   ROUND: 5
SCORES: first_impression=7 engagement=7 navigation=8 scannability=7 visual=7 ia_role=7 conciseness=8 trust=7
OVERALL: 7.3/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile "Speed at a glance" chart (mobile-full, y≈1120-1390 displayed): the new portrait chart fits the width, but its title is cut off inside the PNG itself. It reads "speedup vs scikit-mobilit". You can see the same clipping in the source asset `assets/images/home/home-benchmark-speedup-mobile.png`, where the title runs past the right canvas edge. The x axis starts at 100x, so the 315x/326x bars look small next to 1329x, and all labels are mid-gray small text. Home uses the same file, so it has the same defect.
   FIX: Re-export home-benchmark-speedup-mobile.png. Either center the title over the whole figure with `fig.suptitle` and `bbox_inches="tight"`, or shorten it to "Speedup vs skmob (4M rows)". Make the label color darker (the same as the darker labels used for Home in r4). Optionally start the axis at 1x so bar length reflects magnitude.
2. [first_impression] [P2] Above the fold is still text only on desktop and mobile (desktop-light-fold, mobile-fold): the H1, a one-line lede and six text cards. The first proof (code) appears at about y≈830 on desktop and below the sixth card on mobile. Not fixed since r2.
   FIX: Under the lede in index.md, add a one-line stat strip: "426x median speedup · pandas + Polars · skmob-compatible API". Or move "Switching is one import" above the `.feature-grid`.
3. [scannability] [P2] On mobile, the six trait cards stack into roughly 950px of identical boxes before any code or figure appears (mobile-full, y≈205-915 displayed). Not fixed. The card icons are a known deferral, but the length is a layout problem.
   FIX: Under 480px, show `.feature-grid` as a 2-column compact grid with the card `<p>` hidden. Or cut to four cards by merging "Validated" with "Open source" and "Low memory" with "Fast".
4. [visual] [P2] The mobile code block ("Switching is one import", mobile-full y≈980-1045 displayed) still runs edge to edge with no side gutter, while the cards above and below have 16px. Not fixed.
   FIX: In site CSS at max-width 600px, give `.md-typeset pre` / `.highlight` the same 16px side margin and border-radius as the cards.
5. [ia_role] [P2] "Motifs" (sidebar and third Dig deeper card) is still an analysis how-to filed among the "why fastmob" pages. Not fixed.
   FIX: Move motifs.md under Learn in the zensical.toml nav and drop its card here, leaving Benchmarks and Optimizations as the two Dig deeper cards.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile chart cropped, small gray labels): partial. Phones now get the portrait `<picture>` source, so the skmob/fastmob cropping and "scroll →" are gone. The new image's title is clipped, and the labels are still small and gray.
2 (floating TOC button over Low memory card): fixed. No button appears over the cards in mobile-full or mobile-fold.
3 (text-only above the fold): not fixed.
4 (no card icons / long mobile card stack): not fixed. Icons are deferred, and the stack length remains (issue 3).
5 (Motifs IA fit): not fixed.
6 (mobile code block no side gutter): not fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- Wayfinding earns navigation=8: the active "Features overview" sidebar entry, the highlighted Features tab, the 3-item "On this page" TOC, the Dig deeper card row, and the Previous Home / Next Benchmarks pager.
- Conciseness=8: two arrowed link cards plus four plain trait cards, a 4-line import snippet, one chart with a one-sentence methodology caption, and no repeated links.
- The desktop landscape chart, in light and dark, shows both skmob and fastmob times with "(Nx faster)" labels and a log-scale caption. It is credible, real-data proof. Keep it.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a portrait speedup chart that reads cleanly at 390px with an unclipped title
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-mobile.png (re-export in place)
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: Same 4M Brightkite pandas numbers (5 runs). Horizontal bars of speedup per measure with the value as text. Title "Speedup vs skmob (4M rows)" centered with tight bbox. Labels in a darker gray, >=13px at display size, transparent background.
- purpose: show "any dataframe in, same backend out" (pandas/Polars -> Narwhals -> Arrow buffers -> Rust kernel -> same backend)
  preferred_source: diagram
  suggested_filename: features-backend-flow.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: n/a

CROSS-PAGE NOTES: The clipped title in home-benchmark-speedup-mobile.png also affects Home on phones. The secondary-sidebar fix for the mobile TOC overlap works here; confirm it on the other pages. Mobile code blocks still have no side gutter site-wide. Features child pages have no sidebar icons, while Measures pages do. Known deferrals (P2, not scored): the header logo with the white box, the black header in light mode, the footer "Previous" link crossing sections (Previous=Home here), and no icons on cards.
