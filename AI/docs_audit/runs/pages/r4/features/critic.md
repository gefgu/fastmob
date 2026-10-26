PAGE: docs/src/features/index.md   ROUND: 4
SCORES: first_impression=7 engagement=7 navigation=8 scannability=7 visual=6 ia_role=7 conciseness=8 trust=7
OVERALL: 7.1/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] "Speed at a glance" chart: the page still embeds `home/home-benchmark-speedup.png` (index.md line 45). On mobile (mobile-full, y≈1140-1400 displayed) the chart is still cut off at the right edge: the legend reads "scikit-mobil", the axis caption reads "Execution time on 4M Brightkite", "1208x faste…" is clipped, and every gray skmob bar runs off-screen with no visible skmob time. The new "scroll →" hint helps, but the comparison the chart exists for is hidden on first view. On desktop (light and dark, y≈1020-1350) the measure labels, ticks and axis caption are still small mid-gray text. That is readable in light mode and weak on the dark background. changes.md says Home got a portrait phone chart through `<picture>`. This page did not get it.
   FIX: In index.md, replace the bare `<img>` with the same `<picture>` block Home uses: a `<source media="(max-width: 600px)">` that points at the portrait speedup chart, with the landscape image as the fallback. Then drop the `.figure-scroll` wrapper for this figure. If the Home landscape image was re-exported with larger fonts, point to that file as well.
2. [visual] [P1] Regression or not applied: the floating mobile TOC button is still drawn over the "Low memory" card (mobile-full, right edge, y≈597 displayed), although changes.md says the toggle was hidden site-wide.
   FIX: Check that the CSS rule hiding the floating TOC toggle targets the selector Zensical actually renders at 390px, and that it is in the stylesheet this page loads. Re-screenshot to confirm.
3. [first_impression] [P2] Above the fold (desktop-light-fold, mobile-fold) is still text only: the H1, a one-line lede and six text cards. The chart starts at y≈1000 on desktop and around the 3rd screen on mobile. Not fixed since r2.
   FIX: Add a stat strip of three chips under the lede ("426x median speedup · pandas + Polars · skmob-compatible API"), or move "Switching is one import" above the card grid. This is the cheapest way to get proof above the fold.
4. [visual] [P2] The six trait cards have no icons. On mobile they form a ~950px stack of identical boxes before any code or figure appears (mobile-full, y≈210-930 displayed). This item was deferred in changes.md.
   FIX: Add one lucide icon per `.feature-card` heading (repeat, zap, table, memory-stick, circle-check, github). Under 480px, hide the card `<p>` or use a 2-column compact grid.
5. [ia_role] [P2] "Motifs" (Dig deeper card and sidebar) is still an analysis how-to filed next to the "why fastmob" pages. Not fixed since r2.
   FIX: Move motifs.md to Learn in the zensical.toml nav and drop its card here. Or rename the card to a capability claim with a thumbnail.
6. [visual] [P2] The mobile code block ("Switching is one import") still runs edge to edge with no side gutter, unlike the cards (mobile-full, y≈1000-1065 displayed). It no longer clips, but it looks misaligned.
   FIX: Give `.md-typeset pre` the same 16px side margin as the cards at mobile width (site CSS).

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (chart small fonts, dark contrast, mobile crop): partial. A "scroll →" hint was added under the mobile figure. The same landscape image is still used, so it is still cropped on mobile and still has small gray labels.
2 (text-only above the fold): not fixed.
3 (mobile code block clipped): fixed. The snippet is now `import fastmob as m` and fits. The missing side gutter remains (P2, issue 6).
4 (no card icons; mobile card stack; TOC button overlap): not fixed. Icons are still missing, and the TOC button still overlaps the Low memory card.
5 (duplicate Benchmarks/Optimizations links): fixed. Only "Familiar API" and "Any dataframe" keep arrows. Benchmarks is linked only in the caption and Dig deeper, Optimizations only in Dig deeper.
6 (Motifs IA fit): not fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- Wayfinding: the active "Features overview" sidebar entry, the highlighted top tab, the "On this page" TOC, the "Dig deeper" row and the Previous/Next pager. This earns navigation=8.
- Link affordance is now honest: two arrowed link cards and four plain trait cards, so the page no longer reads as nine equal tiles. With no repeated links, this earns conciseness=8.
- The before/after import snippet plus the captioned real-data chart (5 runs, 4M Brightkite, pandas, linked methodology) are credible proof. Keep both.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a speed chart that is readable at 390px with no cropping. Reuse the portrait variant made for Home.
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-mobile.png (reuse the existing file)
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: Same Brightkite 4M-row pandas numbers. One measure per row, with the skmob and fastmob times and the speedup as text. Fonts >= 13px at display size, transparent background.
- purpose: show "any dataframe in, same backend out" (pandas/Polars -> Narwhals -> Arrow buffers -> Rust kernel -> same backend)
  preferred_source: diagram
  suggested_filename: features-backend-flow.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: n/a

CROSS-PAGE NOTES: Confirm that the CSS hiding the floating mobile TOC toggle actually applies. It is still visible on this page at 390px. Mobile code blocks have no side gutter. The Features child pages have no sidebar icons, while Measures pages do. Known deferrals (P2): the header logo with the white box, the black header in light mode, and the footer "Previous" link crossing sections (here Previous=Home).
