PAGE: docs/src/features/index.md   ROUND: 3
SCORES: first_impression=7 engagement=7 navigation=8 scannability=7 visual=6 ia_role=7 conciseness=7 trust=7
OVERALL: 7.0/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] "Speed at a glance" chart (desktop-light-full and desktop-dark-full, y≈1010-1350): the page still uses the old home-benchmark-speedup.png. Measure names, axis ticks and the x-axis caption are small (~11px) mid-gray text. On the dark page the image keeps its own gray-on-black labels, which are low contrast. On mobile (mobile-full, y≈1150-1380) the new figure-scroll wrapper keeps the text at a readable size, but the chart is cut off at the right edge ("scikit-mobil", "Execution time on 4M Brightkite"), with no visible scroll cue. Most of the gray skmob bars and every skmob time are off-screen, so the comparison the chart exists for is lost. The changes.md note "larger-font log chart" applied to Home, not to this page's image.
   FIX: Point this page at the larger-font chart made for Home in round 2 (or export features-speedup-overview.png at a ~7x4.5in size, fonts >= 13px at 690px display width, transparent background, labels readable on #fff and #0a0a0a). For mobile, use a portrait variant inside `<picture>` with a `(max-width: 600px)` source: one measure per row, with the skmob and fastmob times as text at the end of each bar. If the scroll wrapper stays, add a right-edge fade or a "scroll →" hint to `.figure-scroll` in the theme CSS.
2. [first_impression] [P2] Above the fold (desktop-light-fold, mobile-fold) is still text only: the H1, a one-line lede and six text cards. The chart starts at y≈1010 on desktop and at about the 3rd screen on mobile. Not fixed since r2.
   FIX: Move "Switching is one import" and "Speed at a glance" above the card grid, or add a one-line stat strip under the lede ("426x median speedup · pandas + Polars · drop-in skmob API") styled as three large-number chips.
3. [visual] [P2] Mobile code block (mobile-full, "Switching is one import") is clipped at the right edge: "import jump_len…". The block also runs edge to edge with no side gutter, unlike the cards.
   FIX: Shorten the snippet so it fits ~40 characters, e.g. `import skmob.measures.individual as m` -> `import fastmob as m`, or split the import onto two lines.
4. [visual] [P2] The six feature cards still have no icons (desktop-light-full, y≈300-740). On mobile (mobile-full) they form a ~950px stack of identical boxes before any code or figure appears. The floating TOC button (mobile-full, right edge of the "Low memory" card) overlaps card content.
   FIX: Add one lucide icon per card heading in `.feature-card` (repeat, zap, table, memory-stick, circle-check, github). On mobile, use a 2-column compact grid with the headline and one line each, or hide the card `<p>` under 480px. Move the mobile TOC button so it does not overlap content, or remove it (site CSS).
5. [conciseness] [P2] Benchmarks is still linked three times (Fast card, chart caption, Dig deeper) and Optimizations twice (Low memory card, Dig deeper). With arrows now on every card, the page reads as nine near-equal link tiles.
   FIX: Make the top six cards non-link `<div class="feature-card">` (no arrows), except "Familiar API" and "Any dataframe", which point to Learn pages that are not linked anywhere else. Keep Benchmarks/Optimizations links only in "Dig deeper" and in the caption.
6. [ia_role] [P2] "Motifs" (Dig deeper, sidebar) is still an analysis how-to filed next to the "why fastmob" pages (Benchmarks, Optimizations). Not fixed since r2.
   FIX: Move motifs.md to Learn in the zensical.toml nav and drop its card. Or rename the card to a capability claim ("Daily motifs across 4M check-ins") with a thumbnail, so the reader sees why it is a feature.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (chart illegible on mobile/dark, small fonts): partial. Mobile now scrolls at a readable size, but the chart is cropped with no cue. Desktop fonts and dark-mode contrast are unchanged (same image).
2 (no sidebar self-entry or active marker): fixed. "Features overview" shows in the sidebar, highlighted in light and dark.
3 (inconsistent link affordance): fixed. All cards carry "→".
4 (duplicate links; Validated pointed to Benchmarks): partial. Validated now links to the development setup and Open source links to GitHub. Benchmarks/Optimizations duplication remains (new issue 5).
5 (no card icons; weak card text contrast): partial. Card body text is darker in light mode. Icons are still missing.
6 (Motifs IA fit): not fixed.
7 (text-only above the fold): not fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- Wayfinding is now solid: an active "Features overview" sidebar entry, top tab highlighted, "On this page" TOC, a "Dig deeper" row and a Previous/Next pager. This earns navigation=8.
- The lede plus six short trait cards answer "what and why" with one concrete number (426x, 16 measures, 4M rows), and they render cleanly in light and dark.
- The before/after import snippet and the captioned real-data chart (5 runs, 4M Brightkite, pandas, linked methodology) are credible proof. Keep both.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a speed chart that is legible at desktop width, in dark mode and on a 390px phone without cropping
  preferred_source: notebook-real-data
  suggested_filename: features-speedup-overview.png (plus features-speedup-overview-mobile.png portrait)
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: Same published Brightkite 4M-row pandas numbers as home-benchmark-speedup.png (5 measures). Horizontal log-scale bars, fonts >= 13px at display size, transparent background, label colour passing contrast on #fff and #0a0a0a. Portrait variant ~4x6in with one measure per row and the speedup as large text.
- purpose: show "any dataframe in, same backend out" next to the Any dataframe card (pandas/Polars -> Narwhals -> Arrow buffers -> Rust kernel -> same backend)
  preferred_source: diagram
  suggested_filename: features-backend-flow.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: n/a

CROSS-PAGE NOTES: The `.figure-scroll` wrapper needs a visible scroll affordance (fade or hint) site-wide, because cropped charts on mobile hide their key data. The mobile floating TOC button overlaps card content. Mobile code blocks have no side gutter and clip long lines. The Features child pages still have no sidebar icons, while Measures pages do. The header logo with the white box in dark mode, and the black header in light mode, are known deferrals (P2).
