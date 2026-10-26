PAGE: docs/src/features/index.md   ROUND: 2
SCORES: first_impression=7 engagement=7 navigation=6 scannability=7 visual=6 ia_role=7 conciseness=7 trust=7
OVERALL: 6.8/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile (mobile-full, "Speed at a glance"): the chart shrinks to the 358px column width, so its axis labels, measure names and the "(315x faster)" annotations are unreadably small. The r1 change log says figures scroll sideways on mobile, but this image does not. On desktop (light and dark full) the chart's text is also small, about 9px: measure names and the x-axis caption are hard to read at normal zoom, and in dark mode the gray labels on near-black are low contrast.
   FIX: Re-export home-benchmark-speedup.png (or give Features its own copy, features-speedup-overview.png) with larger fonts (measure labels and annotations >= 13px at the 690px display width) and a transparent or theme-aware background with lighter label text for dark mode (or a `#only-dark` variant). For mobile, either apply the horizontal-scroll figure wrapper with a min-width of ~600px to this image or make a taller portrait variant with the bars stacked.
2. [navigation] [P1] Left sidebar (all desktop shots): the sidebar lists only Benchmarks / Optimizations / Motifs. The "Features overview" entry from zensical.toml does not appear, and nothing is highlighted, so the reader has no "you are here" marker on this page. This is unchanged from r1.
   FIX: The section index is being absorbed by navigation.indexes, so the sidebar has no self-entry. Either drop navigation.indexes for the Features section, or style the section title in the sidebar as an active heading ("Features" above the three children, highlighted when on index.md). Check how the Measures section handles this and make the two consistent.
3. [navigation] [P2] Link affordance is inconsistent. The top six cards are whole-card links with no arrow or hover cue (desktop-light-full, y≈300-720), while the "Dig deeper" cards have "→". A reader cannot tell that "Familiar API" goes to a Learn page or that "Validated" goes to Benchmarks.
   FIX: In features/index.md, either add "→" to the top-grid card headings too, or make the top grid non-link cards and keep the links only in "Dig deeper" plus the caption link. The second option is preferred: less duplication (see issue 4).
4. [conciseness] [P2] The Benchmarks page is linked four times (Fast card, Validated card, the chart caption, and the Dig deeper card) and Optimizations twice (Low memory card, Dig deeper). The "Validated" card claims tests but links to Benchmarks, not to tests or CI, so the link does not prove the claim.
   FIX: Point "Validated" at the CI badge, the test directory on GitHub, or about/development.md#tests. Remove links from the cards whose target already appears in "Dig deeper".
5. [visual] [P2] There are no icons on the cards. The six feature cards are text-only, so they look alike at a glance, and on mobile (mobile-full) they form a ~800px stack of near-identical gray boxes. Card body text in light mode is mid-gray (#777-ish) on white, which is weaker contrast than the lede.
   FIX: Add one lucide icon per card (e.g. lucide/repeat, zap, table, memory-stick, check-circle, github) above or beside each h3 in `.feature-card`. Darken the card `<p>` colour in light mode to the body-text token.
6. [ia_role] [P2] The "Dig deeper" row lists Motifs ("Daily mobility motifs and how they are matched"). This is an analysis capability, not a library trait like the other two children (Benchmarks, Optimizations), so the Features section mixes "why fastmob" pages with a domain how-to.
   FIX: Move motifs.md to Learn (or to Reference > Measures) in the zensical.toml nav and remove its card here. Alternatively, keep it but describe it as a capability ("Find daily motifs in 4M check-ins") so the reader knows why it sits under Features.
7. [first_impression] [P2] Above the fold (desktop-light-fold, mobile-fold) is text-only: the H1, the lede and six text cards. The strongest proof (the speed chart) starts at y≈990 on desktop and is far below the fold on mobile.
   FIX: Move the "Switching is one import" snippet and the chart above the card grid, or put a 2-line headline stat strip ("426x median speedup · pandas + Polars · drop-in skmob API") under the lede so the first screen carries one concrete number visually, not only inside a card.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (bare 500x claim, unlinked): fixed. The number is now "Median 426x ... 16 individual measures on 4M Brightkite rows", linked to Benchmarks, and the chart caption gives the method. The Validated card still links to Benchmarks rather than to tests (see new issue 4).
2 (no figure/code, bullet list): fixed. Card grid, import snippet and real benchmark chart.
3 (H1 mismatch, no lede, not a hub): partial. The H1 "Features", the lede and the "Dig deeper" row are fixed. The sidebar still has no Overview entry or current-page marker (new issue 2).
4 (ordering, overlap): fixed. Familiar API leads, Fast is second, and zero-copy is no longer duplicated.
5 (Fastkit-Mobility brand in page text): partial. The page text uses "fastmob" throughout. The header logo still reads "Fastkit-Mobility", which is a known deferral.
6 (<30KB wheel claim): fixed (removed).
7 (mobile text column): fixed. Cards stack cleanly on mobile, but the chart now regresses on mobile (new issue 1).

WHAT WORKS (max 3, so it is not broken in the next round):
- The lede sentence plus the six-card grid answers "what and why" in one screen, and the cards render cleanly in light, dark and mobile.
- The "Switching is one import" before/after snippet is the single most persuasive block for skmob users. Keep it short.
- The speed chart uses real published numbers, a log scale and per-bar speedup annotations, with a caption that states the dataset, run count and backend and links to the methodology.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a speed chart that stays legible on mobile and in dark mode (same data, larger type)
  preferred_source: notebook-real-data
  suggested_filename: features-speedup-overview.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: Reuse the published benchmark numbers behind home-benchmark-speedup.png (Brightkite 4M rows, pandas, 5 measures). Make a horizontal log-scale bar chart at figsize ~(7,4.5) with font size >= 13, a transparent background, and label colours that pass contrast on both #fff and #0a0a0a. Optionally export a narrow portrait variant for mobile.
- purpose: show "drop-in + any backend" in one glance next to the "Any dataframe" card (pandas/polars in -> Narwhals -> Rust kernel via Arrow buffers -> same backend out)
  preferred_source: diagram
  suggested_filename: features-backend-flow.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: n/a

CROSS-PAGE NOTES: Section index pages under navigation.indexes show no self-entry or active marker in the left sidebar (Features here; check Learn, Reference and About). The Features children (Benchmarks, Optimizations, Motifs) still have no icons, while the Measures pages do. The figure horizontal-scroll wrapper from the r1 theme change does not seem to apply to this image on mobile. Check that the CSS selector matches plain Markdown images and not only `<figure>`. The header logo text and white box in dark mode are a known deferral (P2).
