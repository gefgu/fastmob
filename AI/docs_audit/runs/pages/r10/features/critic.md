PAGE: docs/src/features/index.md   ROUND: 10
SCORES: first_impression=8 engagement=7 navigation=8 scannability=7 visual=6 ia_role=7 conciseness=7 trust=7
OVERALL: 7.1/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P0] The new backend-flow diagram is broken. It renders as a tiny, unreadable strip of boxes in the sixth grid cell, next to the "Open source" card (desktop-light-full and desktop-dark-full, y≈655-680; mobile-full, beside "Open source", y≈650). It is about 200px wide and its labels are about 3px tall. Cause: in index.md the `<div class="figure">` sits before the closing `</div>` of the first `.feature-grid`, so the figure became a grid item. In dark mode it also shows as white boxes on black. A reader sees what looks like a rendering bug next to the trust cards.
   FIX: In docs/src/features/index.md, move the closing `</div>` of the first feature-grid to just after the "Open source" card, so the figure sits outside the grid at full content width. Better still, put it under its own H2 (for example "Any dataframe in, same backend out") after "Switching is one import". Give it a dark-mode variant through `<picture>` with prefers-color-scheme, or a transparent background with light text. Check that the box labels are at least 12px at the 680px content width; if they are not, re-export the image larger.
2. [visual] [P1] Regression on phones: the "Switching is one import" code block bleeds to the left edge of the screen again. Its background starts at x=0 while the right edge stops about 15px short (mobile-full crop at the code block; pixel check: x=0 is gray, x≥380 is white). It is misaligned with the 16px gutter that the headings and cards use.
   FIX: In docs/src/stylesheets/extra.css, phone media query: set `.md-typeset .highlight, .md-typeset pre` to `margin-inline: 0` with `border-radius` kept, so the block spans the 16px gutters on both sides. Check this on Home, Learn and Reference too.
3. [conciseness] [P1] The stat strip still mostly repeats the cards below it: "pandas and Polars" repeats "Any dataframe" and "scikit-mobility-style API" repeats "Familiar API". On desktop the three pills wrap to two rows, and on mobile they take three rows, about 130px, before the first card (mobile-full, y≈270-395). Not fixed since r6, although the "Fast" card was removed.
   FIX: Cut the strip to one pill, "623x median speedup on 16 measures", which is the only fact the cards do not carry. You could instead fold it into the lede as bold text and delete the strip.
4. [visual] [P2] "Open source" reuses the same `</>` icon as "Familiar API" (icon-code.svg), so two of five cards look the same at a glance (desktop fold, cards 1 and 5).
   FIX: Give "Open source" a distinct icon, such as a git-branch or book icon in docs/src/assets/images/icons/, or drop the card and fold "BSD-3, reproduce every benchmark" into the "Validated" card.
5. [visual] [P2] The dark-mode chart category labels ("radius of gyration", "jump lengths" and so on) and the axis caption are mid-gray on near-black and low contrast (desktop-dark-full, y≈1190-1440). The light-mode labels are still light gray as well.
   FIX: Re-export docs/src/assets/images/home/home-benchmark-speedup.png (and its dark variant, if there is one) with label and tick text at ≥ #555 in light and ≥ #bbb in dark. Alternatively, serve a dark-specific PNG through `<picture>`.
6. [scannability] [P2] On mobile, the "Any dataframe →" card wraps its arrow onto a line by itself (mobile-fold, card 2).
   FIX: Wrap the arrow with the last word in a `white-space: nowrap` span, or drop the arrows on the compact phone cards.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (stat strip duplicates cards, 3-row pills on mobile): partial. The "Fast" card is gone, but the strip still repeats two cards and still takes three rows on mobile.
2 (long mobile card stack): fixed. Phone cards are now a compact 2-column grid with titles only.
3 (mobile chart tick collision, light labels): partial. The ticks are now "10x  1000x" and readable. The category labels are still mid-gray, and in dark mode on desktop they are worse.
4 (mobile code block misaligned with gutter): regressed. It now bleeds to x=0 on the left.
5 (Motifs IA fit): fixed. Motifs moved to Learn and its card was removed from Dig deeper.

WHAT WORKS (max 3, so it is not broken in the next round):
- First impression earns 8. The H1, a one-line lede that names Fastkit-Mobility and says what it does, and the speed pill tell a stranger what this is and why it matters within the first 380px, in both themes.
- Navigation earns 8. The footer pager is now section-bound (only "Next: Benchmarks", with no cross-section "Previous: Home"). The nav has three Features children, and the reader gets an active tab, a highlighted sidebar entry, a 3-item TOC and two Dig deeper cards. The new header works: white with a gold rule in light mode and black in dark mode, the logo is not clipped, and the active "Features" tab reads clearly in both themes.
- The desktop scikit-mobility vs Fastkit-Mobility log-scale chart with "(Nx faster)" labels, plus the methodology sentence linking to Benchmarks, is credible proof. The mobile speedup chart's ticks no longer collide.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: a readable "any dataframe in, same backend out" flow (pandas/Polars -> Narwhals -> Arrow buffers -> Rust kernel -> same backend), shown full-width and legible in both themes
  preferred_source: diagram
  suggested_filename: features-backend-flow.png (re-export larger, plus features-backend-flow-dark.png)
  suggested_location: docs/src/assets/images/features/
- purpose: a chart with labels that are readable in dark mode
  preferred_source: notebook-real-data
  suggested_filename: home-benchmark-speedup-dark.png
  suggested_location: docs/src/assets/images/home/
  notebook_recipe: Same 4M Brightkite pandas numbers (5 runs), the same log-scale grouped bars, with text colour #ccc and a transparent background.

CROSS-PAGE NOTES: The header now shows the wordmark twice: the outlined logo image already reads "Fastkit- Mobility", and the "Fastkit-Mobility" site title sits right beside it (desktop header, x≈110-350). Consider an icon-only logo mark next to the title, or hide the title on desktop. The dark header has no gold rule, while the light header does. That may be deliberate, but the two themes look inconsistent. The phone code-block left bleed is site-wide CSS, so check Home, Learn and Reference. The Features child pages still have no sidebar icons.
