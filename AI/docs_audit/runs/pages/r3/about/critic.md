PAGE: docs/src/about/index.md   ROUND: 3
SCORES: first_impression=7 engagement=6 navigation=7 scannability=8 visual=6 ia_role=8 conciseness=8 trust=7
OVERALL: 7.1/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile (mobile-full, y≈775-805, "License and credits"): the floating round TOC button (≡ icon, right edge) still sits on top of body text. It covers the end of "are credited in" on the line just above "THIRD_PARTY_NOTICES.md". The change log says the floating back-to-top button was removed, but this overlap comes from the TOC toggle, a different button, and it is still there. This is the third round with this defect.
   FIX: In docs/src/stylesheets/extra.css, at <=76.1875em, either hide the floating TOC toggle (the header already provides the drawer and search) or give it an opaque background with a shadow and add about 3rem of `padding-right` to `.md-content__inner` so text never runs under it. Check the result on mobile-full at the "License and credits" paragraph.
2. [visual] [P2] Mobile (mobile-full, y≈905-1040): the BibTeX block runs off the right edge. "Gustavo Henrique Bruno" and "human-mobility analy..." are cut off with no visible cue that the block scrolls sideways.
   FIX: In docs/src/about/index.md, break the author and title values onto aligned continuation lines so each line stays under about 42 characters. Or shorten the title field to "fastmob" and move the tagline into a `note` field on its own line.
3. [navigation] [P2] Footer pager (desktop y≈1405-1435, mobile y≈1580-1600): "Previous: Visualization" jumps back into the Reference tab from the first page of About. The pager crosses a tab boundary, so the reader leaves the section they chose.
   FIX: Site-wide, in zensical.toml or extra.css: hide `.md-footer__link--prev` on the first page of each top-level tab (for example, add front matter `hide: [footer]` and render only Next with a small override, or a CSS rule scoped by a page class). Keep "Next: Resources".
4. [trust] [P2] Cite fastmob (desktop y≈725-870): `version` is now present, but there is still no CITATION.cff or DOI (known, deferred). Research users get no "Cite this repository" button on GitHub.
   FIX: Add a CITATION.cff at the repo root that mirrors the BibTeX block, and add one line under the block: "Also available as CITATION.cff". Mint a Zenodo DOI on the next tagged release.
5. [engagement] [P2] The badge row (y≈315-335) is the only visual anchor. If shields.io fails to load, the version, Python range and license disappear from the top of the page.
   FIX: Under the badges, add one plain-text line or a 4-cell table: Version 0.3.2 · BSD-3-Clause · Python 3.9-3.13 · pandas / polars via Narwhals. Do not add imagery.
6. [scannability] [P2] Bug-report checklist (desktop y≈1190-1340): five one-line bullets with loose spacing take about 150px, the tallest block on the page.
   FIX: Tighten the list's vertical spacing with a `.compact` class or `li { margin: .2em 0 }` scoped to this list in extra.css, or merge it into the "Source code and issue tracker" bullet as a nested list.
7. [visual] [P2] Header (all screenshots): the white logo box in dark mode, the "Fastkit-Mobility" logo text and the black header in light mode remain. These are known and deferred.
   FIX: Deferred (brand asset). No action this round.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile TOC button overlapping text): not fixed. It still overlaps "credited in" at y≈790 on mobile. Removing the back-to-top button did not address this toggle.
2 (About missing from its own sidebar, no active marker): fixed. The sidebar now reads About fastmob (highlighted, active), Resources, Development setup, Release Notes, in both themes.
3 (mobile footer bare "Previous" arrow): fixed. The mobile footer now shows "Previous / Visualization" and "Next / Resources" with labels.
4 (BibTeX without version, no CITATION.cff): partial. `version = {0.3.2}` was added. CITATION.cff and DOI are still missing (deferred).
5 (BibTeX clipped on mobile): not fixed. It is still cut off at the right edge.
6 (dense bug-report sentence): fixed. It is now a 5-item checklist.
7 (no at-a-glance facts beyond the badges): not fixed. This was optional.

WHAT WORKS (max 3, so it is not broken in the next round):
- The sidebar now gives a clear "you are here" marker, and the section order (About, Resources, Development setup, Release Notes) is logical.
- The page stays strictly meta: identity, relationship to scikit-mobility, license and credits, citation, maintainer, contributing. It fits in about 1.1 desktop screens, and the right-hand TOC maps 1:1 to its H2s.
- Light and dark themes both have legible link colours (dark gold and yellow) and a readable code block. The bug-report checklist is now scannable.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none. This meta page needs facts, not illustrations.

CROSS-PAGE NOTES:
- The mobile floating TOC toggle overlapping body text is theme-level and has been unresolved since round 1. Fix it once in extra.css. Note that it is the TOC toggle, not the back-to-top button.
- The footer prev/next chain crosses tab boundaries (About → Previous = Reference > Visualization). Decide site-wide whether to suppress "Previous" on each tab's first page.
- Long code blocks clip on mobile without a scroll cue. Consider a right-edge fade or a visible scrollbar on `pre` at narrow widths.
- The header logo text, the white box in dark mode and the black light-mode header remain deferred (brand, P2).
