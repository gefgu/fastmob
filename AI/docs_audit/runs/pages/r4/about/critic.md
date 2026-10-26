PAGE: docs/src/about/index.md   ROUND: 4
SCORES: first_impression=7 engagement=7 navigation=7 scannability=8 visual=7 ia_role=8 conciseness=8 trust=7
OVERALL: 7.4/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile (mobile-fold y≈775-810, mobile-full y≈775-810, "License and credits"): the round floating TOC toggle (≡ icon, right edge) is STILL rendered and sits on top of the "BSD 3-Clause" link text. The r3 change log says it was hidden, and extra.css:458-461 has `.md-sidebar-button { display: none; }`, but the screenshots show the rule has no effect. Either the selector does not match Zensical's element or a theme rule with higher specificity wins. This is the fourth round with this defect.
   FIX: Open the built page (site/about/index.html, or devtools at 390px) and copy the real class of the floating button (likely a `.md-top`/`.md-toc__toggle`/`[data-md-component=...]`-style element, not `.md-sidebar-button`). Target that selector with `display: none !important` inside `@media screen and (max-width: 76.1875em)` in docs/src/stylesheets/extra.css. Re-screenshot mobile-fold and confirm the "BSD 3-Clause license" line is clear. Apply the same check site-wide.
2. [trust] [P2] Cite fastmob (desktop y≈735-965): the BibTeX now has `version`, but there is still no CITATION.cff or DOI (known, deferred). GitHub shows no "Cite this repository" button.
   FIX: Add CITATION.cff at the repo root mirroring the BibTeX, plus one line under the block: "Also available as CITATION.cff". Mint a Zenodo DOI on the next tagged release.
3. [scannability] [P2] Bug-report checklist (desktop y≈1280-1440, mobile y≈1475-1635): five one-line bullets with loose spacing remain the tallest block on the page, longer than all the content above "Maintainer" except the BibTeX.
   FIX: Scope a compact list style (`li { margin: .15em 0 }`) to this list in extra.css with an attr_list class, or collapse the five items into one sentence under "Source code and issue tracker".
4. [navigation] [P2] Footer pager (desktop y≈1500-1530, mobile y≈1700-1725): "Previous: Visualization" leaves the About tab for Reference. This is known and deferred.
   FIX: Deferred (site-wide). When it is addressed, suppress `.md-footer__link--prev` on each tab's first page.
5. [visual] [P2] Header (all screenshots): the white logo box with the "Fastkit-Mobility" text and the black light-mode header remain. These are known and deferred.
   FIX: Deferred (brand asset).

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile floating TOC toggle overlapping text): not fixed. The CSS rule was added, but the button is still visible over "BSD 3-Clause license" at y≈790 on mobile. The selector apparently does not match.
2 (BibTeX clipped on mobile): fixed. Author and note values wrap onto aligned continuation lines, and the longest line (url) now fits within 390px.
3 (footer Previous crossing tabs): not fixed (deferred).
4 (CITATION.cff / DOI): not fixed (deferred). The version field is present.
5 (at-a-glance facts beyond badges): fixed. "Version 0.3.2 · BSD-3-Clause · Python 3.9 or later · pandas and Polars via Narwhals" is shown under the badges, and it survives badge-load failure.
6 (loose bug-report checklist spacing): not fixed.
7 (header logo/black header): deferred.

WHAT WORKS (max 3, so it is not broken in the next round):
- The plain-text facts line under the badges gives version, license, Python range and backends at a glance, independent of shields.io.
- The BibTeX block is now readable on phones with no horizontal clipping, and it keeps aligned fields on desktop.
- The page is strictly meta. The sidebar "About fastmob" active state, the logical section order and the right TOC map 1:1 to the H2s. Both themes have legible link colours.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none. This meta page needs facts, not illustrations.

CROSS-PAGE NOTES:
- The `.md-sidebar-button { display: none; }` rule in extra.css is ineffective. The floating mobile TOC toggle still renders and overlaps body text, which likely affects every page with a TOC. Verify the actual selector in the built HTML and re-check all mobile screenshots.
- Footer prev/next crossing tab boundaries, the header logo box and the black light-mode header remain deferred.
