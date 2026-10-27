PAGE: docs/src/about/index.md   ROUND: 6
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

Score justifications (>= 8):
- first_impression=8: The facts line sits above the fold on both desktop (y≈370-400) and mobile (y≈400-460). It reads "Version 0.3.2 (released 2026-09-19, see release notes) · Alpha · BSD-3-Clause · Python 3.9 or later · pandas and Polars via Narwhals", so in 5 seconds a reader learns what the library is, its maturity, and when it last shipped.
- engagement=8: For a meta page, the concrete "up to 1329x faster on 4M Brightkite rows (see benchmarks)" claim in "Relationship to scikit-mobility" (desktop y≈560) turns the relationship into evidence. It links to real data instead of padding the page with a figure. The claim matches the Home/Features chart.
- navigation=8: The sidebar active state, a right TOC that maps 1:1 to the five H2s, inline links to Features, release notes, benchmarks and Development setup, and the Next: Resources pager give clear next steps. The only defect is the deferred Previous-crosses-tab item.
- scannability=8: The five-bullet checklist is now one sentence (desktop y≈1295-1330, mobile y≈1520-1600). Every section is one short block under an H2, and the eye finds Cite, License or Maintainer at a glance.
- visual=8: Contrast is good in light and dark, links are legible yellow or olive, the badges wrap onto two tidy rows on mobile, and the BibTeX block is aligned in both themes. The only remaining issue is a mobile edge-touch (issue 1).
- ia_role=8: The page is strictly meta, in a logical order: relationship, license, cite, maintainer, contribute. Nothing from the guide or reference leaks in.
- conciseness=8: The whole page fits about 1.3 desktop screens and every block earns its place. The only redundancy is minor (issue 2).
- trust=8: The page now carries a release date, an honest "Alpha" status, a release-notes link, a sourced benchmark claim, third-party credits and a named maintainer. CITATION.cff and a DOI are still missing, but that is deferred and not scored down.

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] Mobile BibTeX block (mobile-full y≈1045-1240): the url line's closing "}" still ends at x≈373 of 375, flush with the viewport edge. The r5 change log says "code blocks keep right padding on phones", but on this page the long url line still overflows to the edge, either scrolling or with no visible padding.
   FIX: Shorten the url line in docs/src/about/index.md so it fits within the 16px gutter. For example, wrap it as `url     = {https://github.com/` / `             gefgu/fastmob}`, the same continuation style already used for author and note. Or confirm that the site-wide `pre > code { padding-right }` rule in extra.css actually applies to `.language-bibtex` blocks.
2. [conciseness] [P2] Hero (desktop y≈315-400): the badge row and the facts line repeat each other. Version, license and Python range each appear twice.
   FIX: Trim the facts line to the information the badges lack: "Released 2026-09-19 ([release notes](...)) · Alpha · pandas and Polars via Narwhals". Keep the badges for version, Python and license.
3. [trust] [P2] Cite fastmob: there is still no CITATION.cff or DOI (known, deferred).
   FIX: Deferred. When done, add "Also available as CITATION.cff" under the block.
4. [navigation] [P2] Footer pager: "Previous: Visualization" crosses into the Reference tab (known, deferred).
   FIX: Deferred (site-wide).
5. [visual] [P2] Header: the white logo box and the black light-mode header remain (known, deferred).
   FIX: Deferred (brand asset).

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (no maturity or release-date signal above the fold): fixed. The facts line now has the release date, a release-notes link and Alpha status.
2 (scikit-mobility relationship not made concrete): fixed. The "up to 1329x faster on 4M Brightkite rows (see benchmarks)" sentence was added.
3 (mobile BibTeX url line flush against the viewport edge): not fixed. The closing "}" still touches the right edge in mobile-full.
4 (loose bug-report checklist): fixed. It is now one sentence.
5 (CITATION.cff / DOI): not fixed (deferred).
6 (footer Previous crossing tabs): not fixed (deferred). On mobile the pager now stacks cleanly.
7 (header logo/black header): not fixed (deferred).
No regressions observed.

WHAT WORKS (max 3, so it is not broken in the next round):
- The facts line (release date, Alpha, release-notes link) gives instant maturity context and survives badge-load failure.
- The single linked benchmark claim in "Relationship to scikit-mobility" provides evidence without bloating a meta page.
- The page is short, five-H2 and strictly meta, with a 1:1 right TOC and clean light, dark and mobile rendering.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none. This meta page needs facts and links, not illustrations.

CROSS-PAGE NOTES:
- Check that the r5 mobile code-block right-padding rule applies to every code language, including bibtex. On About, the long url line still reaches the viewport edge.
- Keep the "released <date>" text in the facts line in sync with release_notes/vX.Y.Z.md and the footer version on each release. The release checklist in CLAUDE.md could mention it.
- Deferred: footer prev/next crossing tabs, header logo box, black light-mode header, CITATION.cff/DOI.
