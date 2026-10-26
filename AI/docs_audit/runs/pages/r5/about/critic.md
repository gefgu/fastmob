PAGE: docs/src/about/index.md   ROUND: 5
SCORES: first_impression=7 engagement=7 navigation=7 scannability=8 visual=8 ia_role=8 conciseness=8 trust=7
OVERALL: 7.5/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [trust] [P1] Hero area (desktop y≈220-385, mobile y≈170-450): nothing above the fold says the project is maintained or how mature it is. There is no "first released / latest release date", no link to Release Notes, and no status word such as alpha or beta. The Stars badge reads "6", and that is the only activity signal a cold visitor gets. A 60-second evaluator cannot tell whether 0.3.2 is a recent release or an abandoned one.
   FIX: Extend the facts line in docs/src/about/index.md to "Version 0.3.2 (released <Mon YYYY>, see [Release Notes](../release-notes/...)) · Beta · BSD-3-Clause · Python 3.9 or later · pandas and Polars via Narwhals". Alternatively, drop the Stars badge and keep only PyPI, Python and License.
2. [engagement] [P2] Whole page: the page is all prose, link lists and one code block, so nothing makes the scikit-mobility relationship concrete. The "Relationship to scikit-mobility" paragraph (desktop y≈470-545) says comparison tests exist but shows nothing.
   FIX: Add one sentence with a link to the Home or Features benchmark, for example "Up to N× faster than scikit-mobility on 4M Brightkite rows ([benchmarks](../features/index.md#...))". Link to existing evidence rather than adding a figure here.
3. [visual] [P2] Mobile BibTeX block (mobile-full y≈965-1160): the code block runs full-bleed, and the url line's closing "}" sits at x≈373 of 375, flush against the viewport edge with no right padding. It is not clipped, but it looks cramped next to the 16px gutter of the body text.
   FIX: In docs/src/stylesheets/extra.css, under the mobile breakpoint, give `.md-typeset pre > code` at least 16px of right padding. Or shorten the url value to `{github.com/gefgu/fastmob}` with a `howpublished` field. The first option is preferred, and it is site-wide.
4. [scannability] [P2] Bug-report checklist (desktop y≈1275-1420, mobile y≈1475-1615): the change log says "tighter list spacing", but the five items still sit at about 31px pitch, the same as r4. The block remains the tallest after the BibTeX.
   FIX: Merge the five bullets into one line under "Contribute and report issues": "Include fastmob/Python versions, dataframe backend and version, a minimal input, expected vs actual output, and the full traceback." Alternatively, apply `{ .compact }` with `li { margin: .1em 0 }` in extra.css.
5. [trust] [P2] Cite fastmob: there is still no CITATION.cff or DOI (known, deferred).
   FIX: Deferred. When done, add "Also available as CITATION.cff" under the block.
6. [navigation] [P2] Footer pager (desktop y≈1480-1510, mobile y≈1680-1700): "Previous: Visualization" crosses into the Reference tab (known, deferred).
   FIX: Deferred (site-wide). Hide `.md-footer__link--prev` on each tab's first page.
7. [visual] [P2] Header: the white logo box and the black light-mode header remain (known, deferred).
   FIX: Deferred (brand asset).

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile floating TOC toggle overlapping "BSD 3-Clause license"): fixed. No floating button appears in mobile-fold or mobile-full, and the License paragraph is clear.
2 (CITATION.cff / DOI): not fixed (deferred).
3 (loose bug-report checklist spacing): not fixed. The spacing looks unchanged from r4 in both desktop and mobile.
4 (footer Previous crossing tabs): not fixed (deferred).
5 (header logo/black header): not fixed (deferred).

WHAT WORKS (max 3, so it is not broken in the next round):
- The page is clean on mobile now that the floating TOC button is gone. The badges wrap onto two tidy rows, and the facts line survives badge-load failure.
- The BibTeX block keeps aligned fields and wrapped continuation lines in both themes and on a 390px phone.
- The page is strictly meta. The sidebar active state, the right TOC mapping 1:1 to the H2s, and the logical order (relationship, license, cite, maintainer, contribute) make it predictable. Link colours are legible in light and dark.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none. This meta page needs facts and links, not illustrations.

CROSS-PAGE NOTES:
- The `.md-sidebar--secondary` fix resolved the mobile overlap here. Confirm the same on every other page's mobile screenshots.
- Code blocks on mobile are full-bleed with no right padding, so long lines touch the viewport edge. A site-wide `pre > code` padding rule in extra.css would fix this everywhere.
- The "tighter list spacing" change from r4 is not visible on this page. Check whether the rule is scoped to a class that is missing from About's lists.
- Deferred: footer prev/next crossing tabs, header logo box, black light-mode header, CITATION.cff/DOI.
