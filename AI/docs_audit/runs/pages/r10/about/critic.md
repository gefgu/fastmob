PAGE: docs/src/about/index.md   ROUND: 10
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

Score justifications (>= 8):
- first_impression=8: Above the fold on desktop (y≈150-390) and mobile (y≈100-445), a reader sees the H1, a three-line definition, four badges and a trimmed facts line: "Released 2026-09-19 (release notes) · Alpha · pandas and Polars via Narwhals". That is enough to tell what the library is, how mature it is and when it last shipped.
- engagement=8: For a meta page, the single linked "up to 2658x faster on 4M Brightkite rows (see benchmarks)" claim (desktop y≈535) is the right amount of proof. The new peer-reviewed paper citation (NetMob 2026, HAL link) is concrete evidence rather than filler.
- navigation=8: The sidebar active pill, a right TOC that maps 1:1 to the five H2s, inline links to Features, release notes, benchmarks and Development setup, and a footer pager kept inside the About section ("Next: Resources" only, no Previous crossing into Reference) all give clear next steps.
- scannability=8: There are five short H2 sections, each one or two blocks long. The bug-report guidance is one sentence and the eye lands on Cite, License or Maintainer at a glance.
- visual=8: Contrast is good in both themes. The new light header (white with a gold rule under the tabs at y≈97) is clean and calmer than the old black bar. The dark header is a seamless black, the logo renders in both themes, and links are legible olive/yellow. The software BibTeX is now hand-wrapped and fits on mobile. One mobile clipping regression remains (issue 1).
- ia_role=8: The page is strictly meta, in a predictable order: relationship, license, cite, maintainer, contribute. No guide or reference content leaks in.
- conciseness=8: The badge/facts duplication flagged in r6 is gone. The page fits in about 2 desktop screens, and the two BibTeX blocks are its only bulk, which is justified.
- trust=8: The page has a release date, an honest Alpha label, a sourced benchmark claim, third-party credits (MoveTK), a named maintainer and now a peer-reviewed paper with a HAL identifier. CITATION.cff and a DOI are still absent.

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] Mobile, Cite section (mobile-full y≈1090-1350): the new @inproceedings block overflows the 375px viewport. "title = {Fastkit-Mobility: A High-Perfor", "address = {Niter{\'o}i, Rio de Janeiro, Br" and "url = {https://inria.hal.science/hal-0" are cut off at the right edge, so the primary citation is unreadable without scrolling sideways inside the block. The @software block below was hand-wrapped and fits, so the two blocks are now inconsistent.
   FIX: In docs/src/about/index.md, wrap the @inproceedings lines the way @software is wrapped. Break the title after "A High-" or put each phrase on its own continuation line, break the address after "Rio de", and split the url as `{https://inria.hal.science/` / `hal-05755990}`. Keep each line at or under about 40 characters.
2. [trust] [P2] Cite section: the two BibTeX entries spell the author differently. The paper uses "Santos, Gustavo H." and the software uses "dos Santos, Gustavo Henrique Bruno", so reference managers will treat them as two people.
   FIX: Use one form in both entries, for example `{dos Santos, Gustavo H.}` or `{Santos, Gustavo H.}`, matching the HAL record.
3. [trust] [P2] There is still no CITATION.cff or DOI for the software release (deferred).
   FIX: Deferred. When it exists, add "Also available as CITATION.cff (GitHub 'Cite this repository')" under the @software block.
4. [visual] [P2] Header (desktop, both themes, x≈115-350): the outlined logo mark already contains the stacked words "Fastkit- Mobility", and the site title "Fastkit-Mobility" sits directly beside it, so the brand name is printed twice in about 230px.
   FIX: Site-wide. Either use the icon-only mark (the F with speed lines) in `logo_header.svg` and keep the text title, or keep the wordmark logo and hide the site title on desktop via extra.css (`.md-header__topic:first-child { display:none }` or the Zensical equivalent).

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile @software BibTeX url flush with the viewport edge): fixed. The url is now wrapped as `{https://github.com/` / `gefgu/fastmob}` and fits. The new @inproceedings block reintroduces the same problem (new issue 1).
2 (badge row and facts line duplicate version, license and Python): fixed. The facts line now reads only release date, Alpha and backends.
3 (CITATION.cff / DOI): partial. A paper citation with a HAL id was added; there is still no CITATION.cff or software DOI.
4 (footer Previous crossing into the Reference tab): fixed. The pager shows only "Next: Resources" and stays inside the About section on desktop and mobile.
5 (white logo box / black light-mode header): fixed. The light header is now white with a gold rule, the logo has no box, and the dark header is black. A minor brand-duplication note remains (new issue 4).
No regressions apart from the new block's mobile overflow.

WHAT WORKS (max 3, so it is not broken in the next round):
- The trimmed facts line under the badges (release date, Alpha, backends) adds information without repeating the badges.
- The new white-and-gold light header and the section-bound pager: calm, on-brand, and the pager never jumps tabs.
- Paper citation first, software citation second, each with a single line of intro: exactly what a researcher needs.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none. This meta page needs facts and links, not illustrations.

CROSS-PAGE NOTES:
- The speed claim changed from "up to 1329x" (r6) to "up to 2658x". Check that Home, Features and Benchmarks all quote the same number and dataset. A single source (a snippet or macro) would prevent drift.
- Header: the logo wordmark and the site title duplicate the brand name (issue 4). On mobile (390px) the header shows only the title, with no logo mark and no visible gold rule, which differs from desktop. Decide whether the gold rule should also appear on mobile for consistency.
- Any long BibTeX or one-line code block across the site should be checked at 375px. Code blocks scroll horizontally rather than wrap, so citation-style blocks need hand wrapping.
- Deferred: CITATION.cff / DOI.
