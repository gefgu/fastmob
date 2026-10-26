PAGE: docs/src/about/index.md   ROUND: 2
SCORES: first_impression=7 engagement=6 navigation=6 scannability=7 visual=6 ia_role=8 conciseness=8 trust=7
OVERALL: 6.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] Mobile (mobile-full, y≈770-800): the floating round TOC button still sits on top of body text in "License and credits". The words "credited in" are covered at the end of the line just above "THIRD_PARTY_NOTICES.md". This is the same defect reported in round 1, now on a different paragraph.
   FIX: In docs/src/stylesheets/extra.css, at <=76.1875em, give the floating TOC button an opaque background (var(--md-default-bg-color)) with a box-shadow, and add right padding to `.md-content__inner` (about 3rem) so text never runs under it. Alternatively, hide the button on pages with 5 or fewer TOC entries.
2. [navigation] [P1] Left sidebar (desktop, x≈123-320, y≈165-237): it lists Resources, Development setup and Release Notes, but not "About fastmob", the page being read. With no active item, the reader has no "you are here" marker inside the section. The footer "Previous: Visualization" (y≈1250) also jumps back into the Reference section, which makes About feel bolted on.
   FIX: Keep `navigation.indexes`, but show the index page as the first sidebar item. One way is a CSS rule in extra.css that renders the section index link as an active top entry. Another is to remove the index binding and list `{ "About fastmob" = "about/index.md" }` as an ordinary first child. Either way, the About sidebar should read: About fastmob (active), Resources, Development setup, Release Notes.
3. [visual] [P1] Mobile footer (mobile-full, y≈1477): "Previous" is still a bare left arrow with no label, while Next is labelled "Resources". This was not fixed from round 1.
   FIX: This is site-wide, in extra.css. Stop hiding `.md-footer__title` for the previous link at narrow widths, or hide `.md-footer__link--prev` entirely on mobile so no orphan arrow remains.
4. [trust] [P2] Cite fastmob (desktop y≈725-850): the BibTeX entry has no `version` field, no DOI, and there is no CITATION.cff in the repo root. For research users, a citation without a version is weak, and GitHub's "Cite this repository" button will not appear.
   FIX: Add `version = {0.3.2}` to the BibTeX block in docs/src/about/index.md, or a note to replace it with the version used. Add a CITATION.cff at the repo root and link it under the block ("Also available as CITATION.cff"). Mint a Zenodo DOI on the next release if feasible.
5. [visual] [P2] On mobile (y≈895-1020), the BibTeX block runs off the right edge ("Gustavo Henrique Bruno" and "human-mobility analy..." are cut off). The block scrolls sideways, but a reader copying by eye cannot tell.
   FIX: Wrap the long values onto aligned continuation lines, or keep each line under about 40 characters (e.g. shorten the title field). Readers can also rely on the copy button, which is already present.
6. [scannability] [P2] "Contribute and report issues" (y≈1013-1160) ends with a dense one-sentence list of six bug-report requirements. This is the one block on the page that the eye cannot scan.
   FIX: Turn the sentence into a compact 5-item checklist (versions, backend, minimal input, expected vs actual, traceback). Or move it into a GitHub issue template and replace the sentence with "Open an issue (the template lists what to include)".
7. [engagement] [P2] Apart from the badge row and the code block, the page is plain headings and prose. That is acceptable for a meta page, but the badge row is the only visual anchor and it is mid-sized and grey-heavy.
   FIX: Do not add imagery. Optionally add a one-line "At a glance" definition list or table directly under the badges: Version 0.3.2 · License BSD-3 · Python 3.9-3.13 · Backends pandas/polars. Then the key facts survive even if shields.io fails to load.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (missing license/maintainer/cite/skmob relationship): fixed. All four sections are present, and the links resolve to LICENSE and fastmob-core/THIRD_PARTY_NOTICES.md. The version is now visible via the badge.
2 ("What it supports" marketing duplication): fixed. The section is removed and replaced by a single "See Features" pointer.
3 (one-item sidebar, no orientation): partial. The sidebar now holds Resources, Development setup and Release Notes, but this page itself is still missing from it, so there is no active marker (issue 2 above).
4 (text only, no visual anchor): fixed. A badge row sits under the intro (PyPI, Python, license, stars).
5 (mobile TOC button overlap and bare Previous arrow): not fixed. Both remain (issues 1 and 3 above).
6 (contribute section mixing links and a dense sentence; duplicate release link): partial. The duplicate link and the vague label are gone, but the dense bug-report sentence remains.
7 (link colour mismatch between themes; logo mismatch): partial. Light links are now a readable dark gold and dark links yellow, which is consistent and AA-legible. The logo text mismatch is deferred (brand).

WHAT WORKS (max 3, so it is not broken in the next round):
- The page now does exactly the meta job: identity, relationship to scikit-mobility, license and credits, citation, maintainer, contributing. The whole page fits in about 1.1 desktop screens with no marketing filler.
- The intro paragraph plus badge row gives a cold reader what/version/license in the first 330px, in both themes.
- The right-hand "On this page" TOC maps one-to-one to short H2 sections. Light and dark contrast is good, and the code block is readable in both themes.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none. This meta page needs facts, not illustrations.

CROSS-PAGE NOTES:
- Mobile floating TOC button overlapping body text, and the bare "Previous" arrow in the mobile footer, are theme-level and unresolved since round 1. Fix both once in extra.css.
- Sections using `navigation.indexes` hide their index page from the sidebar, so on every section overview page (About, Learn, Reference, Features) the reader has no active sidebar item. Decide once, site-wide, whether index pages appear as an explicit first sidebar entry.
- The footer prev/next chain crosses tab boundaries (About "Previous" = Reference > Visualization). Consider hiding prev at a section's first page.
- Header logo text ("Fastkit-Mobility") and the white box in dark mode remain deferred (brand, P2).
