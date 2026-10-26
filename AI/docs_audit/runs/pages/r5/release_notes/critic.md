PAGE: docs/src/release_notes/index.md   ROUND: 5
SCORES: first_impression=8 engagement=7 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 7.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P2] The Highlights column says "faster" twice without saying by how much. Rows v0.2.1 ("Faster stay-location detection and GeoLife loading") and v0.3.0 ("contact-network analysis shares one faster set of kernels") give no figure. The only number on the page is "100M-row inputs". On a library whose whole pitch is speed, the changelog is where a returning reader expects to see the speedups.
   FIX: In docs/src/release_notes/index.md, add one short measured figure to each "faster" cell, taken from the matching vX.Y.Z.md page or the benchmark story (for example "stay-location detection ~Nx faster on 4M Brightkite rows"). If no measured number exists for a row, remove the word "faster" instead of leaving a vague claim. Keep each cell to one line on desktop.
2. [visual] [P2] Mobile (mobile-full, every table row): the Version column takes about 40% of the 390px width. That column only holds "v0.3.2" plus a date, so Highlights is squeezed to about 4 words per line. v0.3.0 wraps to 7 lines and v0.2.1 to 6. The table is about 1150px tall on a phone for 7 rows.
   FIX: In docs/src/stylesheets/extra.css, scope a rule to the release-notes table at <=44.9375em that sets the first column to `width: 1%; white-space: nowrap` (the version cell only, never column 2). Alternatively, reduce its cell padding so Highlights gets about 60-65% of the width.
3. [navigation] [P2] The sidebar label is "Release Notes" (title case) but the H1 is "Release notes" (sentence case). The sibling entries "About fastmob", "Development setup" and "Resources" all use sentence case.
   FIX: In zensical.toml `nav`, rename the entry to `{ "Release notes" = "release_notes/index.md" }`.
4. [visual] [P2] Deferred, noted only: the header logo is clipped ("Fastkit-Mobilit") with a white box, the light-mode header is black, and the footer "Previous: Development setup" link crosses sections.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (dead end, no forward link): fixed. The closing line "Upgrade trouble? Open an issue or see all GitHub releases." appears under Upgrade policy in all three screenshots.
2 (single-child Release Notes nav group): fixed. Release Notes is now a leaf, highlighted in the About sidebar with no expander (desktop light and dark).
3 (mobile floating TOC toggle over the table): fixed. No floating circle appears anywhere in mobile-full.
4 (0.1.x admonition with mismatched styling): fixed. It is now the last table row, "v0.1.0 to v0.1.3", with a date range and inline links to v0.1.1-v0.1.3.
5 (deferred header/logo/footer): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Latest release: v0.3.2 (2026-09-19)" tip, with the upgrade command, a breaking-change statement and pinning advice, answers the three questions a returning user has in one glance. The footer "fastmob 0.3.2" matches it.
- The single, consistent Version/Highlights table: muted dates under each version, wrapping highlights with inline doc links, and 0.1.x folded into one row. Do not reintroduce nowrap on column 2, and do not add admonitions back between rows.
- The Upgrade policy has four honest bullets and a closing line linking to issues and GitHub releases. The page now has an exit and stays short.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures)

CROSS-PAGE NOTES:
- The hidden `.md-sidebar--secondary` fix works on this page at 390px. Keep checking it on the other pages.
- Nav label casing: "Release Notes" is the only title-case leaf under About. Pick sentence case site-wide.
- Carried over: add "add a dated row to docs/src/release_notes/index.md and a vX.Y.Z.md page" to the release checklist in CLAUDE.md so the table stays complete.
