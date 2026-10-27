PAGE: docs/src/release_notes/index.md   ROUND: 6
SCORES: first_impression=8 engagement=7 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 7.9/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P2] Not fixed from round 5. The Highlights cells still say "faster" with no figure. See row v0.2.1 ("Faster stay-location detection and GeoLife loading") and row v0.3.0 ("contact-network analysis shares one faster set of kernels") in all three screenshots. The only number on the page is "100M-row inputs". fastmob sells itself on speed, so vague speed claims in the changelog cost it credibility.
   FIX: In docs/src/release_notes/index.md, give each "faster" cell one measured figure from the matching v0.2.1.md / v0.3.0.md page or the GeoLife benchmark story, for example "stay-location detection ~Nx faster on 4M Brightkite rows". If there is no measured number, delete the word "faster" ("Reworked stay-location detection and GeoLife loading"; "contact-network analysis shares one set of kernels"). Keep each cell to one or two lines on desktop.
2. [visual] [P2] Not fixed from round 5. In mobile-full, the Version column still takes about 40% of the 390px width (the Highlights column starts at about x=180). Highlights wraps to about 3 words per line: v0.3.0 is 7 lines, and v0.2.1 and v0.1.x are 6 lines each. On a phone the table is about 1150px tall for 7 short rows. "fastmob-vis" also breaks mid-token, as "fastmob-" / "vis", in the v0.3.2 row.
   FIX: In docs/src/stylesheets/extra.css, add a rule at max-width 44.9375em scoped to this page's table (for example `.md-typeset table:not([class]) td:first-child` under a page-level class or `[data-md-page="release_notes"]`, or add an `{ .release-table }` attr_list wrapper). It should set `width: 1%; white-space: nowrap` on column 1 only and reduce the left and right cell padding to about 0.6em. Do not put nowrap on column 2.
3. [visual] [P2] Deferred, mention only: the clipped "Fastkit-Mobilit" header logo with its white box, the black light-mode header, and the footer "Previous: Development setup" link crossing sections are all unchanged.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (vague "faster" claims): not fixed. Both cells are unchanged.
2 (mobile Version column too wide): not fixed. The column split is unchanged in mobile-full.
3 (nav label casing): fixed. The sidebar now reads "Release notes" and matches the H1 and the sentence-case siblings (desktop light and dark).
4 (deferred header/logo/footer): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Latest release: v0.3.2 (2026-09-19)" tip, with the upgrade command, a no-breaking-changes statement and pinning advice, matches the footer "fastmob 0.3.2". A returning user gets everything they need in one glance.
- One consistent Version/Highlights table: muted dates under each version, inline doc links, and the 0.1.x releases folded into one row. Contrast is good in both themes.
- The short Upgrade policy ends with a link to issues and GitHub releases, so the page has an exit and stays brief.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures)

CROSS-PAGE NOTES:
- Nav casing is now consistent under About. Check that the other tabs' leaf labels also use sentence case.
- Carried over: add "add a dated row to docs/src/release_notes/index.md and a vX.Y.Z.md page" to the release checklist in CLAUDE.md so the table stays complete.
- Any narrow-viewport fix for two-column tables, a nowrap first column, could be a reusable class shared with the Reference modules table and the Resources glossary.
