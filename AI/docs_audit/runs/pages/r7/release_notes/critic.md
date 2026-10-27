PAGE: docs/src/release_notes/index.md   ROUND: 7
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] Not fixed, although the r6 change log says "narrow date column on phones". In mobile-full the Version column still takes about 45% of the 390px width (Highlights starts at about x=188), so Highlights wraps to 2-4 words per line. The v0.3.0 row is 7 lines and v0.2.1 is 6 lines, which makes the table about 1150px tall on a phone. "fastmob-vis" in the v0.3.2 row still breaks mid-token as "fastmob-" / "vis". The dates already fit (the longest is "2026-07-16 to 07-21"), so the column width comes from the cell padding or table layout, not from the content.
   FIX: In docs/src/stylesheets/extra.css, at max-width 44.9375em, give this table's first column `width: 1%; white-space: nowrap` and cut cell padding to about 0.6em horizontally. Scope it with an attr_list class on a wrapper or a page-level selector, and check that the selector actually matches in the built HTML, since the r6 rule visibly did not apply. Add `white-space: nowrap` to inline `code` inside td so "fastmob-vis" stays whole. Do not put nowrap on column 2.
2. [ia_role] [P2] Naming: the About sidebar on this page (desktop light and dark, top of the left nav) still reads "About fastmob", but the target page's H1 is "About Fastkit-Mobility". Prose labels should say Fastkit-Mobility.
   FIX: In zensical.toml line 67, change `"About fastmob"` to `"About Fastkit-Mobility"` (or just `"About"`).
3. [visual] [P2] Deferred, mention only: the clipped "Fastkit-Mobilit" header logo with its white box, the black light-mode header, and the footer "Previous: Development setup" link crossing sections are all unchanged.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (vague "faster" claims): fixed. v0.2.1 now says "Optimized stay-location detection..." and v0.3.0 says "contact-network analysis shares one set of kernels". The page no longer makes a speed claim without a number.
2 (mobile Version column too wide): not fixed. The column split and the "fastmob-"/"vis" break are unchanged in mobile-full.
3 (nav label casing): still fixed. The label reads "Release notes".
4 (deferred header/logo/footer): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Latest release: v0.3.2 (2026-09-19)" tip gives the upgrade command, says there are no breaking changes and advises pinning. It matches the footer "Fastkit-Mobility 0.3.2", so a returning user gets the answer at a glance.
- The Version/Highlights table is consistent: muted dates, one line per release on desktop, inline links into the reference pages, and the 0.1.x releases folded into one row. Contrast is good in both themes.
- The Upgrade policy is short, states the pre-1.0 contract honestly and ends with issue/release links, so the page has a clear exit.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures)

CROSS-PAGE NOTES:
- The "About fastmob" nav label (zensical.toml) violates the naming rule on every About-section page.
- The narrow-viewport two-column table fix (nowrap first column, nowrap inline code) should be a reusable class shared with the Resources glossary and any other two-column tables.
- Carried over: add "add a dated row to docs/src/release_notes/index.md and a vX.Y.Z.md page" to the release checklist in CLAUDE.md.
