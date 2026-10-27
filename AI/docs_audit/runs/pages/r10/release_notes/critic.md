PAGE: docs/src/release_notes/index.md   ROUND: 10
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] Still not fixed after r7. In mobile-full the Version column takes about 45% of the 390px width (Highlights starts at about x=178 of 368 displayed), so Highlights wraps to 2-4 words per line. The v0.3.0 row runs to 7 lines and v0.2.1 to 6, which makes the table about 1150px tall on a phone. "fastmob-vis" in the v0.3.2 row still breaks as "fastmob-" / "vis", and "stay-location" breaks at its hyphen too.
   FIX: In docs/src/stylesheets/extra.css, at max-width 44.9375em, give the first column of this table `width: 1%; white-space: nowrap` and cut the cell padding to about 0.6em horizontally. Scope it with a wrapper class through attr_list or md_in_html (for example `<div class="release-table" markdown>`), then check in the built HTML that the selector matches. Add `td code { white-space: nowrap; }` so inline code tokens stay whole.
2. [visual] [P2] Header, desktop light and dark: the new outlined logo already reads "Fastkit- / Mobility", and the bold site title "Fastkit-Mobility" sits right next to it, so the name appears twice within 100px. The stacked logo text is also small (about 11px) next to the 17px title.
   FIX: Pick one. Either hide the site-title text on desktop when the wordmark logo is shown (CSS on `.md-header__title` at min-width 76.25em), or switch the logo to the icon-only "F" mark and keep the title. Mobile already shows only the title, so it needs no change.
3. [visual] [P2] Header tabs, desktop light: the active "About" underline is wider than the label (about x=405-469 against text at x=417-457), so it looks offset compared with the gold rule beneath it.
   FIX: In extra.css, make the active-tab indicator match the text width (put the border/`::after` on the link's inline content, or set the left/right inset equal to the tab padding).

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (mobile Version column too wide, "fastmob-"/"vis" break): not fixed. Unchanged in mobile-full.
2 ("About fastmob" sidebar label): fixed. The sidebar reads "About Fastkit-Mobility" in desktop light and dark.
3 (deferred header/logo/footer): fixed, apart from the duplication in issue 2 above. The logo is no longer clipped. The light header is now white with a gold rule, and the dark header is black and blends cleanly with the page. The footer pager shows only "Previous: Development setup", which is inside the About section, and this last page has no "Next" link that would jump out of the section.

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Latest release: v0.3.2 (2026-09-19)" tip shows the upgrade command, says there are no breaking changes and advises pinning. It matches the footer "Fastkit-Mobility 0.3.2" and the repo badge "fastmob-v0.3.2".
- The Version/Highlights table works: muted dates, one or two lines per release on desktop, inline links into the reference pages, and the 0.1.x releases folded into one row. Contrast holds in both themes, with the dark-mode yellow links clearly readable.
- The new header is calm. The gold rule in light mode and the black bar in dark mode frame the page without the old heavy black band, and the pager stays inside the About section.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures)

CROSS-PAGE NOTES:
- Logo wordmark plus site title duplicate the name in the header on every desktop page (issue 2).
- The active-tab underline offset (issue 3) is site-wide.
- A reusable narrow-viewport two-column table class (nowrap first column, nowrap inline code) should be shared with the Resources glossary and the other two-column tables.
- Carried over: add "add a dated row to docs/src/release_notes/index.md and a vX.Y.Z.md page" to the release checklist in CLAUDE.md.
