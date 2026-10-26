PAGE: docs/src/release_notes/index.md   ROUND: 4
SCORES: first_impression=8 engagement=7 navigation=6 scannability=8 visual=7 ia_role=8 conciseness=8 trust=8
OVERALL: 7.5/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [navigation] [P1] The page is still a dead end. In all three full screenshots the only thing after "Upgrade policy" is the footer "Previous: Development setup". There is no forward link and no route to the full per-version history on GitHub or PyPI. This was round 3 issue 5, and it is unchanged.
   FIX: In docs/src/release_notes/index.md, add one closing line under the Upgrade policy bullets, for example "Hit a regression after upgrading? [Open an issue](https://github.com/gefgu/fastmob/issues) · [All tags on GitHub](https://github.com/gefgu/fastmob/releases)". Do not add a new section.
2. [navigation] [P2] In the sidebar (desktop light and dark, left column), "Release Notes" is an expandable group whose only child is "All releases". That is an extra nesting level and an extra click for a single page, and it is the only expandable item under About.
   FIX: In zensical.toml `nav`, make Release Notes a leaf (`{"Release notes" = "release_notes/index.md"}`) next to About fastmob / Resources / Development setup. Keep the per-version pages out of nav; the table links reach them.
3. [visual] [P2] On mobile, the floating TOC toggle is still visible: a white circle at the right edge, about y=775-810 in mobile-fold, sitting on the v0.3.0/v0.2.2 row boundary and touching "Better staypoint detection;". The r3 change log says this toggle was hidden, but it was not hidden on this page. It now has an opaque background, so it no longer hides text, but it still overlaps the table.
   FIX: Check the selector in docs/src/stylesheets/extra.css that was meant to hide the mobile TOC toggle. It does not match the element on this page (it renders as a circular button with a list icon). Hide it at <=76.1875em, or move it into the header bar.
4. [visual] [P2] The "0.1.x series" collapsible (blue box with a paperclip icon, below the table) uses a different component and color than the table directly above it. It reads like an attachment notice, not like older rows of the same changelog.
   FIX: Either (a) add one final table row, `| 0.1.x<br><small>2026-07-16 to 07-21</small> | Initial public release ([v0.1.0](v0.1.0.md)) and maintenance releases [v0.1.1](v0.1.1.md)–[v0.1.3](v0.1.3.md). |`, and delete the admonition, or (b) keep it but switch it to `??? abstract` / `??? info` without the paperclip.
5. [visual] [P2] Deferred, noted only: the header logo is clipped ("Fastkit-Mobilit") with a white box, the light-mode header is black, and the footer "Previous" link crosses sections.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (Highlights column nowrap clipping, P0 regression): fixed. All Highlights cells wrap fully in desktop light, desktop dark, and mobile, and the table stays inside the content column (ends at about x=1057, aligned with the tip admonition).
2 (date wraps mid-value): fixed. Dates render as single-line muted small text under each version ("2026-09-19") in every screenshot.
3 (trust hit from clipped text): fixed. Every highlight and its links ("Spatiotemporal visitation comparison", "Trajectory hierarchy", "preprocessing", "models") are fully visible.
4 (tall rows push Upgrade policy below the fold): fixed. Rows are about 83px on desktop, and Upgrade policy now starts at about y=1035 on a single full page.
5 (dead-end footer, no forward link): not fixed (issue 1 above).
6 (deferred header/logo): deferred, unchanged.
Round 3 cross-page note (mobile floating TOC button over content): partial. The button is now opaque, but it is still shown over the table (issue 3 above).

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Latest release: v0.3.2 (2026-09-19)" tip answers which version, how to upgrade, and whether it breaks, all within the first screen. The footer "fastmob 0.3.2" matches it.
- The two-column Version/Highlights table with muted dates under each version link is compact and readable at 390px. Plain-language highlights with inline doc links make it the best-looking table on the site. Do not reintroduce nowrap rules on column 2.
- The Upgrade policy is four short, honest bullets (pre-1.0 semantics, where migrations live, the vis pin, pinning for reproducibility).

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures)

CROSS-PAGE NOTES:
- The rule that hides the mobile floating TOC toggle does not take effect on this page. Verify it on every page at 390px, not only on the page where it was tested.
- Single-child nav groups (Release Notes > All releases) should be flattened site-wide. Mixed leaf and group items at the same level is one of the known nav weaknesses.
- The release checklist in CLAUDE.md should include "add a dated row to docs/src/release_notes/index.md and a vX.Y.Z.md page", so the table stays complete. This was carried over from round 3 and is not yet done.
