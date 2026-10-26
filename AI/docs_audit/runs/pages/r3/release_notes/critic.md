PAGE: docs/src/release_notes/index.md   ROUND: 3
SCORES: first_impression=7 engagement=6 navigation=7 scannability=6 visual=3 ia_role=7 conciseness=8 trust=6
OVERALL: 6.3/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P0] REGRESSION: the Highlights column no longer wraps, so text runs off the right edge of the table and is cut mid-word in both desktop themes. Examples: "contact-network a…" (v0.3.0), "accepted b…" (v0.2.1), "network and socia…" (v0.2.0). The table also overflows the content column (it extends to about x=1073, past the admonition edge at about 1057). On mobile (mobile-full) every Highlights cell is cut after 2-3 words ("Stable fastmob.vis facade; fast…", "Better staypoint detection; packag…"), so the page's main content is unreadable without horizontal scrolling that nothing signals. Cause: docs/src/stylesheets/extra.css lines 435-437 apply `.md-typeset table:not([class]) td:nth-child(2) { white-space: nowrap; }` site-wide. On this page, column 2 is the long prose column.
   FIX: Delete the `td:nth-child(2) { white-space: nowrap }` rule from extra.css. If some Reference tables need a non-wrapping second column, scope that rule to those tables with a wrapper class, not globally. Keep the `td:first-child code/a` nowrap rule.
2. [visual] [P1] The Version cell still wraps the date mid-value in both themes and on mobile ("2026-" / "09-19"). Every row is now 3 lines tall for one or two lines of content, which makes the table look sparse and ragged. The existing nowrap rule targets only `td:first-child a/code`, and the date is plain text after `<br>`.
   FIX: In docs/src/release_notes/index.md, wrap each date in a span (for example `<span class="rn-date">2026-09-19</span>`, or `<small>`). In extra.css, add `.md-typeset td:first-child small, .rn-date { white-space: nowrap; font-size: .7rem; color: var(--md-default-fg-color--light); }`. Then each row is "v0.3.2" over a muted single-line date.
3. [trust] [P1] Because of issue 1, a first-time reader sees a changelog that looks broken, with sentences cut off and no indication that more text exists. That undoes the credibility the dated, complete table earned in round 2. It is scored separately because it also hides the linked features (the "contact networks" link and the end of v0.2.0's list are invisible).
   FIX: Fixed by issue 1. After the fix, re-screenshot mobile at 390px and confirm every Highlights cell is fully visible without horizontal scroll.
4. [scannability] [P2] The table rows have tall padding (each row about 110px on desktop for one line of Highlights), so six releases fill a whole screen and push "Upgrade policy" below the fold. Once dates fit on one line (issue 2), rows shrink. If they don't, reduce the `td` vertical padding on this page.
   FIX: After issues 1 and 2, check the row height. If it is still more than about 70px, add a `.release-table` wrapper with `td { padding-block: .5em; }`.
5. [navigation] [P2] The footer only offers "Previous: Development setup", with no forward link. The changelog is a dead end, and the per-version pages are reachable only through the table links.
   FIX: Add a one-line closing link under "Upgrade policy" (for example "Something broke after upgrading? [Report it](../about/index.md#…)" or link the GitHub releases page).
6. [visual] [P2] Deferred brand items, noted only: the header logo text is clipped ("Fastkit-Mobilit") with a white box in dark mode, and the header is black in light mode.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (dates wrap mid-value; Version/Released columns squeeze Highlights on mobile): partial. Version and date are merged into one cell as asked, but the date still wraps, and the nowrap rule landed on the Highlights column, which caused a P0 clipping regression (issues 1 and 2 above).
2 (jargon highlights, few links): fixed. The cells are plain-language ("Faster stay-location detection", "scales to 100M-row inputs") and v0.3.0 and v0.2.0 link to docs. The links are only partly visible because of the clipping.
3 (upgrade policy one generic sentence): fixed. There are four concrete bullets, including the pre-1.0 rule, where migrations are documented, the fastmob-vis 0.1.x relation, and pinning. It does not invent a deprecation cycle, which is honest.
4 (mobile floating TOC button overlaps text): partial. The back-to-top button was removed, but a floating TOC/menu button (icon at right, about y=793 in mobile-full) still sits over the v0.2.2/v0.2.1 rows. It overlaps clipped text, so the collision is currently hidden.
5 (sidebar duplicates versions; no About index visible): fixed. The sidebar shows "About fastmob" first and Release Notes contains only "All releases". The footer "Previous: Development setup" remains (issue 5).
6 (small table text): fixed or changed. Table text now renders at near body size in the screenshots.
7 (deferred header/logo): deferred, unchanged.

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Latest release: v0.3.2 (2026-09-19)" tip still answers which version, how to upgrade, and whether it breaks, in one glance at the top.
- The Upgrade policy is now four short, concrete, honest bullets. Keep it as is.
- The sidebar is clean: About fastmob / Resources / Development setup / Release Notes > All releases, with no duplicated version list.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures)

CROSS-PAGE NOTES:
- SITE-WIDE P0 RISK: `.md-typeset table:not([class]) td:nth-child(2) { white-space: nowrap; }` in docs/src/stylesheets/extra.css (lines 435-437) affects every plain Markdown table. Any table whose second column is a description (Reference module tables, API summary tables "API | Description", Resources glossary) will clip or overflow the same way. Audit every page with a 2+ column table after removing or scoping the rule.
- The mobile floating TOC button still floats over content (seen here at right over the table rows). Give it an opaque background, or keep it out of the text column.
- The release checklist in CLAUDE.md should include "add a dated row to docs/src/release_notes/index.md and a vX.Y.Z.md page" so completeness doesn't regress.
