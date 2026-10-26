PAGE: release_notes/index.md   ROUND: 1
SCORES: first_impression=6 engagement=4 navigation=6 scannability=7 visual=7 ia_role=7 conciseness=7 trust=4
OVERALL: 6.0/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [trust] [P0] Version table (all screenshots) jumps from v0.2.0 straight to v0.3.2, but the repo has tags v0.2.1, v0.2.2, v0.3.0 and v0.3.1 with no entries. A visitor checking "what's in the version I have" finds holes, and the page claims to "document user-visible changes", so the gaps read as an unmaintained changelog. No dates appear anywhere either, so the reader can't tell whether the project is active.
   FIX: In docs/src/release_notes/index.md, add a `Released` column (YYYY-MM-DD from the git tag dates). Add rows for v0.3.1, v0.3.0, v0.2.2 and v0.2.1, each with a one-line highlight and a matching vX.Y.Z.md page (a short page is fine, or mark the row "Packaging-only, no API changes"). Add the same pages to the Release Notes block of the `nav` in zensical.toml.
2. [engagement] [P1] Four of the six "Highlights" cells are filler: "Maintenance release." three times (v0.1.3, v0.1.2, v0.1.1) and "Initial public release.". The table body below the fold is mostly repeated text. The v0.1.3 source even says only "Current release documentation is maintained against this version", which is stale. The one release with real content (v0.2.0) gets a comma-run cell with no link into the features it names.
   FIX: Collapse v0.1.0–v0.1.3 into a single row or a closed `??? "0.1.x series"` admonition. Write the v0.2.0 and v0.3.x highlights as 2–3 short items that link to the relevant Reference or Learn pages (for example Trajectory hierarchy -> reference/data-structures.md, network analysis -> reference/network.md).
3. [first_impression] [P1] Above the fold (desktop-light-fold), the first block is a 3-line generic paragraph ("Release notes document user-visible changes..."). The answers the reader actually wants are only implied by the table order: the latest version, what's new in it, and whether it breaks anything.
   FIX: Replace the intro paragraph with a one-line "Latest: **v0.3.2** (date) — `pip install -U fastmob`" callout (an admonition `!!! tip "Latest release"`). Put a one-line "Breaking changes: none" or a link to its migration notes directly under it. Drop the sentence about "the documentation site describes the current release".
4. [ia_role] [P1] The "Upgrade policy" section (bottom of every screenshot) is generic boilerplate with no actual policy. It doesn't say what counts as breaking before 1.0, whether deprecations get a warning cycle, or how the `fastmob`/`fastmob-vis` versions relate. Without that it adds length but no information.
   FIX: Either state a concrete policy in 2–3 bullets (pre-1.0: minor bumps may break, with a deprecation warning for at least one minor release; `fastmob[vis]` pins `fastmob-vis 0.1.x`), or delete the section and put a single "Pin `fastmob==X.Y.Z` for reproducible work" line into the latest-release callout.
5. [navigation] [P2] The left sidebar (desktop) repeats exactly the six version links that are already in the table, so the version list appears twice side by side. The "On this page" TOC has a single entry, "Upgrade policy". Footer "Previous: Development setup" (an About sub-page) is an odd backward link from a changelog.
   FIX: This one is minor and depends on the theme. If old versions are collapsed (issue 2), the nav duplication shrinks by itself. Also consider listing only the last 3 versions plus an "Older" overview link under Release Notes in the zensical.toml nav.
6. [visual] [P2] The table text renders smaller than body copy (desktop, both themes), so the "Highlights" column reads as fine print next to the 16px paragraphs above and below. In light mode the version links are olive/dark-yellow underlined text that looks closer to "visited/disabled" than to "primary link". On mobile the Highlights column wraps v0.2.0 into 4 lines.
   FIX: In docs/src/stylesheets/extra.css, raise the table font-size on this page (or site-wide) to match body text. Check the light-theme link color against the accent token for at least 4.5:1 contrast. The mobile wrap goes away once highlights are short bullets on detail pages (issue 2).

WHAT WORKS (max 3, so it is not broken in the next round):
- One newest-first table with linked version numbers is the right shape for a changelog index. Keep the table and keep reverse-chronological order.
- Dark and light themes both render cleanly. Table borders and row padding are consistent, and nothing is clipped on mobile (390px).
- The page is short and does one job (a changelog index). Don't grow it into a narrative page.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures; dates and real highlights are the missing content)

CROSS-PAGE NOTES:
- The header logo image "Fastkit-Mobility" is clipped at its right edge ("Mobilit") in every desktop screenshot. This is site-wide header and logo sizing.
- The light-theme link color (olive underline) appears site-wide. It should be checked for contrast and for whether it reads as a link.
- The top-nav "Release Notes" tab and the "About" tab both appear as low-value meta sections. Consider moving Release Notes under About (or into the footer) to reduce top-level nav from 7 items to 6.
- Release process: the CLAUDE.md release steps don't include writing a release-notes page, which is likely why tags v0.2.1–v0.3.1 have no entries. Add a "write docs/src/release_notes/vX.Y.Z.md and a table row" step to the release checklist.
