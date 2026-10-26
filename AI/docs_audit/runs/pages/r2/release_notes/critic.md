PAGE: release_notes/index.md   ROUND: 2
SCORES: first_impression=7 engagement=6 navigation=6 scannability=7 visual=6 ia_role=7 conciseness=8 trust=7
OVERALL: 6.8/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P1] The "Released" column wraps dates mid-value on desktop in both themes: "2026-09-" / "09", "2026-08-" / "25", "2026-08-" / "23". Only v0.3.2 and v0.3.1 fit on one line, so the column looks ragged. On mobile (mobile-full) every date wraps. The Version and Released columns also take about 55% of the 390px width, which squeezes Highlights into 5-13 lines per row (v0.2.0 runs to 9 lines).
   FIX: In docs/src/stylesheets/extra.css, add `white-space: nowrap` to the first two table columns on this page (scope it with a page class or `.md-typeset table td:nth-child(-n+2)` under a release-notes wrapper), and let Highlights take the remaining width. For mobile, merge Version and Released into one cell instead ("[v0.3.2](v0.3.2.md)<br><small>2026-09-19</small>"). That gives two columns and roughly doubles the Highlights width.
2. [engagement] [P1] The highlights are now real, but several use internal vocabulary that a cold reader can't parse: "Scalable STVD with a sparse EMD path, shared RECAST contact-network kernels" (v0.3.0), "Arrow-only Rust bindings" (v0.2.0), "wrapper dataframes accepted everywhere" (v0.2.1). Only v0.2.0 links into the docs. v0.3.0 and v0.2.1 name features with no link.
   FIX: In docs/src/release_notes/index.md, rewrite each Highlights cell as the user-facing benefit and link the feature. For example, v0.3.0: "Faster [spatio-temporal visit distance](../reference/measures/...) on large datasets; [contact networks](../reference/measures/contact-network.md)". v0.2.1: "Faster [`stay_locations`](...)". Drop implementation details such as "Arrow-only Rust bindings" and "sparse EMD path" from the index and keep them on the per-version page.
3. [trust] [P1] "Upgrade policy" (bottom) is still one generic sentence. Round 1 asked for concrete bullets. The change log says a "concrete upgrade policy" was added, but the page still doesn't say whether deprecations warn first, how long before removal, or how `fastmob` and `fastmob-vis` versions relate (the v0.3.2 row mentions the 0.1.x pin, but the policy doesn't).
   FIX: Replace the paragraph with 3 bullets. (a) "Pre-1.0: minor bumps (0.x -> 0.y) may break; patch bumps never do." (b) The deprecation rule actually followed (for example "deprecated APIs emit `DeprecationWarning` for one minor release"), or state honestly that there is no deprecation cycle yet. (c) "`fastmob[vis]` pins a compatible `fastmob-vis` 0.1.x."
4. [visual] [P1] On mobile (mobile-full, v0.3.0 row) the floating "On this page" TOC button sits on top of the Highlights text and covers "RECAST conta…ct".
   FIX: This is site-wide. In extra.css, give the floating TOC button an opaque background and shadow, or move it into the bottom gutter so it never overlaps content (for example `bottom: 1rem; right: 1rem`). The column-merge in issue 1 also reduces the collision area.
5. [navigation] [P2] The left sidebar (desktop) still auto-expands all ten version links next to a table that lists the same versions. The page now sits under About, but the sidebar shows Resources / Development setup / Release Notes with no "About fastmob" entry above them, so the reader can't see the parent page. The footer "Previous: Development setup" is still an odd lead-in to a changelog.
   FIX: In zensical.toml, reduce the Release Notes nav group to "All releases" plus the latest 3 versions. Older pages stay reachable from the table and the 0.1.x expander. Check that the About section index ("About fastmob") renders as the clickable section header in the sidebar.
6. [scannability] [P2] Table body text is still smaller than body copy (about 13px against 16px for the "Upgrade policy" paragraph). The same was flagged in round 1 and is unchanged.
   FIX: Set `.md-typeset table:not([class]) { font-size: .8rem }` or larger in extra.css (site-wide; it also helps Reference tables).
7. [visual] [P2] Deferred brand items, noted only: the header logo text is clipped ("Mobilit") with a white box in dark mode, and the header is black in light mode.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (missing versions, no dates): fixed. All tags v0.2.1-v0.3.1 have rows and pages, and the Released dates match the git tag dates.
2 (filler highlights, 0.1.x repetition): partial. 0.1.x is collapsed into an expander and the rows have real content, but most cells don't link and they use jargon (issue 2 above).
3 (generic intro, no latest-release answer): fixed. The "Latest release: v0.3.2 (2026-09-19)" tip shows the install command and "No breaking changes".
4 (upgrade policy boilerplate): partial. It is shorter and states the pre-1.0 rule, but there is still no deprecation rule and no fastmob/fastmob-vis relation.
5 (sidebar duplicates table; odd footer prev): not fixed.
6 (small table text; olive link colour; mobile wrap): partial. The link colour is now a consistent dark gold in light mode and yellow in dark mode, and it reads as a link. The table font size is unchanged and the mobile wrapping is worse (dates now wrap too).

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Latest release" tip above the table answers the reader's three questions (which version, how to upgrade, does it break) in one glance. Keep it first.
- The complete, dated, newest-first table, with the 0.1.x series collapsed into one expander, makes the page short and shows the project is actively maintained.
- Both themes render cleanly. Admonition and table colours are legible in dark mode, and nothing is clipped on desktop.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none (a changelog index does not need figures)

CROSS-PAGE NOTES:
- The mobile floating TOC button overlaps content (issue 4). Check every page with tables or code near the right edge.
- Table font size is smaller than body text site-wide (issue 6).
- The About section sidebar doesn't show the section's own index page ("About fastmob") as a visible entry above Resources. Verify `navigation.indexes` behaviour for the About group.
- Release checklist in CLAUDE.md: add a "write docs/src/release_notes/vX.Y.Z.md and add a dated table row" step so the dates and completeness don't regress.
