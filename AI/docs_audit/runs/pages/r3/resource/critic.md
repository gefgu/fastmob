```
PAGE: docs/src/resource/index.md   ROUND: 3
SCORES: first_impression=6 engagement=4 navigation=8 scannability=7 visual=6 ia_role=6 conciseness=7 trust=6
OVERALL: 6.3/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P1] The page is still entirely text. It has not changed since round 2: three icon-less cards (PyPI / Source code / Get help, desktop-light-fold), two tables and a bullet list. There is no figure anywhere, and next to Home and Features it reads as an appendix.
   FIX: Add a lucide icon to each card heading (`lucide/package`, `lucide/github`, `lucide/life-buoy` or `lucide/bug`) using the card-icon markup Features already uses. Above "Terms used in these docs", reuse the existing Home hierarchy diagram (TrajDataFrame/Positionfixes -> Staypoints -> Locations / Triplegs -> Trips -> Tours) so the glossary has a picture. Optionally add a dataset thumbnail strip above the Datasets table (image request 1).
2. [visual] [P1] Mobile (mobile-full, Datasets and Terms tables): the tables now scroll sideways instead of breaking mid-word. As a result, the most useful column on each table ("How to use it in fastmob", "Class") sits fully off-screen with no scroll cue. The Datasets rows also have about 100px of empty height each, because the hidden column wraps tall. The floating TOC button (mobile-fold, bottom right) still sits on the Datasets table header.
   FIX: For the Datasets table, drop the table on narrow screens. Rewrite it as three short blocks (bold dataset link, one line, then "Use: `load_geolife_trajectories` · stay-point notebook"). That works on every width, and since there are only three rows, it is also cleaner on desktop. For the Terms table, merge Term and Class into one column ("Staypoint — `Staypoints`") so it fits in two columns. Site-wide: hide the floating TOC button below 600px or give it right-edge clearance outside the content box.
3. [visual] [P2] Desktop (desktop-light-full, GeoLife row): the "non-wrapping identifier columns" change did not reach this table. `load_geolife_tr ajectories` still breaks mid-identifier, and `pip install fastmob` wraps across two lines inside the PyPI card.
   FIX: Fixing issue 2 removes the table problem. For the card, put the command on its own line as a `<code>` with `white-space: nowrap`, or shorten the body to "`pip install fastmob` — wheels for Linux, macOS, Windows".
4. [trust] [P2] There is still no license or size information for the datasets, so a researcher cannot tell whether reuse is allowed. Background reading lists three libraries and no papers (scikit-mobility JSS paper, trackintel paper). The new citation pointer is a bare trailing sentence ("To cite fastmob, see About.") that is easy to miss.
   FIX: Add a size/license note per dataset (e.g. "4.5M rows · SNAP terms", "24,876 trajectories · Microsoft Research license"). Add a "Papers" sub-list with 2-3 references under Background reading. Promote the citation line into a fourth card, "Cite fastmob →", that links to about/index.md#cite-fastmob, or into a `!!! note` admonition.
5. [ia_role] [P2] The 8-row glossary is still the largest block on the page and remains reference content. The intro says "ordered along the hierarchy", yet Tour comes before Location, even though Locations are built from Staypoints and Tours depend on Locations (desktop-light-full, Terms table rows 6-7).
   FIX: Move Location up to directly after Staypoint. Better still, move the whole table to reference/data-structures.md and leave one line here: "New to the vocabulary? See Data structures."
6. [first_impression] [P2] Above the fold, the three cards are equal-weight boxes with no visual hook. The cards are reordered as requested (install first), but without icons they read as plain text blocks (desktop-dark-fold).
   FIX: Covered by issue 1 (icons). Also cut the PyPI card body to one line so the three cards are the same height.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (engagement, no icons/figure): not fixed.
2 (visual, mobile TOC overlap / mid-word breaks): partial. Mid-word breaks on mobile are gone because the tables scroll now, but the key columns are hidden off-screen. The TOC button still overlaps the table, and the desktop GeoLife cell still breaks mid-identifier.
3 (visual, dark-mode code-link color / light card text): fixed. Code links are yellow in dark mode (desktop-dark-full, Class column), and the light card body text is readable.
4 (ia_role, glossary placement/order/TrajDataFrame): partial. TrajDataFrame is added and the hierarchy-order intro line is present. The glossary is still on this page and Location is still after Tour.
5 (trust, license/papers/citation): partial. A citation link was added. There is still no dataset license and no papers.
6 (first_impression, card order): fixed for order (PyPI -> Source -> Help). The cards are still visually flat.
7 (navigation, About sidebar missing section index): fixed. The sidebar now lists "About fastmob" first, with Resources highlighted (desktop-light-full, left sidebar).

WHAT WORKS (max 3, so it is not broken in the next round):
- Navigation is now clear. The About sidebar shows the section index first, the right TOC lists the three sections, and the footer has prev/next ("About fastmob" / "Development setup"). Together these earn the 8.
- Link and code-link colors are consistent in both themes (yellow/olive), and the dark-mode tables and cards have good contrast.
- The Datasets table links each source and names the exact fastmob loader or page that uses it.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show readers what each bundled dataset looks like before they pick a loader (Brightkite: global check-in scatter; GeoLife: dense Beijing GPS traces).
  preferred_source: notebook-real-data
  suggested_filename: datasets-thumbnails.png
  suggested_location: docs/src/assets/images/resource/
  notebook_recipe: Load a 10k-row Brightkite sample (tests/shared/data/loc-brightkite_totalCheckins.txt.gz) and one GeoLife user via load_geolife_trajectories. Draw two side-by-side panels in matching style: a world lat/lng scatter for Brightkite and one user's trajectory line over Beijing for GeoLife. Use a transparent or neutral background so it works in both themes.

CROSS-PAGE NOTES:
- The floating mobile TOC button still overlaps content at 390px. The back-to-top button was removed, but this is a separate element. Fix it in extra.css.
- The "non-wrapping identifier columns" table rule does not cover inline code in the last column of three-column tables (GeoLife loader still breaks). Check the selector.
- Wide tables on mobile now scroll horizontally with no visible affordance. Consider a right-edge fade/shadow on `.md-typeset__scrollwrap`, site-wide.
- Header logo/white box in dark mode and the black header in light mode: deferred by design (P2 only).
```
