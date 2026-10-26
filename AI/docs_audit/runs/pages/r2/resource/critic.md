```
PAGE: docs/src/resource/index.md   ROUND: 2
SCORES: first_impression=6 engagement=4 navigation=7 scannability=7 visual=6 ia_role=6 conciseness=7 trust=6
OVERALL: 6.1/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P1] The page is still all text. The three link cards (Get help / PyPI / Source code) are plain boxes with grey body text and no icons (desktop-light-fold, card row). Below them come two tables and a bullet list with no figure. Next to the Home and Features pages, it reads as an appendix.
   FIX: Add a lucide icon to each card heading (e.g. `lucide/bug`/`lucide/message-circle`, `lucide/package`, `lucide/github`) using the same feature-card markup Features uses. Above "Terms used in these docs", reuse the existing Home hierarchy diagram (Positionfixes -> Staypoints/Triplegs -> Locations/Trips -> Tours) so the glossary has a picture. Optionally add one thumbnail per dataset in the Datasets table (image request 1).
2. [visual] [P1] Mobile (mobile-fold, bottom right): the floating TOC button still sits on the Datasets table header and hides "How to use it in fastmob". At 390px the three-column tables break code tokens mid-word: `load_geolife_trajectorie s`, `load_datase t`, `Positionf ixes`, `Staypoint s`, `Location s` (mobile-full, Datasets table and Terms table, right column).
   FIX: Site-wide (extra.css): give the floating TOC button right-margin clearance, or hide it below 600px. Here: on narrow screens set `code { word-break: normal; overflow-wrap: anywhere }` off for table cells, or `white-space: nowrap` for code inside `td`, with horizontal scroll on the table. Alternatively, drop the Datasets table's middle column on mobile by turning each dataset into a short definition list (name + one line + loader).
3. [visual] [P1] In dark mode, code-styled links in the Terms table (`Positionfixes`, `Staypoints`, ...) and the `pip install fastmob` inline code render blue, while every other link is yellow (desktop-dark-full, Terms table Class column versus the "Evaluation measures" link in the same column). One column holds two link colors. In light mode, the card body text is light grey on white and looks disabled (desktop-light-fold, cards).
   FIX: In extra.css, set `.md-typeset a code` color to inherit the link color in both schemes. Darken `.feature-card p` in light mode to the normal body-text token.
4. [ia_role] [P2] The "Terms used in these docs" glossary (8 rows) is the largest block on the page. It is reference content that belongs on reference/data-structures.md. Its order (Tour before Location) does not follow the hierarchy, and TrajDataFrame, the class most readers meet first, is missing.
   FIX: Move the glossary table into reference/data-structures.md (Data structures overview), ordered as the hierarchy diagram: TrajDataFrame/Positionfixes, Staypoints, Locations, Triplegs, Trips, Tours, FlowDataFrame. Here, leave one line: "New to the vocabulary? See the data structures overview." If it stays, at least add TrajDataFrame and reorder.
5. [trust] [P2] The datasets carry no license or size/format info, so a researcher cannot tell whether they can reuse them. Background reading lists three libraries but no papers (scikit-mobility JSS paper, trackintel paper, the mobility-law papers behind the models). The page also does not point to the citation block that now lives on About.
   FIX: Add a "License" column to the Datasets table (Brightkite: SNAP; GeoLife: Microsoft Research license), or a footnote line. Add a "Papers" sub-list with 3-4 references, and a final line "Cite fastmob -> About" linking to about/index.md#citation.
6. [first_impression] [P2] Above the fold, the page jumps from a one-line lead to three equal cards with no sense of priority. The most common action (install) sits in the middle, and "Get help" leads.
   FIX: Order the cards Install (PyPI) -> Source code -> Get help, or add a fourth "Discussions / questions" card if GitHub Discussions is enabled. Keep the lead line.
7. [navigation] [P2] The About sidebar shows Resources, Development setup and Release Notes with no "About fastmob" entry or section heading (desktop-light-full, left sidebar). Readers do not see that they are inside About, apart from the yellow tab.
   FIX: Site-wide: show the section index as the first sidebar item (or a sidebar section title "About"). See cross-page notes.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (ia_role, no external resources): fixed. The page now links GitHub issues, PyPI, source, dataset sources and upstream libraries, and the duplicate-nav table is gone.
2 (trust, cite/license/sk-mob relationship): partial. The license sits in the site footer, and a scikit-mobility relationship line is in Background reading. There is no citation link here.
3 (visual, link color/footer): partial. There is one link color per theme, and the dark footer now matches the page. Code-styled links are still blue in dark mode, which mismatches.
4 (visual, mobile TOC overlap/cramped table): not fixed. The TOC button now covers the Datasets table header, and three-column tables break code mid-word.
5 (engagement, plain text/unused width): partial. The cards use the width, but there are still no icons and no figure.
6 (navigation, glossary links/names/footer label): partial. The terms now link to classes with exact class names, and the footer shows "About fastmob" / "Development setup". TrajDataFrame is missing and the order does not follow the hierarchy.
7 (first_impression, no lead line): fixed.

WHAT WORKS (max 3, so it is not broken in the next round):
- The lead line plus three outward-facing cards make the page's purpose clear in the first screen (desktop-light-fold).
- The Datasets table links each dataset's source and names the fastmob loader or page that uses it. That is exactly what a resources page should do.
- Footer prev/next ("About fastmob" / "Development setup") and the dark-mode footer band are now clear and consistent.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Let readers see what each bundled dataset looks like before picking a loader (Brightkite: global check-in scatter; GeoLife: dense Beijing GPS traces).
  preferred_source: notebook-real-data
  suggested_filename: datasets-thumbnails.png
  suggested_location: docs/src/assets/images/resource/
  notebook_recipe: Load a 10k-row Brightkite sample (cached tests/shared/data/loc-brightkite_totalCheckins.txt.gz) and one GeoLife user via load_geolife_trajectories. Make two side-by-side panels in the same style: a world-extent lat/lng scatter for Brightkite and a single user's trajectory line over Beijing for GeoLife. Export light/dark-neutral backgrounds.

CROSS-PAGE NOTES:
- The floating mobile TOC button overlaps content on narrow screens (site-wide extra.css). It is still present this round.
- In dark mode, links styled as inline code render blue while plain links are yellow. Unify `.md-typeset a code` color site-wide.
- Under navigation.indexes, the About sidebar hides the section index ("About fastmob"). Readers lose the "you are in About" cue. Consider a sidebar section title.
- Header logo text/white box in dark mode and the black header in light mode: deferred by design (P2 only).
```
