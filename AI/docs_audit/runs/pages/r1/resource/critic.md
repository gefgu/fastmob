```
PAGE: docs/src/resource/index.md   ROUND: 1
SCORES: first_impression=4 engagement=3 navigation=5 scannability=6 visual=5 ia_role=4 conciseness=6 trust=4
OVERALL: 4.6/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [ia_role] [P0] The page has no resources of its own. "Choose the right starting point" (desktop-light-full, first table) only re-links five internal pages that are already in the top tabs (Learn, Reference, Features > Benchmarks, Release Notes). Nothing here sends the reader outside the docs: no GitHub repo/issues/discussions, no PyPI page, no citation/BibTeX, no links to the scikit-mobility paper or the mobility-laws literature, and no dataset sources (Brightkite, GeoLife). A visitor who clicks "Resources" expecting the outside world finds a second copy of the navigation and leaves.
   FIX: Rewrite resource/index.md around external and long-lived material. Use four card sections in a `grid cards` block, each with a lucide icon: "Get help" (GitHub issues, discussions), "Install & packages" (PyPI, conda if any, source repo), "Datasets" (Brightkite, GeoLife: source URL, license, which fastmob loader reads it, linking to reference/data.md), and "Background reading" (scikit-mobility paper, trackintel, key papers behind the implemented measures and models). Remove the "Choose the right starting point" table, or cut it to at most two rows that the top nav does not already cover.
2. [trust] [P1] Nothing on the page shows that the project is alive or citable: no version badge, no "how to cite", no license line, no contribution link. The "Data and ecosystem" paragraph is the only trust signal, and it is a hedge ("Fastmob documents its own behavior ... independently") instead of a statement of the scikit-mobility relationship.
   FIX: Add a short "Cite fastmob" block (BibTeX in a code block with copy button) and one line with license plus repo. Replace the hedge paragraph with a two-row comparison: "scikit-mobility: what is compatible / what differs", linking to the migration notes or the Features page.
3. [visual] [P1] Light mode: links in the table and paragraph are olive/khaki on white (desktop-light-full, "Start with a trajectory", "data reference"). They have weak contrast and read as disabled. Dark mode uses blue links, so the two themes do not match. In dark mode the footer band "Made with Zensical" is mid-grey (desktop-dark-full, bottom strip) and clashes with the near-black page.
   FIX: In docs/src/stylesheets/extra.css, set the light-scheme `--md-typeset-a-color` to a color that meets AA on white and matches the dark-mode link hue. Set the dark-scheme footer meta background to the page or header black.
4. [visual] [P1] Mobile (mobile-full): the floating table-of-contents button sits on top of body text and hides the end of "scikit-mobility" in the "Data and ecosystem" paragraph. The two-column table squeezes both columns so every row wraps onto two lines.
   FIX: On narrow screens, either add right padding to the content or move the floating TOC button into the gutter (site-wide, extra.css). On this page, replace the table with a cards or definition-list layout (see issue 1), which stacks cleanly at 390px.
5. [engagement] [P1] The page is plain text: a table, one paragraph, and a bullet list, with no icon, figure, or example. Desktop leaves a large empty left column (desktop-light-full, left of the H1) and the content block fills only about half the width. The page looks unfinished.
   FIX: Use `grid cards` with icons (issue 1) so the width gets used. For the glossary, see image request 1: one small labelled diagram replaces most of the text.
6. [navigation] [P2] The glossary terms are not linked to their reference pages. Its vocabulary also does not match the nav: the glossary says "Position fix" and "Trip leg", the nav says "Positionfixes" and "Triplegs", and "Trip", "Tour", and "TrajDataFrame" are missing. At the bottom of the page, "Next: Overview" is ambiguous; it actually goes to About.
   FIX: Turn each glossary term into a link to its reference/data-structures/*.md page, use the exact class names (Positionfixes, Staypoints, Locations, Triplegs, Trips, Tours, TrajDataFrame, FlowDataFrame), and order them by the hierarchy. Consider moving the glossary to reference/data-structures.md, where it belongs by role, and leaving a link here. The footer label is a site-wide issue (see cross-page notes).
7. [first_impression] [P2] The H1 "Resources" has no lead sentence. The first screen does not say what kind of resources are here or who they are for.
   FIX: Add one lead line under the H1, e.g. "Datasets, citations, community channels, and background reading for fastmob." It should state exactly what the new card sections contain.

PREVIOUS ISSUES STATUS (rounds >= 2): n/a (round 1)

WHAT WORKS (max 3, so it is not broken in the next round):
- Short and calm: the whole page fits in about one desktop screen, and the right-hand "On this page" TOC matches the three H2s.
- The glossary uses a consistent bold-term + em-dash pattern that is quick to scan; keep that rhythm if the glossary stays.
- In dark mode, the table header and borders are legible and the link color is clearly distinguishable.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Show how the data hierarchy fits together (Positionfixes -> Staypoints/Triplegs -> Locations/Trips -> Tours, plus FlowDataFrame for OD) so the glossary terms make sense at a glance.
  preferred_source: diagram
  suggested_filename: data-hierarchy.png
  suggested_location: docs/src/assets/images/resource/
  notebook_recipe: n/a
- purpose: Show what the bundled datasets look like (Brightkite check-ins, GeoLife GPS traces), one thumbnail each for the Datasets cards, so readers see the data before choosing a loader.
  preferred_source: notebook-real-data
  suggested_filename: datasets-thumbnails.png
  suggested_location: docs/src/assets/images/resource/
  notebook_recipe: Load a 10k-row Brightkite sample and one GeoLife user via the fastmob data loaders; scatter lat/lng for Brightkite (world extent) and plot the GeoLife user's trajectory over Beijing; two side-by-side panels, same style.

CROSS-PAGE NOTES:
- The header logo image reads "Fastkit-Mobility" next to the site name "fastmob" (all screenshots, top left). The brand names conflict on every page.
- Light-mode link color (olive on white) has low contrast across the whole site; dark mode uses blue, so the two themes do not match.
- Footer prev/next labels show bare "Overview", which is ambiguous because every section has an "Overview" child. Rename the section index entries in zensical.toml (e.g. "About", "Learn overview") or show the section name in the footer.
- Resources is the only leaf item among the top-level tabs that sit between section tabs (nav: Reference [section], Resources [leaf], About [section]). This is the mixed leaf/section pattern. Either merge Resources into About as a child, or give About and Resources the same shape.
- Dark-mode footer "Made with Zensical" band is mid-grey and clashes with the black header and body.
- On mobile, the floating TOC button covers body text (site-wide CSS).
```
