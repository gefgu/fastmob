# Round 0 changes (site-wide)

Critic: 4.4/10 REVISE (see critic.md).

Applied
- P0 nav: Learn and Release Notes moved to the native nested-list nav form
  (the `children = [...]` form is unsupported, so Learn had no sidebar and no
  active tab). Overview is the first entry (navigation.indexes).
- Release Notes nav now lists v0.3.2 and v0.2.0 (pages already existed).
- `navigation.footer` enabled (prev/next links on every page).
- Light theme: Learn cards use a neutral white surface with 1px border; code
  background neutral grey instead of cream. Example output blocks scroll
  horizontally instead of clipping.
- Learn index H1 "How-to guides" -> "Learn" to match the tab.

Declined / deferred
- Collapse 7 top tabs to 5 (P1, ia_role): changes site IA; needs user decision.
- Black header in light mode: kept, it is the brand identity (user decision).
- Logo reads "Fastkit Mobilit" (live Lato text clipped in the SVG, white box in
  dark mode): needs the source re-exported from Inkscape with text converted to
  paths and a transparent background; not editable here.
- Individual measures page split / docstring rendering (P1): handled as a
  Reference-page task in later rounds.
- Home hero, Key Features cards, benchmark figure: Home round 1.
- Footer content, Reference sidebar cut-off, icon consistency: later rounds.
