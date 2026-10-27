PAGE: learn/index.md   ROUND: 10
SCORES: first_impression=8 engagement=8 navigation=8 scannability=8 visual=8 ia_role=8 conciseness=8 trust=8
OVERALL: 8.0/10   VERDICT: PASS
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [visual] [P2] Dark mode: the Route and map-match thumbnail is a bright, near-white map tile. It is the only light panel in a grid of dark ones, at desktop-dark-full row 2, centre card (y~985-1100), and it pulls the eye away from the hero. Every other thumb follows the dark thumb surface.
   FIX: Ship a dark variant, learn-thumb-route-dark.png, with dark-grey roads on the dark thumb surface, and swap it in with a `#only-dark`/`#only-light` image pair or a `prefers-color-scheme` `<picture>` in docs/src/learn/index.md. Otherwise, apply `filter: invert(.9) hue-rotate(180deg)` to that one thumb in dark mode, in extra.css.
2. [visual] [P2] Header: the brand name appears twice side by side. The outlined logo already reads "Fastkit- Mobility" (stacked, about 11px, so tiny), and it is followed straight away by the site title "Fastkit-Mobility" (desktop-light-fold and desktop-dark-fold, x~115-350, y~10-38). It reads as a stutter, and the logo's own text is too small to read.
   FIX: Use the glyph-only mark (the gold "F" bolts) in the header logo slot and keep the text title. Alternatively, keep the wordmark logo and hide `.md-header__title` on desktop. Do one or the other, not both.
3. [visual] [P2] In dark mode the header has no separation from the page. The light theme gets a gold rule under the tabs (desktop-light-fold y~97), but the dark header is the same black as the body and has no rule, so only the active-tab underline marks the edge (desktop-dark-fold y~96).
   FIX: Give the dark header the same 1px gold rule as light, perhaps at reduced opacity, or make the header surface a step above the body background (for example #111 against #000). Set this in docs/src/stylesheets/extra.css under `[data-md-color-scheme="slate"] .md-header`.
4. [visual] [P2] Some thumbnail details are still faint or small.
   - Light hero: the pale-yellow early half of the trajectory almost vanishes on the #f3 panel (desktop-light-fold, thumb x~560-600). It is clearly visible in dark mode.
   - Backends SVG: the centre node "Fastkit Rust" is still noticeably smaller and greyer than the pandas/Polars labels.
   FIX:
   - Darken the lightest colour stop of the hero colormap, for example to #e8c860 or darker, in learn-thumb-trajectory.png.
   - Set the centre node's font-size and weight in learn-thumb-backends.svg to match the side boxes.
5. [engagement] [P2] The notebook thumbnails repeat the guide thumbnails. The stay-point notebook thumb is visually the same picture as "Create staypoints and locations", and the laws notebook thumb is close to "Fit and evaluate". Seen side by side, it looks like reuse rather than new content.
   FIX: Make each notebook thumb show what is unique to the notebook. For the stay-point notebook, overlay trackintel staypoints as hollow rings next to the gold Fastkit-Mobility ones. For the laws notebook, show 2-3 candidate laws in distinct colours.
6. [trust] [P2] Cards still have no time or effort estimates (carried over, optional).
   FIX: Append "· ~N min" to each `how-to-card__meta` line.
7. [conciseness] [P2] Mobile: the page is about 4650px tall. All nine cards stack with full-width 16:9 thumbs, so reaching the notebooks takes about 8 screens.
   FIX: Below 600px, render the non-hero cards as a horizontal row: a 96px square thumb on the left and the title and meta on the right. Hide the description `<p>` at that width. Keep the hero card as it is.

PREVIOUS ISSUES STATUS (rounds >= 2): (compared against r6; there is no r7-r9 critic file for this page)
1 (dark thumb panels invisible): fixed. Every thumb now sits on a visible dark-grey panel, except the Route thumb (new issue 1).
2 (tiny in-image text): mostly fixed. The axis labels are gone from the measures and fit thumbs. The Backends centre node is still small (issue 4).
3 (no time estimates): not fixed (optional).
4 (text-only notebook cards): fixed. Both notebooks now have real-data thumbnails, though they are close to the guide thumbs (issue 5).
5 (Previous pager crossing into Features): fixed. The footer now shows only "Next: Start with a trajectory", kept inside the Learn section.
No regressions in layout. The header change helps in light mode: white with a gold rule gives clean separation and replaces the old heavy black bar. In dark mode the header looks flatter (issue 3).

WHAT WORKS (max 3, so it is not broken in the next round):
- The "Start here" hero, with a real GeoLife figure, is above the fold on desktop and mobile, and the section-bound "Next: Start with a trajectory" pager backs it up. That makes the path unambiguous (first_impression=8, navigation=8).
- A uniform hero, 3x2 guide grid and 2-up notebook grid. Card titles match the sidebar one to one, every card has a consistent "Data: ..." meta line, and every card now has a thumbnail (scannability=8, ia_role=8, engagement=8).
- Theme parity: thumb panels, card borders and the gold accent now read correctly in both light and dark. The light header (white with a gold rule) is calm and matches the card accent colour (visual=8). The intro is three sentences with no filler (conciseness=8).

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: A dark-mode variant of the map-matching thumbnail, so it does not glare in the dark grid.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-route-dark.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Same road network and trace as learn-thumb-route.png. Roads in #555 on a transparent or #262626 background, noisy GPS points in red, matched path in gold. 16:9, no text.
- purpose: Make the stay-point notebook thumb distinct by showing the trackintel comparison.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-notebook-staypoints.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: GeoLife. Positionfixes in grey, Fastkit-Mobility staypoints as filled gold dots, trackintel staypoints as hollow dark rings on the same axes. Zoom to one district so the overlap is visible. 16:9, no text.

CROSS-PAGE NOTES:
- Header: the duplicated brand (logo wordmark plus title text) and the missing dark-mode header rule are site-wide (issues 2 and 3).
- Any map-tile thumbnail with a light basemap, on Home, Features or elsewhere, needs a dark variant. Check for the same glare effect.
- The section-bound pager works well here. Confirm that the last page of each section shows only "Previous" and no cross-section "Next".
- The filename typo `learn/comparision-to-mobility-laws.md` was noted in earlier rounds. The nav now points to `learn/compare-to-mobility-laws.md`, so check that any old URL redirects.
