PAGE: learn/index.md   ROUND: 1
SCORES: first_impression=6 engagement=3 navigation=6 scannability=6 visual=6 ia_role=6 conciseness=6 trust=5
OVERALL: 5.5/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [engagement] [P0] The page is text only: 8 identical text cards and no figure, output or code. On desktop-light-full the grid runs from y=313 to y=1590, and on mobile-full it is about 2,500px of white boxes. Nothing shows what a trajectory, staypoint, map-match or fitted law looks like, so the gallery pattern from the scikit-learn benchmark is missing.
   FIX: Add a thumbnail slot at the top of each `.how-to-card` (`<img class="how-to-card__thumb" src="../assets/images/learn/<slug>.png" alt="">`, 16:9, object-fit: cover) and style it in docs/src/stylesheets/extra.css. Make the images from each guide's own data (see IMAGE REQUESTS). If there is only time for three, do Scenarios 01, 04 and 06 first.
2. [ia_role] [P1] The hub does not match its section. The sidebar lists 9 guides, but the grid has 8 cards: "Compare mobility laws notebook" (sidebar, last item) has no card. The two notebooks are also handled differently: one gets a "NOTEBOOK EXAMPLE" card and the other gets nothing. A reader who scans the cards never finds the second notebook.
   FIX: In docs/src/learn/index.md, split the grid under two H2s: "Recipes" (Scenarios 01-07) and "Notebook examples" (stay-point detection GeoLife, plus a new card for learn/comparision-to-mobility-laws.md). Every sidebar entry should then have exactly one card.
3. [scannability] [P1] Each guide has three different names, so the reader cannot match a card to its sidebar entry. Example: sidebar "Start with a trajectory", card H2 "Inspect a minimal trajectory", link "Start with a trajectory ->", and the page H1 (round 0) "Analyze a simple trajectory". The same happens for all 8 cards, e.g. "Turn pings into places" vs "Create staypoints and locations". The "SCENARIO 0n" eyebrow adds a fourth label that tells the reader nothing.
   FIX: Use the nav label as the card H2 and the target page's H1. Replace the per-card link text with a plain "Open ->" (or make the whole card the link and drop the line). Replace the "SCENARIO 0n" eyebrow with useful metadata: topic tag + reading time, or "Recipe" / "Notebook".
4. [first_impression] [P1] Above the fold (desktop-light-fold), the reader sees an H1, a two-line intro and four cards, but no sense of order or starting point. The intro talks about "the complete parameter contract" (internal jargon) and does not say "start here", which guide is for a beginner, or what the guides need (pip install, sample data).
   FIX: Replace the intro with one sentence plus a starting point, e.g. "New to fastmob? Start with *Start with a trajectory* (5 min), then follow the order below." Visually mark card 01 as the starting point (accent border or a "Start here" badge). Drop the "parameter contract" phrase.
5. [conciseness] [P2] The bottom tip "Use the right level of detail" (desktop-light-full, y~1640-1735) repeats the intro's second sentence ("the API reference remains the complete parameter contract"). Its body text is also set smaller than the page body, so it reads as an afterthought.
   FIX: Delete the `!!! tip` block from docs/src/learn/index.md and keep a single "Need every parameter? See the API reference" link in the intro.
6. [visual] [P2] The cards have uneven internal spacing and wrapping. Scenario 04 and 05 cards have a large empty gap above their links (desktop-light-full, y~790-870 and y~1130-1215), and Scenario 06's link wraps so the arrow sits alone on its own line (y~1228; same on mobile-full). Light-mode link/eyebrow colour is olive (#6b6b00-ish), which looks muddy next to the bright yellow active "Learn" tab in the header. The right-hand column (x>1060) is empty on desktop, so the grid is squeezed to about 690px while 400px of whitespace sits unused.
   FIX: Once the link text is shortened (issue 3), the wrap goes away. Give the card description `flex: 1` and anchor the footer link with `margin-top: auto` so the gaps line up across cards. Darken the light-mode accent to a true dark gold (e.g. #8a6d00) in extra.css. Hide the empty TOC column on this hub page (`hide: [toc]` in front matter) so the grid can widen to 3 columns at desktop width.
7. [trust] [P2] There are no signs of effort or maintenance: no time estimates, no "requires: pip install fastmob" note, no dataset names except on the GeoLife card, and no runnable/Binder badge, even though notebook pages have one. The reader cannot tell which recipes run on real data.
   FIX: Add a small metadata row to each card (e.g. "5 min · synthetic data", "10 min · GeoLife · Binder"), using a `.how-to-card__meta` span.

PREVIOUS ISSUES STATUS (rounds >= 2): n/a (round 1 for this page; round-0 site notes: sidebar fixed, light cards fixed, H1 "Learn" fixed, missing mobility-laws card NOT fixed, three-names problem NOT fixed, eyebrow/thumbnail NOT fixed, duplicated tip NOT fixed)

WHAT WORKS (max 3, so it is not broken in the next round):
- The left sidebar now renders, the "Learn" tab is highlighted, and the footer shows Previous/Next, so the reader knows where they are.
- Light-mode cards now use a white surface with a thin border, and dark-mode cards use a dark surface with yellow accents. The two themes are consistent and easy to read.
- On mobile (390px), the cards stack cleanly in one column with no horizontal scroll and good padding.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: Card thumbnail for "Start with a trajectory", showing what a trajectory and its first measures look like (points, jumps, radius of gyration).
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-trajectory.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Use the 6-row DataFrame from learn/analyze-a-simple-trajectory.md, plus fastmob.jump_lengths and fastmob.radius_of_gyration. Plot the points connected in time order and a dashed circle of radius rg around the centroid. Use a 16:9 aspect ratio, no axes, and a transparent background.
- purpose: Card thumbnail for "Create staypoints and locations", showing raw GPS pings collapsing into staypoint clusters.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-staypoints.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Use one GeoLife user-day from the stay-point notebook. Plot positionfixes as small grey dots and the output of Positionfixes.generate_staypoints() as large accent-coloured circles. Use a 16:9 aspect ratio and no axes.
- purpose: Card thumbnail for "Route and map-match trajectories", showing a noisy GPS trace snapped onto a road network.
  preferred_source: notebook-real-data
  suggested_filename: learn-thumb-map-match.png
  suggested_location: docs/src/assets/images/learn/
  notebook_recipe: Use the network example from learn/network-analysis.md. Plot road edges in light grey, the raw trace as red dots, and the map-matched route as a thick accent line. Use a 16:9 aspect ratio and no axes.

CROSS-PAGE NOTES:
- The Learn index footer shows "Previous: Motifs" (from Features), so the prev/next chain crosses section boundaries. This is acceptable, but a reader arriving from Features reads it as unrelated.
- Light-mode accent: the olive link colour (cards here, links site-wide) differs from the bright yellow active tab in the header. Pick one dark-gold accent token for light mode in extra.css.
- The nav label "Compare mobility laws notebook" points to `learn/comparision-to-mobility-laws.md`. The filename has a typo ("comparision"), which shows up in the URL. Rename it, with a redirect if the URL is already public.
- Generic text casing: the intro writes "Fastmob" while the brand and header use "fastmob". Standardise across all pages.
