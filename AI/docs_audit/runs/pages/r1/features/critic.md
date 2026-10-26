PAGE: features/index.md   ROUND: 1
SCORES: first_impression=4 engagement=2 navigation=5 scannability=5 visual=6 ia_role=4 conciseness=6 trust=3
OVERALL: 4.4/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [trust] [P0] "Rust-accelerated core: 500x median speedup" (bullet 2) is a bare headline number: it does not say against what (skmob?), on which measures, at what row count, or on what hardware, and it links nowhere. The Benchmarks page sits one click away in the sidebar but is not referenced. A skeptical reader reads this as marketing inflation. "Measured validation" (bullet 4) makes the same move: it claims reproducibility but links to no tests, coverage report or benchmark.
   FIX: Put a context line on the number ("median 500x vs scikit-mobility 1.3.1 across N measures, 1k-4M Brightkite rows; see Benchmarks") and link it to features/benchmarks.md. Link "Measured validation" to the Benchmarks page and the About > Development setup page (or the CI badge). Drop any claim that has no link target.
2. [engagement] [P0] The whole page is one bulleted list of 6 prose items with no figure, no code and no benchmark chart (desktop-light-full: everything below y≈560 is empty white space). It is a feature overview with nothing that demonstrates a feature.
   FIX: Swap the list for a 2x3 grid of Zensical cards (`<div class="grid cards" markdown>`), one per feature, each with a lucide icon, a one-line claim and a "-> link" to its proof page. Under the grid, add (a) one small speedup bar chart (fastmob vs skmob, log scale, 4M rows) reused from the Benchmarks page and (b) a 4-line before/after code snippet: `import skmob.measures.individual as m` -> `import fastmob as m`, the same call on a polars DataFrame.
3. [ia_role] [P1] The page has no lead paragraph, and its H1 "Key Features" does not match its nav label "Overview" (sidebar: Overview is not even listed as an entry; the left nav shows only Benchmarks/Optimizations/Motifs, so the reader cannot tell which page they are on). It also does not act as a hub for its three children: it never says what Benchmarks, Optimizations and Motifs contain.
   FIX: Retitle the H1 to "Features" (or rename the nav entry to match), add a one-sentence lead ("fastmob runs the scikit-mobility API on a Rust core, on any dataframe"), and end the page with a "Dig deeper" row of 3 cards: Benchmarks (numbers), Optimizations (how), Motifs (what that page covers).
4. [first_impression] [P1] Above the fold the reader gets only a title and dense list items, with no hook or visual anchor. Bullets 1 and 3 (backend-agnostic, drop-in API) are the strongest differentiators but read as the same weight as "Lightweight". Bullets 2 and 5 also overlap (zero-copy is named in both).
   FIX: Order the cards by reader value: Drop-in skmob API, Speed (with number + link), Any dataframe, Zero-copy/memory, Validated, Lightweight. Merge the zero-copy wording out of the speed card so each card makes one distinct claim.
5. [trust] [P1] Brand inconsistency: the header logo reads "Fastkit-Mobility" and bullet 5 says "Fastkit-Mobility process the data", while the site title, nav and every other bullet say "fastmob". A newcomer can't tell whether these are two products.
   FIX: In docs/src/features/index.md, use "fastmob" throughout (or define the relationship once: "fastmob (Fastkit-Mobility)"). Flag the logo text for the site-wide pass.
6. [trust] [P2] "Python Wheels ship with less than 30KB" is an unlikely size for a wheel that bundles a compiled Rust extension, so it reads as an error and weakens the other claims.
   FIX: Check the real wheel size from `dist/` and state it ("~X MB wheel, no GDAL/GEOS runtime deps"), or remove the bullet.
7. [visual] [P2] Mobile (mobile-full): six long multi-line bullets build a ~600px text column with no breaks. The bold label and its description run together, so scanning is slow.
   FIX: This goes away with the card grid from issue 2: cards stack one per row on mobile, each with an icon and a short heading on its own line.

WHAT WORKS (max 3, so it is not broken in the next round):
- Bold-label-then-description bullets give a basic scannable rhythm, and the page is short.
- Light and dark themes both render with good text contrast, and the active "Features" tab gets a clear yellow highlight.
- The footer Previous/Next (Home / Benchmarks) gives a working next step.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- purpose: show the size of the speed claim in one glance (fastmob vs skmob runtime per measure at 4M rows)
  preferred_source: notebook-real-data
  suggested_filename: features-speedup-overview.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: Brightkite 4M rows; run skmob and fastmob jump_lengths, radius_of_gyration, filter, stay_locations (or load existing benchmark JSON from scripts/run_benchmarks.sh --benchmark-json); horizontal log-scale bar chart of runtime per measure, annotate the speedup factor, light/dark-friendly palette.
- purpose: show "drop-in + any backend" visually, so that one input (pandas or polars) goes through the same fastmob call to the same-backend output
  preferred_source: diagram
  suggested_filename: features-backend-flow.png
  suggested_location: docs/src/assets/images/features/
  notebook_recipe: n/a (simple left-to-right diagram: pandas/polars DataFrame -> Narwhals -> Rust kernel (zero-copy Arrow buffers) -> result in caller's backend)

CROSS-PAGE NOTES: The header logo text "Fastkit-Mobility" conflicts with the "fastmob" site title on every page. The section-index "Overview" entry does not show in the left sidebar, so on index pages the sidebar has no "you are here" marker. In dark mode the footer "Made with Zensical" band is a mid-gray (#5a5a5a-ish) strip that clashes with the near-black page. The Features section's children (Benchmarks, Optimizations, Motifs) have no icons, while the Measures section uses lucide icons, so icon use is inconsistent.
