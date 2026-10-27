SNAPSHOT: lr1  ROUND: 1  SCOPE: pilot
VERDICT: REVISE

GATES:
- API catalogue/details 100% (campaign-wide): PENDING. Pilot slice: FAIL (one gap). Preprocessing: 16/16 catalogue rows plus detail entries. Individual measures: 33 `:::` detail entries but only 32 catalogue rows. `work_location` has a detail entry (docs/src/reference/measures/individual.md:192) but no row in the api-table block, so it also has no Example link, even though coverage.csv marks it covered. All catalogue anchors and Example links on both pilot pages resolve in ./site; I checked them with a script against the rendered ids. The ledger itself is wrong here: inventory.md says "Symbols without a catalogue anchor in the built site: 214". That check uses canonical module paths (for example `fastmob.measures.individual.jump_lengths.jump_lengths` as a module path), while the site uses public ids. A lenient alias/short-name check finds only 14 symbols missing anchors, all in `fastmob.social`, which is outside the pilot.
- ≥80% executed example coverage: PENDING. The ledger reports 24/186 = 12.9%. The pilot count is slightly overstated: `recency_rank` is called in cell-18 of measure-individual-mobility, but its result is never displayed or interpreted. The contract requires the result to be "inspected and explained".
- Isolated Run All, complete setup, honest timing: FAIL for clean-and-detect-stops and measure-individual-mobility. PASS with caveats for analyze-a-simple-trajectory.
  - Evidence: execution/*.status shows exit=0 for all three, wall 22/30/30 s, Python 3.10.12. Saved outputs in the canonical .ipynb match the executed copies cell for cell, except the cleared pip cell.
  - The runs used a locally built 0.3.2 wheel. The notebooks themselves run `%pip install -q fastmob`, which today resolves to a PyPI 0.3.2 that has no `fastmob.data.load_geolife` or `load_brightkite`. The contract requires an explicit reproducible pin when an API is unreleased. The notebooks have none and do not disclose the mismatch: a grep finds no "release", "unreleased" or "wheel" text in the notebook prose.
  - The timing text contains invented estimates, which the contract forbids. See LR1-05.
- Colab/Binder launch and portability: FAIL.
  - Launch badges point at `gefgu/fastmob/blob/main/docs/src/learn/<nb>.ipynb` (scripts/build_notebook_docs.py:29-32).
  - `git show main:docs/src/learn/clean-and-detect-stops.ipynb` fails. On the current branch, `git status` lists all three pilot notebooks as untracked (`??`), so pushing the branch would not publish them either.
  - binder/requirements.txt has unpinned `fastmob`, `pandas`, `matplotlib`, which resolves to the release without the loaders.
  - Hosted execution, reported separately: Colab not attempted; Binder not run. Neither is claimed.
- Conversion, docs tests, strict build, anchors, light/dark/mobile: FAIL, because of mobile.
  - Build (strict) and tests/docs (8 passed): reported in the brief; I did not rerun them.
  - Anchors: PASS for the pilot pages, the notebook "Related API" links, and the reference Example links (script check against ./site).
  - Mobile: FAIL. screens/learn_measure-individual-mobility/mobile-full.png is 608 px wide at a 390 px viewport. The `describe()` output tables overflow the page, starting at y≈3230. See LR1-07.

COVERAGE: 24/186 = 12.9% as claimed. Pilot-strict count is 23/186 = 12.4%, because recency_rank's output is not inspected. Catalogue: 200/214 symbols have a rendered anchor by alias (14 `fastmob.social` gaps, outside the pilot). Pilot catalogue rows: 48/49 (work_location missing). Inventory "214 missing anchors" is a faulty check, not evidence.

SCORES: comprehension=6 navigation=7 scannability=7 role=7 examples=5 visual=6
MEAN: 6.3

EVIDENCE:
- comprehension (6): Every notebook opens with Question / You will get / Prerequisites / Data, and each output cell is followed by an interpretation in plain words. This is a strong pattern. But several statements contradict the library:
  - `maximum_distance` is defined as "the longest distance between any two of a person's records/check-ins". The docstring defines it as the max of consecutive jumps, and Alice's printed 1.507 km equals her second jump; her first-to-third point distance is about 2.5 km.
  - The stop and cluster mechanisms are mis-described (LR1-03).
  - The home, work and k-RoG readings are mis-described (LR1-04).
  - A novice cannot detect any of these from the page.
- navigation (7): Learn overview → "2. Preparing trajectories" → "Clean noisy GPS data and find stops" is two selections. Sidebar labels match the overview headings (screens/learn/desktop-light-fold.png). Each notebook ends with Next steps and Related API, and those links resolve. Weaknesses:
  - The Learn sidebar does not mark which entries are notebooks.
  - Function detail entries carry no link to their worked example; only the catalogue row has one.
  - The individual page's intro link says "Measure people and populations" while the page title is "Measuring people and populations".
- scannability (7): Reference overview puts the module catalogue first, with a one-line purpose and two example symbols per module (screens/reference/desktop-dark-fold.png). Preprocessing opens with the API/Description/Example table. On Individual measures, though:
  - a 5-row "Question / Representative APIs" table pushes the Catalogue heading below the fold (screens/reference_measures_individual/desktop-light-fold.png);
  - the right TOC lists many symbols twice ("distance_straight_line / distance_straight_line", "diversity / diversity", "frequency_rank / frequency_rank");
  - the Preprocessing table follows neither alphabetical nor workflow order (filter, compress, stay_locations, cluster, latlng_to_h3, … segment, simplify).
- role (7): Concept pages (clean-and-segment, staypoints-and-locations) explain choices and link out. Reference pages specify and link back to Learn. The reference overview keeps its module diagram at the end, not ahead of the catalogue. Two defects:
  - individual-and-collective-measures.md is only a lookup table of names and units. It defines no measure (radius of gyration, entropy, k-RoG) and gives no research meaning, so the concept role falls entirely on the notebook.
  - reference/measures/individual.md repeats a question-to-API table that belongs in Learn.
- examples (5): The GeoLife and Brightkite workflows are realistic and coherent. Method choices are explicit: the 200 km/h threshold with a 60/120/200/400 sensitivity sweep; stays run with and without `no_data_for_minutes`; null-island removal with the count printed (8,321 rows). Every run ends with a measured timing table. They still fall short:
  - Neither real-data notebook can run from its launch link today (LR1-02).
  - The prose gives unmeasured runtimes (LR1-05).
  - Several interpretations are wrong (LR1-01/03/04).
- visual (6): The layout is calm and dark mode is legible (screens/learn_clean-and-detect-stops/desktop-dark-full.png; the matplotlib figure renders on white and stays readable). Problems:
  - Notebook DataFrame outputs render as raw, borderless-padding HTML tables with harsh 1 px grids and a misaligned index header. In the cluster output, "uid" sits on its own row.
  - At 390 px, the measure notebook's describe() tables widen the page to 608 px.
  - On mobile, the Preprocessing catalogue squeezes Description to one word per line and pushes the Example column off-screen (screens/reference_preprocessing/mobile-fold.png).
  - The first screen of every notebook is the timing boilerplate code, not content or output (screens/learn_analyze-a-simple-trajectory/desktop-light-fold.png).

TASK TRACES:
1. Clean noisy GPS data → stops.
   - Path: Learn overview → sidebar "2. Preparing trajectories" (1) → "Cleaning and segmentation" (2) concept page → "Clean noisy GPS data and find stops" (3). The notebook is also reachable directly at selection 2 from the overview list. Assumptions (UTC, WGS 84, thresholds) are stated in the notebook and in the concept page's "Assumptions to state in your methods".
   - Launch: the Colab badge points to a `main` path where the notebook does not exist (dead end, verified via git). If the reader copies the notebook, `%pip install fastmob` installs 0.3.2 and `from fastmob.data import load_geolife` fails at cell 4 (inferred from the brief's disclosure; not executed by me).
   - Interpretation: present after every step. Continuing to stop detection happens in the same notebook ("Detect stops"). The "Stops and locations" concept page comes after the notebook, but cleaning page's Next steps says "Then detect stops: [Stops and locations]", which is a concept page with no executable stop detection: an ambiguous choice.
   - Novice: blocked at launch, and would take "stays are cut where the data stop" at face value. Experienced reader: would spot the unpinned install, and may notice that the stop count fell from 205 to 72 rather than rising, as cutting would predict.
2. Measure with units and meaning → run → full parameters.
   - Path: Learn → "3. Measuring mobility" (1) → "Measuring people and populations" (2): a question/measure/unit table gives km, seconds and bits. Then "Measure individual mobility" (3).
   - Units are explicit, including the UTC caveat for home/work. Research meaning is wrong for maximum_distance and misleading for k_radius_of_gyration, home_location and work_location.
   - Run: same launch failure as task 1.
   - Parameters: the Related API links use absolute production URLs (https://gefgu.github.io/fastmob/reference/measures/individual/#…). They resolve in the local ./site, but the deployed site may not yet carry this build (UNVERIFIED). In the local build they land on the module heading just above the function entry.
3. Reference symbol lookup.
   - Path: Reference → Measures (1) → Individual (2) → scroll past the question table to the Catalogue → `maximum_distance` row → detail entry (in-page anchor). The detail says "maximum distance covered in a single movement" while its Example link is titled "Find the farthest point". The two contradict each other.
   - From the detail entry there is no example link back (rendered HTML for jump_lengths contains no /learn/ link). The reader must return to the table.
   - `work_location` has no catalogue row at all.
   - Search: `stay_locations` → reference/preprocessing/#fastmob.preprocessing.stay_locations, plus the TrajDataFrame method (search.json). One search plus one selection: PASS.
   - Outside the popular modules: Reference overview → Privacy → `location_risk` anchor exists, but the privacy page has no catalogue table (campaign scope, PENDING).
4. From a notebook alone.
   - Dependencies: shown (`%pip install -q fastmob pandas matplotlib`), but unpinned and insufficient.
   - Data provenance: cited with links. GeoLife says "check terms of use". Brightkite has no license statement.
   - Run order is linear. Expected results are stated up front. Limitations are listed.
   - Timing: the final cell separates setup / download and load / analysis / plotting / other / total (for example GeoLife: 13.93 / 8.48 / 0.34 / 0.19 / 1.23 / 24.17 s). The header prose still says "About three minutes in Colab, mostly the data download", which contradicts both the measurement and the setup-versus-download split.
   - Cache status is stated as a conditional, not measured.

TOP ISSUES (max 7):
- ID: LR1-01  PRIORITY: P0  DIMENSION/GATE: comprehension / scientific correctness
  OBSERVED: `maximum_distance` is described wrongly in three places:
    - analyze-a-simple-trajectory cell-12: "the longest distance between any two of a person's records"; heading "Find the farthest point"; cell-0 "the farthest they get from it"; cell-14 "the maximum distance the extreme".
    - measure-individual-mobility cell-6: "the longest distance between any two check-ins".
    - docs/reference_examples.json gives the reference Example title "Find the farthest point".
  The docstring (fastmob/measures/individual/maximum_distance.py:29-41) defines it as the largest distance between consecutive fixes. The saved output agrees with the docstring: Alice 1.506992 equals jumps[1]; Bob 6.286276 equals jumps[0].
  READER IMPACT: Researchers will report the wrong quantity, a largest single jump, as an individual's spatial extent, and will compare it with the radius of gyration as if it were a diameter. A misleading scientific result.
  FIX: Rename the section to "Find the longest single jump". Define it as the largest straight-line distance between consecutive records. Note that `max_distance_from_home` or a pairwise extent answers the "how far apart" question. Edit cell-0, cell-12 and cell-14 in analyze-a-simple-trajectory.ipynb and cell-6/cell-8 in measure-individual-mobility.ipynb. Change the reference_examples.json title, then regenerate the tables.
  VERIFY: grep finds no "any two" or "farthest point" near maximum_distance in docs/src/learn. The prose numbers match max(jump_lengths) for each user. The catalogue Example title matches the docstring.
- ID: LR1-02  PRIORITY: P0  DIMENSION/GATE: notebook execution / Colab-Binder portability
  OBSERVED:
    - The notebooks call `fastmob.data.load_geolife` and `load_brightkite` after an unpinned `%pip install -q fastmob`, which resolves to PyPI 0.3.2, a release without these loaders.
    - binder/requirements.txt is unpinned.
    - The launch badges target `main`, where `git show main:docs/src/learn/clean-and-detect-stops.ipynb` fails. All three notebooks are untracked (`??`) even on docs/critic-loop.
    - The isolated runs used a locally built wheel, not what the notebook declares.
    - None of this is disclosed in the notebooks.
  READER IMPACT: Every reader who clicks "Open in Colab" or "launch binder" on the two real-data notebooks hits a 404 or an AttributeError. The core "launch the example" journey is blocked.
  FIX: Either release the loaders, then pin `fastmob>=<new version>` in the pip cell and in binder/requirements.txt, or pin an installable git revision (`pip install "fastmob @ git+https://github.com/gefgu/fastmob@<sha>"`, with build requirements disclosed) in both places. Commit the notebooks and point the badges at a ref that contains them (scripts/build_notebook_docs.py:29-32). Until then, add a visible admonition under the badges stating the version requirement.
  VERIFY: A fresh venv that runs only the notebook's own pip cell (no preinstalled wheel) executes all cells with exit 0. `git ls-tree <badge ref> docs/src/learn/` lists the three notebooks. Binder requirements resolve to a version that exports the loaders.
- ID: LR1-03  PRIORITY: P1  DIMENSION/GATE: comprehension / examples
  OBSERVED: Stop and place mechanisms are mis-described in clean-and-detect-stops and staypoints-and-locations.md.
    - (a) Cell-16/18 say `no_data_for_minutes` "ends a stay" and "stays are cut where the data stop, so there are fewer stops (72 instead of 205)". In fastmob-core/src/preprocessing/stay_locations.rs:113-120, a gap resets the anchor and drops the in-progress candidate stay; it is not closed or truncated. That is why the count falls.
    - (b) Cell-19 says "`cluster_radius_km` sets how close stops must be to count as the same place", and the concept page says the cluster radius "decides whether two nearby shops count as one location". `cluster` (fastmob/preprocessing/_cluster.py docstring) assigns stops to H3 cells and joins edge-adjacent occupied cells; `cluster_radius_km` only selects a grid resolution.
    - (c) Cell-21 mentions label -1 "if present", which cannot occur with the default `min_samples=1`.
  READER IMPACT: Readers will write methods sections claiming truncated stays and distance-threshold clustering, and will misjudge which stays they lose and how places merge (grid-boundary effects).
  FIX: In the notebook cell-16/18, say that a stay spanning a silence longer than the limit is discarded, and that the remaining 72 stops are those fully covered by data. In cell-19 and staypoints-and-locations.md "Choices", describe H3-grid clustering with `cluster_radius_km` as an approximate scale, mention `h3_resolution`, and note the boundary effect. Drop or qualify the -1 remark.
  VERIFY: The prose matches the docstrings and the Rust behaviour. A reviewer reading only the notebook can predict that adding the silence limit lowers the stop count.
- ID: LR1-04  PRIORITY: P1  DIMENSION/GATE: comprehension / coverage evidence
  OBSERVED: Several readings in measure-individual-mobility do not match the functions.
    - Cell-25: "Someone who never checks in at home has no home here". home_location falls back to the most-visited place across all hours (docstring), which is why 306/306 users get a home.
    - Cell-20: "`work_location` uses 8 to 18". The function also restricts to Monday–Friday (work_location.py:30-35, weekday<5), and the weekday is computed in UTC too.
    - Cell-8: k-RoG with k=2 "shows that most check-ins happen around two places". A small k-RoG only shows the two most-visited places are close together; user 0's top place gets 10% of visits.
    - Cell-18/19: `recency_rank` output is never displayed or interpreted, yet coverage.csv counts it as covered.
  READER IMPACT: Readers would treat fallback homes as night-time evidence, misread the work definition, and over-interpret k-RoG. The coverage ledger is inflated by one.
  FIX:
    - State the home fallback, and show how many users had no night check-ins.
    - Add "Monday–Friday" to the work definition.
    - Rephrase k-RoG as "the two most-visited places lie within a few km of each other", or add the returners/explorers ratio rg_k/rg with a citation.
    - Display one user's top recency ranks and explain them, or mark recency_rank uncovered.
  VERIFY: Each claim cites a value printed in a cell. coverage.csv is consistent with the inspected outputs.
- ID: LR1-05  PRIORITY: P1  DIMENSION/GATE: notebook timing honesty / examples
  OBSERVED:
    - Header estimates are not measurements: "runs top to bottom in about a minute" (analyze); "About three minutes in Colab, mostly the data download" (clean); "About three minutes in Colab" (measure). Measured totals are 16.18, 24.17 and 24.75 s locally, and setup exceeds download in both real-data runs (13.93 vs 8.48; 11.69 vs 11.04).
    - analyze-a-simple-trajectory has no analysis phase, so the measures fall into an "other" bucket.
    - The "other" row is labelled "kernel start, plotting overhead". Kernel start precedes `_NOTEBOOK_START` and cannot be included.
    - Cache status is a conditional sentence, not a detected state.
    - Setup was timed against a preinstalled local wheel.
  READER IMPACT: The contract forbids invented runtime claims. Readers cannot tell what a cold Colab run costs, and the phase table is mislabelled.
  FIX: Replace the header estimates with the measured values and environment ("measured 24 s locally, Python 3.10, 2026-10-27; Colab unmeasured"). Wrap the measure calls in `phase("analysis")`. Relabel "other" as "untimed cells and display". Have the loader or the cell report whether the cache file existed before the run. Re-measure setup after LR1-02 with a true cold install.
  VERIFY: The header numbers equal the final timing-cell output. Every code cell that does work is inside a named phase. Cache status prints "cold" or "cached" from a check.
- ID: LR1-06  PRIORITY: P1  DIMENSION/GATE: navigation / catalogue gate
  OBSERVED:
    - `work_location` is missing from the Individual catalogue table (33 details vs 32 rows), so a covered function has no reference-to-example link.
    - Detail entries have no worked-example link; the rendered jump_lengths entry contains no /learn/ href. The task "reach full details, follow its worked-example link" fails from the detail entry.
    - The inventory's anchor check reports all 214 symbols missing, a false negative that makes the gate ledger unusable.
    - On Individual measures, the Catalogue sits below a question table, and the TOC duplicates module and function entries.
  READER IMPACT: Lookup readers landing on a function from search or a Learn link cannot reach its example. The gate evidence cannot be trusted.
  FIX: Make scripts/update_reference_tables.py include `work_location`, which is probably skipped because of how it is exported. Inject a "Worked example: <link>" line into each detail entry from docs/reference_examples.json (mkdocstrings template or docstring "See also"). Fix inventory anchor detection to use public alias ids. Move the question table below the Catalogue, or into the Learn concept page. Suppress the module-level headings that create duplicate TOC entries.
  VERIFY: For every covered symbol, both the table row and the rendered detail entry link to the example anchor. Inventory reports 0 missing anchors for the pilot modules. Individual page TOC has one entry per function.
- ID: LR1-07  PRIORITY: P1  DIMENSION/GATE: visual / mobile gate
  OBSERVED:
    - At 390 px, learn/measure-individual-mobility renders 608 px wide (mobile-full.png). The describe() and network tables are unwrapped raw HTML tables that overflow the page; the code blocks clip beside them.
    - All notebook DataFrame outputs use unstyled grid tables with no cell padding and a misaligned index header (crop of clean-and-detect-stops dark full, cluster table).
    - The Preprocessing catalogue on mobile squeezes Description to one word per line and hides the Example column (reference_preprocessing/mobile-fold.png).
  READER IMPACT: Horizontal page scrolling on phones. Numeric results, the main evidence of the notebooks, are hard to read in both themes.
  FIX: In the notebook converter or the shared CSS, wrap output tables in a horizontally scrollable container and style them like the site tables (padding, right-aligned numbers, theme borders). Give the catalogue a mobile layout: stack Example under Description, or set a minimum column width.
  VERIFY: Every mobile-full screenshot is 390 px wide. Output tables scroll inside their container. The catalogue Example link is visible without horizontal scrolling at 390 px.

PREVIOUS ISSUES: none (first review of this campaign).

PRESERVE:
- The notebook opening contract (Question / You will get / Prerequisites / Data), with an interpretation paragraph after each output and an "Interpretation and limits" section. For example, the GeoLife subset caveat "3 of 182 users" and the Brightkite non-random-sample note.
- Explicit method-choice demonstrations backed by real outputs: the speed-threshold sweep {60: 5640, 120: 449, 200: 193, 400: 58}, the with/without silence-limit comparison, and the null-island removal with a printed count.
- The Reference overview's module-first catalogue, and the generated API/Description/Example tables with resolving anchors, plus symbol-level search results.

UNCERTAINTIES:
- I did not rerun the build, tests or notebooks; I relied on the supplied status files and the saved ./site.
- The isolated execution logs are 3-line nbconvert logs with no pip output. I cannot see how the local wheel was injected, or whether setup timing reflects any real install.
- Colab and Binder hosted execution are unverified, as declared. I did not check whether the production site at gefgu.github.io has the anchors used by the notebooks' absolute "Related API" links.
- The failure on PyPI 0.3.2 is inferred from the brief's disclosure, not executed by me.
- Campaign-wide gates (80% coverage, 100% catalogue, all-notebook evidence) are PENDING, by pilot scope.
