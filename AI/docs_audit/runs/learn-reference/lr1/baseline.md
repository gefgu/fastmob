# Baseline, snapshot lr1 (Stage 0)

- Source snapshot: branch `docs/critic-loop` at `c2d681f`, plus the uncommitted pilot work of this round.
- Pre-existing uncommitted changes owned by the user (not touched, not committed): Rust sources under `fastmob-core/` and `fastmob-py/`, `fastmob/__init__.py` (adds `radius_of_gyration_lazy`), `fastmob/measures/*`, `tests/correctness/test_public_api.py`, `tests/correctness/individual/test_radius_of_gyration_lazy.py`, `.gitignore` (`hidden/`), `docs/src/assets/logo_header.svg`.
- Dependencies: Python 3.10.12 (isolated runs), fastmob 0.3.2 (locally built wheel, includes the user's uncommitted Rust edits), pandas 2.3.3. The published 0.3.2 on PyPI does not contain `fastmob.data.load_geolife` / `load_brightkite`, which this round added.
- Public API inventory (`inventory.md`, `coverage.csv`): 212 unique canonical symbols, 184 functions and methods plus 28 classes before the two new loaders; 186 functions and methods after. 80% target: 149.
- Baseline example coverage: 0 notebooks followed the contract. Two hand-kept walkthroughs and two notebooks existed (a GeoLife-vs-trackintel benchmark that downloads 313 MB, and the mobility-laws notebook); neither had isolated-run evidence or a timing table.
- Baseline findings (from reading the pages, before editing):
  - `learn/analyze-a-simple-trajectory.md`, `clean-and-segment.md`, `staypoints-and-locations.md`, `individual-and-collective-measures.md` were hand-written snippets with no executed outputs.
  - `reference/preprocessing.md` documented 8 of 16 public preprocessing functions (`segment`, `simplify`, `identify_locations`, `predict_transport_mode`, `calculate_modal_split`, `create_activity_flag`, `cdr_to_trips_df`, `cdr_to_visitation_df` had no entry). 26 public symbols across the library had no catalogue anchor.
  - `reference/index.md` led with a toy walkthrough and a module diagram before the catalogue.
  - Learn index was a flat list of cards with no theme grouping.
- Evidence limits: Colab and Binder links resolve against `main`, which does not contain this work; hosted execution was not attempted. No human reader testing.
