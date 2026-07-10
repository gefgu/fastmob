---
name: TDD migration step completion status
description: Which steps of the fkmob TDD migration plan are done, ON HOLD, or blocked
type: project
---

Plan file: /home/gustavo/agents_transport_netmob/docs/features/fkmob-tdd-migration-plan.md

| Step | Feature | Status | Notes |
|------|---------|--------|-------|
| 1 | OD Matrix | DONE | fkmob/measures/od.py |
| 2 | Visitation Law Binning | ON HOLD | Explicitly deferred by plan |
| 3 | Power-Law Fitting | DONE | fkmob/measures/mobility_laws.py |
| 4 | Activity Transition Matrix | DONE | fkmob/measures/activity.py |
| 5 | Intermittance + degree-of-return | DONE | fkmob/measures/individual.py |
| 6 | Regularity | DONE | fkmob/measures/individual.py |
| 7 | Diversity (fast_diversity) | DONE | fkmob/measures/individual.py; requires pydivsufsort |
| 8 | Entropy + Predictability | DONE | fkmob/measures/individual.py |
| 9 | Motifs (no networkx) | DONE | fkmob/measures/motifs.py |

**Why:** Migration plan extracted mobility measures from mobility_analysis/ into fkmob/ with TDD. All steps 1-9 are now complete.

**How to apply:** When asked about migration progress, all steps through 9 are done. No steps remain unless new ones are added to the plan.

## Key algorithmic notes discovered during implementation

- Kontoyiannis entropy estimator has finite-sample variance — constant sequences do NOT return exactly 0.0 for small n (n=4 gives ~1.14 bits). The correct invariant is constant < varied, not constant == 0.
- fast_diversity (pydivsufsort) uses sum((n-sa)-lcp), NOT total-sum(lcp). Constant sequences still have diversity > 0 for small n (e.g., n=4 gives 0.4).
- Motif library: 17 canonical directed graphs; brute-force isomorphism (permutations) is fast enough up to n=6 nodes.
- pyproject.toml had a duplicate [build-system] section (fixed 2026-04-18).
- 5 pre-existing test failures: polars-to-pandas requires pyarrow which is not installed in .venv. These are in test_jump_lengths.py and test_radius_of_gyration.py.
