# RECAST Social Relationships

`fastmob.social` implements RECAST (Random rElationship ClASsifier sTrategy)
for temporal physical contacts inferred from globally assigned staypoints.

`recast_from_staypoints` uses UTC calendar days as its event snapshots. This
matches RECAST's human-mobility setting and avoids exposing an arbitrary time
delta as a social parameter. A daily edge is created only when two users' stay
intervals overlap at a shared global location for at least
`min_minutes_for_encounter` (default: 5 minutes). It then uses T-RND: each
daily event graph is independently randomized by exact degree-preserving
double-edge swaps. The pooled randomized persistence and topological-overlap
distributions define both thresholds at `p_rnd`.

Every observed aggregate edge is returned as one of:

- `Friends`: social persistence and social overlap.
- `Bridges`: social persistence and random-like overlap.
- `Acquaintances`: random-like persistence and social overlap.
- `Random`: random-like persistence and overlap.

The defaults reproduce the paper's five randomized replicas, while keeping
the graph construction, randomization, and metric calculations in Rust.

---

::: fastmob.social.RecastClass
    options:
      show_source: false

---

::: fastmob.social.RecastResult
    options:
      show_source: false

---

::: fastmob.social.recast_from_staypoints
    options:
      show_source: false
