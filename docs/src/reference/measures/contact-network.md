---
icon: lucide/network
---

# RECAST Social Relationships

`fastmob.social` implements RECAST (Random rElationship ClASsifier sTrategy)
for temporal physical contacts inferred from globally assigned staypoints.

`recast_from_staypoints` uses UTC calendar days as its event snapshots. This
matches RECAST's human-mobility setting and avoids exposing an arbitrary time
delta as a social parameter. A daily edge is created only when two users' stay
intervals overlap at a shared global location for at least
`min_minutes_for_encounter` (default: 5 minutes). It then uses the paper's
T-RND: each daily event graph is independently sampled with the degree-product
RND probability. This preserves degrees in expectation rather than exactly in
every replica. The pooled randomized persistence and topological-overlap
distributions define both thresholds at `p_rnd`.

Every observed aggregate edge is returned as one of:

- `Friends`: social persistence and social overlap.
- `Bridges`: social persistence and random-like overlap.
- `Acquaintances`: random-like persistence and social overlap.
- `Random`: random-like persistence and overlap.

The defaults reproduce the paper's five randomized replicas, while keeping
the graph construction, randomization, and metric calculations in Rust.

RECAST parallelizes independent `(day, location)` contact buckets and T-RND
replica/window work through Rayon. The public API intentionally has no worker
parameter: use the `RAYON_NUM_THREADS` environment variable to bound the
process-wide Rayon pool when coordinating CPU use with other workloads.

`validate_recast_from_staypoints` adds the validation artifacts from the paper
without expanding into its application-specific routing experiments: pooled
null distributions for the Figure 4 CCDFs, full cumulative clustering curves
(Figure 3), and random-edge-only clustering curves (Figure 8).  The dedicated
`temporal_graph_from_staypoints`, `rnd`, and `t_rnd` APIs expose the underlying
Arrow-backed event graphs for independent inspection.

## Raw co-presence networks

`co_presence_graph_from_staypoints` returns raw contacts from the same daily
event graphs RECAST uses. It applies globally assigned locations, stay
interval overlap, and `min_minutes_for_encounter`, and returns both the daily
`RecastTemporalGraph` and its aggregate `NetworkGraph` with per-edge daily
persistence. Groups are not capped.

The raw contact result does not estimate random thresholds or assign
relationship classes. Those steps belong to RECAST. Generic graph diagnostics
such as clustering coefficients and topological overlap remain available for
the aggregate graph.

---

::: fastmob.social.RecastClass
    options:
      show_source: false

---

::: fastmob.social.RecastResult
    options:
      show_source: false

---

::: fastmob.social.RecastTemporalGraph
    options:
      show_source: false

---

::: fastmob.social.RecastValidation
    options:
      show_source: false

---

::: fastmob.social.recast_from_staypoints
    options:
      show_source: false

---

::: fastmob.social.validate_recast_from_staypoints
    options:
      show_source: false

---

::: fastmob.social.NetworkGraph
    options:
      show_source: false

---

::: fastmob.social.ContactNetworkResult
    options:
      show_source: false

---

::: fastmob.social.co_presence_graph_from_staypoints
    options:
      show_source: false

---

::: fastmob.social.clustering_coefficients
    options:
      show_source: false

---

::: fastmob.social.topological_overlap
    options:
      show_source: false
