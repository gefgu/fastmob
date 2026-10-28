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

## Catalogue

<!-- api-table:start -->
**Build contact networks**

| API | Description | Example |
| --- | --- | --- |
| [`co_presence_graph_from_staypoints`](#fastmob.social.co_presence_graph_from_staypoints) | Build the aggregate contact network from the co-presence of users at the same location. | [Build the contact network](../../learn/contact-networks.md#build-the-contact-network) |
| [`temporal_graph_from_staypoints`](#fastmob.social.temporal_graph_from_staypoints) | Build the daily contact graphs from staypoints and a global location catalogue, without classifying. | [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation) |
| [`graph_from_edges`](#fastmob.social.graph_from_edges) | Build a normalized undirected graph from a list of edges. | [A random baseline](../../learn/contact-networks.md#a-random-baseline) |
| [`NetworkGraph`](#fastmob.social.NetworkGraph) | An undirected graph stored as sorted, dense integer edge arrays. |  |
| [`NetworkGraph.degrees`](#fastmob.social.NetworkGraph.degrees) | Return the number of contacts of each node, as an integer array of length `node_count`. |  |
| [`ContactNetworkResult`](#fastmob.social.ContactNetworkResult) | Daily RECAST event graphs plus their raw aggregate contact network. |  |

**Network statistics**

| API | Description | Example |
| --- | --- | --- |
| [`clustering_coefficients`](#fastmob.social.clustering_coefficients) | Return the local clustering coefficient of every node. | [Structure](../../learn/contact-networks.md#structure-clustering-and-neighbourhood-overlap) |
| [`topological_overlap`](#fastmob.social.topological_overlap) | Return the neighbourhood overlap of every edge. | [Structure](../../learn/contact-networks.md#structure-clustering-and-neighbourhood-overlap) |
| [`distribution_summary`](#fastmob.social.distribution_summary) | Summarize a sample with stable statistics after dropping non-finite values. | [Structure](../../learn/contact-networks.md#structure-clustering-and-neighbourhood-overlap) |
| [`safe_wasserstein`](#fastmob.social.safe_wasserstein) | Return the 1-D Wasserstein distance between two samples, or `None` when it is undefined. | [A random baseline](../../learn/contact-networks.md#a-random-baseline) |

**Null models**

| API | Description | Example |
| --- | --- | --- |
| [`degree_preserving_random_graph`](#fastmob.social.degree_preserving_random_graph) | Sample a random graph whose node degrees match the given degrees in expectation (Chung-Lu model). | [A random baseline](../../learn/contact-networks.md#a-random-baseline) |
| [`random_persistence`](#fastmob.social.random_persistence) | Sample the persistence that edges of a degree-preserving random baseline would have. | [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation) |
| [`rnd`](#fastmob.social.rnd) | Generate one randomized copy of a daily contact graph (the paper's RND null model). | [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation) |
| [`t_rnd`](#fastmob.social.t_rnd) | Generate independent randomized replicas of every daily contact graph (the T-RND null model). | [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation) |

**Recast: re-timing a contact network**

| API | Description | Example |
| --- | --- | --- |
| [`recast_from_staypoints`](#fastmob.social.recast_from_staypoints) | Classify contacts as Friends, Bridges, Acquaintances or Random (the RECAST method). | [Classify the contacts](../../learn/contact-networks.md#classify-the-contacts-friends-bridges-acquaintances-random) |
| [`validate_recast_from_staypoints`](#fastmob.social.validate_recast_from_staypoints) | Run the RECAST classifier and return it together with the validation distributions. | [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation) |
| [`RecastClass`](#fastmob.social.RecastClass) | Relationship classes defined by RECAST Algorithm 1. |  |
| [`RecastResult`](#fastmob.social.RecastResult) | Classification arrays aligned one-for-one with aggregate observed edges. |  |
| [`RecastResult.mask`](#fastmob.social.RecastResult.mask) | Return a boolean array that is true for the contacts classified as `relationship`. |  |
| [`RecastTemporalGraph`](#fastmob.social.RecastTemporalGraph) | Packed Arrow event graphs; offsets delimit the edges for each day. |  |
| [`RecastTemporalGraph.event`](#fastmob.social.RecastTemporalGraph.event) | Return the contact graph of time step `index` (one day), raising `IndexError` if out of range. |  |
| [`RecastEventGraph`](#fastmob.social.RecastEventGraph) | One Arrow-backed, undirected event graph with integer node ids. |  |
| [`RecastClusteringComparison`](#fastmob.social.RecastClusteringComparison) | Cumulative clustering coefficients for observed G_t and T-RND G^R_t. |  |
| [`RecastValidation`](#fastmob.social.RecastValidation) | Section 4/5 RECAST validation artifacts, all in Arrow arrays. |  |
<!-- api-table:end -->

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

::: fastmob.social.co_presence_graph_from_staypoints
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Build the contact network](../../learn/contact-networks.md#build-the-contact-network)

---

::: fastmob.social.temporal_graph_from_staypoints
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation)

---

::: fastmob.social.graph_from_edges
    options:
      show_source: false

<!-- example-link -->**Worked example:** [A random baseline](../../learn/contact-networks.md#a-random-baseline)

---

::: fastmob.social.NetworkGraph
    options:
      show_source: false

---

::: fastmob.social.ContactNetworkResult
    options:
      show_source: false

---

::: fastmob.social.clustering_coefficients
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Structure](../../learn/contact-networks.md#structure-clustering-and-neighbourhood-overlap)

---

::: fastmob.social.topological_overlap
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Structure](../../learn/contact-networks.md#structure-clustering-and-neighbourhood-overlap)

---

::: fastmob.social.distribution_summary
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Structure](../../learn/contact-networks.md#structure-clustering-and-neighbourhood-overlap)

---

::: fastmob.social.safe_wasserstein
    options:
      show_source: false

<!-- example-link -->**Worked example:** [A random baseline](../../learn/contact-networks.md#a-random-baseline)

---

::: fastmob.social.degree_preserving_random_graph
    options:
      show_source: false

<!-- example-link -->**Worked example:** [A random baseline](../../learn/contact-networks.md#a-random-baseline)

---

::: fastmob.social.random_persistence
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation)

---

::: fastmob.social.rnd
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation)

---

::: fastmob.social.t_rnd
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation)

---

::: fastmob.social.recast_from_staypoints
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Classify the contacts](../../learn/contact-networks.md#classify-the-contacts-friends-bridges-acquaintances-random)

---

::: fastmob.social.validate_recast_from_staypoints
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Look inside](../../learn/contact-networks.md#look-inside-daily-graphs-and-their-randomisation)

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

::: fastmob.social.RecastEventGraph
    options:
      show_source: false

---

::: fastmob.social.RecastClusteringComparison
    options:
      show_source: false

---

::: fastmob.social.RecastValidation
    options:
      show_source: false
