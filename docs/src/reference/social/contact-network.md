# Social Contact Network

Build a co-presence (contact) network from raw mobility data, validate it
against a degree-preserving random-graph null model, and infer which
co-presence pairs represent genuine social ties rather than incidental
contact.

| API | Description |
| --- | --- |
| [`NetworkGraph`](#fastmob.social.NetworkGraph) | Undirected graph as a plain edge list. |
| [`co_presence_graph_from_visits`](#fastmob.social.co_presence_graph_from_visits) | Build a co-presence graph directly from raw mobility data. |
| [`graph_from_edges`](#fastmob.social.graph_from_edges) | Build a graph from a small/moderate edge collection. |
| [`clustering_coefficients`](#fastmob.social.clustering_coefficients) | Per-node clustering coefficient. |
| [`topological_overlap`](#fastmob.social.topological_overlap) | Per-edge topological overlap (Jaccard similarity of neighborhoods). |
| [`degree_preserving_random_graph`](#fastmob.social.degree_preserving_random_graph) | Chung-Lu-style degree-preserving null model. |
| [`random_persistence`](#fastmob.social.random_persistence) | Synthetic per-edge persistence for a random null-model graph. |
| [`distribution_summary`](#fastmob.social.distribution_summary) | Summary stats of a metric distribution. |
| [`infer_social_ties`](#fastmob.social.infer_social_ties) | Infer genuine social ties from a co-presence graph. |

## Social-tie inference methodology

`infer_social_ties` promotes a co-presence edge to an inferred social tie
when its persistence (how often the pair co-occurs) is high enough *and*
its topological overlap (Jaccard similarity of neighbor sets) is
significantly higher than random chance would produce -- a
Fournet & Barrat-style random-graph-baseline significance test: sample many
random node pairs in the same graph, take the `(1 - random_chance_probability)`
quantile of their topological overlap as the null-model threshold, and only
promote pairs whose overlap clears it.

This method has been validated against real observed co-presence data in a
production simulation pipeline: it produced a plausible ~103 mean friendship
degree on a dense real-world urban dataset at `random_chance_probability=1e-3`.
It can also degenerate on coarser, fixed-grid location data -- the random
baseline saturates when co-presence is too dense to separate "social" from
"random" contact at that spatial resolution, collapsing toward either an
empty or a near-complete inferred graph. Always inspect the output graph's
mean degree against your domain's expectations when applying this to a new
dataset or spatial resolution, rather than assuming the default parameters
transfer.

---

::: fastmob.social.NetworkGraph
    options:
      show_source: false

---

::: fastmob.social.co_presence_graph_from_visits
    options:
      show_source: false

---

::: fastmob.social.graph_from_edges
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

---

::: fastmob.social.degree_preserving_random_graph
    options:
      show_source: false

---

::: fastmob.social.random_persistence
    options:
      show_source: false

---

::: fastmob.social.distribution_summary
    options:
      show_source: false

---

::: fastmob.social.infer_social_ties
    options:
      show_source: false
