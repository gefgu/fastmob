"""Social-network analysis and social-tie inference.

This namespace contains methodology that derives contact networks and social
ties from mobility observations.  It is deliberately separate from
``fastmob.measures.collective`` so social analysis has a stable, dedicated
home.
"""

from .contact_network import (
    NetworkGraph,
    clustering_coefficients,
    co_presence_graph_from_visits,
    co_presence_graph_from_staypoints,
    degree_preserving_random_graph,
    distribution_summary,
    graph_from_edges,
    infer_social_ties,
    random_persistence,
    safe_wasserstein,
    topological_overlap,
)

__all__ = [
    "NetworkGraph",
    "clustering_coefficients",
    "co_presence_graph_from_visits",
    "co_presence_graph_from_staypoints",
    "degree_preserving_random_graph",
    "distribution_summary",
    "graph_from_edges",
    "infer_social_ties",
    "random_persistence",
    "safe_wasserstein",
    "topological_overlap",
]
