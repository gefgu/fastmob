"""Social-network analysis and social-tie inference.

This namespace contains methodology that derives contact networks and social
ties from mobility observations.  It is deliberately separate from
``fastmob.measures.collective`` so social analysis has a stable, dedicated
home.
"""

from .contact_network import (
    ContactNetworkResult,
    NetworkGraph,
    clustering_coefficients,
    co_presence_graph_from_staypoints,
    degree_preserving_random_graph,
    distribution_summary,
    graph_from_edges,
    random_persistence,
    safe_wasserstein,
    topological_overlap,
)
from .recast import (
    RecastClass,
    RecastClusteringComparison,
    RecastEventGraph,
    RecastResult,
    RecastTemporalGraph,
    RecastValidation,
    recast_from_staypoints,
    rnd,
    t_rnd,
    temporal_graph_from_staypoints,
    validate_recast_from_staypoints,
)

__all__ = [
    "ContactNetworkResult",
    "NetworkGraph",
    "RecastClass",
    "RecastClusteringComparison",
    "RecastEventGraph",
    "RecastResult",
    "RecastTemporalGraph",
    "RecastValidation",
    "clustering_coefficients",
    "co_presence_graph_from_staypoints",
    "degree_preserving_random_graph",
    "distribution_summary",
    "graph_from_edges",
    "random_persistence",
    "recast_from_staypoints",
    "rnd",
    "safe_wasserstein",
    "t_rnd",
    "temporal_graph_from_staypoints",
    "topological_overlap",
    "validate_recast_from_staypoints",
]
