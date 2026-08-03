"""Compatibility imports for social-network methodology.

Social contact-network analysis now lives in :mod:`fastmob.social`.  Import
from that namespace in new code; these names remain available here so
existing collective-measure callers do not break.
"""

from fastmob.social.contact_network import (
    NetworkGraph,
    clustering_coefficients,
    co_presence_graph_from_visits,
    degree_preserving_random_graph,
    distribution_summary,
    graph_from_edges,
    infer_social_ties,
    random_persistence,
    topological_overlap,
)

__all__ = [
    "NetworkGraph",
    "clustering_coefficients",
    "co_presence_graph_from_visits",
    "degree_preserving_random_graph",
    "distribution_summary",
    "graph_from_edges",
    "infer_social_ties",
    "random_persistence",
    "topological_overlap",
]
