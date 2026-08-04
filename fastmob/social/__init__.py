"""Social-network analysis and social-tie inference.

This namespace contains methodology that derives contact networks and social
ties from mobility observations.  It is deliberately separate from
``fastmob.measures.collective`` so social analysis has a stable, dedicated
home.
"""

from .recast import RecastClass, RecastResult, recast_from_staypoints

__all__ = [
    "RecastClass",
    "RecastResult",
    "recast_from_staypoints",
]
