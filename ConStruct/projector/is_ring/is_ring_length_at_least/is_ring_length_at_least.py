###############################################################################
#
# Ring length "at least" constraint functionality for molecular graph generation
# This is the flipped mechanism of edge-deletion for edge-insertion transitions
# Uses ALL simple rings (unique simple cycles) for structural ring detection
#
###############################################################################

import warnings

import networkx as nx
from ConStruct.projector.graph_cycles import (
    enumerate_simple_cycles_unique,
    max_ring_length_exceeds,
    max_simple_cycle_length,
)

__all__ = ["has_rings_of_length_at_least", "ring_length_at_least_projector", "get_max_ring_length_at_least", "get_min_ring_length_at_least"]


def has_rings_of_length_at_least(graph, min_length):
    """Return whether the maximum simple-cycle length is at least ``min_length``."""
    if min_length < 3:
        raise ValueError("The minimum ring length must be at least 3.")
    return max_ring_length_exceeds(graph, min_length - 1)


def ring_length_at_least_projector(graph, min_length):
    """
    Edge-Insertion Projector: If the graph has no rings of length >= min_length, 
    add edges to create larger rings.
    SIMPLE APPROACH: Create a ring of exact length using existing nodes.
    Uses ALL simple rings (unique simple cycles) for detection.
    """
    cycles = list(enumerate_simple_cycles_unique(graph))
    
    # Check if we already have a ring of sufficient length
    for cycle in cycles:
        if len(cycle) >= min_length:
            return graph  # Already satisfies constraint
    
    # Need to create a ring of at least min_length
    nodes = list(graph.nodes())
    
    # Strategy: Create a ring of exactly min_length using existing nodes
    # This ensures we get a ring of the required length without tensor issues
    
    if len(nodes) >= min_length:
        # Create a ring of min_length using the first min_length nodes
        for i in range(min_length):
            u = nodes[i]
            v = nodes[(i + 1) % min_length]
            graph.add_edge(u, v)
    else:
        # Not enough nodes to create a ring of min_length
        # Create the largest possible ring with available nodes
        if len(nodes) >= 3:
            for i in range(len(nodes)):
                u = nodes[i]
                v = nodes[(i + 1) % len(nodes)]
                graph.add_edge(u, v)
        else:
            # Not enough nodes for any ring, add edges to create a chain
            for i in range(len(nodes) - 1):
                graph.add_edge(nodes[i], nodes[i + 1])
    
    return graph


def get_min_ring_length_at_least(graph):
    """Deprecated compatibility alias for :func:`get_max_ring_length_at_least`."""
    warnings.warn(
        "get_min_ring_length_at_least() is deprecated; use get_max_ring_length_at_least().",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_max_ring_length_at_least(graph)


def get_max_ring_length_at_least(graph):
    """Return the maximum simple-cycle length, or zero for an acyclic graph."""
    return max_simple_cycle_length(graph)
