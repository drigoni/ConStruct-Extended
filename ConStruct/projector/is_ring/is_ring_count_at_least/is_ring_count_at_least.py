###############################################################################
#
# Ring count "at least" constraint functionality for molecular graph generation
# This is the flipped mechanism of edge-deletion for edge-insertion transitions
# Uses ALL simple rings (unique simple cycles) for structural counting
#
###############################################################################

import networkx as nx
from ConStruct.projector.graph_cycles import (
    enumerate_simple_cycles_unique,
    count_simple_cycles,
    simple_cycle_count_exceeds,
)

__all__ = ["has_at_least_n_rings", "ring_count_at_least_projector", "count_rings_at_least"]


def has_at_least_n_rings(graph, n):
    """Return True if the graph has at least n rings (all simple rings)."""
    if n < 0:
        raise ValueError("The minimum ring count must be non-negative.")
    if n == 0:
        return True
    return simple_cycle_count_exceeds(graph, n - 1)


def ring_count_at_least_projector(graph, min_rings):
    """
    Edge-Insertion Projector: If the graph has fewer than min_rings, add edges to create rings.
    AGGRESSIVE APPROACH: Add edges strategically to create the required number of rings.
    Uses ALL simple rings (unique simple cycles) for ring count.
    """
    cycles = list(enumerate_simple_cycles_unique(graph))
    current_rings = len(cycles)
    
    if current_rings >= min_rings:
        return graph  # Already satisfies constraint
    
    # Need to add rings
    rings_needed = min_rings - current_rings
    
    # Strategy: Add edges to create rings
    # 1. First, try to connect disconnected components
    components = list(nx.connected_components(graph))
    
    # Keep trying until we have enough rings
    attempts = 0
    max_attempts = rings_needed * 10  # Prevent infinite loops
    
    while current_rings < min_rings and attempts < max_attempts:
        attempts += 1
        
        if len(components) <= 1:
            # Need to create a ring within a single component
            nodes = list(graph.nodes())
            
            # Strategy 1: Try to add edges to create new rings
            ring_created = False
            
            # Try to find a path and close it
            for start_node in nodes:
                for end_node in nodes:
                    if start_node != end_node and not graph.has_edge(start_node, end_node):
                        try:
                            path = nx.shortest_path(graph, start_node, end_node)
                            if len(path) >= 3:
                                # Add edge to close the path into a ring
                                graph.add_edge(start_node, end_node)
                                ring_created = True
                                break
                        except nx.NetworkXNoPath:
                            continue
                if ring_created:
                    break
            
            # Strategy 2: If no ring created, try adding edges to create new rings
            if not ring_created:
                for i, node1 in enumerate(nodes):
                    for j, node2 in enumerate(nodes):
                        if i < j and not graph.has_edge(node1, node2):
                            # Check if adding this edge creates a new ring
                            graph.add_edge(node1, node2)
                            new_cycles = list(enumerate_simple_cycles_unique(graph))
                            if len(new_cycles) > current_rings:
                                ring_created = True
                                break
                            else:
                                # Remove the edge if it didn't create a ring
                                graph.remove_edge(node1, node2)
                    if ring_created:
                        break
            
            # Strategy 3: If still no ring created, add a new node and create a ring
            if not ring_created and len(nodes) >= 3:
                # Add a new node and connect it to create a new triangle
                new_node = max(nodes) + 1
                # Connect to two existing nodes to create a new triangle
                graph.add_edge(new_node, nodes[0])
                graph.add_edge(new_node, nodes[1])
                ring_created = True
        else:
            # Connect two components to create a ring
            if len(components) >= 2:
                comp1, comp2 = components[0], components[1]
                node1 = list(comp1)[0]
                node2 = list(comp2)[0]
                graph.add_edge(node1, node2)
                components = list(nx.connected_components(graph))
        
        # Update ring count and components
        current_rings = count_simple_cycles(graph)
        components = list(nx.connected_components(graph))
    
    return graph


def count_rings_at_least(graph):
    """Return the number of rings (all simple rings) in the graph for 'at least' constraints."""
    return count_simple_cycles(graph) 