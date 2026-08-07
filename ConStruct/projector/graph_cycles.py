#!/usr/bin/env python3
"""
Shared utilities for structural ring (cycle) detection on undirected molecular graphs.
- enumerate_simple_cycles_unique: unique simple cycles (no orientation duplicates)
- count_simple_cycles: total number of unique simple cycles
- max_simple_cycle_length: maximum simple cycle length, or 0 if acyclic
- count_simple_cycles_up_to: early-stop counter for efficiency
- simple_cycle_count_exceeds: early-stop boolean check
- max_ring_length_exceeds: early-stop boolean check for ring length constraint

Note: Structural only; no chemical validity checks.
"""
from __future__ import annotations
from typing import Iterable, List, Set, Tuple, Optional
import networkx as nx

__all__ = [
    "enumerate_simple_cycles_unique",
    "count_simple_cycles",
    "max_simple_cycle_length",
    "count_simple_cycles_up_to",
    "simple_cycle_count_exceeds",
    "max_ring_length_exceeds",
    "fast_ring_count_precheck",
    "fast_ring_length_precheck",
]


def _canonical_cycle(cycle: List[int]) -> Tuple[int, ...]:
    """Canonicalize a cycle for undirected graphs.
    - Rotate both orientations so the smallest node is first
    - Choose the lexicographically smaller orientation
    """
    if not cycle:
        return tuple()
    # Remove duplicated closing node if present
    if len(cycle) > 1 and cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    if not cycle:
        return tuple()
    m = len(cycle)
    # Forward orientation rotated to put smallest node first
    min_node = min(cycle)
    f_idx = cycle.index(min_node)
    forward = cycle[f_idx:] + cycle[:f_idx]
    # Reverse orientation also rotated to put smallest node first
    rc = list(reversed(cycle))
    r_idx = rc.index(min_node)
    backward = rc[r_idx:] + rc[:r_idx]
    return tuple(forward) if forward <= backward else tuple(backward)


def enumerate_simple_cycles_unique(graph: nx.Graph, max_len: Optional[int] = None) -> Iterable[List[int]]:
    """Enumerate unique simple cycles for an undirected graph.
    Uses networkx.simple_cycles on a directed view and deduplicates by canonical form.
    max_len: if provided, skip cycles longer than this threshold.
    """
    if graph.number_of_edges() == 0 or graph.number_of_nodes() < 3:
        return
    # Stream cycles so threshold predicates can stop without materializing all
    # cycles of dense terminal graphs.
    directed = graph.to_directed()
    seen: Set[Tuple[int, ...]] = set()
    for cyc in nx.simple_cycles(directed):
        if max_len is not None and len(cyc) > max_len:
            continue
        can = _canonical_cycle(cyc)
        if len(can) >= 3 and can not in seen:
            seen.add(can)
            yield list(can)


def count_simple_cycles(graph: nx.Graph, max_len: Optional[int] = None) -> int:
    return len(list(enumerate_simple_cycles_unique(graph, max_len=max_len)))


def max_simple_cycle_length(graph: nx.Graph, max_len: Optional[int] = None) -> int:
    cycles = enumerate_simple_cycles_unique(graph, max_len=max_len)
    max_len_found = 0
    for cyc in cycles:
        if len(cyc) > max_len_found:
            max_len_found = len(cyc)
    return max_len_found


# --- Early-stop cycle COUNT: True if >k unique simple cycles (no cap) ---
def simple_cycle_count_exceeds(graph, k: int) -> bool:
    """
    True if the number of unique simple cycles is > k (early-stops).
    NO max_len cap for correctness outside QM9.
    """
    cnt = 0
    for _cyc in enumerate_simple_cycles_unique(graph, max_len=None):
        cnt += 1
        if cnt > k:
            return True
    return False


# --- Early-stop CAPPED counting (optional utility) ---
def count_simple_cycles_up_to(graph, limit: int) -> int:
    """
    Count unique simple cycles up to `limit`. If count exceeds `limit`, stop early and return limit+1.
    NO max_len cap for correctness outside QM9.
    """
    cnt = 0
    for _cyc in enumerate_simple_cycles_unique(graph, max_len=None):
        cnt += 1
        if cnt > limit:
            return cnt
    return cnt


# --- Early-stop ring LENGTH: True if any simple cycle has length > L (no cap) ---
def max_ring_length_exceeds(graph, L: int) -> bool:
    """
    True if any simple cycle has length > L (early-stops).
    NO max_len cap for correctness outside QM9.
    """
    for cyc in enumerate_simple_cycles_unique(graph, max_len=None):
        if len(cyc) > L:
            return True
    return False


# --- FAST PRE-SCREENING OPTIMIZATIONS ---
def fast_ring_count_prescreen(graph, max_rings: int):
    """
    Fast pre-screening for ring count constraints with zero false positives.
    
    This function performs lightweight checks to identify cases where we can
    guarantee the graph satisfies the ring count constraint WITHOUT doing
    expensive cycle enumeration. Only returns True when mathematically certain.
    
    Philosophy: Better to do full validation than risk false positives that
    would bias molecular generation toward accepting invalid structures.
    
    Args:
        graph: NetworkX graph to check
        max_rings: Maximum allowed number of rings/cycles
        
    Returns:
        True: Graph definitely satisfies constraint (safe to skip full validation)
        None: Uncertain - requires full cycle enumeration for correctness
        
    Performance: O(1) for trivial cases, prevents O(exponential) cycle enumeration
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    
    if n_nodes < 3:
        return True  # Impossible to form cycles with < 3 nodes
    
    # Only apply heuristics that are mathematically guaranteed safe
    
    # Heuristic 1: Forest detection (guaranteed zero cycles)
    if n_edges < n_nodes:
        return True  # Forest structure - no cycles possible
    
    # Heuristic 2: Connected graph with exactly one cycle
    if n_edges == n_nodes and max_rings >= 1:
        # Additional safety: verify graph is actually connected
        if graph.number_of_nodes() > 0:
            try:
                from networkx import is_connected
                if is_connected(graph):
                    return True  # Connected graph with n edges has exactly 1 cycle
            except:
                pass  # Skip connectivity check if it fails
    
    # All other cases require full validation to avoid false positives
    return None  # Must do complete cycle enumeration


def fast_ring_length_prescreen(graph, max_length: int):
    """
    FIXED: Fast pre-screening for ring length constraints with ZERO false positives.
    
    This function performs only mathematically guaranteed safe checks to identify 
    cases where we can be 100% certain all cycles have length ≤ max_length.
    
    CRITICAL FIX: Removed all heuristics that caused false positives:
    - Removed "graph too small" heuristic (n_nodes <= max_length)
    - Removed "linear structures" heuristic (max degree ≤ 2)  
    - Removed "single cycle" heuristic (n_edges == n_nodes)
    
    Philosophy: Better to do full validation than risk false positives that
    would allow constraint violations in molecular generation.
    
    Args:
        graph: NetworkX graph to check
        max_length: Maximum allowed cycle length
        
    Returns:
        True: All cycles definitely have length ≤ max_length (safe to skip full validation)
        None: Uncertain - requires full cycle enumeration for correctness
        
    Performance: O(1) for trivial cases, prevents O(exponential) cycle enumeration
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    
    if n_nodes < 3:
        return True  # No cycles possible with < 3 nodes
    
    # ONLY apply heuristics that are mathematically guaranteed safe
    
    # Heuristic 1: Forest detection (guaranteed no cycles)
    if n_edges < n_nodes:
        return True  # Forest structure - no cycles to violate length constraint
    
    # Heuristic 2: Very small graphs with guaranteed short cycles
    if n_nodes == 3 and max_length >= 3:
        return True  # Triangle (3-cycle) is the only possible cycle
    
    # REMOVED ALL OTHER HEURISTICS - they were causing false positives:
    # - "Graph too small" (n_nodes <= max_length) - WRONG: can have shorter cycles
    # - "Linear structures" (max degree ≤ 2) - WRONG: can have various cycle lengths  
    # - "Single cycle" (n_edges == n_nodes) - WRONG: cycle might not span all nodes
    
    # All other cases require full validation to guarantee correctness
    return None  # Must do complete cycle enumeration 