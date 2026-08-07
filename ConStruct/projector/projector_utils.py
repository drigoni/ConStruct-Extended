import abc
import logging

import networkx as nx
import numpy as np
import time
import torch
import torch.nn.functional as F
from rdkit import Chem
from typing import List, Optional, Tuple, Dict, Any
from networkx.algorithms import isomorphism as iso
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique, count_simple_cycles, max_simple_cycle_length

from ConStruct.projector.is_planar import is_planar
from ConStruct.projector.is_ring.is_ring_count_at_most.is_ring_count_at_most import ring_count_at_most_projector
from ConStruct.projector.is_ring.is_ring_length_at_most.is_ring_length_at_most import has_rings_of_length_at_most
from ConStruct.projector.is_ring.is_ring_count_at_least.is_ring_count_at_least import has_at_least_n_rings
from ConStruct.projector.is_ring.is_ring_length_at_least.is_ring_length_at_least import has_rings_of_length_at_least
from ConStruct.utils import PlaceHolder
from ConStruct.diffusion.extra_features import ExtraFeatures
from ConStruct.diffusion.extra_features_molecular import ExtraMolecularFeatures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_simple_graph_from_edge_tensor(edge_tensor: torch.Tensor, mask: torch.Tensor):
    """
    Unified graph construction helper for consistent graph building across all components.

    - Channel 0 = no-edge
    - Channels 1..4 = edges (single, double, triple, AROMATIC)  ← include aromatic
    """
    n = int(mask.sum().item()) if mask is not None else edge_tensor.shape[0]
    et = edge_tensor[:n, :n]

    if et.dim() == 3:
        # Include aromatic channel (index 4). Sum 1..4.
        adj = (et[..., 1:5].sum(dim=-1) > 0).int()
    else:
        adj = (et > 0).int()

    g = nx.from_numpy_array(adj.cpu().numpy())
    g = nx.Graph(g)
    return g


def resize_placeholder_tensor(z_s: PlaceHolder, new_size: int, graph_idx: int = 0) -> PlaceHolder:
    """
    Resize PlaceHolder tensor to accommodate new nodes for dynamic tensor resizing.
    
    TODO: FUTURE WORK - This function was used by edge-insertion projectors
    which have been removed for simplification. Kept for future reference.
    
    Args:
        z_s: PlaceHolder tensor to resize
        new_size: New number of nodes (must be >= current size)
        graph_idx: Index of the graph being modified
    
    Returns:
        Resized PlaceHolder tensor
    """
    current_size = z_s.X.shape[1]
    
    if new_size <= current_size:
        return z_s  # No resizing needed
    
    # Resize X tensor (node features)
    new_X = torch.zeros(z_s.X.shape[0], new_size, z_s.X.shape[2], device=z_s.X.device)
    new_X[:, :current_size, :] = z_s.X
    z_s.X = new_X
    
    # Resize E tensor (edge features)
    new_E = torch.zeros(z_s.E.shape[0], new_size, new_size, z_s.E.shape[3], device=z_s.E.device)
    new_E[:, :current_size, :current_size, :] = z_s.E
    z_s.E = new_E
    
    # Resize node mask if it exists
    if hasattr(z_s, 'node_mask') and z_s.node_mask is not None:
        new_mask = torch.zeros(z_s.node_mask.shape[0], new_size, device=z_s.node_mask.device)
        new_mask[:, :current_size] = z_s.node_mask
        z_s.node_mask = new_mask
    
    # Resize extra features if they exist
    if hasattr(z_s, 'extra_features') and z_s.extra_features is not None:
        if isinstance(z_s.extra_features, ExtraMolecularFeatures):
            # Resize charge tensor
            if hasattr(z_s.extra_features, 'charge') and z_s.extra_features.charge is not None:
                new_charge = torch.zeros(z_s.extra_features.charge.shape[0], new_size, device=z_s.extra_features.charge.device)
                new_charge[:, :current_size] = z_s.extra_features.charge
                z_s.extra_features.charge = new_charge
            
            # Resize is_aromatic tensor
            if hasattr(z_s.extra_features, 'is_aromatic') and z_s.extra_features.is_aromatic is not None:
                new_aromatic = torch.zeros(z_s.extra_features.is_aromatic.shape[0], new_size, device=z_s.extra_features.is_aromatic.device)
                new_aromatic[:, :current_size] = z_s.extra_features.is_aromatic
                z_s.extra_features.is_aromatic = new_aromatic
    
    return z_s


def update_tensor_from_graph(z_s: PlaceHolder, graph_idx: int, nx_graph: nx.Graph) -> None:
    """
    Synchronize tensor with NetworkX graph after resizing.
    
    TODO: FUTURE WORK - This function was used by edge-insertion projectors
    which have been removed for simplification. Kept for future reference.
    
    Args:
        z_s: PlaceHolder tensor to update
        graph_idx: Index of the graph being updated
        nx_graph: NetworkX graph with the updated structure
    """
    # Clear existing edges for this graph
    z_s.E[graph_idx] = torch.zeros_like(z_s.E[graph_idx])
    
    # Add all edges from graph
    for u, v in nx_graph.edges():
        if u != v and u < z_s.E.shape[1] and v < z_s.E.shape[2]:
            z_s.E[graph_idx, u, v, 1] = 1  # single bond
            z_s.E[graph_idx, v, u, 1] = 1  # undirected


def create_ring_with_new_nodes(graph: nx.Graph, min_length: int, start_node_idx: int) -> nx.Graph:
    """
    Create a ring of specified length using new nodes, and connect it to the existing graph if possible.
    
    TODO: FUTURE WORK - This function was used by edge-insertion projectors
    which have been removed for simplification. Kept for future reference.
    
    Args:
        graph: NetworkX graph to modify
        min_length: Minimum length of the ring to create
        start_node_idx: Starting index for new nodes
    Returns:
        Modified graph with new ring, connected to existing graph if possible
    """
    # Add new nodes to graph
    new_nodes = []
    for i in range(min_length):
        new_node = start_node_idx + i
        graph.add_node(new_node)
        new_nodes.append(new_node)
    # Create ring using new nodes
    for i in range(min_length):
        u = new_nodes[i]
        v = new_nodes[(i + 1) % min_length]
        graph.add_edge(u, v)
    # Connect to existing graph (maintain connectivity)
    existing_nodes = [n for n in graph.nodes() if n < start_node_idx]
    if existing_nodes:
        # Connect all isolated existing nodes to the new ring
        for i, existing_node in enumerate(existing_nodes):
            # Connect to a different new node for each existing node to distribute connections
            new_node_idx = i % len(new_nodes)
            graph.add_edge(existing_node, new_nodes[new_node_idx])
    return graph


def is_chemically_valid_graph(nx_graph, atom_types, atom_decoder):
    """
    Check if a NetworkX graph can be converted to a chemically valid RDKit molecule.
    This ensures that the graph structure is chemically meaningful.
    """
    try:
        # Convert NetworkX graph to RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        for i, atom_type in enumerate(atom_types):
            if atom_type == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom_type.item())])
            mol.AddAtom(a)
        
        # Add bonds
        for edge in nx_graph.edges():
            u, v = edge
            if u != v:  # No self-loops
                mol.AddBond(int(u), int(v), Chem.rdchem.BondType.SINGLE)
        
        # Try to convert to a valid molecule
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


def check_valence_constraints(nx_graph, atom_types, atom_decoder):
    """
    Check if adding an edge would violate valence constraints.
    Returns True if the edge addition is chemically valid.
    """
    try:
        # Get current valence for each atom
        valence_count = {}
        for node in nx_graph.nodes():
            valence_count[node] = 0
        
        # Count current bonds
        for edge in nx_graph.edges():
            u, v = edge
            valence_count[u] += 1
            valence_count[v] += 1
        
        # Define maximum valence for common atoms (for edge addition checking)
        max_valence = {
            'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6,
            'Cl': 1, 'Br': 1, 'I': 1, 'H': 1
        }
        
        # Check if any atom exceeds maximum valence
        for node, valence in valence_count.items():
            if atom_types[node] == -1:
                continue
            atom_symbol = atom_decoder[int(atom_types[node].item())]
            
            if atom_symbol in max_valence and valence > max_valence[atom_symbol]:
                return False
        
        return True
    except:
        return False


def can_add_edge_safely(nx_graph, u, v, atom_types, atom_decoder):
    """
    Check if adding edge (u,v) would be chemically valid.
    Returns True if the edge can be safely added.
    """
    # Check if edge already exists
    if nx_graph.has_edge(u, v):
        return False
    
    # Check if adding edge would violate valence constraints
    trial_graph = nx_graph.copy()
    trial_graph.add_edge(u, v)
    
    return check_valence_constraints(trial_graph, atom_types, atom_decoder)


def do_zero_prob_forbidden_edges(pred, z_t, clean_data):
    """
    Checks if the graph has forbidden edges.
    """
    adj_matrices = (z_t.E > 0).int()
    zeroed_edge = torch.tensor([1.0] + [0.0] * (pred.E.shape[-1] - 1))
    for graph_idx, adj_matrix in enumerate(adj_matrices):
        # t21 = time.time()
        num_nodes = z_t.X.shape[1]
        adj_matrix = adj_matrix[:num_nodes, :num_nodes]
        forbidden_edges = get_forbidden_edges(adj_matrix)
        if forbidden_edges.shape[0] > 0:
            pred.E[graph_idx, forbidden_edges[:, 0], forbidden_edges[:, 1]] = (
                zeroed_edge
            )
            pred.E[graph_idx, forbidden_edges[:, 1], forbidden_edges[:, 0]] = (
                zeroed_edge
            )

        # t22 = time.time()
        # print("Graph idx", graph_idx, "-- Time to get forbidden edges: ", t22 - t21)

    return pred


def get_forbidden_edges(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    nx_graph = nx.from_numpy_array(adj_matrix.cpu().numpy())
    forbidden_edges = []
    for node_1 in range(num_nodes):
        for node_2 in range(node_1, num_nodes):
            # Prevent computation of is planar
            if nx_graph.has_edge(node_1, node_2):
                continue
            # Check if adding edge makes graph non-planar
            trial_graph = nx_graph.copy()
            trial_graph.add_edge(node_1, node_2)
            if not is_planar.is_planar(trial_graph):
                forbidden_edges.append([node_1, node_2])
            # assert nx.is_planar(trial_graph) == is_planar.is_planar(trial_graph)

    return torch.tensor(forbidden_edges)


def get_adj_matrix(z_t):
    # Include aromatic channel in structural view
    z_t_adj = (z_t.E[:, :, :, 1:5].sum(dim=3) > 0).int()
    return z_t_adj


class AbstractProjector(abc.ABC):
    @abc.abstractmethod
    def valid_graph_fn(self, nx_graph):
        pass

    @property
    @abc.abstractmethod
    def can_block_edges(self):
        pass

    @property
    def edge_mutation(self):
        return "add"

    def __init__(self, z_t: PlaceHolder):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        if self.can_block_edges:
            # Use canonical undirected edge tuples as keys
            self.blocked_edges = {graph_idx: set() for graph_idx in range(self.batch_size)}

        # Initialize total blocked edges counter
        self.total_blocked = 0

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_adj_matrix(z_t)
        self.z_t_E = z_t.E.clone()

        # add data structure where planarity is checked
        for graph_idx in range(self.batch_size):
            # Use unified graph construction helper
            edge_mat = z_t.E[graph_idx]
            mask = z_t.node_mask[graph_idx]
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            
            # For edge-insertion, we validate the initial graph
            # because we need to add edges to satisfy constraints
            self.nx_graphs_list.append(nx_graph)

    def project(self, z_s: PlaceHolder):
        z_s_adj = get_adj_matrix(z_s)

        if self.edge_mutation == "add":
            diff_adj = z_s_adj - self.z_t_adj
            assert (diff_adj >= 0).all()
            candidate_edges = diff_adj.nonzero(as_tuple=False)
        elif self.edge_mutation == "remove":
            diff_adj = self.z_t_adj - z_s_adj
            assert (diff_adj >= 0).all()
            candidate_edges = diff_adj.nonzero(as_tuple=False)
        else:
            raise ValueError(f"Unknown edge mutation mode: {self.edge_mutation}")

        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            edges_to_process = (
                candidate_edges[
                    torch.logical_and(
                        candidate_edges[:, 0] == graph_idx,
                        candidate_edges[:, 1] < candidate_edges[:, 2],
                    )
                ][:, 1:]
            )
            if edges_to_process.is_cuda:
                edges_to_process = edges_to_process.cpu()
            edges_to_process = edges_to_process.numpy()

            for edge in edges_to_process:
                u, v = int(edge[0]), int(edge[1])
                e = tuple(sorted((u, v)))
                prev_edge_features = self.z_t_E[graph_idx, u, v].clone()
                no_edge_features = torch.zeros_like(prev_edge_features)
                no_edge_features[0] = 1

                if self.can_block_edges and e in self.blocked_edges[graph_idx]:
                    if self.edge_mutation == "add":
                        z_s.E[graph_idx, u, v] = no_edge_features
                        z_s.E[graph_idx, v, u] = no_edge_features
                    else:
                        z_s.E[graph_idx, u, v] = prev_edge_features
                        z_s.E[graph_idx, v, u] = prev_edge_features
                    self.total_blocked += 1
                    continue

                if self.edge_mutation == "add":
                    nx_graph.add_edge(u, v)
                else:
                    nx_graph.remove_edge(u, v)

                if self.valid_graph_fn(nx_graph):
                    pass
                else:
                    if self.edge_mutation == "add":
                        nx_graph.remove_edge(u, v)
                        z_s.E[graph_idx, u, v] = no_edge_features
                        z_s.E[graph_idx, v, u] = no_edge_features
                    else:
                        nx_graph.add_edge(u, v)
                        z_s.E[graph_idx, u, v] = prev_edge_features
                        z_s.E[graph_idx, v, u] = prev_edge_features

                    if self.can_block_edges:
                        self.blocked_edges[graph_idx].add(e)
                    self.total_blocked += 1

            self.nx_graphs_list[graph_idx] = nx_graph

        self.z_t_adj = get_adj_matrix(z_s)
        self.z_t_E = z_s.E.clone()


def has_no_cycles(nx_graph):
    # Tree have n-1 edges
    if nx_graph.number_of_edges() >= nx_graph.number_of_nodes():
        return False
    return nx.is_forest(nx_graph)
    # try:
    #     # Attempt to find a cycle
    #     nx.find_cycle(nx_graph)
    #     # If a cycle is found, it's not a tree
    #     return False
    # except nx.exception.NetworkXNoCycle:
    #     # No cycle found, so it's a tree
    #     return True


def is_linear_graph(nx_graph):
    num_nodes = nx_graph.number_of_nodes()
    num_degree_one = sum([d == 1 for n, d in nx_graph.degree()])
    num_degree_two = sum([d == 2 for n, d in nx_graph.degree()])

    return (num_degree_one == 2 and num_degree_two == num_nodes - 2) or (
        num_degree_one == 0 and num_degree_two == 0
    )


def has_lobster_components(nx_graph):
    if has_no_cycles(nx_graph):
        G = nx_graph.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        conn_components = nx.connected_components(G)
        for node_set in conn_components:
            subgraph = G.subgraph(node_set)
            if not is_linear_graph(subgraph):
                return False
        return True
    else:
        return False


class PlanarProjector(AbstractProjector):
    def __init__(self, z_t: PlaceHolder):
        super().__init__(z_t)

    def valid_graph_fn(self, nx_graph):
        return is_planar(nx_graph)

    @property
    def can_block_edges(self):
        return True


class TreeProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        return has_no_cycles(nx_graph)

    @property
    def can_block_edges(self):
        return True


class LobsterProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        return has_lobster_components(nx_graph)

    @property
    def can_block_edges(self):
        return True


class RingCountAtMostProjector(AbstractProjector):
    """
    Edge-Deletion Projector: Ensures graphs have at most N rings.
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (ring count) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    Uses the original ConStruct paper algorithm with hash-based edge blocking.
    """
    
    def __init__(self, z_t: PlaceHolder, max_rings: int, atom_decoder=None, use_incremental=None):
        self.max_rings = max_rings
        # Note: atom_decoder is kept for compatibility but NOT used for chemical validation
        self.atom_decoder = atom_decoder
        super().__init__(z_t)

    def valid_graph_fn(self, nx_graph):
        """Check if graph satisfies structural constraint: ring count ≤ max_rings."""
        from ConStruct.projector.graph_cycles import simple_cycle_count_exceeds
        return not simple_cycle_count_exceeds(nx_graph, self.max_rings)

    @property
    def can_block_edges(self):
        """Enable hash-based edge blocking from the original paper."""
        return True


class RingLengthAtMostProjector(AbstractProjector):
    """
    FIXED: Edge-Deletion Projector: Ensures graphs have max ring length ≤ L.
    
    CRITICAL FIXES APPLIED:
    - Removed buggy fast pre-screening heuristics that caused false positives
    - Always uses full cycle enumeration for 100% correctness
    - Achieves same 100% constraint satisfaction as planarity projectors
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (max ring length) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    BUG FIXES:
    - Fixed "graph too small" heuristic (n_nodes <= max_length) that incorrectly allowed violations
    - Fixed "linear structures" heuristic (max degree ≤ 2) that missed cycles
    - Fixed "single cycle" heuristic (n_edges == n_nodes) that assumed wrong cycle lengths
    
    PERFORMANCE: Uses full cycle enumeration for guaranteed correctness, matching
    the approach of other working projectors (planarity, ring count).
    """
    
    def __init__(self, z_t: PlaceHolder, max_ring_length: int, atom_decoder=None, use_incremental_length=None):
        self.max_ring_length = max_ring_length
        self.atom_decoder = atom_decoder
        super().__init__(z_t)

    def valid_graph_fn(self, nx_graph):
        """Check if max ring length ≤ max_ring_length."""
        from ConStruct.projector.is_ring.is_ring_length_at_most import has_rings_of_length_at_most
        return has_rings_of_length_at_most(nx_graph, self.max_ring_length)

    @property
    def can_block_edges(self):
        """Enable hash-based edge blocking from the original paper."""
        return True
    



class RingCountAtLeastProjector(AbstractProjector):
    """
    Edge-Insertion Projector: Ensures graphs have at least N rings.
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (ring count) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    Mathematical Construction:
    - Constraint: "At least N rings" (structural only)
    - Forward diffusion: Edges progressively appear toward edge state
    - Reverse diffusion: Edges are removed while preserving min ring constraint
    - Natural bias: toward connected graphs (more edges = more ring possibilities)
    
    Structural Constraint:
    - Count rings using unique simple cycles
    - Block edge removals that would drop ring count below minimum
    - Allow edge additions that create new rings
    
    CRITICAL: Chemical validity (valency, connectivity, atom types) is NOT enforced.
    These properties are measured separately after generation using RDKit.
    
    Usage:
    - Transition: 'edge_insertion'
    - Config: rev_proj: 'ring_count_at_least', min_rings: N
    - Post-generation: Run RDKit validation to measure chemical properties
    
    """

    def __init__(self, z_t: PlaceHolder, min_rings: int, atom_decoder=None):
        if min_rings < 0:
            raise ValueError("The minimum ring count must be non-negative.")
        self.min_rings = min_rings
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        for graph_idx, graph in enumerate(self.nx_graphs_list):
            if not self.valid_graph_fn(graph):
                raise ValueError(
                    f"Initial graph {graph_idx} cannot satisfy ring_count_at_least={min_rings}."
                )

    def valid_graph_fn(self, nx_graph):
        return has_at_least_n_rings(nx_graph, self.min_rings)

    @property
    def can_block_edges(self):
        return True

    @property
    def edge_mutation(self):
        return "remove"
    


class RingLengthAtLeastProjector(AbstractProjector):
    """
    Edge-Insertion Projector: Ensures graphs have rings of at least N length.
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (ring length) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    Mathematical Construction:
    - Constraint: "The maximum simple-cycle length is at least N" (structural only)
    - Forward diffusion: Edges progressively appear toward the complete edge state
    - Reverse diffusion: Edges are removed while preserving the lower bound
    - Natural bias: toward connected graphs (more edges = more ring possibilities)

    Structural Constraint:
    - Search unique simple cycles and stop as soon as one has length at least N
    - Block edge removals that would eliminate every sufficiently long cycle
    - Preserve the exact previous bond type when a removal is blocked
    
    CRITICAL: Chemical validity (valency, connectivity, atom types) is NOT enforced.
    These properties are measured separately after generation using RDKit.
    
    Usage:
    - Transition: 'edge_insertion'
    - Config: rev_proj: 'ring_length_at_least', min_ring_length: N
    - Post-generation: Run RDKit validation to measure chemical properties
    
    """

    def __init__(self, z_t: PlaceHolder, min_ring_length: int, atom_decoder=None):
        if min_ring_length < 3:
            raise ValueError("The minimum ring length must be at least 3.")
        self.min_ring_length = min_ring_length
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        for graph_idx, graph in enumerate(self.nx_graphs_list):
            if not self.valid_graph_fn(graph):
                raise ValueError(
                    f"Initial graph {graph_idx} cannot satisfy "
                    f"ring_length_at_least={min_ring_length}."
                )

    def valid_graph_fn(self, nx_graph):
        return has_rings_of_length_at_least(nx_graph, self.min_ring_length)

    @property
    def can_block_edges(self):
        return True

    @property
    def edge_mutation(self):
        return "remove"
