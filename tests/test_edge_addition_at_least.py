import unittest
from types import SimpleNamespace

import networkx as nx
import torch
from omegaconf import OmegaConf

from ConStruct.diffusion.distributions import DistributionNodes
from ConStruct.diffusion.noise_model import EdgeInsertionTransition
from ConStruct.metrics.sampling_metrics import (
    ring_count_at_least_satisfaction_ratio,
    ring_count_satisfaction_ratio,
    ring_length_at_least_satisfaction_ratio,
    ring_length_satisfaction_ratio,
)
from ConStruct.projector.is_ring.is_ring_count_at_least import has_at_least_n_rings
from ConStruct.projector.is_ring.is_ring_length_at_least import has_rings_of_length_at_least
from ConStruct.projector.projector_utils import (
    RingCountAtLeastProjector,
    RingLengthAtLeastProjector,
)
from ConStruct.reporting.report_builder import constraint_caption, collect_structural
from ConStruct.utils import PlaceHolder


def dense_placeholder(graph, edge_classes=5, bond_type=2):
    n = graph.number_of_nodes()
    edge_types = torch.zeros((1, n, n, edge_classes))
    edge_types[..., 0] = 1
    for u, v in graph.edges():
        edge_types[0, u, v] = 0
        edge_types[0, v, u] = 0
        edge_types[0, u, v, bond_type] = 1
        edge_types[0, v, u, bond_type] = 1
    return PlaceHolder(
        X=torch.ones((1, n, 1)),
        charges=torch.zeros((1, n, 0)),
        E=edge_types,
        y=torch.zeros((1, 0)),
        node_mask=torch.ones((1, n), dtype=torch.bool),
    )


class EdgeAdditionAtLeastTests(unittest.TestCase):
    def test_conditioned_node_distribution(self):
        distribution = DistributionNodes({2: 2, 3: 3, 5: 5})
        torch.manual_seed(7)
        expected = distribution.m.sample((20,))
        torch.manual_seed(7)
        actual = distribution.sample_n(20, torch.device("cpu"))
        self.assertTrue(torch.equal(actual, expected))
        samples = distribution.sample_n(200, torch.device("cpu"), min_nodes=4)
        self.assertTrue((samples == 5).all())
        with self.assertRaises(ValueError):
            distribution.sample_n(1, torch.device("cpu"), min_nodes=6)

    def test_structural_predicates(self):
        self.assertFalse(has_at_least_n_rings(nx.path_graph(5), 1))
        self.assertTrue(has_at_least_n_rings(nx.cycle_graph(4), 1))
        self.assertTrue(has_at_least_n_rings(nx.complete_graph(4), 3))
        fused = nx.Graph([(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)])
        self.assertTrue(has_at_least_n_rings(fused, 3))
        self.assertTrue(has_rings_of_length_at_least(fused, 4))
        self.assertFalse(has_rings_of_length_at_least(nx.cycle_graph(4), 5))
        self.assertTrue(has_rings_of_length_at_least(nx.cycle_graph(5), 5))

    def test_ring_count_projector_restores_removed_bond_type(self):
        z_t = dense_placeholder(nx.cycle_graph(4), bond_type=3)
        projector = RingCountAtLeastProjector(z_t, min_rings=1)
        z_s = z_t.copy()
        z_s.E = z_t.E.clone()
        z_s.E[0, 0, 1] = torch.tensor([1, 0, 0, 0, 0])
        z_s.E[0, 1, 0] = torch.tensor([1, 0, 0, 0, 0])
        projector.project(z_s)
        self.assertEqual(torch.argmax(z_s.E[0, 0, 1]).item(), 3)
        self.assertTrue(projector.nx_graphs_list[0].has_edge(0, 1))

    def test_ring_count_projector_accepts_safe_removal(self):
        z_t = dense_placeholder(nx.complete_graph(4), bond_type=2)
        projector = RingCountAtLeastProjector(z_t, min_rings=1)
        z_s = z_t.copy()
        z_s.E = z_t.E.clone()
        z_s.E[0, 0, 1] = torch.tensor([1, 0, 0, 0, 0])
        z_s.E[0, 1, 0] = torch.tensor([1, 0, 0, 0, 0])
        projector.project(z_s)
        self.assertEqual(torch.argmax(z_s.E[0, 0, 1]).item(), 0)
        self.assertFalse(projector.nx_graphs_list[0].has_edge(0, 1))

    def test_ring_length_projector_restores_required_cycle(self):
        z_t = dense_placeholder(nx.cycle_graph(5), bond_type=4)
        projector = RingLengthAtLeastProjector(z_t, min_ring_length=5)
        z_s = z_t.copy()
        z_s.E = z_t.E.clone()
        z_s.E[0, 0, 1] = torch.tensor([1, 0, 0, 0, 0])
        z_s.E[0, 1, 0] = torch.tensor([1, 0, 0, 0, 0])
        projector.project(z_s)
        self.assertEqual(torch.argmax(z_s.E[0, 0, 1]).item(), 4)
        self.assertTrue(projector.nx_graphs_list[0].has_edge(0, 1))

    def test_all_types_terminal_graph(self):
        cfg = OmegaConf.create({
            "model": {
                "diffusion_steps": 5,
                "transition": "edge_insertion",
                "nu": {"x": 1, "c": 1, "e": 1, "y": 1},
            }
        })
        transition = EdgeInsertionTransition(
            cfg,
            x_marginals=torch.tensor([0.4, 0.6]),
            e_marginals=torch.tensor([0.7, 0.1, 0.1, 0.05, 0.05]),
            charges_marginals=torch.tensor([1.0]),
            y_classes=0,
        )
        self.assertEqual(transition.E_marginals[0].item(), 0)
        self.assertTrue(torch.allclose(transition.E_marginals[1:], torch.tensor([1 / 3, 1 / 3, 1 / 6, 1 / 6])))
        z_t = transition.sample_limit_dist(torch.tensor([[True, True, True, False]]))
        adjacency = z_t.E[0, :3, :3, 1:].sum(dim=-1) > 0
        self.assertEqual(adjacency.sum().item(), 6)
        self.assertEqual(z_t.E[0, 3, :, 1:].sum().item(), 0)
        self.assertEqual(z_t.E[0, :, 3, 1:].sum().item(), 0)
        self.assertEqual(z_t.E[0].diagonal(dim1=0, dim2=1)[1:].sum().item(), 0)
        self.assertTrue(torch.equal(z_t.E, z_t.E.transpose(1, 2)))
        qt = transition.get_Qt(torch.tensor([2]))
        qt_bar = transition.get_Qt_bar(torch.tensor([2]))
        self.assertTrue(torch.allclose(qt.E.sum(dim=-1), torch.ones_like(qt.E[..., 0])))
        self.assertTrue(torch.allclose(qt_bar.E.sum(dim=-1), torch.ones_like(qt_bar.E[..., 0])))

    def test_returned_tensor_metrics_and_reports(self):
        satisfied = dense_placeholder(nx.cycle_graph(5)).collapse(torch.tensor([]))
        violated = dense_placeholder(nx.path_graph(5)).collapse(torch.tensor([]))
        generated = [satisfied, violated]
        self.assertEqual(ring_count_at_least_satisfaction_ratio(generated, 1).tolist(), [1, 0])
        self.assertEqual(ring_length_at_least_satisfaction_ratio(generated, 5).tolist(), [1, 0])
        self.assertEqual(ring_count_satisfaction_ratio(generated, 0).tolist(), [0, 1])
        self.assertEqual(ring_length_satisfaction_ratio(generated, 4).tolist(), [0, 1])
        self.assertEqual(constraint_caption("ring_count_at_least", {"min_rings": 2}), "Ring count ≥ 2")
        self.assertEqual(constraint_caption("ring_count_at_most", {"max_rings": 2}), "Ring count ≤ 2")
        self.assertEqual(constraint_caption("ring_length_at_least", {"min_ring_length": 5}), "Maximum ring length ≥ 5")
        self.assertEqual(constraint_caption("ring_length_at_most", {"max_ring_length": 5}), "Maximum ring length ≤ 5")
        cfg = OmegaConf.create({"model": {"rev_proj": "ring_count_at_least"}})
        rows = collect_structural(
            "test",
            {"test_sampling/ring_count_satisfaction": 0.5},
            2,
            ring_count_counts=[1, 0, 1],
            min_rings=2,
            cfg=cfg,
        )
        self.assertIn(("Ring count <2 (%)", "50.0%"), rows)


if __name__ == "__main__":
    unittest.main()
