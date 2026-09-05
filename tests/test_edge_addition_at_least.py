import unittest
from types import SimpleNamespace

import networkx as nx
import torch
from omegaconf import OmegaConf

from ConStruct.diffusion.distributions import DistributionNodes
from ConStruct.diffusion.noise_model import EdgeInsertionTransition
from ConStruct.metrics.sampling_metrics import (
    joint_constraint_satisfaction_ratio,
    ring_count_at_least_satisfaction_ratio,
    ring_count_satisfaction_ratio,
    ring_length_at_least_satisfaction_ratio,
    ring_length_satisfaction_ratio,
)
from ConStruct.projector.is_ring.is_ring_count_at_least import has_at_least_n_rings
from ConStruct.projector.is_ring.is_ring_length_at_least import has_rings_of_length_at_least
from ConStruct.projector.projector_utils import (
    JointAtLeastProjector,
    RingCountAtLeastProjector,
    RingLengthAtLeastProjector,
)
from ConStruct.projector.constraints import resolve_constraints
from ConStruct.diffusion_model_discrete import DiscreteDenoisingDiffusion
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
    def test_constraint_normalization_variants(self):
        cases = [
            ({"constraints": []}, []),
            (
                {"constraints": [{"type": "ring_count_at_least", "min_rings": 1}]},
                [{"type": "ring_count_at_least", "min_rings": 1}],
            ),
            (
                {"constraints": [{"type": "ring_length_at_least", "min_ring_length": 4}]},
                [{"type": "ring_length_at_least", "min_ring_length": 4}],
            ),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    [spec.to_dict() for spec in resolve_constraints(OmegaConf.create(raw))],
                    expected,
                )

    def test_constraint_normalization_and_legacy_merge(self):
        cfg = OmegaConf.create({
            "rev_proj": "ring_count_at_least",
            "min_rings": 2,
            "constraints": [
                {"type": "ring_count_at_least", "min_rings": 2},
                {"type": "ring_length_at_least", "min_ring_length": 5},
            ],
        })
        specs = resolve_constraints(cfg)
        self.assertEqual(
            [spec.to_dict() for spec in specs],
            [
                {"type": "ring_count_at_least", "min_rings": 2},
                {"type": "ring_length_at_least", "min_ring_length": 5},
            ],
        )

        cfg.constraints[0].min_rings = 1
        with self.assertRaisesRegex(ValueError, "Conflicting thresholds"):
            resolve_constraints(cfg)

    def test_constraint_normalization_rejects_invalid_entries(self):
        invalid = [
            {"constraints": [{"type": "ring_count_at_least", "min_rings": 0}]},
            {"constraints": [{"type": "ring_length_at_least", "min_ring_length": 2}]},
            {"constraints": [{"type": "unknown", "min_rings": 1}]},
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                resolve_constraints(OmegaConf.create(raw))

        unsupported_mixed = OmegaConf.create({
            "rev_proj": "planar",
            "constraints": [{"type": "ring_count_at_least", "min_rings": 1}],
        })
        with self.assertRaisesRegex(ValueError, "Multiple constraints"):
            resolve_constraints(unsupported_mixed)

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

    def test_joint_projector_blocks_deletion_that_violates_either_constraint(self):
        count_graph = nx.cycle_graph(5)
        count_graph.add_edge(0, 2)
        count_state = dense_placeholder(count_graph, bond_type=3)
        count_projector = JointAtLeastProjector(
            count_state, min_rings=2, min_ring_length=5
        )
        count_proposal = count_state.copy()
        count_proposal.E = count_state.E.clone()
        count_proposal.E[0, 0, 2] = torch.tensor([1, 0, 0, 0, 0])
        count_proposal.E[0, 2, 0] = torch.tensor([1, 0, 0, 0, 0])
        count_projector.project(count_proposal)
        self.assertEqual(torch.argmax(count_proposal.E[0, 0, 2]).item(), 3)

        length_graph = nx.disjoint_union_all([
            nx.cycle_graph(4), nx.cycle_graph(3), nx.cycle_graph(3)
        ])
        length_state = dense_placeholder(length_graph, bond_type=4)
        length_projector = JointAtLeastProjector(
            length_state, min_rings=2, min_ring_length=4
        )
        length_proposal = length_state.copy()
        length_proposal.E = length_state.E.clone()
        length_proposal.E[0, 0, 1] = torch.tensor([1, 0, 0, 0, 0])
        length_proposal.E[0, 1, 0] = torch.tensor([1, 0, 0, 0, 0])
        length_projector.project(length_proposal)
        self.assertEqual(torch.argmax(length_proposal.E[0, 0, 1]).item(), 4)

    def test_combined_minimum_sampling_nodes(self):
        model = DiscreteDenoisingDiffusion.__new__(DiscreteDenoisingDiffusion)
        model.nodes_dist = DistributionNodes({2: 1, 3: 1, 4: 1, 5: 1})
        model.constraints = resolve_constraints(OmegaConf.create({
            "constraints": [
                {"type": "ring_count_at_least", "min_rings": 2},
                {"type": "ring_length_at_least", "min_ring_length": 5},
            ]
        }))
        self.assertEqual(model._minimum_feasible_sampling_nodes(), 5)

        model.constraints = resolve_constraints(OmegaConf.create({
            "constraints": [
                {"type": "ring_length_at_least", "min_ring_length": 6}
            ]
        }))
        with self.assertRaisesRegex(ValueError, "no support"):
            model._minimum_feasible_sampling_nodes()

    def test_final_joint_assertion_checks_both_constraints(self):
        model = DiscreteDenoisingDiffusion.__new__(DiscreteDenoisingDiffusion)
        model.use_projection = True
        model.cfg = OmegaConf.create({
            "model": {
                "constraints": [
                    {"type": "ring_count_at_least", "min_rings": 2},
                    {"type": "ring_length_at_least", "min_ring_length": 5},
                ]
            }
        })
        model.constraints = resolve_constraints(model.cfg.model)

        valid_graph = nx.cycle_graph(5)
        valid_graph.add_edge(0, 2)
        model._assert_final_constraint(dense_placeholder(valid_graph).collapse(torch.tensor([])))
        with self.assertRaisesRegex(AssertionError, "ring_count_at_least"):
            model._assert_final_constraint(
                dense_placeholder(nx.cycle_graph(5)).collapse(torch.tensor([]))
            )

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
        count = ring_count_at_least_satisfaction_ratio(generated, 1)
        length = ring_length_at_least_satisfaction_ratio(generated, 5)
        self.assertEqual(
            joint_constraint_satisfaction_ratio(count, length).tolist(), [1.0, 0.0]
        )
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
