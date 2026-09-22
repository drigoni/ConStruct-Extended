"""Conditional classes must preserve topology without retaining stale classes."""
import unittest
from unittest.mock import patch

import networkx as nx
import torch

from test_edge_addition_at_least import dense_placeholder
from ConStruct.projector.projector_utils import (
    JointAtLeastProjector, RingCountAtLeastProjector, RingLengthAtLeastProjector,
)


class BlockedDeletionResamplingTests(unittest.TestCase):
    def projectors(self, state):
        return [
            RingCountAtLeastProjector(state, 1),
            RingLengthAtLeastProjector(state, 3),
            JointAtLeastProjector(state, 1, 3),
        ]

    def delete(self, graph, edge):
        proposal = dense_placeholder(graph, bond_type=2)
        u, v = edge
        proposal.E[0, u, v] = proposal.E[0, v, u] = torch.tensor([1., 0, 0, 0, 0])
        return proposal

    def test_new_and_cached_blocks_resample_and_sync_state(self):
        graph = nx.cycle_graph(3)
        state = dense_placeholder(graph)
        for projector in self.projectors(state):
            with self.subTest(projector=type(projector).__name__):
                for selected_class in (1, 4):
                    proposal = self.delete(graph, (0, 1))
                    probabilities = torch.zeros_like(proposal.E)
                    probabilities[..., 0] = .99
                    probabilities[..., selected_class] = .01
                    projector.project(proposal, probabilities, True)
                    self.assertEqual(proposal.E[0, 0, 1].argmax().item(), selected_class)
                    torch.testing.assert_close(proposal.E, proposal.E.transpose(1, 2))
                    torch.testing.assert_close(projector.z_t_E, proposal.E)
                    self.assertTrue(projector.valid_graph_fn(projector.nx_graphs_list[0]))
                self.assertEqual(projector.total_blocked, 2)

    def test_renormalizes_positive_probabilities(self):
        graph = nx.cycle_graph(3)
        proposal = self.delete(graph, (0, 1))
        probabilities = torch.zeros_like(proposal.E)
        probabilities[0, 0, 1] = torch.tensor([.9, .02, .03, .05, 0])
        observed = []
        original_log = torch.Tensor.log
        def capture_log(tensor):
            observed.append(tensor.clone())
            return original_log(tensor)
        with patch.object(torch.Tensor, 'log', capture_log):
            RingCountAtLeastProjector(dense_placeholder(graph), 1).project(
                proposal, probabilities, True
            )
        torch.testing.assert_close(observed[0], torch.tensor([.2, .3, .5, 0]))
        self.assertIn(proposal.E[0, 0, 1].argmax().item(), (1, 2, 3))

    def test_zero_mass_and_legacy_keep_previous_class(self):
        graph = nx.cycle_graph(3)
        for enabled in (False, True):
            proposal = self.delete(graph, (0, 1))
            probabilities = torch.zeros_like(proposal.E)
            probabilities[..., 0] = 1
            projector = RingCountAtLeastProjector(dense_placeholder(graph), 1)
            projector.project(proposal, probabilities, enabled)
            self.assertEqual(proposal.E[0, 0, 1].argmax().item(), 2)

    def test_allowed_deletion_and_positive_update_remain_unchanged(self):
        graph = nx.cycle_graph(3)
        graph.add_edge(2, 3)
        proposal = self.delete(graph, (2, 3))
        proposal.E[0, 0, 1] = proposal.E[0, 1, 0] = torch.tensor([0., 0, 0, 1, 0])
        expected = proposal.E.clone()
        probabilities = torch.zeros_like(proposal.E)
        probabilities[..., 1] = 1
        RingCountAtLeastProjector(dense_placeholder(graph), 1).project(
            proposal, probabilities, True
        )
        torch.testing.assert_close(proposal.E, expected)

    def test_padding_and_diagonal_stay_masked(self):
        graph = nx.cycle_graph(3)
        graph.add_node(3)
        state = dense_placeholder(graph).mask()
        state.node_mask[0, 3] = False
        state.mask()
        proposal = self.delete(graph, (0, 1))
        proposal.node_mask = state.node_mask.clone()
        proposal.mask()
        probabilities = torch.zeros_like(proposal.E)
        probabilities[..., 1] = 1
        RingCountAtLeastProjector(state, 1).project(proposal, probabilities, True)
        self.assertEqual(proposal.E[0, 3].count_nonzero().item(), 0)
        self.assertEqual(proposal.E[0, :, 3].count_nonzero().item(), 0)
        self.assertEqual(proposal.E[0].diagonal(dim1=0, dim2=1).count_nonzero().item(), 0)

    def test_missing_probabilities_fail_before_mutation(self):
        graph = nx.cycle_graph(3)
        proposal = self.delete(graph, (0, 1))
        with self.assertRaisesRegex(ValueError, 'matching edge probabilities'):
            RingCountAtLeastProjector(dense_placeholder(graph), 1).project(
                proposal, resample_blocked_deletions=True
            )


if __name__ == '__main__':
    unittest.main()
