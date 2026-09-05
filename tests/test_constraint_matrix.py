import json
import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import torch

from ConStruct.analysis.run_constraint_matrix import (
    METRICS,
    PROFILE_BY_NAME,
    PROFILES,
    ArtifactMismatchError,
    _retry,
    evaluate_batches,
    generation_command,
    load_trimmed_batches,
    render_aggregate,
    validate_generation_artifacts,
)
from ConStruct.utils import PlaceHolder


def graph_batch(graphs):
    max_nodes = max(graph.number_of_nodes() for graph in graphs)
    batch_size = len(graphs)
    x = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    charges = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    edges = torch.full((batch_size, max_nodes, max_nodes), -1, dtype=torch.long)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    for index, graph in enumerate(graphs):
        nodes = graph.number_of_nodes()
        node_mask[index, :nodes] = True
        edges[index, :nodes, :nodes] = 0
        for left, right in graph.edges():
            edges[index, left, right] = 1
            edges[index, right, left] = 1
    return PlaceHolder(
        X=x,
        charges=charges,
        E=edges,
        y=torch.zeros((batch_size, 0)),
        node_mask=node_mask,
    )


def artifact_metadata(checkpoint, profile, samples, seed=0):
    return {
        "checkpoint": str(checkpoint.resolve()),
        "constraints": profile.constraints,
        "dataset": "qm9",
        "device_count": 1,
        "generated_samples": samples,
        "projection_enabled": profile.projection_enabled,
        "requested_samples": samples,
        "seed": seed,
    }


class ConstraintMatrixTests(unittest.TestCase):
    def test_predefined_grid_contains_all_nine_profiles(self):
        self.assertEqual(len(PROFILES), 9)
        self.assertEqual(
            {(profile.min_rings, profile.min_ring_length) for profile in PROFILES},
            {
                (None, None), (None, 4), (None, 5),
                (1, None), (1, 4), (1, 5),
                (2, None), (2, 4), (2, 5),
            },
        )

    def test_cross_evaluation_computes_all_five_metrics(self):
        graph_with_two_cycles = nx.cycle_graph(5)
        graph_with_two_cycles.add_edge(0, 2)
        batch = graph_batch(
            [nx.path_graph(2), nx.cycle_graph(3), graph_with_two_cycles]
        )

        result = evaluate_batches([batch], ["C", "N", "O", "F"])

        self.assertEqual(result["num_graphs"], 3)
        self.assertAlmostEqual(result["metrics"]["molecular_validity_pct"], 100.0)
        self.assertAlmostEqual(
            result["metrics"]["ring_count_at_least_1_pct"], 200 / 3
        )
        self.assertAlmostEqual(
            result["metrics"]["ring_count_at_least_2_pct"], 100 / 3
        )
        self.assertAlmostEqual(
            result["metrics"]["ring_length_at_least_4_pct"], 100 / 3
        )
        self.assertAlmostEqual(
            result["metrics"]["ring_length_at_least_5_pct"], 100 / 3
        )

    def test_artifact_validation_and_exact_trimming(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "epoch=1.ckpt"
            checkpoint.touch()
            profile = PROFILE_BY_NAME["count_1_length_disabled"]
            sample_dir = root / profile.name / "seed_0"
            sample_dir.mkdir(parents=True)
            batches = [
                graph_batch([nx.path_graph(2), nx.cycle_graph(3)]),
                graph_batch([nx.cycle_graph(4), nx.cycle_graph(5)]),
            ]
            with (sample_dir / "generated_samples_rank0.pkl").open("wb") as handle:
                pickle.dump(batches, handle)
            with (sample_dir / "sampling_metrics.json").open("w") as handle:
                json.dump(
                    {"metadata": artifact_metadata(checkpoint, profile, 3)}, handle
                )

            metadata = validate_generation_artifacts(
                sample_dir, profile, checkpoint, seed=0, requested_samples=3
            )
            self.assertEqual(metadata["generated_samples"], 3)
            trimmed = load_trimmed_batches(sample_dir, requested_samples=3)
            self.assertEqual(sum(batch.X.shape[0] for batch in trimmed), 3)

            with self.assertRaises(ArtifactMismatchError):
                validate_generation_artifacts(
                    sample_dir,
                    PROFILE_BY_NAME["count_2_length_disabled"],
                    checkpoint,
                    seed=0,
                    requested_samples=3,
                )

    def test_generation_command_quotes_checkpoint_equals(self):
        command = generation_command(
            Path("/app"),
            "training/qm9_no_constraint_edge_addition",
            PROFILE_BY_NAME["count_1_length_disabled"],
            Path("/tmp/epoch=389.ckpt"),
            0,
            10000,
            Path("/tmp/samples"),
        )
        override = next(item for item in command if item.startswith("general.test_only"))
        self.assertEqual(override, "general.test_only='/tmp/epoch=389.ckpt'")

    def test_retry_writes_structured_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            error_path = Path(temporary) / "errors" / "metrics.json"

            def fail(_attempt):
                raise ValueError("deterministic failure")

            with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
                _retry(
                    fail,
                    attempts=2,
                    error_path=error_path,
                    stage="metrics",
                    profile=PROFILES[0],
                )
            payload = json.loads(error_path.read_text())
            self.assertEqual(payload["attempts"], 2)
            self.assertEqual(payload["error_type"], "ValueError")
            self.assertIn("deterministic failure", payload["traceback"])

    def test_aggregate_writes_json_csv_grids_and_heatmaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.touch()
            cells = {}
            for index, profile in enumerate(PROFILES):
                cells[profile.name] = {
                    "profile": profile.name,
                    "enforced_min_rings": profile.min_rings,
                    "enforced_min_ring_length": profile.min_ring_length,
                    "num_graphs": 10,
                    "metrics": {
                        metric: float(index * 10) for metric, _ in METRICS
                    },
                }

            render_aggregate(root, cells, checkpoint, seed=0, requested_samples=10)

            self.assertTrue((root / "metrics.json").is_file())
            self.assertTrue((root / "metrics.csv").is_file())
            payload = json.loads((root / "metrics.json").read_text())
            self.assertEqual(len(payload["cells"]), 9)
            self.assertEqual(len(payload["grids"]), 5)
            for metric, _ in METRICS:
                slug = metric.removesuffix("_pct")
                self.assertTrue((root / "grids" / f"{slug}.md").is_file())
                self.assertGreater((root / "heatmaps" / f"{slug}.png").stat().st_size, 0)
                self.assertGreater((root / "heatmaps" / f"{slug}.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
